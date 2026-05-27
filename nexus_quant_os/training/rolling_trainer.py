"""
training/rolling_trainer.py — 滾動窗口再訓練系統
===================================================

讓模型持續適應最新市場，而非只學過一個時代。

核心機制：
    使用 5 年訓練 + 1 年驗證的滾動窗口，每季步進一次。

    Window 1: [2010 ─── Train ─── 2015] → [2015 ─ Val ─ 2016]
    Window 2: [2011 ─── Train ─── 2016] → [2016 ─ Val ─ 2017]
    ...

    回測時，在時間 T 使用 val_end ≤ T 的最新 Checkpoint。

Author : Nexus Quant OS — Model Training Division
"""

from __future__ import annotations

import logging
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.training.train_moe import (
    ASSET_UNIVERSE,
    N_FEATURES,
    FEATURE_COLS,
    FRED_API_KEY,
    TRAIN_RATIO,
    INPUT_DIM,
    OUTPUT_DIM,
    HIDDEN_DIM,
    NUM_EXPERTS,
    TOP_K,
    AUX_LOSS_COEFF,
    BATCH_SIZE,
    EPOCHS,
    build_daily_dataset,
    build_raw_daily_returns,
    train_moe_router,
    save_checkpoint,
    load_checkpoint,
    compute_val_metrics,
)

logger = logging.getLogger("nexus_quant_os.training.rolling_trainer")


# ═════════════════════════════════════════════════════════════════════
# 數據結構
# ═════════════════════════════════════════════════════════════════════

@dataclass
class TrainWindow:
    """描述一個滾動訓練窗口。"""
    window_id: int
    train_start: str          # 'YYYY-MM-DD'
    train_end: str
    val_start: str
    val_end: str

    @property
    def train_days(self) -> int:
        return (pd.Timestamp(self.train_end) - pd.Timestamp(self.train_start)).days

    @property
    def val_days(self) -> int:
        return (pd.Timestamp(self.val_end) - pd.Timestamp(self.val_start)).days

    def __repr__(self) -> str:
        return (
            f"Window-{self.window_id}: "
            f"Train[{self.train_start} → {self.train_end}] "
            f"Val[{self.val_start} → {self.val_end}]"
        )


@dataclass
class WindowResult:
    """單一窗口的訓練結果。"""
    window: TrainWindow
    val_sharpe: float = 0.0
    val_mse: float = 0.0
    val_hit_rate: float = 0.0
    checkpoint_path: Optional[Path] = None
    train_samples: int = 0
    val_samples: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.checkpoint_path is not None


# ═════════════════════════════════════════════════════════════════════
# 滾動訓練器
# ═════════════════════════════════════════════════════════════════════

class RollingTrainer:
    """滾動窗口再訓練系統。

    Usage
    -----
    >>> trainer = RollingTrainer(train_window_years=5, val_window_years=1)
    >>> windows = trainer.generate_windows('2010-01-01', '2025-01-01')
    >>> results = trainer.train_all_windows(daily_prices, macro_data)
    """

    def __init__(
        self,
        train_window_years: int = 5,
        val_window_years: int = 1,
        step_months: int = 3,
        asset_universe: list[str] | None = None,
        checkpoint_dir: Path | None = None,
        epochs: int = 150,
        batch_size: int = 64,
    ) -> None:
        self.train_window_years = train_window_years
        self.val_window_years = val_window_years
        self.step_months = step_months
        self.asset_universe = asset_universe or ASSET_UNIVERSE
        self.epochs = epochs
        self.batch_size = batch_size

        # Checkpoint 目錄
        if checkpoint_dir is None:
            self.checkpoint_dir = (
                _PROJECT_ROOT / "nexus_quant_os" / "models" / "checkpoints" / "rolling"
            )
        else:
            self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._windows: list[TrainWindow] = []
        self._results: list[WindowResult] = []

    def generate_windows(
        self,
        start_date: str,
        end_date: str,
    ) -> list[TrainWindow]:
        """生成滾動訓練/驗證窗口序列。

        Parameters
        ----------
        start_date : str
            第一個訓練窗口的起始日期。
        end_date : str
            最後一個驗證窗口不可超過此日期。

        Returns
        -------
        list[TrainWindow]
            按時間順序的窗口列表。
        """
        windows: list[TrainWindow] = []
        current_start = pd.Timestamp(start_date)
        final_date = pd.Timestamp(end_date)
        window_id = 0

        while True:
            train_end = current_start + pd.DateOffset(years=self.train_window_years)
            val_start = train_end
            val_end = val_start + pd.DateOffset(years=self.val_window_years)

            # 超出總範圍 → 停止
            if val_end > final_date:
                break

            windows.append(TrainWindow(
                window_id=window_id,
                train_start=current_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
            ))

            window_id += 1
            current_start += pd.DateOffset(months=self.step_months)

        self._windows = windows
        logger.info("生成 %d 個滾動窗口", len(windows))
        for w in windows:
            logger.info("  %s", w)

        return windows

    def _prepare_aligned_data(
        self,
        daily_prices: pd.DataFrame,
        macro_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """PiT 對齊所有資產（與 train_moe.py 一致）。"""
        frames = []
        for asset in sorted(self.asset_universe):
            ap = daily_prices[daily_prices["asset_id"] == asset].copy()
            am = macro_data[macro_data["asset_id"] == asset].copy()

            if ap.empty or am.empty:
                logger.warning("資產 %s 無數據，跳過", asset)
                continue

            aligned, _ = enforce_pit_alignment(
                daily_prices=ap,
                macro_fundamental_data=am,
                timestamp_col="timestamp",
                asset_col="asset_id",
                max_drift_days=45,
                drop_unmatched=True,
                preserve_right_timestamp=True,
            )
            frames.append(aligned)

        aligned_df = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["asset_id", "timestamp"])
            .reset_index(drop=True)
        )
        return aligned_df

    def _slice_aligned_data(
        self,
        aligned_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """依時間範圍切片已對齊的數據。"""
        mask = (
            (aligned_df["timestamp"] >= pd.Timestamp(start_date)) &
            (aligned_df["timestamp"] < pd.Timestamp(end_date))
        )
        return aligned_df[mask].copy().reset_index(drop=True)

    def _train_single_window(
        self,
        window: TrainWindow,
        aligned_df: pd.DataFrame,
    ) -> WindowResult:
        """訓練單一窗口。

        Parameters
        ----------
        window : TrainWindow
            當前窗口描述。
        aligned_df : pd.DataFrame
            完整已對齊數據。

        Returns
        -------
        WindowResult
            訓練結果。
        """
        t0 = time.time()
        logger.info("開始訓練 %s", window)

        try:
            # ── 切片訓練和驗證數據 ────────────────────────────────
            train_df = self._slice_aligned_data(
                aligned_df, window.train_start, window.train_end,
            )
            val_df = self._slice_aligned_data(
                aligned_df, window.val_start, window.val_end,
            )

            if train_df.empty or val_df.empty:
                return WindowResult(
                    window=window,
                    error=f"數據不足：train={len(train_df)} val={len(val_df)}",
                    elapsed_seconds=time.time() - t0,
                )

            # ── 構建數據集 ────────────────────────────────────────
            # 合併訓練和驗證數據（build_daily_dataset 自行切割）
            combined = pd.concat([train_df, val_df], ignore_index=True)
            combined = combined.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

            X, y_norm, y_raw, dates, assets = build_daily_dataset(combined)

            if len(X) < 50:
                return WindowResult(
                    window=window,
                    error=f"數據點過少：{len(X)} < 50",
                    elapsed_seconds=time.time() - t0,
                )

            # 找到訓練/驗證的分割點
            train_end_ts = pd.Timestamp(window.train_end)
            split_mask = dates < train_end_ts
            split_idx = int(split_mask.sum())

            if split_idx < 20 or (len(X) - split_idx) < 10:
                return WindowResult(
                    window=window,
                    error=f"分割後數據不足：train={split_idx} val={len(X)-split_idx}",
                    elapsed_seconds=time.time() - t0,
                )

            X_train = X[:split_idx]
            y_train = y_norm[:split_idx]
            y_raw_train = y_raw[:split_idx]
            X_val = X[split_idx:]
            y_val = y_norm[split_idx:]
            y_raw_val = y_raw[split_idx:]

            logger.info(
                "  Window-%d 數據：train=%d val=%d assets=%s",
                window.window_id, len(X_train), len(X_val), assets,
            )

            # ── 訓練 ─────────────────────────────────────────────
            router, scaler = train_moe_router(
                X_train, y_train, y_raw_train,
                X_val, y_val, y_raw_val,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )

            # ── 驗證指標 ──────────────────────────────────────────
            val_m = compute_val_metrics(
                router, X_val, y_val, y_raw_val,
                scaler, torch.device("cpu"),
            )

            # ── 儲存 Checkpoint ───────────────────────────────────
            ckpt_name = (
                f"rolling_w{window.window_id:03d}_"
                f"{window.train_start.replace('-','')}_{window.val_end.replace('-','')}.pt"
            )
            ckpt_path = self.checkpoint_dir / ckpt_name
            save_checkpoint(
                router, scaler, assets, val_m["sharpe"],
                path=ckpt_path,
            )

            elapsed = time.time() - t0
            result = WindowResult(
                window=window,
                val_sharpe=val_m["sharpe"],
                val_mse=val_m["mse"],
                val_hit_rate=val_m["hit_rate"],
                checkpoint_path=ckpt_path,
                train_samples=len(X_train),
                val_samples=len(X_val),
                elapsed_seconds=elapsed,
            )

            logger.info(
                "  Window-%d 完成：Sharpe=%.4f  MSE=%.6f  "
                "HitRate=%.1f%%  耗時=%.1fs",
                window.window_id, val_m["sharpe"], val_m["mse"],
                val_m["hit_rate"] * 100, elapsed,
            )
            return result

        except Exception as e:
            logger.exception("Window-%d 訓練失敗", window.window_id)
            return WindowResult(
                window=window,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )

    def train_all_windows(
        self,
        daily_prices: pd.DataFrame,
        macro_data: pd.DataFrame,
    ) -> list[WindowResult]:
        """依序訓練所有窗口。

        Parameters
        ----------
        daily_prices : pd.DataFrame
            完整歷史價格數據。
        macro_data : pd.DataFrame
            完整歷史總經數據。

        Returns
        -------
        list[WindowResult]
            每個窗口的訓練結果。
        """
        if not self._windows:
            raise ValueError("請先呼叫 generate_windows() 生成窗口")

        SEP = "═" * 72
        print(f"\n{SEP}")
        print(f"  滾動訓練開始：{len(self._windows)} 個窗口")
        print(f"  訓練窗口：{self.train_window_years} 年")
        print(f"  驗證窗口：{self.val_window_years} 年")
        print(f"  步進：每 {self.step_months} 個月")
        print(f"{SEP}\n")

        # 一次性對齊所有數據
        print(">>> 對齊所有資產數據...")
        aligned_df = self._prepare_aligned_data(daily_prices, macro_data)
        print(f"    對齊後共 {len(aligned_df):,} rows\n")

        results: list[WindowResult] = []
        for i, window in enumerate(self._windows):
            print(f"\n{'─' * 72}")
            print(f"  訓練窗口 {i+1}/{len(self._windows)}: {window}")
            print(f"{'─' * 72}")

            result = self._train_single_window(window, aligned_df)
            results.append(result)

            if result.success:
                print(f"  ✅ Sharpe={result.val_sharpe:+.4f}  "
                      f"HitRate={result.val_hit_rate:.1%}  "
                      f"耗時={result.elapsed_seconds:.1f}s")
            else:
                print(f"  ❌ 失敗：{result.error}")

        self._results = results
        self._print_summary_table(results)

        return results

    def _print_summary_table(self, results: list[WindowResult]) -> None:
        """輸出訓練結果摘要表。"""
        SEP = "═" * 72

        print(f"\n{SEP}")
        print("  滾動訓練結果摘要")
        print(f"{SEP}")

        header = (
            f"  {'Window':>8}  {'Period':<25}  "
            f"{'Sharpe':>8}  {'MSE':>8}  {'Hit%':>6}  {'Status':>8}"
        )
        print(header)
        print("  " + "─" * 68)

        sharpes = []
        for r in results:
            period = f"{r.window.train_start}→{r.window.val_end}"
            if r.success:
                print(
                    f"  W-{r.window.window_id:>3d}    {period:<25}  "
                    f"{r.val_sharpe:>+8.4f}  {r.val_mse:>8.6f}  "
                    f"{r.val_hit_rate*100:>5.1f}%  {'✅':>8}"
                )
                sharpes.append(r.val_sharpe)
            else:
                print(
                    f"  W-{r.window.window_id:>3d}    {period:<25}  "
                    f"{'N/A':>8}  {'N/A':>8}  {'N/A':>6}  {'❌':>8}"
                )

        if sharpes:
            print(f"\n  平均 Sharpe : {np.mean(sharpes):+.4f}")
            print(f"  最佳 Sharpe : {np.max(sharpes):+.4f}")
            print(f"  最差 Sharpe : {np.min(sharpes):+.4f}")
            print(f"  成功率     : {len(sharpes)}/{len(results)}")

    def rolling_backtest(
        self,
        daily_prices: pd.DataFrame,
        macro_data: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> dict:
        """使用滾動訓練的 Checkpoint 進行走前回測。

        核心規則：在時間 T，只使用 val_end ≤ T 的最新 Checkpoint。

        Parameters
        ----------
        daily_prices : pd.DataFrame
            完整價格數據。
        macro_data : pd.DataFrame
            完整總經數據。
        start_date : str
            回測起始日期。
        end_date : str
            回測結束日期。

        Returns
        -------
        dict
            回測績效指標。
        """
        if not self._results:
            raise ValueError("請先呼叫 train_all_windows() 訓練所有窗口")

        # 過濾出成功的結果，按 val_end 排序
        valid_results = [
            r for r in self._results if r.success
        ]
        valid_results.sort(key=lambda r: r.window.val_end)

        if not valid_results:
            raise ValueError("沒有成功的訓練結果可用於回測")

        SEP = "═" * 72
        print(f"\n{SEP}")
        print(f"  滾動回測：{start_date} → {end_date}")
        print(f"  可用 Checkpoint：{len(valid_results)} 個")
        print(f"{SEP}\n")

        # ── 對齊數據 ──────────────────────────────────────────────
        aligned_df = self._prepare_aligned_data(daily_prices, macro_data)

        # 切片回測期
        bt_df = self._slice_aligned_data(aligned_df, start_date, end_date)
        X, y_norm, y_raw, dates, assets = build_daily_dataset(bt_df)

        if len(X) == 0:
            raise ValueError("回測期無足夠數據")

        # ── 預載入所有模型 ────────────────────────────────────────
        models = {}
        for r in valid_results:
            assert r.checkpoint_path is not None
            router, scaler, _, _ = load_checkpoint(r.checkpoint_path)
            models[r.window.val_end] = (router, scaler)

        # ── 逐日回測 ──────────────────────────────────────────────
        daily_returns: list[float] = []
        model_switches = 0
        current_model_key = None

        for t in range(len(X)):
            date = dates[t]
            date_str = date.strftime("%Y-%m-%d")

            # 找到 val_end ≤ date 的最新 Checkpoint
            eligible = [
                key for key in models.keys()
                if pd.Timestamp(key) <= date
            ]

            if not eligible:
                # 沒有可用的 Checkpoint → 等權配置
                daily_returns.append(float(y_raw[t].mean()))
                continue

            best_key = max(eligible)  # 最近的
            if best_key != current_model_key:
                model_switches += 1
                current_model_key = best_key

            router, scaler = models[best_key]

            # 推論
            x_scaled = scaler.transform(X[t:t+1]).astype(np.float32)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            with torch.no_grad():
                out = router(x_tensor)
                w = out.combined_output[0].numpy()

            # 正規化
            abs_sum = float(np.abs(w).sum()) + 1e-8
            w_norm = w / abs_sum

            # 計算日報酬
            port_ret = float((w_norm * y_raw[t]).sum())
            daily_returns.append(port_ret)

        # ── 計算績效 ──────────────────────────────────────────────
        returns = np.array(daily_returns)
        total_ret = float(np.prod(1 + returns) - 1)
        n_days = len(returns)
        ann_ret = float(returns.mean() * 252)
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = ann_ret / (ann_vol + 1e-8)

        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        max_dd = float(dd.min())

        cagr = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)

        results = {
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "model_switches": model_switches,
            "n_days": n_days,
            "ann_vol": ann_vol,
        }

        print(f"\n  滾動回測結果：")
        print(f"    總報酬  : {total_ret:+.2%}")
        print(f"    CAGR    : {cagr:+.2%}")
        print(f"    Sharpe  : {sharpe:+.4f}")
        print(f"    Max DD  : {max_dd:.2%}")
        print(f"    模型切換: {model_switches} 次")
        print(f"    交易日  : {n_days}")

        return results
