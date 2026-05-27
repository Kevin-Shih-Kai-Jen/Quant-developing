"""
backtest.py — Nexus Quant OS Walk-Forward Backtest
====================================================

在驗證集（Out-of-Sample）上逐日模擬投資組合，輸出：
    - 總報酬、CAGR、年化 Sharpe、最大回撤、Calmar、勝率
    - SPY 買進持有 基準對比
    - 月度報酬熱力圖（文字版）
    - 文字版股權曲線

執行方式（Docker 容器內）：
    docker compose run --rm quant_engine python backtest.py

Author : Nexus Quant OS — Quant Research
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.training.train_moe import (
    ASSET_UNIVERSE, DATA_START, FRED_API_KEY, TRAIN_RATIO,
    build_daily_dataset, build_raw_daily_returns,
    find_latest_checkpoint, load_checkpoint,
)
from nexus_quant_os.portfolio.weight_smoother import WeightSmoother, SmootherConfig
from nexus_quant_os.portfolio.regime_allocator import RegimeAllocator, RegimeAllocatorConfig
from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer, OptimizerConfig


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("nexus_quant_os.backtest")

# ─── 回測參數 ────────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS = 5      # 5 bps 單邊交易成本（保守估計）
TRADING_DAYS_PER_YEAR = 252
GROSS_EXPOSURE = 1.0           # 總絕對權重 = 1（市場中性格局）


# ─── 績效指標計算 ─────────────────────────────────────────────────────────────

def compute_metrics(returns: np.ndarray, label: str = "Strategy") -> dict:
    """從每日報酬序列計算標準化績效指標。"""
    n = len(returns)
    if n == 0:
        return {}

    total_ret   = float(np.prod(1 + returns) - 1)
    cagr        = float((1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    ann_ret     = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol     = float(returns.std()  * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe      = ann_ret / (ann_vol + 1e-8)

    equity      = np.cumprod(1 + returns)
    peak        = np.maximum.accumulate(equity)
    dd          = (equity - peak) / (peak + 1e-8)
    max_dd      = float(dd.min())
    calmar      = cagr / (abs(max_dd) + 1e-8)

    win_rate    = float((returns > 0).mean())
    best_day    = float(returns.max())
    worst_day   = float(returns.min())

    return {
        "label":        label,
        "n_days":       n,
        "total_return": total_ret,
        "cagr":         cagr,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "calmar":       calmar,
        "win_rate":     win_rate,
        "best_day":     best_day,
        "worst_day":    worst_day,
        "equity":       equity,
    }


# ─── 文字版股權曲線 ────────────────────────────────────────────────────────────

def print_equity_curve(
    equity_strat: np.ndarray,
    equity_spy:   np.ndarray,
    dates:        pd.DatetimeIndex,
    width:        int = 60,
    height:       int = 12,
) -> None:
    """Print ASCII equity curve for strategy vs SPY."""
    # 月末取樣點（降低密度）
    df = pd.DataFrame({
        "strat": equity_strat,
        "spy":   equity_spy,
    }, index=dates)
    monthly = df.resample("ME").last()

    strat_vals = monthly["strat"].values
    spy_vals   = monthly["spy"].values
    date_labels = monthly.index

    all_vals = np.concatenate([strat_vals, spy_vals])
    y_min, y_max = all_vals.min() * 0.98, all_vals.max() * 1.02
    y_range = y_max - y_min + 1e-8

    n_pts = len(strat_vals)
    canvas = [[" "] * (width + 6) for _ in range(height + 2)]

    def map_y(v):
        return height - 1 - int(round((v - y_min) / y_range * (height - 1)))

    def map_x(i):
        return int(round(i / max(n_pts - 1, 1) * (width - 1))) + 5

    # Plot SPY (dots) and strategy (stars)
    for i in range(n_pts - 1):
        x0, x1 = map_x(i), map_x(i + 1)
        y0s, y1s = map_y(strat_vals[i]),  map_y(strat_vals[i + 1])
        y0b, y1b = map_y(spy_vals[i]),    map_y(spy_vals[i + 1])
        for x in range(x0, x1 + 1):
            alpha = (x - x0) / max(x1 - x0, 1)
            ys = int(y0s + alpha * (y1s - y0s))
            yb = int(y0b + alpha * (y1b - y0b))
            if 0 <= ys < height and 5 <= x < width + 5:
                canvas[ys][x] = "*"
            if 0 <= yb < height and 5 <= x < width + 5:
                if canvas[yb][x] == " ":
                    canvas[yb][x] = "."

    # Y-axis labels
    for row in range(height):
        v = y_max - row / (height - 1) * y_range
        canvas[row][0] = f"{v:.2f}"[:4]

    SEP = "─" * 72
    print(f"\n  Equity Curve  ── ★ Strategy  ·· SPY Buy-and-Hold\n")
    print("  " + SEP)
    for canvas_row in canvas[:-1]:
        print("  " + "".join(str(c) for c in canvas_row))

    # X-axis: quarter labels
    x_labels = " " * 5
    quarter_idx = [i for i, d in enumerate(date_labels) if d.month in (1, 4, 7, 10)]
    prev_x = -99
    for qi in quarter_idx:
        x = map_x(qi)
        gap = x - prev_x
        if gap >= 8:
            label = date_labels[qi].strftime("%b'%y")
            x_labels += " " * max(0, x - len(x_labels)) + label
            prev_x = x + len(label)
    print("  " + x_labels)
    print("  " + SEP)


# ─── 月度報酬表 ────────────────────────────────────────────────────────────────

def print_monthly_returns(
    strategy_returns: np.ndarray,
    dates:            pd.DatetimeIndex,
) -> None:
    """Print monthly return table."""
    df = pd.DataFrame({"ret": strategy_returns}, index=dates)
    monthly = (df + 1).resample("ME").prod() - 1

    years  = sorted(monthly.index.year.unique())
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    print("\n  Monthly Returns (Strategy)\n")
    header = f"  {'Year':>4s}  " + "  ".join(f"{m:>5s}" for m in months) + "  Annual"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for year in years:
        row = f"  {year:>4d}  "
        annual = 1.0
        for m in range(1, 13):
            mask = (monthly.index.year == year) & (monthly.index.month == m)
            val  = monthly.loc[mask, "ret"].values
            if len(val) > 0:
                r = float(val[0])
                annual *= (1 + r)
                sign = "+" if r >= 0 else ""
                row += f"{sign}{r*100:.1f}%  "
            else:
                row += "  ---  "
        ann = annual - 1
        sign = "+" if ann >= 0 else ""
        row += f"{sign}{ann*100:.1f}%"
        print(row)


# ─── 主回測函數 ────────────────────────────────────────────────────────────────

def run_backtest(start_date: str | None = None, end_date: str | None = None) -> None:
    SEP  = "═" * 72
    THIN = "─" * 72

    bt_start = start_date or DATA_START
    bt_end   = end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    print(f"\n{SEP}")
    print("  NEXUS QUANT OS — Walk-Forward Backtest")
    print(f"  Assets    : {ASSET_UNIVERSE}")
    print(f"  Data      : {bt_start} → {bt_end}")
    print(f"  TC        : {TRANSACTION_COST_BPS} bps/trade")
    print(f"  Split     : {int(TRAIN_RATIO*100)}% train / {100 - int(TRAIN_RATIO*100)}% OOS")
    print(f"{SEP}\n")

    # ── 1. 載入數據 ──────────────────────────────────────────────────
    print(">>> [1/5] 載入真實市場數據...")
    if not FRED_API_KEY:
        print("❌ FRED_API_KEY 未設定")
        sys.exit(1)

    daily_prices, macro_data = load_all_data(
        tickers=ASSET_UNIVERSE,
        fred_api_key=FRED_API_KEY,
        start=bt_start,
        end=bt_end,
    )

    # ── 1b. 自動過濾資產（處理早期年份部分資產尚未上市） ──────────
    available_assets = sorted(daily_prices["asset_id"].unique().tolist())
    # 只保留有足夠數據（至少 252 交易日 = 1 年）的資產
    valid_assets = []
    for asset in available_assets:
        n_days = len(daily_prices[daily_prices["asset_id"] == asset])
        if n_days >= 252:
            valid_assets.append(asset)
        else:
            print(f"    ⚠️ {asset}: 僅 {n_days} 天數據，不足 1 年，排除")
    if not valid_assets:
        print("❌ 無任何資產有足夠數據")
        sys.exit(1)
    if set(valid_assets) != set(ASSET_UNIVERSE):
        excluded = set(ASSET_UNIVERSE) - set(valid_assets)
        print(f"    📋 有效資產: {valid_assets}")
        print(f"    🚫 排除（數據不足）: {excluded}")
    # 用 valid_assets 取代全域 ASSET_UNIVERSE（回測期間限定）
    bt_assets = valid_assets

    # ── 2. PiT 對齊 ───────────────────────────────────────────────────
    print(">>> [2/5] PiT 對齊...")
    frames = []
    for asset in bt_assets:
        ap = daily_prices[daily_prices["asset_id"] == asset].copy()
        am = macro_data[macro_data["asset_id"] == asset].copy()
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

    # ── 3. 構建數據集 ─────────────────────────────────────────────────
    print(">>> [3/5] 構建橫截面特徵 + 原始報酬...")
    X, y_norm, _, dates, assets = build_daily_dataset(aligned_df)
    y_raw, dates_raw, _      = build_raw_daily_returns(aligned_df)

    # 確保日期對齊
    split_idx = int(len(X) * TRAIN_RATIO)
    X_val     = X[split_idx:]
    y_raw_val = y_raw[split_idx:]
    dates_val = dates[split_idx:]

    spy_idx   = assets.index("SPY") if "SPY" in assets else 0
    if split_idx > 0:
        print(f"    訓練期 : {dates[:split_idx][0].date()} ~ {dates[:split_idx][-1].date()} "
              f"({split_idx} 天)")
    else:
        print(f"    訓練期 : 無 (100% OOS 測試)")
        
    print(f"    驗證期 : {dates_val[0].date()} ~ {dates_val[-1].date()} "
          f"({len(X_val)} 天)")

    # ── 4. 載入 Checkpoint ────────────────────────────────────────────
    print(">>> [4/5] 載入訓練好的 Router...")
    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("❌ 未找到 checkpoint，請先執行訓練腳本")
        sys.exit(1)

    router, scaler, trained_assets, firewall = load_checkpoint(ckpt_path)
    router.eval()
    print(f"    Checkpoint: {ckpt_path.name}")
    print(f"    Assets    : {trained_assets}")

    # ── 5. Walk-Forward 模擬 ──────────────────────────────────────────
    print(">>> [5/5] Walk-Forward 回測模擬...")

    X_val_sc  = scaler.transform(X_val).astype(np.float32)
    X_tensor  = torch.tensor(X_val_sc, dtype=torch.float32)

    with torch.no_grad():
        out     = router(X_tensor)
        weights = out.combined_output.numpy()  # [T_val, N_assets]

    # 每天：縮放至 gross exposure = 1（市場中性方向）
    abs_sum   = np.abs(weights).sum(axis=1, keepdims=True) + 1e-8
    w_scaled  = weights / abs_sum * GROSS_EXPOSURE   # [T_val, N_assets]

    # ── v2.0 Phase 2: 政體自適應配置 ────────────────────────────────────
    # 使用 Firewall 的 HMM 偏測器判斷每天的市場政體
    min_len = len(w_scaled)
    regime_labels: list[str] = []
    regime_confs: list[float] = []
    bear_probs = np.zeros(min_len, dtype=np.float32)
    danger_probs = np.zeros(min_len, dtype=np.float32)
    
    if firewall is not None:
        allocator = RegimeAllocator(
            assets=assets,
            config=RegimeAllocatorConfig(
                min_confidence=0.65,
                max_prior_blend=0.25,
                transition_smoothing=0.8,
            ),
        )
        from nexus_quant_os.data_pipelines.feature_engineer import engineer_features
        spy_df_bt = aligned_df[aligned_df["asset_id"] == "SPY"].copy().reset_index(drop=True)
        spy_feat_matrix, _ = engineer_features(spy_df_bt)
        # 使用回測的驗證期日期範圍來切片 SPY 特徵（避免索引不匹配）
        spy_dates = aligned_df[aligned_df["asset_id"] == "SPY"]["timestamp"].dt.normalize()
        val_start_date = dates_val[0]
        val_end_date = dates_val[-1]
        date_mask = (spy_dates >= val_start_date) & (spy_dates <= val_end_date)
        # 在 engineer_features 處理後的 df 中，用日期範圍取子集
        spy_val_feat = spy_feat_matrix[date_mask.values[-len(spy_feat_matrix):]] if len(date_mask) > len(spy_feat_matrix) else spy_feat_matrix[-len(dates_val):]

        # 確保 SPY 特徵長度與回測期一致
        min_len = min(len(w_scaled), len(spy_val_feat))
        y_raw_val = y_raw_val[:min_len]
        dates_val = dates_val[:min_len]

        for t in range(min_len):
            start_idx = max(0, t - 20)
            obs = spy_val_feat[start_idx : t + 1]
            if len(obs) < 5:
                regime_labels.append("NEUTRAL")
                regime_confs.append(0.0)
                continue
            hmm_pred = firewall.hmm_detector.predict(obs)
            regime_labels.append(hmm_pred.regime_label)
            bear_probs[t] = hmm_pred.bear_probability
            danger_probs[t] = hmm_pred.danger_probability
            # 根據政體類型計算對應的 confidence
            label = hmm_pred.regime_label
            if "BEAR" in label:
                conf = hmm_pred.bear_probability
            elif "EXTREME" in label or "SHOCK" in label:
                conf = hmm_pred.danger_probability
            else:
                conf = max(0.0, 1.0 - hmm_pred.bear_probability - hmm_pred.danger_probability)
            regime_confs.append(conf)

    # ── v2.0 Phase 3: 投組優化器 (Risk Parity / Constrained MVO) ────────────
    optimizer = PortfolioOptimizer(
        n_assets=len(assets),
        config=OptimizerConfig(
            max_single_weight=0.35,
            gross_exposure=GROSS_EXPOSURE,
            min_cash=0.05,
            dispersion_threshold=0.45,
        ),
        asset_names=assets,
    )
    expert_util = out.expert_utilisation.numpy()
    w_optimized = optimizer.optimize_series(
        moe_weight_series=w_scaled[:min_len],
        expert_utilisation_series=expert_util,
        returns_history=y_raw[:split_idx + min_len],  # Pass full history up to current val length
        lookback=60,
        hmm_bear_probs=bear_probs[:min_len],
    )
    print(f"    ✓ 投組優化器已套用（Risk Parity / MVO 雙層）")

    if firewall is not None:
        w_regime = allocator.blend_series(
            moe_weight_series=w_optimized,
            regime_labels=regime_labels,
            regime_confidences=np.array(regime_confs),
            hmm_bear_probs=bear_probs,
            hmm_danger_probs=danger_probs,
        )
        print(f"    ✓ 政體配置已套用（{len(set(regime_labels))} 個政體）")
    else:
        w_regime = w_optimized
        print("    ⚠ 無 Firewall，跳過政體配置")

    # ── v2.0 Phase 1: 權重平滑器 ──────────────────────────────────────
    smoother = WeightSmoother(
        n_assets=len(assets),
        config=SmootherConfig(
            alpha=0.30,
            min_rebalance_threshold=0.04,
            max_single_turnover=0.08,
            min_hold_days=5,
            signal_stability_window=3,
        ),
    )
    
    # 計算每日的 expert_utilization (基於 top_k_indices)
    top_k_indices = out.top_k_indices.numpy()[:min_len]  # [T, K]
    E = len(expert_util)
    daily_expert_util = np.zeros((min_len, E), dtype=np.float32)
    for t in range(min_len):
        for k in range(top_k_indices.shape[1]):
            daily_expert_util[t, top_k_indices[t, k]] = 1.0

    # 計算 VIX series (以 SPY realised_vol * sqrt(252) 為 proxy)
    vol_idx = 1  # realised_vol is index 1 in feature_matrix
    vix_series = spy_val_feat[:, vol_idx] * np.sqrt(252) * 100
    
    # 計算 VIX 5MA
    vix_5ma_series = pd.Series(vix_series).rolling(window=5, min_periods=1).mean().values

    w_smooth = smoother.smooth_series(
        w_regime,
        expert_utilization_series=daily_expert_util,
        vix_series=vix_series,
        vix_5ma_series=vix_5ma_series,
    )
    print(f"    ✓ 權重平滑已套用（含 VIX 動態視窗與 Expert-0 豁免）")

    # 計算每日權重變動 (Turnover)
    w_prev = np.vstack([np.zeros((1, w_smooth.shape[1])), w_smooth[:-1]])
    w_diff = w_smooth - w_prev  # [T_val, N_assets]

    # 投資組合每日報酬
    tc_cost     = TRANSACTION_COST_BPS * 1e-4          # bps → decimal
    daily_tc    = np.abs(w_diff).sum(axis=1) * tc_cost # [T_val] 每日總交易費
    port_ret    = (w_smooth * y_raw_val).sum(axis=1)    # [T_val]
    port_ret_tc = port_ret - daily_tc                   # 扣除交易成本

    # SPY 買進持有基準
    spy_ret     = y_raw_val[:, spy_idx]                 # [T_val]

    # ── 績效指標 ──────────────────────────────────────────────────────
    strat_metrics = compute_metrics(port_ret_tc, "Router Strategy (net of TC)")
    spy_metrics   = compute_metrics(spy_ret, "SPY Buy-and-Hold")

    # ─── 輸出報告 ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  OUT-OF-SAMPLE PERFORMANCE REPORT")
    print(f"  Period : {dates_val[0].date()} → {dates_val[-1].date()}  "
          f"({len(dates_val)} trading days)")
    print(f"{SEP}")

    header = f"\n  {'Metric':<28} {'Strategy':>14}  {'SPY B&H':>12}"
    print(header)
    print("  " + THIN)

    rows = [
        ("Total Return",     strat_metrics["total_return"],   spy_metrics["total_return"],   "{:+.1%}"),
        ("CAGR",             strat_metrics["cagr"],           spy_metrics["cagr"],           "{:+.2%}"),
        ("Ann. Volatility",  strat_metrics["ann_vol"],        spy_metrics["ann_vol"],        "{:.2%}"),
        ("Sharpe Ratio",     strat_metrics["sharpe"],         spy_metrics["sharpe"],         "{:+.3f}"),
        ("Max Drawdown",     strat_metrics["max_drawdown"],   spy_metrics["max_drawdown"],   "{:.2%}"),
        ("Calmar Ratio",     strat_metrics["calmar"],         spy_metrics["calmar"],         "{:+.3f}"),
        ("Win Rate",         strat_metrics["win_rate"],       spy_metrics["win_rate"],       "{:.1%}"),
        ("Best Day",         strat_metrics["best_day"],       spy_metrics["best_day"],       "{:+.2%}"),
        ("Worst Day",        strat_metrics["worst_day"],      spy_metrics["worst_day"],      "{:+.2%}"),
    ]
    for name, sv, bv, fmt in rows:
        sv_str = fmt.format(sv)
        bv_str = fmt.format(bv)
        # Mark if strategy is better
        better = ""
        if name in ("Total Return", "CAGR", "Sharpe Ratio", "Calmar Ratio", "Win Rate", "Best Day"):
            better = " ✓" if sv > bv else ""
        elif name in ("Ann. Volatility", "Max Drawdown"):
            better = " ✓" if sv < bv else ""  # lower is better (less negative)
        elif name == "Worst Day":
            better = " ✓" if sv > bv else ""  # closer to 0 is better
        print(f"  {name:<28} {sv_str:>14}  {bv_str:>12}{better}")

    total_drag = daily_tc.sum()
    print(f"\n  Transaction Cost ({TRANSACTION_COST_BPS}bps/trade):  "
          f"{total_drag:.2%} total drag (Turnover based)")

    # ─── 月度報酬 ─────────────────────────────────────────────────────
    print_monthly_returns(port_ret_tc, dates_val)

    # ─── 股權曲線 ─────────────────────────────────────────────────────
    print_equity_curve(
        strat_metrics["equity"],
        spy_metrics["equity"],
        dates_val,
    )

    # ─── Expert 使用率 ────────────────────────────────────────────────
    expert_util = out.expert_utilisation.numpy()
    print(f"\n  Expert Utilisation (avg over {len(X_val)} days):")
    for i, util in enumerate(expert_util):
        bar = "█" * int(util * 30)
        print(f"    Expert-{i}  {util:.3f}  {bar}")

    print(f"\n{SEP}")
    print(f"  Backtest complete  ·  Checkpoint: {ckpt_path.name}")
    print(f"{SEP}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus Quant OS Walk-Forward Backtest")
    parser.add_argument("--start", type=str, default=None,
                        help="回測數據起始日期，例如 2000-01-01（預設使用訓練設定）")
    parser.add_argument("--end", type=str, default=None,
                        help="回測數據結束日期，例如 2020-12-31（預設使用今天）")
    args = parser.parse_args()
    run_backtest(start_date=args.start, end_date=args.end)
