"""
portfolio/weight_smoother.py — 投資組合權重平滑器
====================================================

防禦層的最後一道關卡：在模型推論後、交易下單前，
透過 EMA 平滑 + 最小換手閾值，大幅降低不必要的交易。

核心機制：
    1. EMA 平滑 (Exponential Moving Average):
       smooth_weight[t] = α × raw_weight[t] + (1 - α) × smooth_weight[t-1]
       α 越低越平滑但反應越慢

    2. 最小換手閾值 (Min Rebalance Threshold):
       如果 |Δw| < threshold，則不交易（維持原部位）

預期效果：
    v1.0 日均換手率 ~8% → v2.0 目標 < 1.5%
    v1.0 交易成本拖累 31.76% → v2.0 目標 < 5%

Author : Nexus Quant OS — Risk Engineering Division
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("nexus_quant_os.portfolio.weight_smoother")


# ═════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════

@dataclass
class SmootherConfig:
    """權重平滑器配置。

    Parameters
    ----------
    alpha : float
        EMA 平滑係數，範圍 (0, 1]。
        越小越平滑（權重變化越慢），越大越敏銳（快速跟隨模型輸出）。
        建議：0.2 ~ 0.4。
    min_rebalance_threshold : float
        最小調倉閾值。如果某資產的權重變動 |Δw| 小於此值，
        則不執行交易（維持當前部位）。
        建議：0.01 ~ 0.05。
    max_single_turnover : float
        單一資產單日最大換手幅度。
        防止因模型突然翻轉而產生巨大交易成本。
        建議：0.05 ~ 0.15。
    min_hold_days : int
        最小持倉天數。在持倉方向確定後，至少持有 N 天才允許反轉。
        防止短期噪音導致的來回交易（whipsaw）。
        建議：3 ~ 7。
    signal_stability_window : int
        信號穩定度視窗。只有當信號方向連續 N 天一致時，
        才啟動調倉。過濾掉短期噪音信號。
        建議：2 ~ 5。
    """
    alpha: float = 0.55
    min_rebalance_threshold: float = 0.02
    max_single_turnover: float = 0.20
    min_hold_days: int = 3
    signal_stability_window: int = 2


# ═════════════════════════════════════════════════════════════════════
# 核心類別
# ═════════════════════════════════════════════════════════════════════

class WeightSmoother:
    """投資組合權重平滑器。

    在模型輸出原始權重後，透過 EMA 平滑 + 閾值過濾，
    將「神經質的頻繁調倉」轉化為「穩定的漸進式部位調整」。

    Usage
    -----
    >>> smoother = WeightSmoother(n_assets=7)
    >>> for day in trading_days:
    ...     raw = model.predict(features[day])
    ...     smooth = smoother.smooth(raw)
    ...     execute_trade(smooth)

    Batch 模式（回測用）
    >>> smooth_series = smoother.smooth_series(raw_weight_series)
    """

    def __init__(
        self,
        n_assets: int,
        config: SmootherConfig | None = None,
    ) -> None:
        self.n_assets = n_assets
        self.config = config or SmootherConfig()
        self._prev_smooth: np.ndarray | None = None
        self._step_count: int = 0
        # 持倉方向追蹤：每個資產的持倉天數計數器
        self._hold_counter: np.ndarray = np.zeros(n_assets, dtype=np.int32)
        # 信號穩定度緩衝：追蹤連續 N 天的信號方向
        self._signal_direction_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        """重置內部狀態（回測新窗口起始時呼叫）。"""
        self._prev_smooth = None
        self._step_count = 0
        self._hold_counter = np.zeros(self.n_assets, dtype=np.int32)
        self._signal_direction_buffer = []

    def _check_signal_stability(self, delta: np.ndarray, dynamic_window: int | np.ndarray) -> np.ndarray:
        """檢查信號穩定度：只有連續 N 天方向一致的資產才允許調倉。

        Parameters
        ----------
        delta : np.ndarray
            本日信號方向 (EMA - prev)，shape [N_assets]。
        dynamic_window : int | np.ndarray
            動態視窗大小，若是陣列則形狀為 [N_assets]。

        Returns
        -------
        np.ndarray (bool)
            shape [N_assets]，True = 信號穩定，允許調倉。
        """
        # 動態視窗可能是一個陣列，我們先取最大值以決定 buffer 長度
        max_w = np.max(dynamic_window) if isinstance(dynamic_window, np.ndarray) else dynamic_window
        if max_w <= 1:
            # 如果最大視窗都 <= 1，則直接全部放行 (注意：這裡如果是 array，可能有部分 > 1)
            # 但為了簡化，如果傳入標量 <=1，直接放行
            if isinstance(dynamic_window, (int, float, np.integer, np.floating)) and dynamic_window <= 1:
                return np.ones(self.n_assets, dtype=bool)

        # 記錄信號方向（+1/-1/0）
        direction = np.sign(delta)
        self._signal_direction_buffer.append(direction)

        # 保持緩衝長度不超過最大視窗大小
        if len(self._signal_direction_buffer) > max_w:
            self._signal_direction_buffer = self._signal_direction_buffer[-max_w:]

        stable = np.zeros(self.n_assets, dtype=bool)
        
        # 處理標量視窗
        if np.isscalar(dynamic_window):
            dyn_w_array = np.full(self.n_assets, dynamic_window, dtype=int)
        else:
            dyn_w_array = np.asarray(dynamic_window, dtype=int)

        # 對每個資產個別判斷
        for i in range(self.n_assets):
            w = dyn_w_array[i]
            if w <= 1:
                stable[i] = True
                continue
                
            if len(self._signal_direction_buffer) < w:
                stable[i] = False
                continue
                
            # 檢查最近 w 天的方向是否全部一致
            recent = [buf[i] for buf in self._signal_direction_buffer[-w:]]
            direction_sum = abs(sum(recent))
            stable[i] = (direction_sum >= w)
            
        return stable

    def smooth(
        self, 
        raw_weights: np.ndarray,
        expert_utilization: np.ndarray | None = None,
        vix_value: float | None = None,
        vix_5ma: float | None = None,
    ) -> np.ndarray:
        """對單一時間步的原始權重執行平滑。

        整合三道濾網：
          1. EMA 平滑 — 抑制信號噪音
          2. 信號穩定度 — 連續 N 天同方向才調倉
          3. 最小持倉天數 — 持倉未滿 N 天不允許反轉
          4. 最大換手限制 + 最小閾值 — 控制單日交易量

        Parameters
        ----------
        raw_weights : np.ndarray
            模型輸出的原始權重，shape [N_assets]。

        Returns
        -------
        np.ndarray
            平滑後的最終交易權重，shape [N_assets]。
        """
        cfg = self.config
        raw = np.asarray(raw_weights, dtype=np.float64).copy()

        # ── NaN/Inf 防護 ──────────────────────────────────────────
        if np.any(~np.isfinite(raw)):
            logger.warning("WeightSmoother: NaN/Inf detected in raw weights, replacing with 0.0")
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ── 第一天：直接使用模型輸出（無歷史可平滑）──────────────
        if self._prev_smooth is None:
            self._prev_smooth = raw.copy()
            self._step_count = 1
            self._hold_counter = np.ones(self.n_assets, dtype=np.int32)
            logger.debug("WeightSmoother: Day 1 — using raw weights directly.")
            return raw.copy()

        # ── EMA 平滑 ─────────────────────────────────────────────
        ema = cfg.alpha * raw + (1.0 - cfg.alpha) * self._prev_smooth

        # ── 最大單日換手限制 ──────────────────────────────────────
        delta = ema - self._prev_smooth
        clipped_delta = np.clip(
            delta,
            -cfg.max_single_turnover,
            cfg.max_single_turnover,
        )
        ema_capped = self._prev_smooth + clipped_delta

        # ── 動態視窗與動態持倉設定 ─────────────────────────────────
        dynamic_window = np.full(self.n_assets, cfg.signal_stability_window, dtype=int)
        dynamic_hold = np.full(self.n_assets, cfg.min_hold_days, dtype=int)

        if vix_value is not None:
            is_risk_off = raw < self._prev_smooth
            is_risk_on = raw > self._prev_smooth
            
            if vix_value > 25.0:
                if vix_5ma is not None and vix_value < vix_5ma:
                    # VIX 動能：高波且正在下降（V 型反轉），強烈抄底信號，加倉縮短至 1
                    dynamic_window = np.where(is_risk_off, 1, np.where(is_risk_on, 1, cfg.signal_stability_window))
                    dynamic_hold = np.where(is_risk_off, 0, np.where(is_risk_on, 1, cfg.min_hold_days))
                else:
                    # 恐慌逃命 (VIX>25)：減倉立即生效 (w=1, hold=0)，加倉防死貓 (w=4)
                    dynamic_window = np.where(is_risk_off, 1, 4)
                    dynamic_hold = np.where(is_risk_off, 0, cfg.min_hold_days)
            elif vix_value < 15.0:
                # 低波加倉 (VIX<15)：減倉防假跌 (w=3)，加倉加速上車 (w=2)
                dynamic_window = np.where(is_risk_off, 3, 2)
        
        # ── 特權快車道 (Expert-0 Bypass) ───────────────────────────
        if expert_utilization is not None and len(expert_utilization) > 0:
            if expert_utilization[0] > 0.6:  # Expert-0 利用率 > 60%
                dynamic_window = np.ones(self.n_assets, dtype=int)
                dynamic_hold = np.full(self.n_assets, 2, dtype=int)

        # ── 濾網 1：信號穩定度檢查 ────────────────────────────────
        # 只有連續 N 天同方向的信號才允許通過
        signal_stable = self._check_signal_stability(delta, dynamic_window)

        # ── 濾網 2：最小持倉天數鎖定 ──────────────────────────────
        # 如果持倉未滿 min_hold_days 且方向要反轉，則鎖定
        # 使用量級閾值而非 sign()：長倉 (>0.05)=+1, 出場 (<0.01)=-1, 中間=0
        # sign() 在純多頭組合永遠 ≥ 0，無法偵測「縮倉→清倉」的方向反轉
        new_direction = np.where(ema_capped > 0.05, 1, np.where(ema_capped < 0.01, -1, 0))
        old_direction = np.where(self._prev_smooth > 0.05, 1, np.where(self._prev_smooth < 0.01, -1, 0))
        direction_reversal = (new_direction != old_direction) & (old_direction != 0)
        hold_locked = direction_reversal & (self._hold_counter < dynamic_hold)

        # ── 濾網 3：最小換手閾值 ──────────────────────────────────
        small_change = np.abs(ema_capped - self._prev_smooth) < cfg.min_rebalance_threshold

        # ── 合併所有濾網：任一濾網觸發 → 維持原部位 ──────────────
        blocked = small_change | hold_locked | (~signal_stable)
        final = np.where(blocked, self._prev_smooth, ema_capped)

        # ── 更新持倉計數器 ────────────────────────────────────────
        position_changed = ~np.isclose(final, self._prev_smooth, atol=1e-8)
        # 部位有變 → 重置計數器，無變 → 計數器 +1
        self._hold_counter = np.where(position_changed, 1, self._hold_counter + 1)

        # ── 更新狀態 ─────────────────────────────────────────────
        self._prev_smooth = final.copy()
        self._step_count += 1

        if self._step_count % 50 == 0:
            n_blocked = int(blocked.sum())
            n_stable = int(signal_stable.sum())
            n_hold_locked = int(hold_locked.sum())
            logger.debug(
                "WeightSmoother: step=%d  blocked=%d/%d  "
                "stable=%d  hold_locked=%d",
                self._step_count, n_blocked, self.n_assets,
                n_stable, n_hold_locked,
            )

        # 確保權重總和為 1.0
        total = final.sum()
        if total > 0:
            final = final / total

        return final.copy()

    def smooth_series(
        self,
        raw_weight_series: np.ndarray,
        expert_utilization_series: np.ndarray | None = None,
        vix_series: np.ndarray | None = None,
        vix_5ma_series: np.ndarray | None = None,
    ) -> np.ndarray:
        """對整個時間序列批次平滑（回測專用）。

        Parameters
        ----------
        raw_weight_series : np.ndarray
            原始權重序列，shape [T, N_assets]。
        expert_utilization_series : np.ndarray | None
            Expert 利用率序列，shape [T, E]。
        vix_series : np.ndarray | None
            VIX 值序列，shape [T]。

        Returns
        -------
        np.ndarray
            平滑後權重序列，shape [T, N_assets]。
        """
        self.reset()
        T, N = raw_weight_series.shape
        assert N == self.n_assets, \
            f"Asset count mismatch: expected {self.n_assets}, got {N}"

        smoothed = np.zeros_like(raw_weight_series, dtype=np.float64)
        for t in range(T):
            eu = expert_utilization_series[t] if expert_utilization_series is not None else None
            vix = vix_series[t] if vix_series is not None else None
            vix5 = vix_5ma_series[t] if vix_5ma_series is not None else None
            smoothed[t] = self.smooth(raw_weight_series[t], eu, vix, vix5)

        total_raw_turnover = float(
            np.abs(np.diff(raw_weight_series, axis=0)).sum()
        )
        total_smooth_turnover = float(
            np.abs(np.diff(smoothed, axis=0)).sum()
        )
        reduction = 1.0 - (total_smooth_turnover / (total_raw_turnover + 1e-8))

        logger.info(
            "WeightSmoother batch: T=%d  raw_turnover=%.2f  "
            "smooth_turnover=%.2f  reduction=%.1f%%",
            T, total_raw_turnover, total_smooth_turnover, reduction * 100,
        )

        return smoothed


# ═════════════════════════════════════════════════════════════════════
# Torch-compatible Turnover Penalty（嵌入損失函數用）
# ═════════════════════════════════════════════════════════════════════

def compute_turnover_penalty_torch(
    weights: "torch.Tensor",
    cost_coeff: float = 1.0,
) -> "torch.Tensor":
    """計算批次內的換手率懲罰（PyTorch 可微分）。

    在時序批次中，weights[t] 和 weights[t-1] 的差異即為換手率。
    由於訓練時使用的是打亂後的 mini-batch，我們改用「每個 batch
    內相鄰樣本的權重差異」作為換手率的近似。

    Parameters
    ----------
    weights : torch.Tensor
        預測權重，shape [B, N_assets]。
    cost_coeff : float
        交易成本係數（乘法因子）。

    Returns
    -------
    torch.Tensor
        標量損失值。
    """
    import torch

    if weights.shape[0] < 2:
        return torch.tensor(0.0, device=weights.device, requires_grad=True)

    # 相鄰樣本間的絕對權重變化
    diffs = torch.abs(weights[1:] - weights[:-1])  # [B-1, N_assets]
    turnover_per_step = diffs.sum(dim=1)            # [B-1]
    mean_turnover = turnover_per_step.mean()

    return mean_turnover * cost_coeff
