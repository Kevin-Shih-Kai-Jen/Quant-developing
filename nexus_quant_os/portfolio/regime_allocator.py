"""
portfolio/regime_allocator.py — 政體自適應配置器
=================================================

根據 HMM 政體（Market Regime）動態調整投資組合的「基底配置」，
讓系統在熊市也能賺錢，而不僅僅是「減倉避險」。

核心邏輯：
    最終權重 = 政體先驗 × β + MoE 預測 × (1 - β)
    β = regime_confidence（HMM 後驗機率）

    BULL_LOW_VOL     → 進攻型：重科技股 (QQQ/NVDA)
    BEAR_HIGH_VOL    → 防禦型：重債券+黃金 (TLT/GLD)
    EXTREME_SHOCK    → 100% 現金（防火牆強制）

Author : Nexus Quant OS — Portfolio Engineering Division
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("nexus_quant_os.portfolio.regime_allocator")


# ═════════════════════════════════════════════════════════════════════
# 政體先驗權重表
# ═════════════════════════════════════════════════════════════════════

# 基於歷史分析的政體最優配置
# 在牛市中，科技股（QQQ, NVDA, AVGO）歷史表現遠超大盤
# 在熊市中，美國公債（TLT）和黃金（GLD）提供避險收益
REGIME_PRIORS: Dict[str, Dict[str, float]] = {
    # 牛市低波動：進攻型配置，重倉成長股
    "BULL_LOW_VOL": {
        "AVGO": 0.10,
        "GLD":  0.10,
        "IWM":  0.10,
        "NVDA": 0.15,
        "QQQ":  0.25,
        "SPY":  0.20,
        "TLT":  0.10,
    },
    # 熊市高波動：防禦型配置，重倉避險資產
    "BEAR_HIGH_VOL": {
        "AVGO": 0.00,
        "GLD":  0.40,
        "IWM":  0.00,
        "NVDA": 0.00,
        "QQQ":  0.05,
        "SPY":  0.05,
        "TLT":  0.50,
    },
    # 極端衝擊：全現金（由防火牆強制執行，此處僅為記錄）
    # 使用等權 1/N 使先驗在數學上有效（防止 normalization 產生 NaN）
    "EXTREME_SHOCK": {
        "AVGO": 1/7,
        "GLD":  1/7,
        "IWM":  1/7,
        "NVDA": 1/7,
        "QQQ":  1/7,
        "SPY":  1/7,
        "TLT":  1/7,
    },
    # 中性 / 不確定：等權配置
    "NEUTRAL": {
        "AVGO": 1/7,
        "GLD":  1/7,
        "IWM":  1/7,
        "NVDA": 1/7,
        "QQQ":  1/7,
        "SPY":  1/7,
        "TLT":  1/7,
    },
}


# ═════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════

@dataclass
class RegimeAllocatorConfig:
    """政體配置器參數。

    Parameters
    ----------
    min_confidence : float
        HMM 政體判定的最低信心閾值。
        低於此值時，回退到等權配置（不確定市場不賭方向）。
    max_prior_blend : float
        政體先驗的最大混合比例。
        即使 HMM 非常確定，先驗最多也只佔這個比例。
        建議 0.3 ~ 0.6。
    transition_smoothing : float
        政體轉換時的平滑係數。
        防止政體在邊界處反覆切換（chattering）。
    """
    min_confidence: float = 0.50
    max_prior_blend: float = 0.25
    transition_smoothing: float = 0.8


# ═════════════════════════════════════════════════════════════════════
# 核心類別
# ═════════════════════════════════════════════════════════════════════

class RegimeAllocator:
    """政體自適應配置器。

    根據 HMM 偵測到的市場政體，在 MoE 模型預測的基礎上，
    混合進「政體先驗權重」，使系統在不同市場環境下都有對應的策略。

    Usage
    -----
    >>> allocator = RegimeAllocator(assets=['AVGO','GLD','IWM','NVDA','QQQ','SPY','TLT'])
    >>> final_weights = allocator.blend(
    ...     moe_weights=model_output,
    ...     regime_label='BEAR_HIGH_VOL',
    ...     regime_confidence=0.75,
    ... )
    """

    def __init__(
        self,
        assets: list[str],
        config: RegimeAllocatorConfig | None = None,
        regime_priors: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self.assets = sorted(assets)  # 確保字母排序
        self.config = config or RegimeAllocatorConfig()
        self.priors = regime_priors or REGIME_PRIORS
        self._prev_regime: str | None = None
        self._prev_blend: float = 0.0

        # 將先驗字典轉為 numpy array（按 assets 排序）
        self._prior_arrays: Dict[str, np.ndarray] = {}
        for regime, weight_dict in self.priors.items():
            arr = np.array([
                weight_dict.get(asset, 0.0) for asset in self.assets
            ], dtype=np.float64)
            # 正規化（確保和為 1）
            total = arr.sum()
            if total > 0:
                arr = arr / total
            self._prior_arrays[regime] = arr

        logger.info(
            "RegimeAllocator initialized: %d assets, %d regimes, "
            "max_blend=%.2f, min_confidence=%.2f",
            len(self.assets), len(self.priors),
            self.config.max_prior_blend, self.config.min_confidence,
        )

    def _map_regime_label(self, label: str) -> str:
        """將各種 HMM 輸出的標籤映射到先驗表的 key。

        HMM 可能輸出 'Regime_0', 'BULL_LOW_VOL' 等不同格式。
        """
        label_upper = label.upper().replace(" ", "_")

        # 直接匹配
        if label_upper in self.priors:
            return label_upper

        # 關鍵字匹配
        if "BULL" in label_upper or "LOW_VOL" in label_upper:
            return "BULL_LOW_VOL"
        if "BEAR" in label_upper or "HIGH_VOL" in label_upper:
            return "BEAR_HIGH_VOL"
        if "EXTREME" in label_upper or "SHOCK" in label_upper or "CRASH" in label_upper:
            return "EXTREME_SHOCK"

        # 無法匹配 → 回退到中性
        logger.warning(
            "RegimeAllocator: Unknown regime label '%s' → fallback to NEUTRAL",
            label,
        )
        return "NEUTRAL"

    def blend(
        self,
        moe_weights: np.ndarray,
        regime_label: str,
        regime_confidence: float,
        hmm_bear_prob: float = 0.0,
        hmm_danger_prob: float = 0.0,
        asset_order: list[str] | None = None,
    ) -> np.ndarray:
        """將 MoE 預測權重與政體先驗混合。

        Parameters
        ----------
        moe_weights : np.ndarray
            MoE Router 的原始預測權重，shape [N_assets]。
        regime_label : str
            HMM 偵測到的當前政體名稱。
        regime_confidence : float
            HMM 對當前政體判定的信心（後驗機率）。
        hmm_bear_prob : float
            HMM 判定熊市的後驗機率。
        hmm_danger_prob : float
            HMM 判定極端危險的後驗機率。
        asset_order : list[str] | None
            moe_weights 的資產順序。若與 self.assets 不同，
            會自動重排 moe_weights 以匹配 self.assets。

        Returns
        -------
        np.ndarray
            混合後的最終配置權重，shape [N_assets]。
        """
        cfg = self.config
        moe = np.asarray(moe_weights, dtype=np.float64).copy()

        # ── 重排 moe_weights 以匹配 self.assets 排序 ─────────────
        if asset_order is not None and list(asset_order) != self.assets:
            order_map = {name: idx for idx, name in enumerate(asset_order)}
            reordered = np.zeros_like(moe)
            for i, asset in enumerate(self.assets):
                if asset in order_map:
                    reordered[i] = moe[order_map[asset]]
            moe = reordered

        # ── 映射政體標籤 ───────────────────────────────────────────
        mapped_regime = self._map_regime_label(regime_label)

        # ── 計算混合比例 β ─────────────────────────────────────────
        # β = min(regime_confidence, max_prior_blend) if confidence > min_confidence
        if regime_confidence < cfg.min_confidence:
            # 信心不足 → 不混合先驗，完全信任模型
            beta = 0.0
        else:
            # 信心越高，先驗比例越大（但不超過 max_prior_blend）
            raw_beta = (regime_confidence - cfg.min_confidence) / (1.0 - cfg.min_confidence + 1e-8)
            beta = min(raw_beta, cfg.max_prior_blend)

        # ── 政體轉換平滑 ───────────────────────────────────────────
        # 防止政體在邊界處反覆切換
        if self._prev_regime is not None and mapped_regime != self._prev_regime:
            # 轉換期間：平滑地從舊政體過渡到新政體
            beta = cfg.transition_smoothing * self._prev_blend + \
                   (1.0 - cfg.transition_smoothing) * beta
            logger.info(
                "RegimeAllocator: Regime transition %s → %s  "
                "(smoothed β=%.3f)",
                self._prev_regime, mapped_regime, beta,
            )

        # ── 取得先驗 ──────────────────────────────────────────────
        prior = self._prior_arrays.get(mapped_regime)
        if prior is None:
            logger.warning(
                "RegimeAllocator: No prior for '%s', using model output directly.",
                mapped_regime,
            )
            beta = 0.0
            prior = np.zeros(len(self.assets), dtype=np.float64)

        # ── 混合：final = prior × β + moe × (1 - β) ──────────────
        blended = prior * beta + moe * (1.0 - beta)

        # ── 正規化（保持 gross exposure = 原始值）──────────────────
        original_gross = float(np.abs(moe).sum())
        blended_gross = float(np.abs(blended).sum())
        if blended_gross > 1e-8 and original_gross > 1e-8:
            blended = blended * (original_gross / blended_gross)

        # ── 更新狀態 ──────────────────────────────────────────────
        self._prev_regime = mapped_regime
        self._prev_blend = beta

        logger.debug(
            "RegimeAllocator: regime=%s  confidence=%.3f  β=%.3f  "
            "bear_prob=%.3f  danger_prob=%.3f",
            mapped_regime, regime_confidence, beta,
            hmm_bear_prob, hmm_danger_prob,
        )

        return blended

    def blend_series(
        self,
        moe_weight_series: np.ndarray,
        regime_labels: list[str],
        regime_confidences: np.ndarray,
        hmm_bear_probs: np.ndarray | None = None,
        hmm_danger_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """批次混合整個時間序列的權重（回測用）。

        Parameters
        ----------
        moe_weight_series : np.ndarray
            MoE 權重序列，shape [T, N_assets]。
        regime_labels : list[str]
            每天的政體標籤，length T。
        regime_confidences : np.ndarray
            每天的政體信心，shape [T]。
        hmm_bear_probs : np.ndarray, optional
            熊市機率序列，shape [T]。
        hmm_danger_probs : np.ndarray, optional
            極端危險機率序列，shape [T]。

        Returns
        -------
        np.ndarray
            混合後權重序列，shape [T, N_assets]。
        """
        self._prev_regime = None
        self._prev_blend = 0.0

        T = len(moe_weight_series)
        blended = np.zeros_like(moe_weight_series, dtype=np.float64)

        if hmm_bear_probs is None:
            hmm_bear_probs = np.zeros(T)
        if hmm_danger_probs is None:
            hmm_danger_probs = np.zeros(T)

        for t in range(T):
            blended[t] = self.blend(
                moe_weights=moe_weight_series[t],
                regime_label=regime_labels[t],
                regime_confidence=float(regime_confidences[t]),
                hmm_bear_prob=float(hmm_bear_probs[t]),
                hmm_danger_prob=float(hmm_danger_probs[t]),
            )

        # 統計政體分佈
        regime_counts: Dict[str, int] = {}
        for label in regime_labels:
            mapped = self._map_regime_label(label)
            regime_counts[mapped] = regime_counts.get(mapped, 0) + 1

        logger.info(
            "RegimeAllocator batch: T=%d  regime_dist=%s",
            T, regime_counts,
        )

        return blended

    def reset(self) -> None:
        """重置內部狀態。"""
        self._prev_regime = None
        self._prev_blend = 0.0
