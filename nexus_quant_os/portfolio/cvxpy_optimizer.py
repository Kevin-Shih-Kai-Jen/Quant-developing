"""
portfolio/optimizer.py — 雙層投組優化器
========================================

Posterior-probability-driven portfolio optimiser.
在 MoE Router 預測的基礎上，透過數學優化產生最終的交易權重。

雙層架構：
    Layer 1: Risk Parity 回退
        當專家分歧度高（Expert Dispersion > threshold）時，
        不信任模型預測，回退到等風險貢獻配置。

    Layer 2: 約束優化 (Constrained MVO)
        當專家意見一致時，使用均值方差優化 (Mean-Variance Optimization)
        在以下約束條件下最大化 Sharpe Ratio：
        - 最大單一資產權重 ≤ 30%
        - 總絕對曝險 ≤ 1.0
        - 最小現金水位 ≥ 5%

Author : Nexus Quant OS — Portfolio Engineering Division
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nexus_quant_os.portfolio.risk_parity import (
    compute_risk_parity_weights,
    compute_inverse_volatility_weights,
)

logger = logging.getLogger("nexus_quant_os.portfolio.optimizer")


# ═════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════

@dataclass
class OptimizerConfig:
    """投組優化器配置。

    Parameters
    ----------
    max_single_weight : float
        最大單一資產權重（防止過度集中）。
    max_drawdown_target : float
        目標最大回撤，用於風險縮放。
    gross_exposure : float
        總絕對權重上限。
    min_cash : float
        最小現金水位（1 - sum(|w|)）。
    dispersion_threshold : float
        Expert 利用率分歧度閾值。
        超過此值時回退到 Risk Parity。
    ann_factor : float
        年化因子（252 個交易日）。
    """
    max_single_weight: float = 0.30
    dynamic_weight_lambda: float = 0.5  # 用於動態放大上限
    max_drawdown_target: float = 0.20
    gross_exposure: float = 1.0
    min_cash: float = 0.05
    dispersion_threshold: float = 0.30
    ann_factor: float = 252.0


# ═════════════════════════════════════════════════════════════════════
# 核心類別
# ═════════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """雙層投組優化器。

    Usage
    -----
    >>> optimizer = PortfolioOptimizer(n_assets=7)
    >>> final = optimizer.optimize(
    ...     moe_weights=raw_weights,
    ...     expert_utilisation=expert_util,
    ...     cov_matrix=cov,
    ...     returns_history=hist_returns,
    ... )
    """

    def __init__(
        self,
        n_assets: int,
        config: OptimizerConfig | None = None,
        asset_names: list[str] | None = None,
    ) -> None:
        self.n_assets = n_assets
        self.config = config or OptimizerConfig()
        self.asset_names = asset_names or []
        
        # 尋找特殊資產的索引
        self.idx_shy = self.asset_names.index("SHY") if "SHY" in self.asset_names else -1
        self.idx_sh = self.asset_names.index("SH") if "SH" in self.asset_names else -1
        self.idx_psq = self.asset_names.index("PSQ") if "PSQ" in self.asset_names else -1

    def optimize(
        self,
        moe_weights: np.ndarray,
        expert_utilisation: np.ndarray,
        cov_matrix: np.ndarray,
        returns_history: np.ndarray | None = None,
        hmm_bear_prob: float = 0.0,
    ) -> np.ndarray:
        """雙層優化主入口。

        Parameters
        ----------
        moe_weights : np.ndarray
            MoE Router 原始預測權重，shape [N_assets]。
        expert_utilisation : np.ndarray
            各 Expert 的利用率，shape [N_experts]。
        cov_matrix : np.ndarray
            資產共變異數矩陣，shape [N, N]。
        returns_history : np.ndarray, optional
            歷史日報酬，shape [T, N_assets]。
            用於估算預期報酬（MVO 需要）。
        hmm_bear_prob : float
            熊市機率，用於控制反向 ETF。

        Returns
        -------
        np.ndarray
            優化後權重，shape [N_assets]。
        """
        cfg = self.config

        # ── Layer 1：檢查 Expert 分歧度 → 決定是否回退 ────────────
        dispersion = float(np.std(expert_utilisation))
        use_risk_parity = dispersion > cfg.dispersion_threshold

        if use_risk_parity:
            logger.info(
                "PortfolioOptimizer: Expert 分歧度=%.4f > 閾值=%.4f "
                "→ 回退到 Risk Parity",
                dispersion, cfg.dispersion_threshold,
            )
            try:
                weights = compute_risk_parity_weights(cov_matrix)
            except Exception as e:
                logger.warning(
                    "Risk Parity 失敗(%s)，回退到反波動率加權", e,
                )
                weights = compute_inverse_volatility_weights(cov_matrix)
        else:
            # ── Layer 2：約束優化 (Constrained MVO) ───────────────
            logger.info(
                "PortfolioOptimizer: Expert 分歧度=%.4f ≤ 閾值=%.4f "
                "→ 使用 Constrained MVO",
                dispersion, cfg.dispersion_threshold,
            )

            # 預期報酬：使用 MoE 預測作為信號方向
            expected_returns = moe_weights.copy()

            # 如果有歷史報酬，混合歷史均值（shrinkage）
            if returns_history is not None and len(returns_history) >= 20:
                hist_mean = returns_history[-60:].mean(axis=0) * cfg.ann_factor
                # Shrinkage: 60% MoE signal + 40% historical mean
                expected_returns = 0.6 * expected_returns + 0.4 * hist_mean

            try:
                weights = self._constrained_mvo(expected_returns, cov_matrix, moe_weights, hmm_bear_prob)
            except Exception as e:
                logger.warning(
                    "Constrained MVO 失敗(%s)，回退到 Risk Parity", e,
                )
                try:
                    weights = compute_risk_parity_weights(cov_matrix)
                except Exception:
                    weights = compute_inverse_volatility_weights(cov_matrix)

        # ── 約束後處理 ────────────────────────────────────────────
        weights = self._apply_constraints(weights)

        logger.debug(
            "PortfolioOptimizer: mode=%s  dispersion=%.4f  "
            "weights=[%s]",
            "RiskParity" if use_risk_parity else "MVO",
            dispersion,
            ", ".join(f"{w:.4f}" for w in weights),
        )

        return weights

    def _constrained_mvo(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        moe_weights: np.ndarray,
        hmm_bear_prob: float,
    ) -> np.ndarray:
        """約束均值方差優化 (CVXPY + OSQP)。

        Parameters
        ----------
        expected_returns : np.ndarray
            預期報酬向量，shape [N]。
        cov_matrix : np.ndarray
            共變異數矩陣，shape [N, N]。

        Returns
        -------
        np.ndarray
            MVO 最優權重，shape [N]。
        """
        import cvxpy as cp
        cfg = self.config
        N = self.n_assets
        cov = cov_matrix.astype(np.float64)

        # Ridge 正則化 (消除共線性)
        cov = cov + 1e-4 * np.eye(N)

        mu = expected_returns.astype(np.float64)
        mu_scale = np.abs(mu).max() + 1e-8
        mu_norm = mu / mu_scale

        # 動態上限：Base + λ * max(0, Confidence - threshold)
        upper_bounds = np.zeros(N)
        for i in range(N):
            conf = float(moe_weights[i])
            dyn_max = cfg.max_single_weight + cfg.dynamic_weight_lambda * max(0.0, conf - 0.80)
            dyn_max = min(dyn_max, 0.50)  # 絕對上限 50%
            
            # 針對特殊資產 (SHY, SH, PSQ)
            if i in (self.idx_sh, self.idx_psq):
                if hmm_bear_prob < 0.50:
                    upper_bounds[i] = 0.0  # 非熊市禁止做空
                else:
                    upper_bounds[i] = 0.15 # 熊市允許單一反向 ETF 最多 15%
            elif i == self.idx_shy:
                upper_bounds[i] = 1.0  # 現金特權兜底
            else:
                upper_bounds[i] = dyn_max  # 多頭資產動態上限

        # 保證凸優化有解：確保所有資產上限總和大於等於 1.0
        # (排除現金，因為現金已經是 1.0。若排除現金後風險資產總和小於 1.0，則等比放大)
        risk_asset_indices = [i for i in range(N) if i not in (self.idx_shy, self.idx_sh, self.idx_psq)]
        risk_bound_sum = sum(upper_bounds[i] for i in risk_asset_indices)
        if risk_bound_sum < 1.0 and risk_bound_sum > 0:
            scale = 1.0 / risk_bound_sum
            for i in risk_asset_indices:
                upper_bounds[i] = min(upper_bounds[i] * scale, 1.0)

        w = cp.Variable(N)
        gamma = 2.0  # 風險趨避係數
        
        # 目標函數：最大化風險調整後報酬
        objective = cp.Maximize(mu_norm.T @ w - gamma * cp.quad_form(w, cov))
        
        constraints = [
            w >= 0,
            cp.sum(w) == 1.0,  # 強制 100% 資金分配 (含現金)
            w <= upper_bounds,
        ]
        
        # 總反向 ETF 上限約束 (SH + PSQ <= 15%)
        if self.idx_sh != -1 and self.idx_psq != -1:
            constraints.append(w[self.idx_sh] + w[self.idx_psq] <= 0.15)
            
        # 政體動態資金利用率
        if hmm_bear_prob < 0.50:
            # BULL: 強迫 100% 資金參與多頭市場，不留現金
            if self.idx_shy != -1:
                constraints.append(w[self.idx_shy] == 0.0)
                
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, max_iter=4000)
        except Exception as e:
            logger.error("CVXPY solve error: %s", e)
            
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            return np.array(w.value)
        else:
            logger.warning("MVO (CVXPY) 未收斂狀態: %s，回退全現金", prob.status)
            w_fallback = np.zeros(N)
            if self.idx_shy != -1:
                w_fallback[self.idx_shy] = 1.0
            else:
                w_fallback[:] = 1.0 / N
            return w_fallback

    def _apply_constraints(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """後處理：清理小數誤差。 CVXPY 已保證所有硬約束，
        此處只需確保總和為 1 且沒有極小負值。
        """
        w = weights.copy().astype(np.float64)
        
        # 清除小於 1e-4 的數值
        w[w < 1e-4] = 0.0
        
        # 重新歸一化以彌補捨去誤差
        total = w.sum()
        if total > 0:
            w = w / total
            
        return w

    def optimize_series(
        self,
        moe_weight_series: np.ndarray,
        expert_utilisation_series: np.ndarray,
        returns_history: np.ndarray,
        lookback: int = 60,
        hmm_bear_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """批次優化整個時間序列（回測用）。

        Parameters
        ----------
        moe_weight_series : np.ndarray
            MoE 權重序列，shape [T, N_assets]。
        expert_utilisation_series : np.ndarray
            Expert 利用率序列，shape [T, N_experts] 或 [N_experts]。
        returns_history : np.ndarray
            完整歷史報酬，shape [T_total, N_assets]。
        lookback : int
            共變異數矩陣計算的回望窗口。
        hmm_bear_probs : np.ndarray | None
            熊市機率序列，shape [T]。

        Returns
        -------
        np.ndarray
            優化後權重序列，shape [T, N_assets]。
        """
        T, N = moe_weight_series.shape
        optimized = np.zeros_like(moe_weight_series, dtype=np.float64)

        # 如果 expert_utilisation 是固定的（非時序），擴展為 [T, N_experts]
        if expert_utilisation_series.ndim == 1:
            expert_util_2d = np.tile(expert_utilisation_series, (T, 1))
        else:
            expert_util_2d = expert_utilisation_series

        offset = len(returns_history) - T
        if offset < 0:
            offset = 0

        for t in range(T):
            # 動態計算共變異數矩陣（使用回望窗口）
            start = max(0, t + offset - lookback)
            end_idx = t + offset
            
            if end_idx - start < 20:
                # 數據不足 → 使用等權
                optimized[t] = moe_weight_series[t]
                continue

            hist_window = returns_history[start:end_idx]
            cov = np.cov(hist_window, rowvar=False)

            # 確保共變異數矩陣有效
            if np.isnan(cov).any() or np.isinf(cov).any():
                optimized[t] = moe_weight_series[t]
                continue

            bp = hmm_bear_probs[t] if hmm_bear_probs is not None else 0.0
            optimized[t] = self.optimize(
                moe_weights=moe_weight_series[t],
                expert_utilisation=expert_util_2d[t],
                cov_matrix=cov,
                returns_history=hist_window,
                hmm_bear_prob=float(bp),
            )

        logger.info(
            "PortfolioOptimizer batch: T=%d  N=%d  lookback=%d",
            T, N, lookback,
        )

        return optimized
