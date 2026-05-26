"""
portfolio/risk_parity.py — 等風險貢獻配置 (Risk Parity)
=========================================================

當 MoE Router 的專家意見分歧過大時，Portfolio Optimizer 會
回退到 Risk Parity 配置：讓每個資產對整體投組的風險貢獻相等。

數學原理：
    風險貢獻 = w_i × (Σ × w)_i / σ_p
    其中 Σ = 共變異數矩陣, σ_p = 投組標準差

    Risk Parity 要求：RC_1 = RC_2 = ... = RC_N
    即每個資產對總風險的邊際貢獻相等

演算法：
    使用 Spinu (2013) 的迭代重新加權法 (Newton-Raphson variant):
    w_i^{new} = w_i × (1/N × σ_p) / (Σ × w)_i

Author : Nexus Quant OS — Portfolio Engineering Division
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("nexus_quant_os.portfolio.risk_parity")


def compute_risk_parity_weights(
    cov_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    risk_budget: np.ndarray | None = None,
) -> np.ndarray:
    """計算等風險貢獻 (Risk Parity) 投組權重。

    Parameters
    ----------
    cov_matrix : np.ndarray
        資產共變異數矩陣，shape [N, N]。
        必須為對稱半正定矩陣。
    max_iter : int
        最大迭代次數。
    tol : float
        收斂容差（相鄰兩次迭代的權重最大變化）。
    risk_budget : np.ndarray, optional
        各資產的風險預算比例，shape [N]。
        預設為 None → 等權風險預算（每資產 1/N）。

    Returns
    -------
    np.ndarray
        Risk Parity 權重，shape [N]，正值且加總為 1.0。

    Notes
    -----
    若共變異數矩陣為奇異或有負特徵值，會自動加入
    微小對角正則化 (ridge) 以確保數值穩定。
    """
    N = cov_matrix.shape[0]
    assert cov_matrix.shape == (N, N), \
        f"共變異數矩陣必須為方陣，got shape {cov_matrix.shape}"

    # ── 數值穩定性：正則化 ────────────────────────────────────────
    cov = cov_matrix.copy().astype(np.float64)

    # 確保對稱
    cov = (cov + cov.T) / 2.0

    # 檢查正定性，若有負特徵值則加正則化
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-12:
        ridge = abs(eigvals.min()) + 1e-8
        cov += np.eye(N) * ridge
        logger.warning(
            "Risk Parity: 共變異數矩陣非正定（min_eigval=%.2e），"
            "已加入 ridge=%.2e 正則化",
            eigvals.min(), ridge,
        )

    # ── 風險預算 ──────────────────────────────────────────────────
    if risk_budget is None:
        budget = np.ones(N, dtype=np.float64) / N
    else:
        budget = np.asarray(risk_budget, dtype=np.float64)
        budget = budget / budget.sum()  # 正規化

    # ── 初始權重：等權 ────────────────────────────────────────────
    w = np.ones(N, dtype=np.float64) / N

    # ── 迭代求解 ──────────────────────────────────────────────────
    for iteration in range(max_iter):
        # 投組變異數的梯度 = Σ × w
        sigma_w = cov @ w                    # [N]
        # 投組標準差
        port_var = float(w @ sigma_w)
        port_std = np.sqrt(max(port_var, 1e-16))

        # 邊際風險貢獻 (Marginal Risk Contribution)
        mrc = sigma_w / port_std             # [N]

        # 風險貢獻 (Risk Contribution)
        rc = w * mrc                         # [N]
        rc_sum = rc.sum() + 1e-16

        # 目標：每個資產的風險貢獻 = budget_i × port_std
        target_rc = budget * port_std

        # 更新權重：Newton-like step
        # w_new = w × target_rc / rc
        ratio = target_rc / (rc + 1e-16)
        w_new = w * ratio

        # 正規化
        w_new = np.maximum(w_new, 1e-8)      # 防止負權重
        w_new = w_new / w_new.sum()

        # 收斂檢查
        change = float(np.max(np.abs(w_new - w)))
        w = w_new

        if change < tol:
            logger.debug(
                "Risk Parity 收斂：iter=%d  max_change=%.2e",
                iteration + 1, change,
            )
            break
    else:
        logger.warning(
            "Risk Parity 未在 %d 次迭代內收斂（max_change=%.2e）",
            max_iter, change,
        )

    # ── 驗證結果 ──────────────────────────────────────────────────
    final_sigma_w = cov @ w
    final_port_std = np.sqrt(max(float(w @ final_sigma_w), 1e-16))
    final_rc = w * final_sigma_w / final_port_std
    rc_dispersion = float(np.std(final_rc) / (np.mean(final_rc) + 1e-16))

    logger.info(
        "Risk Parity 結果：N=%d  port_std=%.6f  "
        "RC_dispersion=%.4f  weights=[%s]",
        N, final_port_std, rc_dispersion,
        ", ".join(f"{x:.4f}" for x in w),
    )

    return w


def compute_inverse_volatility_weights(
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """簡化版反波動率加權。

    比 Risk Parity 更快但不考慮相關性。
    權重 = (1/σ_i) / Σ(1/σ_j)

    Parameters
    ----------
    cov_matrix : np.ndarray
        共變異數矩陣，shape [N, N]。

    Returns
    -------
    np.ndarray
        反波動率權重，shape [N]。
    """
    vols = np.sqrt(np.diag(cov_matrix) + 1e-16)
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()
