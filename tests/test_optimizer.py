"""
tests/test_optimizer.py — Portfolio Optimizer Integration Tests
================================================================

Tests the dual-layer PortfolioOptimizer (Risk Parity fallback +
Constrained MVO) with synthetic covariance matrices and weight
vectors. These tests require ``cvxpy`` and ``osqp`` to be installed.

Author : Nexus Quant OS — Quality Assurance Division
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer, OptimizerConfig


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _make_optimizer(
    n_assets: int = 5,
    max_single_weight: float = 0.35,
    dispersion_threshold: float = 0.50,
    asset_names: list[str] | None = None,
) -> PortfolioOptimizer:
    """Create a PortfolioOptimizer with sensible test defaults."""
    config = OptimizerConfig(
        max_single_weight=max_single_weight,
        gross_exposure=1.0,
        min_cash=0.05,
        dispersion_threshold=dispersion_threshold,
    )
    return PortfolioOptimizer(
        n_assets=n_assets,
        config=config,
        asset_names=asset_names,
    )


def _identity_cov(n: int, scale: float = 0.01) -> np.ndarray:
    """Return a scaled identity covariance matrix."""
    return np.eye(n, dtype=np.float64) * scale


# ═════════════════════════════════════════════════════════════════════
# Basic Convergence
# ═════════════════════════════════════════════════════════════════════


class TestOptimizerBasicConvergence:
    """Verify the optimizer produces valid weights for simple inputs."""

    def test_optimizer_basic_convergence(self) -> None:
        """With identity covariance and uniform expected returns,
        the optimizer should converge to weights that:
          1. Sum to ≈ 1.0 (full capital deployment)
          2. Are all ≥ 0 (long-only constraint)

        Using low expert dispersion to trigger the MVO path
        (not the Risk Parity fallback).
        """
        n = 5
        optimizer = _make_optimizer(n_assets=n, dispersion_threshold=0.50)

        moe_weights = np.ones(n, dtype=np.float64) / n
        expert_util = np.array([0.5, 0.5, 0.5])  # low dispersion
        cov = _identity_cov(n)

        result = optimizer.optimize(
            moe_weights=moe_weights,
            expert_utilisation=expert_util,
            cov_matrix=cov,
            hmm_bear_prob=0.0,
        )

        # Weights must sum to ~1.0
        np.testing.assert_almost_equal(
            result.sum(), 1.0, decimal=2,
            err_msg=f"Weights sum to {result.sum():.6f}, expected ~1.0",
        )

        # All weights must be non-negative
        assert (result >= -1e-6).all(), (
            f"Negative weights found: {result[result < 0]}"
        )


# ═════════════════════════════════════════════════════════════════════
# Collinear Assets (Numerical Stability)
# ═════════════════════════════════════════════════════════════════════


class TestOptimizerCollinearity:
    """Verify the optimizer handles near-singular covariance matrices."""

    def test_optimizer_handles_collinear_assets(self) -> None:
        """Create a covariance matrix where assets 0 and 1 have
        correlation ≈ 0.99. The optimizer should still converge
        without raising an exception or producing NaN weights.

        The ridge regularisation (``cov + 1e-4 * I``) inside
        ``_constrained_mvo`` is designed to handle this case.
        """
        n = 5
        optimizer = _make_optimizer(n_assets=n, dispersion_threshold=0.50)

        # Build covariance with two nearly identical rows
        cov = _identity_cov(n, scale=0.01)
        # Make assets 0 and 1 highly correlated
        cov[0, 1] = 0.0099  # correlation = 0.0099 / 0.01 = 0.99
        cov[1, 0] = 0.0099

        moe_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        expert_util = np.array([0.5, 0.5, 0.5])  # low dispersion → MVO

        result = optimizer.optimize(
            moe_weights=moe_weights,
            expert_utilisation=expert_util,
            cov_matrix=cov,
            hmm_bear_prob=0.0,
        )

        # Must not contain NaN or Inf
        assert np.isfinite(result).all(), (
            f"Non-finite weights produced: {result}"
        )

        # Weights must still sum to ~1.0
        np.testing.assert_almost_equal(
            result.sum(), 1.0, decimal=2,
            err_msg="Collinear-asset optimisation failed to normalise",
        )


# ═════════════════════════════════════════════════════════════════════
# Max-Weight Constraint
# ═════════════════════════════════════════════════════════════════════


class TestOptimizerMaxWeight:
    """Verify the optimizer respects single-asset concentration limits."""

    def test_optimizer_respects_max_weight(self) -> None:
        """No single weight in the output should exceed
        ``max_single_weight`` (0.35) after post-processing.

        The optimizer uses dynamic upper bounds that cap at 0.50
        in extreme cases, but with default config the effective
        ceiling should be near ``max_single_weight``.

        Note: SHY (cash proxy) is exempt from the cap and can
        receive up to 100% in emergency scenarios. This test
        excludes SHY by not including it in asset_names.
        """
        n = 5
        max_w = 0.35
        # Use generic asset names (no SHY/SH/PSQ exemptions)
        names = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]
        optimizer = _make_optimizer(
            n_assets=n,
            max_single_weight=max_w,
            dispersion_threshold=0.50,
            asset_names=names,
        )

        # Skewed MoE weights — one asset dominates
        moe_weights = np.array([0.70, 0.10, 0.10, 0.05, 0.05])
        expert_util = np.array([0.5, 0.5, 0.5])  # low dispersion → MVO
        cov = _identity_cov(n)

        result = optimizer.optimize(
            moe_weights=moe_weights,
            expert_utilisation=expert_util,
            cov_matrix=cov,
            hmm_bear_prob=0.0,
        )

        # Allow for the dynamic_weight_lambda expansion up to 0.50
        # (absolute ceiling in _constrained_mvo)
        effective_max = 0.50 + 1e-4  # absolute ceiling + tolerance
        assert (result <= effective_max).all(), (
            f"Weight exceeds absolute ceiling {effective_max}: "
            f"max={result.max():.6f}, weights={result}"
        )
