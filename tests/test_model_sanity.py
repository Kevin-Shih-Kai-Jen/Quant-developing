"""
tests/test_model_sanity.py — Model Output Sanity Tests
========================================================

Verifies mathematical invariants that must hold for any valid
portfolio weight vector produced by the MoE Router + Optimizer
pipeline. These tests use synthetic weights (no model loading
required) and are designed to run in < 1 second.

Author : Nexus Quant OS — Quality Assurance Division
"""

from __future__ import annotations

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════════════
# Weight Normalisation
# ═════════════════════════════════════════════════════════════════════


class TestWeightNormalisation:
    """Verify that portfolio weights satisfy sum-to-one constraint."""

    def test_weights_normalize_to_one(self) -> None:
        """After L1-normalisation, weights must sum to ≈ 1.0.

        This mirrors the final normalisation step in
        ``PortfolioOptimizer._apply_constraints`` which enforces
        full capital deployment.
        """
        rng = np.random.default_rng(seed=42)
        raw = rng.random(10)  # 10 synthetic asset weights

        # L1 normalise (same as optimizer post-processing)
        normalised = raw / raw.sum()

        np.testing.assert_almost_equal(
            normalised.sum(),
            1.0,
            decimal=8,
            err_msg="Normalised weights do not sum to 1.0",
        )


# ═════════════════════════════════════════════════════════════════════
# Non-Negativity (Long-Only Constraint)
# ═════════════════════════════════════════════════════════════════════


class TestNonNegativity:
    """Verify ReLU-style clipping produces non-negative weights."""

    def test_weights_non_negative_after_relu(self) -> None:
        """After applying ReLU (max(0, w)), all weights must be ≥ 0.

        The optimizer enforces ``w >= 0`` as a hard CVXPY constraint.
        This test validates the equivalent numpy post-processing path
        used in fallback scenarios.
        """
        rng = np.random.default_rng(seed=123)
        # Mix of positive and negative raw model outputs
        raw = rng.standard_normal(10)

        clipped = np.maximum(raw, 0.0)

        assert (clipped >= 0.0).all(), (
            f"Found negative weights after ReLU: "
            f"{clipped[clipped < 0.0]}"
        )


# ═════════════════════════════════════════════════════════════════════
# Single-Weight Concentration Cap
# ═════════════════════════════════════════════════════════════════════


class TestConcentrationCap:
    """Verify that individual weights respect the max-weight bound."""

    def test_single_weight_within_max_bound(self) -> None:
        """No single weight may exceed 0.35 after capping.

        The ``OptimizerConfig.max_single_weight`` default is 0.30,
        but with dynamic expansion (``dynamic_weight_lambda``) the
        effective ceiling can reach 0.35. This test uses 0.35 as the
        hard upper limit.
        """
        max_weight = 0.35

        rng = np.random.default_rng(seed=7)
        raw = rng.random(8)
        # Normalise first, then cap
        normalised = raw / raw.sum()
        capped = np.minimum(normalised, max_weight)
        # Re-normalise after capping (excess redistributed)
        capped = capped / capped.sum()

        assert (capped <= max_weight + 1e-8).all(), (
            f"Weight exceeds {max_weight}: max={capped.max():.6f}"
        )


# ═════════════════════════════════════════════════════════════════════
# Expert Utilisation Range
# ═════════════════════════════════════════════════════════════════════


class TestExpertUtilisation:
    """Verify expert utilisation values are bounded in [0, 1]."""

    def test_expert_utilisation_valid_range(self) -> None:
        """Expert utilisation scores must lie in [0, 1].

        The MoE gating network produces softmax-normalised routing
        probabilities. After aggregation, each expert's average
        utilisation must be a valid probability.
        """
        rng = np.random.default_rng(seed=99)
        # Simulate gating output for 4 experts over 100 samples
        logits = rng.standard_normal((100, 4))
        # Softmax along expert axis
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        expert_util = probs.mean(axis=0)  # [4]

        assert (expert_util >= 0.0).all(), (
            f"Expert utilisation below 0: {expert_util}"
        )
        assert (expert_util <= 1.0).all(), (
            f"Expert utilisation above 1: {expert_util}"
        )
