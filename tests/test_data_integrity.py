"""
tests/test_data_integrity.py — Data Integrity & Configuration Tests
=====================================================================

Validates that core data constants, feature definitions, and universe
configuration are consistent and free of forward-looking bias.

These tests catch silent regressions that slip through when constants
are refactored across modules (e.g. main.py ↔ train_moe.py).

Author : Nexus Quant OS — Quality Assurance Division
"""

from __future__ import annotations


# ═════════════════════════════════════════════════════════════════════
# Feature Column Integrity
# ═════════════════════════════════════════════════════════════════════


class TestFeatureColumns:
    """Ensure FEATURE_COLS is correctly defined and leak-free."""

    def test_no_forward_leakage(self) -> None:
        """``forward_return`` must NEVER appear in FEATURE_COLS.

        If it does, the model has access to future information at
        training time, invalidating all backtest results.
        """
        from nexus_quant_os.training.train_moe import FEATURE_COLS

        assert "forward_return" not in FEATURE_COLS, (
            "CRITICAL: 'forward_return' found in FEATURE_COLS — "
            "this constitutes forward-looking bias (data leakage)."
        )

    def test_feature_cols_expected_count(self) -> None:
        """FEATURE_COLS must contain exactly 9 features.

        4 technical (daily_return, realised_vol, high_low_spread,
        volume_zscore) + 5 macro (cpi_yoy, unemployment_rate,
        pmi_manufacturing, credit_spread, yield_curve_slope).
        """
        from nexus_quant_os.training.train_moe import FEATURE_COLS

        assert len(FEATURE_COLS) == 9, (
            f"Expected 9 features (4 tech + 5 macro), got {len(FEATURE_COLS)}: "
            f"{FEATURE_COLS}"
        )


# ═════════════════════════════════════════════════════════════════════
# Asset Universe Integrity
# ═════════════════════════════════════════════════════════════════════


class TestAssetUniverse:
    """Validate the investable asset universe configuration."""

    def test_asset_universe_contains_expected(self) -> None:
        """ASSET_UNIVERSE must include the core ETF tickers.

        SPY (S&P 500), QQQ (NASDAQ-100), GLD (Gold), and TLT
        (20+ Year Treasuries) form the backbone of the multi-asset
        allocation strategy.
        """
        from nexus_quant_os.training.train_moe import ASSET_UNIVERSE

        required = {"SPY", "QQQ", "GLD", "TLT"}
        missing = required - set(ASSET_UNIVERSE)
        assert not missing, (
            f"Core assets missing from ASSET_UNIVERSE: {missing}"
        )

    def test_asset_universe_sorted(self) -> None:
        """ASSET_UNIVERSE must be alphabetically sorted.

        Sorted order is required for reproducible cross-sectional
        feature concatenation — assets are horizontally stacked in
        alphabetical order to build the [N_assets × N_features]
        input vector.
        """
        from nexus_quant_os.training.train_moe import ASSET_UNIVERSE

        assert ASSET_UNIVERSE == sorted(ASSET_UNIVERSE), (
            f"ASSET_UNIVERSE is not sorted: {ASSET_UNIVERSE}. "
            "This will cause feature ↔ asset misalignment."
        )


# ═════════════════════════════════════════════════════════════════════
# Temporal Configuration
# ═════════════════════════════════════════════════════════════════════


class TestTemporalConfig:
    """Validate data start date and temporal parameters."""

    def test_data_start_is_2020(self) -> None:
        """DATA_START must begin in 2020.

        The training window intentionally covers the COVID crash
        (2020-03), the subsequent recovery, and the 2022–2024
        rate-hiking cycle — all essential regime transitions for
        HMM calibration.
        """
        from nexus_quant_os.training.train_moe import DATA_START

        assert DATA_START.startswith("2020"), (
            f"DATA_START should begin in 2020 to capture COVID-era "
            f"regime transitions, got '{DATA_START}'"
        )
