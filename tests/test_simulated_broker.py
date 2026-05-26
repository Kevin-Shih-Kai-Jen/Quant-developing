"""
tests/test_simulated_broker.py — SimulatedBroker Unit Tests
=============================================================

Tests the paper-trading execution layer: order management,
cash accounting, position reconciliation, and snapshot persistence.

All market data is mocked via ``pytest-mock`` patching of
``yfinance.download`` — no live network calls are made.

Author : Nexus Quant OS — Quality Assurance Division
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from nexus_quant_os.execution.broker_router import SimulatedBroker
from nexus_quant_os.execution.broker_base import OrderIntent


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════


def _mock_yfinance_download(mocker, prices: dict[str, float]):
    """Patch ``yfinance.download`` to return a static price DataFrame.

    Handles both single-ticker (flat columns) and multi-ticker
    (MultiIndex columns) formats, matching real yfinance behavior.
    """
    today = pd.Timestamp.today().normalize()

    def _fake_download(tickers, **kwargs):
        # yfinance receives tickers as a space-separated string
        if isinstance(tickers, str):
            symbols = tickers.strip().split()
        else:
            symbols = list(tickers)

        if len(symbols) == 1:
            # Single ticker: flat columns (Close, Open, High, Low, Volume)
            sym = symbols[0]
            price = prices.get(sym, 100.0)
            df = pd.DataFrame(
                {
                    "Close": [price],
                    "Open": [price * 0.99],
                    "High": [price * 1.01],
                    "Low": [price * 0.98],
                    "Volume": [10_000_000],
                },
                index=pd.DatetimeIndex([today]),
            )
        else:
            # Multi ticker: MultiIndex columns
            data = {}
            for sym in symbols:
                price = prices.get(sym, 100.0)
                data[("Close", sym)] = [price]
                data[("Open", sym)] = [price * 0.99]
                data[("High", sym)] = [price * 1.01]
                data[("Low", sym)] = [price * 0.98]
                data[("Volume", sym)] = [10_000_000]
            df = pd.DataFrame(data, index=pd.DatetimeIndex([today]))

        return df

    mock = mocker.patch(
        "nexus_quant_os.execution.broker_router.yfinance.download",
        side_effect=_fake_download,
    )
    return mock


def _create_broker(db_path: str, initial_capital: float = 100_000.0) -> SimulatedBroker:
    """Instantiate a SimulatedBroker with test defaults and zero slippage."""
    return SimulatedBroker(
        initial_capital=initial_capital,
        slippage_bps=0.0,   # Zero slippage for deterministic tests
        commission_bps=0.0,  # Zero commission for deterministic tests
        db_path=db_path,
    )


# ═════════════════════════════════════════════════════════════════════
# Initial State
# ═════════════════════════════════════════════════════════════════════


class TestInitialState:
    """Verify the broker starts with correct initial conditions."""

    def test_initial_capital(self, tmp_db: str) -> None:
        """A freshly created broker must have account equity equal
        to the configured ``initial_capital`` (default 100,000 USD).
        """
        broker = _create_broker(tmp_db, initial_capital=100_000.0)
        account = broker.get_account()
        assert account.equity == pytest.approx(100_000.0, abs=0.01), (
            f"Initial equity should be 100,000, got {account.equity}"
        )
        assert account.cash == pytest.approx(100_000.0, abs=0.01), (
            f"Initial cash should be 100,000, got {account.cash}"
        )


# ═════════════════════════════════════════════════════════════════════
# Order Execution
# ═════════════════════════════════════════════════════════════════════


class TestOrderExecution:
    """Verify that buy orders correctly update cash and positions."""

    def test_buy_deducts_cash(
        self,
        tmp_db: str,
        sample_prices: dict[str, float],
        mocker,
    ) -> None:
        """Buying SPY via execute() should reduce available cash."""
        _mock_yfinance_download(mocker, sample_prices)
        broker = _create_broker(tmp_db)
        initial_cash = broker.get_account().cash

        # Create a BUY intent and execute
        intent = OrderIntent(
            symbol="SPY",
            side="BUY",
            qty=10.0,
            reason="test buy",
        )
        results = broker.execute([intent])

        # Should have one filled result
        assert len(results) >= 1, "Expected at least one order result"
        assert results[0].status == "FILLED", f"Order status: {results[0].status}"

        # Cash should decrease
        new_cash = broker.get_account().cash
        assert new_cash < initial_cash, (
            f"Cash should decrease after BUY. Before: {initial_cash}, After: {new_cash}"
        )


# ═════════════════════════════════════════════════════════════════════
# Order Sequencing (SELL before BUY)
# ═════════════════════════════════════════════════════════════════════


class TestOrderSequencing:
    """Verify that reconcile() produces SELL orders before BUY orders."""

    def test_sell_before_buy_ordering(
        self,
        tmp_db: str,
        sample_prices: dict[str, float],
        mocker,
    ) -> None:
        """When reconciling from a position in SPY to a position in GLD,
        the SELL SPY intent must come before the BUY GLD intent.
        """
        _mock_yfinance_download(mocker, sample_prices)
        broker = _create_broker(tmp_db)

        # First, buy some SPY to have a position to sell
        buy_intent = OrderIntent(symbol="SPY", side="BUY", qty=50.0, reason="setup")
        broker.execute([buy_intent])

        # Now reconcile to a different allocation (reduce SPY, add GLD)
        target_weights = {"SPY": 0.10, "GLD": 0.40, "TLT": 0.30, "QQQ": 0.20}
        intents = broker.reconcile(target_weights)

        # Find SELL and BUY intents
        sell_indices = [i for i, intent in enumerate(intents) if intent.side == "SELL"]
        buy_indices = [i for i, intent in enumerate(intents) if intent.side == "BUY"]

        if sell_indices and buy_indices:
            assert max(sell_indices) < min(buy_indices), (
                f"SELL intents (indices {sell_indices}) should come before "
                f"BUY intents (indices {buy_indices})"
            )


# ═════════════════════════════════════════════════════════════════════
# Position Reconciliation
# ═════════════════════════════════════════════════════════════════════


class TestReconciliation:
    """Verify target-weight reconciliation logic."""

    def test_reconcile_calculates_delta(
        self,
        tmp_db: str,
        sample_weights: dict[str, float],
        sample_prices: dict[str, float],
        mocker,
    ) -> None:
        """Given target weights and no current positions,
        reconciliation should produce BUY intents for all assets.
        """
        _mock_yfinance_download(mocker, sample_prices)
        broker = _create_broker(tmp_db)

        intents = broker.reconcile(sample_weights)

        # Starting from cash-only, all intents should be BUY
        assert len(intents) > 0, "Expected non-empty intent list"
        for intent in intents:
            assert intent.side == "BUY", (
                f"Expected BUY intent for {intent.symbol}, got {intent.side}"
            )
            assert intent.qty > 0, (
                f"Expected positive qty for {intent.symbol}, got {intent.qty}"
            )


# ═════════════════════════════════════════════════════════════════════
# Daily Snapshot Persistence
# ═════════════════════════════════════════════════════════════════════


class TestSnapshotPersistence:
    """Verify that daily portfolio snapshots are recorded."""

    def test_daily_snapshot(
        self,
        tmp_db: str,
        sample_prices: dict[str, float],
        mocker,
    ) -> None:
        """After calling ``take_daily_snapshot()``, the performance
        history should contain at least one entry.
        """
        _mock_yfinance_download(mocker, sample_prices)
        broker = _create_broker(tmp_db)

        result = broker.take_daily_snapshot()

        # Should return a dict with equity and cash
        assert "equity" in result, "Snapshot should contain 'equity'"
        assert "cash" in result, "Snapshot should contain 'cash'"
        assert result["equity"] == pytest.approx(100_000.0, abs=1.0)

        # Performance history should have the entry
        history = broker.get_performance_history()
        assert len(history) >= 1, (
            f"Expected ≥1 snapshot in history, found {len(history)}"
        )


# ═════════════════════════════════════════════════════════════════════
# Duplicate Prevention
# ═════════════════════════════════════════════════════════════════════


class TestDuplicatePrevention:
    """Verify that duplicate same-day trades are prevented."""

    def test_no_duplicate_same_day(
        self,
        tmp_db: str,
        sample_prices: dict[str, float],
        mocker,
    ) -> None:
        """Executing the same BUY order twice on the same calendar day
        should only produce one FILLED and one CANCELLED result.
        """
        _mock_yfinance_download(mocker, sample_prices)
        broker = _create_broker(tmp_db)

        intent = OrderIntent(symbol="SPY", side="BUY", qty=10.0, reason="test dup")

        # Execute first time
        results_1 = broker.execute([intent])
        assert len(results_1) == 1
        assert results_1[0].status == "FILLED"

        # Execute same order again (same symbol + side + same day)
        results_2 = broker.execute([intent])
        assert len(results_2) == 1
        assert results_2[0].status == "CANCELLED", (
            f"Duplicate same-day order should be CANCELLED, got {results_2[0].status}"
        )
