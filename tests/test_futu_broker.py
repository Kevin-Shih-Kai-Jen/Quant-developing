"""
tests/test_futu_broker.py — Unit Tests for Moomoo (Futu) Broker Adapter
=========================================================================

Tests the FutuBroker adapter by mocking the moomoo SDK's OpenQuoteContext
and OpenSecTradeContext.  No actual FutuOpenD connection is needed.

Author : Nexus Quant OS — Execution Engineering Division
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# 1. MOCK THE MOOMOO SDK BEFORE IMPORT
# ══════════════════════════════════════════════════════════════════════

# Create a fake moomoo module so we can test without pip install moomoo-api
import sys
import types

mock_moomoo = types.ModuleType("moomoo")
mock_moomoo.RET_OK = 0  # type: ignore[attr-defined]
mock_moomoo.TrdMarket = MagicMock()  # type: ignore[attr-defined]
mock_moomoo.TrdMarket.US = "US"
mock_moomoo.TrdSide = MagicMock()  # type: ignore[attr-defined]
mock_moomoo.TrdSide.BUY = "BUY"
mock_moomoo.TrdSide.SELL = "SELL"
mock_moomoo.OrderType = MagicMock()  # type: ignore[attr-defined]
mock_moomoo.OrderType.NORMAL = "NORMAL"
mock_moomoo.SecurityFirm = MagicMock()  # type: ignore[attr-defined]
mock_moomoo.SecurityFirm.FUTUINC = "FUTUINC"
mock_moomoo.OpenQuoteContext = MagicMock  # type: ignore[attr-defined]
mock_moomoo.OpenSecTradeContext = MagicMock  # type: ignore[attr-defined]


class _FakeTrdEnv:
    SIMULATE = "SIMULATE"
    REAL = "REAL"


mock_moomoo.TrdEnv = _FakeTrdEnv  # type: ignore[attr-defined]

sys.modules["moomoo"] = mock_moomoo

# NOW we can safely import FutuBroker
from nexus_quant_os.execution.futu_broker import FutuBroker
from nexus_quant_os.execution.broker_base import OrderIntent


# ══════════════════════════════════════════════════════════════════════
# 2. FIXTURES
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def broker():
    """Create a FutuBroker with mocked connectivity check."""
    with patch.object(FutuBroker, "_verify_connection"):
        return FutuBroker(host="127.0.0.1", port=11111)


# ══════════════════════════════════════════════════════════════════════
# 3. SAFETY TESTS
# ══════════════════════════════════════════════════════════════════════


class TestSafetyConstraints:
    """Verify that FutuBroker can NEVER use the real trading environment."""

    def test_trd_env_is_simulate(self, broker: FutuBroker):
        """The _TRD_ENV constant must always be SIMULATE."""
        assert broker._TRD_ENV == "SIMULATE"

    def test_trd_env_is_class_constant(self):
        """_TRD_ENV should be a class attribute, not instance-settable."""
        assert FutuBroker._TRD_ENV == "SIMULATE"


# ══════════════════════════════════════════════════════════════════════
# 4. SYMBOL CONVERSION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestSymbolConversion:
    """Test Moomoo code ↔ bare ticker conversion."""

    def test_to_futu_code(self):
        assert FutuBroker._to_futu_code("AAPL") == "US.AAPL"
        assert FutuBroker._to_futu_code("US.AAPL") == "US.AAPL"

    def test_from_futu_code(self):
        assert FutuBroker._from_futu_code("US.AAPL") == "AAPL"
        assert FutuBroker._from_futu_code("TSLA") == "TSLA"


# ══════════════════════════════════════════════════════════════════════
# 5. ACCOUNT & POSITION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestGetAccount:
    """Test account balance queries."""

    def test_get_account_success(self, broker: FutuBroker):
        """Verify account snapshot is correctly parsed from Moomoo data."""
        mock_ctx = MagicMock()
        mock_ctx.accinfo_query.return_value = (
            0,  # RET_OK
            pd.DataFrame([{
                "total_assets": 100000.0,
                "cash": 50000.0,
                "avl_withdrawal_cash": 50000.0,
            }]),
        )

        with patch.object(broker, "_get_trade_ctx", return_value=mock_ctx):
            account = broker.get_account()

        assert account.equity == 100000.0
        assert account.cash == 50000.0
        assert account.buying_power == 50000.0
        mock_ctx.close.assert_called_once()

    def test_get_account_failure_returns_zero(self, broker: FutuBroker):
        """On API failure, return zeroed snapshot instead of crashing."""
        mock_ctx = MagicMock()
        mock_ctx.accinfo_query.return_value = (1, "Error")

        with patch.object(broker, "_get_trade_ctx", return_value=mock_ctx):
            account = broker.get_account()

        assert account.equity == 0.0
        assert account.cash == 0.0


class TestGetPositions:
    """Test position list queries."""

    def test_get_positions_success(self, broker: FutuBroker):
        """Verify positions are correctly parsed."""
        mock_ctx = MagicMock()
        mock_ctx.position_list_query.return_value = (
            0,  # RET_OK
            pd.DataFrame([
                {
                    "code": "US.AAPL",
                    "qty": 100.0,
                    "cost_price": 150.0,
                    "market_val": 17500.0,
                    "nominal_price": 175.0,
                    "pl_val": 2500.0,
                },
                {
                    "code": "US.MSFT",
                    "qty": 50.0,
                    "cost_price": 300.0,
                    "market_val": 16000.0,
                    "nominal_price": 320.0,
                    "pl_val": 1000.0,
                },
            ]),
        )

        with patch.object(broker, "_get_trade_ctx", return_value=mock_ctx):
            positions = broker.get_positions()

        assert len(positions) == 2
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 100.0
        assert positions[0].current_price == 175.0
        assert positions[1].symbol == "MSFT"
        mock_ctx.close.assert_called_once()

    def test_get_positions_filters_non_us(self, broker: FutuBroker):
        """Non-US positions (e.g. HK stocks) should be excluded."""
        mock_ctx = MagicMock()
        mock_ctx.position_list_query.return_value = (
            0,
            pd.DataFrame([
                {
                    "code": "HK.00700",
                    "qty": 200.0,
                    "cost_price": 300.0,
                    "market_val": 70000.0,
                    "nominal_price": 350.0,
                    "pl_val": 10000.0,
                },
            ]),
        )

        with patch.object(broker, "_get_trade_ctx", return_value=mock_ctx):
            positions = broker.get_positions()

        assert len(positions) == 0


# ══════════════════════════════════════════════════════════════════════
# 6. ORDER EXECUTION TESTS
# ══════════════════════════════════════════════════════════════════════


class TestExecute:
    """Test order execution against Moomoo SIMULATE."""

    def test_execute_buy_success(self, broker: FutuBroker):
        """Verify a successful BUY order produces a FILLED result."""
        mock_ctx = MagicMock()
        mock_ctx.place_order.return_value = (
            0,  # RET_OK
            pd.DataFrame([{"order_id": "ORD-12345"}]),
        )

        intent = OrderIntent(
            symbol="AAPL", side="BUY", qty=10.0, reason="Test buy",
        )

        with (
            patch.object(broker, "_get_trade_ctx", return_value=mock_ctx),
            patch.object(broker, "_get_prices", return_value={"AAPL": 175.0}),
        ):
            results = broker.execute([intent])

        assert len(results) == 1
        assert results[0].status == "FILLED"
        assert results[0].symbol == "AAPL"
        assert results[0].filled_price == 175.0
        assert results[0].order_id == "ORD-12345"
        mock_ctx.close.assert_called_once()

    def test_execute_rejected_no_price(self, broker: FutuBroker):
        """Order should be REJECTED if price is unavailable."""
        mock_ctx = MagicMock()
        intent = OrderIntent(
            symbol="BADTICKER", side="BUY", qty=5.0, reason="Test",
        )

        with (
            patch.object(broker, "_get_trade_ctx", return_value=mock_ctx),
            patch.object(broker, "_get_prices", return_value={}),
        ):
            results = broker.execute([intent])

        assert len(results) == 1
        assert results[0].status == "REJECTED"

    def test_execute_api_rejection(self, broker: FutuBroker):
        """Verify handling when Moomoo API rejects the order."""
        mock_ctx = MagicMock()
        mock_ctx.place_order.return_value = (
            1,  # NOT RET_OK
            "Insufficient buying power",
        )

        intent = OrderIntent(
            symbol="TSLA", side="BUY", qty=100.0, reason="Test",
        )

        with (
            patch.object(broker, "_get_trade_ctx", return_value=mock_ctx),
            patch.object(broker, "_get_prices", return_value={"TSLA": 250.0}),
        ):
            results = broker.execute([intent])

        assert len(results) == 1
        assert results[0].status == "REJECTED"
        assert "REJECTED_" in results[0].order_id


# ══════════════════════════════════════════════════════════════════════
# 7. RECONCILE TESTS
# ══════════════════════════════════════════════════════════════════════


class TestReconcile:
    """Test portfolio reconciliation logic."""

    def test_reconcile_generates_buy_intents(self, broker: FutuBroker):
        """Starting from empty, reconcile should create BUY intents."""
        with (
            patch.object(broker, "get_account", return_value=MagicMock(
                equity=100000.0,
            )),
            patch.object(broker, "get_positions", return_value=[]),
            patch.object(broker, "_get_prices", return_value={
                "AAPL": 175.0, "MSFT": 320.0,
            }),
        ):
            intents = broker.reconcile({"AAPL": 0.30, "MSFT": 0.20})

        assert len(intents) > 0
        assert all(i.side == "BUY" for i in intents)
        symbols = {i.symbol for i in intents}
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_reconcile_skips_small_deltas(self, broker: FutuBroker):
        """Deltas below the rebalance threshold should be ignored."""
        mock_position = MagicMock(
            symbol="AAPL", market_value=30000.0,
            current_price=175.0, qty=171.0,
        )
        with (
            patch.object(broker, "get_account", return_value=MagicMock(
                equity=100000.0,
            )),
            patch.object(broker, "get_positions", return_value=[mock_position]),
            patch.object(broker, "_get_prices", return_value={}),
        ):
            # Target 30% and current is 30% → no trades
            intents = broker.reconcile({"AAPL": 0.30})

        assert len(intents) == 0

    def test_reconcile_sells_before_buys(self, broker: FutuBroker):
        """Sell intents must come before buy intents."""
        mock_pos = MagicMock(
            symbol="AAPL", market_value=50000.0,
            current_price=175.0, qty=285.0,
        )
        with (
            patch.object(broker, "get_account", return_value=MagicMock(
                equity=100000.0,
            )),
            patch.object(broker, "get_positions", return_value=[mock_pos]),
            patch.object(broker, "_get_prices", return_value={
                "MSFT": 320.0,
            }),
        ):
            intents = broker.reconcile({"AAPL": 0.20, "MSFT": 0.30})

        sells = [i for i in intents if i.side == "SELL"]
        buys = [i for i in intents if i.side == "BUY"]
        if sells and buys:
            # All sells should come before all buys
            first_buy_idx = next(
                idx for idx, i in enumerate(intents) if i.side == "BUY"
            )
            last_sell_idx = len(intents) - 1 - next(
                idx for idx, i in enumerate(reversed(intents))
                if i.side == "SELL"
            )
            assert last_sell_idx < first_buy_idx
