"""
execution/futu_broker.py — Moomoo (Futu) Paper-Trading Broker Adapter
======================================================================

Connects to a locally-running **FutuOpenD** gateway to execute simulated
trades on the Moomoo platform.  This adapter implements the same
``BrokerBase`` interface as ``SimulatedBroker``, so the rest of the
Nexus Quant OS pipeline works without modification.

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  SAFETY CONSTRAINT: This adapter is HARDCODED to the       ║
║      SIMULATE (paper-trading) environment.  Real-money trading  ║
║      is architecturally impossible through this class.          ║
╚══════════════════════════════════════════════════════════════════╝

Prerequisites
-------------
1. Install: ``pip install moomoo-api``
2. Download & run **Moomoo OpenD** (FutuOpenD) on 127.0.0.1:11111.
3. Log in to OpenD with your Moomoo account credentials.

Data-flow position::

    Risk Firewall  →  reconcile(target_weights)
                   →  execute(intents)
                   →  Moomoo SIMULATE environment

Author : Nexus Quant OS — Execution Engineering Division
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from moomoo import (
        OpenQuoteContext,
        OpenSecTradeContext,
        RET_OK,
        TrdEnv,
        TrdMarket,
        TrdSide,
        OrderType,
        ModifyOrderOp,
        SecurityFirm,
    )

    _MOOMOO_AVAILABLE = True
except ImportError:
    _MOOMOO_AVAILABLE = False

from nexus_quant_os.execution.broker_base import (
    AccountSnapshot,
    BrokerBase,
    OrderIntent,
    OrderResult,
    Position,
)

logger = logging.getLogger("nexus_quant_os.execution.futu_broker")

# US Eastern timezone
try:
    from zoneinfo import ZoneInfo
    _ET: Any = ZoneInfo("America/New_York")
except ImportError:
    _ET = timezone(timedelta(hours=-5))  # EST fallback (conservative)


# ═════════════════════════════════════════════════════════════════════
# MOOMOO BROKER ADAPTER (SIMULATE ONLY)
# ═════════════════════════════════════════════════════════════════════


class FutuBroker(BrokerBase):
    """Moomoo/Futu broker adapter — **SIMULATION ONLY**.

    Connects to a local FutuOpenD gateway for US equity paper trading.
    All ``trd_env`` arguments are hardcoded to ``TrdEnv.SIMULATE``.

    Parameters
    ----------
    host : str
        FutuOpenD gateway address.  Default ``'127.0.0.1'``.
    port : int
        FutuOpenD gateway port.  Default ``11111``.
    security_firm : SecurityFirm, optional
        The security firm to use.  Default ``SecurityFirm.FUTUINC`` for
        US-based Moomoo.
    min_order_value : float
        Minimum notional order value (USD) to avoid micro-orders.
        Default ``50.0``.
    rebalance_threshold : float
        Minimum weight delta to trigger a trade.  Default ``0.02`` (2%).

    Raises
    ------
    ImportError
        If ``moomoo-api`` is not installed.
    ConnectionError
        If FutuOpenD is not running on the specified host/port.
    """

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  This constant CANNOT be overridden. Real trading is        ║
    # ║  architecturally blocked at every method call.              ║
    # ╚══════════════════════════════════════════════════════════════╝
    _TRD_ENV = TrdEnv.SIMULATE if _MOOMOO_AVAILABLE else None

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        security_firm: Any = None,
        min_order_value: float = 50.0,
        rebalance_threshold: float = 0.02,
    ) -> None:
        if not _MOOMOO_AVAILABLE:
            raise ImportError(
                "moomoo-api is not installed.  Run: pip install moomoo-api"
            )

        self._host = host
        self._port = port
        self._security_firm = security_firm or SecurityFirm.FUTUINC
        self._min_order_value = min_order_value
        self._rebalance_threshold = rebalance_threshold

        # Verify connectivity on startup
        self._verify_connection()
        logger.info(
            "FutuBroker initialised — SIMULATE mode on %s:%d",
            host, port,
        )

    # ── Connection helpers ────────────────────────────────────────

    def _verify_connection(self) -> None:
        """Verify that FutuOpenD is reachable."""
        import socket
        # Pre-check TCP connection to fail fast and avoid the internal moomoo-api infinite retry loop
        try:
            with socket.create_connection((self._host, self._port), timeout=1.0):
                pass
        except Exception as exc:
            raise ConnectionError(
                f"Cannot establish TCP connection to FutuOpenD at {self._host}:{self._port}. "
                f"Is FutuOpenD running and listening?  Error: {exc}"
            ) from exc

        ctx = None
        try:
            ctx = OpenQuoteContext(host=self._host, port=self._port)
            ret, data = ctx.get_global_state()
            if ret != RET_OK:
                raise ConnectionError(
                    f"FutuOpenD returned error: {data}"
                )
            logger.info("FutuOpenD connection verified (state: OK)")
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to FutuOpenD at {self._host}:{self._port}. "
                f"Is OpenD running and logged in?  Error: {exc}"
            ) from exc
        finally:
            if ctx is not None:
                ctx.close()

    def _get_quote_ctx(self) -> OpenQuoteContext:
        """Create a fresh quote context (caller must close)."""
        return OpenQuoteContext(host=self._host, port=self._port)

    def _get_trade_ctx(self) -> OpenSecTradeContext:
        """Create a fresh trade context for US market (caller must close)."""
        return OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=self._host,
            port=self._port,
            security_firm=self._security_firm,
        )

    # ── Order hygiene ─────────────────────────────────────────────

    def cancel_all_pending(self) -> int:
        """Cancel all SUBMITTED (pending) orders in the SIMULATE account.

        This prevents cash from being frozen by stale limit orders that
        were placed outside market hours and never filled.

        Returns
        -------
        int
            Number of orders successfully cancelled.
        """
        ctx = None
        cancelled = 0
        try:
            ctx = self._get_trade_ctx()
            ret, orders = ctx.order_list_query(trd_env=self._TRD_ENV)
            if ret != RET_OK:
                logger.error("order_list_query failed: %s", orders)
                return 0

            pending = orders[orders["order_status"] == "SUBMITTED"]
            if pending.empty:
                logger.info("No pending orders to cancel")
                return 0

            logger.info(
                "Found %d pending orders — cancelling to free cash",
                len(pending),
            )

            for _, row in pending.iterrows():
                order_id = str(row["order_id"])
                code = str(row.get("code", "?"))
                ret2, _ = ctx.modify_order(
                    modify_order_op=ModifyOrderOp.CANCEL,
                    order_id=order_id,
                    qty=0,
                    price=0,
                    trd_env=self._TRD_ENV,
                )
                if ret2 == RET_OK:
                    logger.info("Cancelled pending order: %s (id=%s)", code, order_id)
                    cancelled += 1
                else:
                    logger.warning("Failed to cancel order %s", order_id)
                time.sleep(0.5)  # rate limit

        except Exception as exc:
            logger.error("cancel_all_pending failed: %s", exc)
        finally:
            if ctx is not None:
                ctx.close()

        return cancelled

    # ── Symbol conversion ─────────────────────────────────────────

    @staticmethod
    def _to_futu_code(symbol: str) -> str:
        """Convert bare ticker to Moomoo code format.

        'AAPL' → 'US.AAPL'
        """
        if symbol.startswith("US."):
            return symbol
        return f"US.{symbol}"

    @staticmethod
    def _from_futu_code(futu_code: str) -> str:
        """Convert Moomoo code format to bare ticker.

        'US.AAPL' → 'AAPL'
        """
        if futu_code.startswith("US."):
            return futu_code[3:]
        return futu_code

    # ── Price helper ──────────────────────────────────────────────

    def _get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current market prices from Moomoo quote API.

        Parameters
        ----------
        symbols : list[str]
            Bare tickers (e.g. ``['AAPL', 'MSFT']``).

        Returns
        -------
        dict[str, float]
            Symbol → latest price.  Symbols that fail are omitted.
        """
        if not symbols:
            return {}

        futu_codes = [self._to_futu_code(s) for s in symbols]
        prices: dict[str, float] = {}
        ctx = None

        try:
            ctx = self._get_quote_ctx()
            ret, data = ctx.get_market_snapshot(futu_codes)
            if ret != RET_OK:
                logger.error("get_market_snapshot failed: %s", data)
                return prices

            for _, row in data.iterrows():
                code = str(row.get("code", ""))
                last_price = row.get("last_price", 0.0)
                sym = self._from_futu_code(code)
                if last_price and float(last_price) > 0:
                    prices[sym] = float(last_price)
                else:
                    logger.warning("No valid price for %s", sym)
        except Exception as exc:
            logger.error("Price fetch failed: %s", exc)
        finally:
            if ctx is not None:
                ctx.close()

        return prices

    # ── BrokerBase implementation ─────────────────────────────────

    def get_account(self) -> AccountSnapshot:
        """Query the Moomoo SIMULATE account for balances.

        Returns
        -------
        AccountSnapshot
            Current equity, cash, and buying power from the paper
            trading account.
        """
        ctx = None
        try:
            ctx = self._get_trade_ctx()
            ret, data = ctx.accinfo_query(trd_env=self._TRD_ENV)
            if ret != RET_OK:
                logger.error("accinfo_query failed: %s", data)
                return AccountSnapshot(
                    equity=0.0, cash=0.0, buying_power=0.0,
                    timestamp=datetime.now(timezone.utc),
                )

            row = data.iloc[0]
            equity = float(row.get("total_assets", 0.0))
            cash = float(row.get("cash", 0.0))
            buying_power = float(row.get("avl_withdrawal_cash", cash))

            return AccountSnapshot(
                equity=equity,
                cash=cash,
                buying_power=buying_power,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.error("get_account failed: %s", exc)
            return AccountSnapshot(
                equity=0.0, cash=0.0, buying_power=0.0,
                timestamp=datetime.now(timezone.utc),
            )
        finally:
            if ctx is not None:
                ctx.close()

    def get_positions(self) -> list[Position]:
        """Query all open positions from the Moomoo SIMULATE account.

        Returns
        -------
        list[Position]
            One ``Position`` per held US equity.
        """
        positions: list[Position] = []
        ctx = None
        try:
            ctx = self._get_trade_ctx()
            ret, data = ctx.position_list_query(trd_env=self._TRD_ENV)
            if ret != RET_OK:
                logger.error("position_list_query failed: %s", data)
                return positions

            for _, row in data.iterrows():
                code = str(row.get("code", ""))
                # Only include US positions
                if not code.startswith("US."):
                    continue

                qty = float(row.get("qty", 0))
                if qty <= 0:
                    continue

                cost_price = float(row.get("cost_price", 0))
                market_val = float(row.get("market_val", 0))
                current_price = float(row.get("nominal_price", 0))
                pl_val = float(row.get("pl_val", 0))

                cost_basis = qty * cost_price if cost_price > 0 else 1.0
                pl_pct = (pl_val / cost_basis) * 100.0 if cost_basis > 0 else 0.0

                positions.append(Position(
                    symbol=self._from_futu_code(code),
                    qty=qty,
                    avg_cost=cost_price,
                    current_price=current_price,
                    market_value=market_val,
                    unrealized_pl=pl_val,
                    unrealized_pl_pct=pl_pct,
                ))
        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
        finally:
            if ctx is not None:
                ctx.close()

        return positions

    def reconcile(
        self,
        target_weights: dict[str, float],
    ) -> list[OrderIntent]:
        """Compare current holdings to target and produce trade intents.

        Sells are placed before buys to free up cash.

        Parameters
        ----------
        target_weights : dict[str, float]
            ``{symbol: target_weight}`` where weight is a fraction
            of total portfolio equity.

        Returns
        -------
        list[OrderIntent]
            Sorted: sells first, then buys.
        """
        account = self.get_account()
        equity = account.equity

        if equity <= 0:
            logger.warning("Reconcile skipped — equity is %.2f", equity)
            return []

        positions = self.get_positions()
        current_values: dict[str, float] = {
            p.symbol: p.market_value for p in positions
        }
        current_prices: dict[str, float] = {
            p.symbol: p.current_price for p in positions
        }
        current_qty: dict[str, float] = {
            p.symbol: p.qty for p in positions
        }

        # Fetch prices for new symbols not yet in portfolio
        new_symbols = [
            s for s in target_weights if s not in current_prices
        ]
        if new_symbols:
            fresh_prices = self._get_prices(new_symbols)
            current_prices.update(fresh_prices)

        sells: list[OrderIntent] = []
        buys: list[OrderIntent] = []

        # Build intents for each target position
        all_symbols = set(list(target_weights.keys()) + list(current_values.keys()))
        for symbol in all_symbols:
            target_w = target_weights.get(symbol, 0.0)
            current_val = current_values.get(symbol, 0.0)
            current_w = current_val / equity if equity > 0 else 0.0
            delta_w = target_w - current_w

            price = current_prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(
                    "Skipping %s — no price available", symbol,
                )
                continue

            # Skip if delta is below threshold
            if abs(delta_w) < self._rebalance_threshold:
                continue

            delta_value = delta_w * equity
            delta_qty = abs(delta_value) / price

            # Skip micro-orders
            if abs(delta_value) < self._min_order_value:
                continue

            # For sells, cap to held quantity
            if delta_w < 0:
                held = current_qty.get(symbol, 0.0)
                delta_qty = min(delta_qty, held)
                if delta_qty < 0.01:
                    continue

                # Moomoo requires integer quantities for US stocks
                delta_qty = int(delta_qty)
                if delta_qty < 1:
                    continue

                sells.append(OrderIntent(
                    symbol=symbol,
                    side="SELL",
                    qty=float(delta_qty),
                    reason=(
                        f"Rebalance: current {current_w:.1%} → "
                        f"target {target_w:.1%} (Δ{delta_w:+.1%})"
                    ),
                ))
            else:
                # Moomoo requires integer quantities for US stocks
                delta_qty = int(delta_qty)
                if delta_qty < 1:
                    continue

                buys.append(OrderIntent(
                    symbol=symbol,
                    side="BUY",
                    qty=float(delta_qty),
                    reason=(
                        f"Rebalance: current {current_w:.1%} → "
                        f"target {target_w:.1%} (Δ{delta_w:+.1%})"
                    ),
                ))

        # Sells first to free cash
        return sells + buys

    def execute(
        self,
        intents: list[OrderIntent],
    ) -> list[OrderResult]:
        """Submit orders to Moomoo SIMULATE environment.

        Parameters
        ----------
        intents : list[OrderIntent]
            Orders to execute (typically from ``reconcile()``).

        Returns
        -------
        list[OrderResult]
            One result per intent.
        """
        if not intents:
            return []

        results: list[OrderResult] = []
        ctx = None

        try:
            ctx = self._get_trade_ctx()

            for intent in intents:
                futu_code = self._to_futu_code(intent.symbol)
                side = (
                    TrdSide.BUY if intent.side == "BUY"
                    else TrdSide.SELL
                )

                # Get current price for the limit order
                prices = self._get_prices([intent.symbol])
                price = prices.get(intent.symbol)
                if price is None or price <= 0:
                    logger.warning(
                        "REJECTED %s %s — no price available",
                        intent.side, intent.symbol,
                    )
                    results.append(OrderResult(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        filled_price=0.0,
                        commission=0.0,
                        timestamp=datetime.now(timezone.utc),
                        order_id="NO_PRICE",
                        status="REJECTED",
                    ))
                    continue

                # Ensure integer quantity (US equities)
                qty = max(1, int(intent.qty))

                logger.info(
                    "Placing %s order: %s x%d @ $%.2f (SIMULATE)",
                    intent.side, intent.symbol, qty, price,
                )

                # ── Place the order ───────────────────────────────
                # Use MARKET order for simulation to ensure fills.
                # Some Moomoo simulate environments only support
                # NORMAL (limit) — fallback if MARKET fails.
                ret, data = ctx.place_order(
                    price=price,
                    qty=qty,
                    code=futu_code,
                    trd_side=side,
                    order_type=OrderType.NORMAL,
                    trd_env=self._TRD_ENV,  # ALWAYS SIMULATE
                )

                if ret == RET_OK:
                    order_id = str(data.iloc[0].get("order_id", ""))
                    logger.info(
                        "Order SUBMITTED: %s %s x%d — order_id=%s",
                        intent.side, intent.symbol, qty, order_id,
                    )
                    results.append(OrderResult(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=float(qty),
                        filled_price=price,
                        commission=0.0,  # Moomoo simulate has no commission
                        timestamp=datetime.now(timezone.utc),
                        order_id=order_id,
                        status="SUBMITTED",
                    ))
                else:
                    error_msg = str(data)
                    logger.error(
                        "Order REJECTED: %s %s x%d — %s",
                        intent.side, intent.symbol, qty, error_msg,
                    )
                    results.append(OrderResult(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=float(qty),
                        filled_price=0.0,
                        commission=0.0,
                        timestamp=datetime.now(timezone.utc),
                        order_id=f"REJECTED_{error_msg[:50]}",
                        status="REJECTED",
                    ))

                # Rate limit: Moomoo has a 30 req/30s limit
                time.sleep(1.0)

        except Exception as exc:
            logger.error("Execute failed: %s", exc)
            # Mark remaining intents as cancelled
            executed_symbols = {r.symbol for r in results}
            for intent in intents:
                if intent.symbol not in executed_symbols:
                    results.append(OrderResult(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        filled_price=0.0,
                        commission=0.0,
                        timestamp=datetime.now(timezone.utc),
                        order_id="ERROR",
                        status="CANCELLED",
                    ))
        finally:
            if ctx is not None:
                ctx.close()

        return results

    def is_market_open(self) -> bool:
        """Check if US market is currently in regular session.

        Uses the same ET-based logic as the simulated broker:
        weekdays, 9:30–16:00 ET.

        Returns
        -------
        bool
            ``True`` if NYSE is open for regular trading.
        """
        now_et = datetime.now(_ET)

        # Weekends
        if now_et.weekday() >= 5:
            return False

        market_open = now_et.replace(
            hour=9, minute=30, second=0, microsecond=0,
        )
        market_close = now_et.replace(
            hour=16, minute=0, second=0, microsecond=0,
        )
        return market_open <= now_et < market_close

    def get_portfolio_value(self) -> float:
        """Return total portfolio equity from Moomoo SIMULATE account.

        Returns
        -------
        float
            Total equity in USD.
        """
        return self.get_account().equity

    def take_daily_snapshot(self) -> None:
        """Daily snapshot — not yet implemented for FutuBroker."""
        logger.warning("take_daily_snapshot() not implemented for FutuBroker")

    def get_performance_history(self, days: int = 30) -> list:
        """Performance history — not yet implemented for FutuBroker."""
        logger.warning("get_performance_history() not implemented for FutuBroker")
        return []
