"""
execution/broker_router.py — Simulated Paper-Trading Broker Engine
====================================================================

A fully self-contained paper-trading broker that implements the
``BrokerBase`` interface.  All state is persisted in a local SQLite
database so the simulated portfolio survives process restarts.

Features
--------
- SQLite-backed account, positions, trades, and daily snapshots.
- Configurable slippage and commission models (basis-point based).
- 5-minute price cache via ``yfinance`` (no paid data feed needed).
- Reconciliation: sells first to free cash, then buys.
- Duplicate-trade guard: same symbol + side + calendar day → skip.
- NYSE market-hours check (9:30–16:00 ET, weekdays only).

Data-flow position::

    Risk Firewall  →  reconcile(target_weights)
                   →  execute(intents)
                   →  SQLite ← take_daily_snapshot()

Author : Nexus Quant OS — Execution Engineering Division
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any

import yfinance

from nexus_quant_os.execution.broker_base import (
    AccountSnapshot,
    BrokerBase,
    OrderIntent,
    OrderResult,
    Position,
)

logger = logging.getLogger("nexus_quant_os.execution.broker_router")

# Project root: three levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# US Eastern timezone offset (standard: UTC-5, daylight: UTC-4)
# Using a simple fixed approach — production would use ``zoneinfo``.
try:
    from zoneinfo import ZoneInfo
    _ET: Any = ZoneInfo("America/New_York")
except ImportError:  # Python < 3.9 fallback
    _ET = timezone(timedelta(hours=-4))  # EDT approximation


# ═════════════════════════════════════════════════════════════════════
# 1. SCHEMA
# ═════════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS account (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    cash          REAL    NOT NULL,
    initial_capital REAL  NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol        TEXT    PRIMARY KEY,
    qty           REAL    NOT NULL DEFAULT 0,
    avg_cost      REAL    NOT NULL DEFAULT 0,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id      TEXT    PRIMARY KEY,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    qty           REAL    NOT NULL,
    filled_price  REAL    NOT NULL,
    commission    REAL    NOT NULL,
    executed_at   TEXT    NOT NULL,
    trade_date    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_snapshots (
    snapshot_date TEXT    PRIMARY KEY,
    equity        REAL    NOT NULL,
    cash          REAL    NOT NULL,
    positions_json TEXT   NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


# ═════════════════════════════════════════════════════════════════════
# 2. PRICE CACHE
# ═════════════════════════════════════════════════════════════════════

@dataclass
class _PriceCacheEntry:
    """In-memory cache entry for a batch of ticker prices."""

    prices: dict[str, float]
    fetched_at: datetime


class _PriceCache:
    """TTL-based price cache backed by ``yfinance``.

    Parameters
    ----------
    ttl_seconds : int
        Maximum age of cached prices before a re-fetch (default 300 s).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, _PriceCacheEntry] = {}

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return current prices for *symbols*, using cache when valid.

        Parameters
        ----------
        symbols : list[str]
            Ticker symbols to price.

        Returns
        -------
        dict[str, float]
            Mapping of symbol → latest price.  Symbols that cannot be
            priced are omitted from the result.
        """
        now = datetime.now(timezone.utc)
        result: dict[str, float] = {}
        stale: list[str] = []

        for sym in symbols:
            entry = self._cache.get(sym)
            if entry and (now - entry.fetched_at) < self._ttl:
                if sym in entry.prices:
                    result[sym] = entry.prices[sym]
                    continue
            stale.append(sym)

        if stale:
            fresh = self._fetch(stale)
            fetch_time = datetime.now(timezone.utc)
            for sym, price in fresh.items():
                self._cache[sym] = _PriceCacheEntry(
                    prices={sym: price}, fetched_at=fetch_time,
                )
            result.update(fresh)

        return result

    @staticmethod
    def _fetch(symbols: list[str]) -> dict[str, float]:
        """Download latest prices from Yahoo Finance.

        Parameters
        ----------
        symbols : list[str]
            Tickers to download.

        Returns
        -------
        dict[str, float]
            Symbol → close price.
        """
        prices: dict[str, float] = {}
        if not symbols:
            return prices

        tickers_str = " ".join(symbols)
        logger.debug("yfinance fetch: %s", tickers_str)

        try:
            df = yfinance.download(
                tickers_str,
                period="1d",
                progress=False,
                threads=False,
            )
            if df.empty:
                logger.warning("yfinance returned empty DataFrame for %s", tickers_str)
                return prices

            # yfinance returns MultiIndex columns for multiple tickers:
            #   ('Close', 'AAPL'), ('Close', 'MSFT'), ...
            # For a single ticker the columns are just: 'Close', 'Open', ...
            if len(symbols) == 1:
                sym = symbols[0]
                if "Close" in df.columns:
                    val = df["Close"].iloc[-1]
                    if hasattr(val, "item"):
                        val = val.item()
                    val = float(val)
                    if not math.isnan(val):
                        prices[sym] = val
                    else:
                        logger.warning("NaN price for %s — skipping", sym)
            else:
                for sym in symbols:
                    try:
                        col = ("Close", sym)
                        if col in df.columns:
                            val = df[col].iloc[-1]
                            if hasattr(val, "item"):
                                val = val.item()
                            val = float(val)
                            if not math.isnan(val):
                                prices[sym] = val
                            else:
                                logger.warning("NaN price for %s — skipping", sym)
                    except (KeyError, IndexError):
                        logger.warning("No price data for %s", sym)

        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)

        return prices


# ═════════════════════════════════════════════════════════════════════
# 3. SIMULATED BROKER
# ═════════════════════════════════════════════════════════════════════

class SimulatedBroker(BrokerBase):
    """Paper-trading broker with SQLite persistence.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance for a fresh account (default $100k).
    slippage_bps : float
        Max slippage in basis points applied symmetrically around the
        mid-price (default 5 bps = 0.05%).
    commission_bps : float
        Commission rate in basis points of notional value (default 5 bps).
    db_path : str | Path | None
        Path to the SQLite database file.  Defaults to
        ``<project_root>/data/paper_trading.db``.

    Usage
    -----
    >>> broker = SimulatedBroker(initial_capital=50_000)
    >>> intents = broker.reconcile({"AAPL": 0.25, "MSFT": 0.20})
    >>> results = broker.execute(intents)
    >>> broker.take_daily_snapshot()
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 5.0,
        db_path: str | Path | None = None,
    ) -> None:
        self._initial_capital = initial_capital
        self._slippage_bps = slippage_bps
        self._commission_bps = commission_bps

        # Resolve database path
        if db_path is None:
            self._db_path = _PROJECT_ROOT / "data" / "paper_trading.db"
        else:
            self._db_path = Path(db_path)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Price cache (5-minute TTL)
        self._price_cache = _PriceCache(ttl_seconds=300)

        # Initialise database
        self._init_db()

        logger.info(
            "SimulatedBroker ready — db=%s  capital=%.2f  "
            "slippage=%.1f bps  commission=%.1f bps",
            self._db_path, initial_capital,
            slippage_bps, commission_bps,
        )

    # ── Database helpers ──────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and seed the account if empty."""
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA_SQL)

            # Seed account on first run
            row = conn.execute("SELECT COUNT(*) AS cnt FROM account").fetchone()
            if row["cnt"] == 0:
                now_iso = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO account (id, cash, initial_capital, created_at, updated_at) "
                    "VALUES (1, ?, ?, ?, ?)",
                    (self._initial_capital, self._initial_capital, now_iso, now_iso),
                )
                logger.info(
                    "Seeded new account with $%.2f", self._initial_capital,
                )
            conn.commit()
        finally:
            conn.close()

    def _get_cash(self) -> float:
        """Read current cash balance from SQLite."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            return float(row["cash"]) if row else 0.0
        finally:
            conn.close()

    def _update_cash(self, new_cash: float, conn: sqlite3.Connection) -> None:
        """Update cash balance (caller must commit)."""
        conn.execute(
            "UPDATE account SET cash = ?, updated_at = ? WHERE id = 1",
            (new_cash, datetime.now(timezone.utc).isoformat()),
        )

    # ── Price helper ──────────────────────────────────────────────

    def _get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices with 5-minute cache.

        Parameters
        ----------
        symbols : list[str]
            Ticker symbols to price.

        Returns
        -------
        dict[str, float]
            Symbol → latest close price.
        """
        if not symbols:
            return {}
        return self._price_cache.get_prices(symbols)

    # ── BrokerBase implementation ─────────────────────────────────

    def get_account(self, _positions: list[Position] | None = None) -> AccountSnapshot:
        """Return a snapshot of the simulated account.

        Equity is computed as cash + sum of all position market values.

        Parameters
        ----------
        _positions : list[Position], optional
            Pre-fetched positions to avoid redundant DB/API calls.
        """
        cash = self._get_cash()
        positions = _positions if _positions is not None else self.get_positions()
        market_value = sum(p.market_value for p in positions)
        equity = cash + market_value

        return AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=cash,  # No margin in simulation
            timestamp=datetime.now(timezone.utc),
        )

    def get_positions(self) -> list[Position]:
        """Return all open positions with live-priced valuations."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT symbol, qty, avg_cost FROM positions WHERE qty != 0"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        symbols = [r["symbol"] for r in rows]
        prices = self._get_current_prices(symbols)

        positions: list[Position] = []
        for row in rows:
            sym = row["symbol"]
            qty = float(row["qty"])
            avg_cost = float(row["avg_cost"])
            current_price = prices.get(sym, avg_cost)  # Fallback to cost
            market_value = qty * current_price
            cost_basis = qty * avg_cost
            unrealized_pl = market_value - cost_basis
            unrealized_pl_pct = (
                (unrealized_pl / abs(cost_basis)) * 100.0
                if abs(cost_basis) > 1e-8
                else 0.0
            )

            positions.append(Position(
                symbol=sym,
                qty=qty,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pl_pct,
            ))

        return positions

    def reconcile(
        self,
        target_weights: dict[str, float],
    ) -> list[OrderIntent]:
        """Generate order intents to reach the target allocation.

        Logic
        -----
        1. Compute total equity (cash + positions).
        2. For each symbol, compute ``delta = equity * target_weight - current_value``.
        3. If ``|delta| < $10`` → skip (de-minimis).
        4. Sort: sells first (free cash), then buys.

        Parameters
        ----------
        target_weights : dict[str, float]
            ``symbol → fraction`` of equity.

        Returns
        -------
        list[OrderIntent]
            Sorted: sells before buys.
        """
        # Fetch positions once and reuse for account snapshot (fix S7: avoid double fetch)
        positions = self.get_positions()
        account = self.get_account(_positions=positions)
        equity = account.equity

        if equity <= 0:
            logger.warning("Reconcile skipped — equity is %.2f", equity)
            return []

        # Current position values by symbol
        current_values: dict[str, float] = {p.symbol: p.market_value for p in positions}
        current_prices: dict[str, float] = {p.symbol: p.current_price for p in positions}

        # Fetch prices for new symbols not in positions
        new_symbols = [s for s in target_weights if s not in current_prices]
        if new_symbols:
            fresh_prices = self._get_current_prices(new_symbols)
            current_prices.update(fresh_prices)

        sells: list[OrderIntent] = []
        buys: list[OrderIntent] = []

        # Handle symbols to sell (in positions but not in targets, or reduced)
        all_symbols = set(target_weights.keys()) | set(current_values.keys())

        for sym in all_symbols:
            target_value = equity * target_weights.get(sym, 0.0)
            current_value = current_values.get(sym, 0.0)
            delta = target_value - current_value

            # De-minimis filter
            if abs(delta) < 10.0:
                continue

            price = current_prices.get(sym, 0.0)
            if price <= 0:
                logger.warning("Cannot reconcile %s — no price available", sym)
                continue

            qty = abs(delta) / price

            if delta < 0:
                sells.append(OrderIntent(
                    symbol=sym,
                    side="SELL",
                    qty=round(qty, 4),
                    reason=f"Reduce {sym}: current=${current_value:.0f} → target=${target_value:.0f}",
                ))
            else:
                buys.append(OrderIntent(
                    symbol=sym,
                    side="BUY",
                    qty=round(qty, 4),
                    reason=f"Increase {sym}: current=${current_value:.0f} → target=${target_value:.0f}",
                ))

        # Sells first, then buys
        intents = sells + buys

        logger.info(
            "Reconciled %d intents (%d sells, %d buys) — equity=%.2f",
            len(intents), len(sells), len(buys), equity,
        )
        return intents

    def execute(
        self,
        intents: list[OrderIntent],
    ) -> list[OrderResult]:
        """Simulate order execution with slippage and commission.

        For each intent:
        1. Check for duplicate trade (same symbol + side + date → skip).
        2. Fetch current price, apply random slippage.
        3. Deduct commission from proceeds / add to cost.
        4. Update position and cash in SQLite atomically.
        5. Record trade.

        Parameters
        ----------
        intents : list[OrderIntent]
            Orders to execute.

        Returns
        -------
        list[OrderResult]
            One result per intent.
        """
        results: list[OrderResult] = []
        # Fix S1: use ET timezone for trade date, not system local time
        today_str = datetime.now(_ET).date().isoformat()

        # Fix M1: batch-fetch all prices upfront instead of per-intent
        all_symbols = list({intent.symbol for intent in intents})
        prefetched_prices = self._get_current_prices(all_symbols)

        conn = self._get_conn()
        try:
            for intent in intents:
                # ── Duplicate guard ───────────────────────────────
                dup = conn.execute(
                    "SELECT trade_id FROM trades "
                    "WHERE symbol = ? AND side = ? AND trade_date = ?",
                    (intent.symbol, intent.side, today_str),
                ).fetchone()

                if dup:
                    logger.info(
                        "Duplicate trade skipped: %s %s on %s",
                        intent.side, intent.symbol, today_str,
                    )
                    results.append(OrderResult(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        filled_price=0.0,
                        commission=0.0,
                        timestamp=datetime.now(timezone.utc),
                        order_id=str(uuid.uuid4()),
                        status="CANCELLED",
                    ))
                    continue

                # ── Price with slippage ───────────────────────────
                base_price = prefetched_prices.get(intent.symbol)

                if base_price is None or base_price <= 0:
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
                        order_id=str(uuid.uuid4()),
                        status="REJECTED",
                    ))
                    continue

                slip_factor = 1.0 + random.uniform(
                    -self._slippage_bps, self._slippage_bps,
                ) / 10_000.0
                filled_price = base_price * slip_factor

                # ── Commission ────────────────────────────────────
                notional = intent.qty * filled_price
                commission = notional * (self._commission_bps / 10_000.0)

                # ── Update cash & position ────────────────────────
                cash = float(
                    conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()["cash"]
                )

                if intent.side == "BUY":
                    total_cost = notional + commission
                    if total_cost > cash:
                        # Fix C4: solve algebraically for max qty under cash
                        # cash = qty * price * (1 + comm_rate)
                        # qty = cash / (price * (1 + comm_rate))
                        comm_rate = self._commission_bps / 10_000.0
                        adjusted_qty = cash / (filled_price * (1.0 + comm_rate))
                        if adjusted_qty < 0.01:
                            logger.warning(
                                "REJECTED BUY %s — insufficient cash (%.2f)",
                                intent.symbol, cash,
                            )
                            results.append(OrderResult(
                                symbol=intent.symbol,
                                side="BUY",
                                qty=intent.qty,
                                filled_price=filled_price,
                                commission=0.0,
                                timestamp=datetime.now(timezone.utc),
                                order_id=str(uuid.uuid4()),
                                status="REJECTED",
                            ))
                            continue
                        adjusted_qty = round(adjusted_qty, 4)
                        logger.info(
                            "BUY %s qty reduced %.2f → %.2f (cash constraint)",
                            intent.symbol, intent.qty, adjusted_qty,
                        )
                        intent = OrderIntent(
                            symbol=intent.symbol,
                            side="BUY",
                            qty=adjusted_qty,
                            reason=intent.reason + " [cash-adjusted]",
                        )
                        notional = intent.qty * filled_price
                        commission = notional * (self._commission_bps / 10_000.0)
                        total_cost = notional + commission

                    new_cash = cash - total_cost
                    self._upsert_position_buy(
                        conn, intent.symbol, intent.qty, filled_price,
                    )

                else:  # SELL
                    # Fix C1+C2: cap sell qty to actual held position
                    held_row = conn.execute(
                        "SELECT qty FROM positions WHERE symbol = ?",
                        (intent.symbol,),
                    ).fetchone()
                    held_qty = float(held_row["qty"]) if held_row else 0.0

                    if held_qty <= 1e-8:
                        logger.warning(
                            "REJECTED SELL %s — no position held",
                            intent.symbol,
                        )
                        results.append(OrderResult(
                            symbol=intent.symbol,
                            side="SELL",
                            qty=intent.qty,
                            filled_price=filled_price,
                            commission=0.0,
                            timestamp=datetime.now(timezone.utc),
                            order_id=str(uuid.uuid4()),
                            status="REJECTED",
                        ))
                        continue

                    actual_qty = min(intent.qty, held_qty)
                    if actual_qty < intent.qty:
                        logger.info(
                            "SELL %s qty capped %.4f → %.4f (position limit)",
                            intent.symbol, intent.qty, actual_qty,
                        )
                        intent = OrderIntent(
                            symbol=intent.symbol,
                            side="SELL",
                            qty=round(actual_qty, 4),
                            reason=intent.reason + " [position-capped]",
                        )
                        notional = intent.qty * filled_price
                        commission = notional * (self._commission_bps / 10_000.0)

                    proceeds = notional - commission
                    new_cash = cash + proceeds
                    self._upsert_position_sell(conn, intent.symbol, intent.qty)

                self._update_cash(new_cash, conn)

                # ── Record trade ──────────────────────────────────
                order_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO trades "
                    "(trade_id, symbol, side, qty, filled_price, commission, "
                    "executed_at, trade_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        order_id, intent.symbol, intent.side,
                        intent.qty, filled_price, commission,
                        datetime.now(timezone.utc).isoformat(), today_str,
                    ),
                )

                results.append(OrderResult(
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=intent.qty,
                    filled_price=filled_price,
                    commission=commission,
                    timestamp=datetime.now(timezone.utc),
                    order_id=order_id,
                    status="FILLED",
                ))

                logger.info(
                    "FILLED %s %s %.4f @ %.4f  commission=%.4f  cash=%.2f",
                    intent.side, intent.symbol, intent.qty,
                    filled_price, commission, new_cash,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            # Fix C7: clear results to prevent reporting FILLED for rolled-back trades
            results.clear()
            raise
        finally:
            conn.close()

        return results

    def is_market_open(self) -> bool:
        """Check if current time falls within NYSE regular trading hours.

        NYSE hours: 9:30 – 16:00 Eastern Time, Monday–Friday.
        Holidays are NOT checked (would require a calendar data source).

        Returns
        -------
        bool
            ``True`` if the market is considered open.
        """
        now_et = datetime.now(_ET)

        # Weekend check (Monday=0 .. Sunday=6)
        if now_et.weekday() >= 5:
            return False

        # Time bounds
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        # Fix S2: market closes AT 16:00, not after
        return market_open <= now_et < market_close

    def get_portfolio_value(self) -> float:
        """Return total portfolio equity (cash + positions)."""
        return self.get_account().equity

    # ── Extended methods (not in BrokerBase) ──────────────────────

    def take_daily_snapshot(self) -> dict[str, Any]:
        """Record today's equity, cash, and positions into daily_snapshots.

        Returns
        -------
        dict
            The snapshot record that was persisted.
        """
        import json

        account = self.get_account()
        positions = self.get_positions()
        # Fix S1: use ET timezone for snapshot date
        today_str = datetime.now(_ET).date().isoformat()

        positions_data = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
            }
            for p in positions
        ]

        snapshot = {
            "date": today_str,
            "equity": account.equity,
            "cash": account.cash,
            "positions": positions_data,
        }

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO daily_snapshots "
                "(snapshot_date, equity, cash, positions_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    today_str,
                    account.equity,
                    account.cash,
                    json.dumps(positions_data),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(
            "Daily snapshot saved: date=%s equity=%.2f cash=%.2f positions=%d",
            today_str, account.equity, account.cash, len(positions),
        )
        return snapshot

    def get_performance_history(self) -> list[dict[str, Any]]:
        """Return all daily snapshots for charting / analysis.

        Returns
        -------
        list[dict]
            Chronologically ordered list of daily snapshots.
        """
        import json

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT snapshot_date, equity, cash, positions_json "
                "FROM daily_snapshots ORDER BY snapshot_date"
            ).fetchall()
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "date": row["snapshot_date"],
                "equity": float(row["equity"]),
                "cash": float(row["cash"]),
                "positions": json.loads(row["positions_json"]),
            })

        return results

    def reset_account(self, initial_capital: float | None = None) -> None:
        """Reset the simulated account to a clean state.

        Useful for testing — drops all positions, trades, and snapshots,
        then re-seeds the account with the specified capital.

        Parameters
        ----------
        initial_capital : float, optional
            New starting capital.  Defaults to the original value.
        """
        # Fix S3: 0.0 is falsy, use explicit None check
        capital = self._initial_capital if initial_capital is None else initial_capital
        now_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM daily_snapshots")
            conn.execute(
                "UPDATE account SET cash = ?, initial_capital = ?, "
                "updated_at = ? WHERE id = 1",
                (capital, capital, now_iso),
            )
            conn.commit()
        finally:
            conn.close()

        self._initial_capital = capital
        logger.info("Account reset — capital=%.2f", capital)

    # ── Internal position helpers ─────────────────────────────────

    @staticmethod
    def _upsert_position_buy(
        conn: sqlite3.Connection,
        symbol: str,
        qty: float,
        price: float,
    ) -> None:
        """Insert or update a position after a BUY fill.

        Computes new weighted-average cost basis.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT qty, avg_cost FROM positions WHERE symbol = ?",
            (symbol,),
        ).fetchone()

        if row and float(row["qty"]) != 0:
            old_qty = float(row["qty"])
            old_cost = float(row["avg_cost"])
            new_qty = old_qty + qty
            new_avg = ((old_qty * old_cost) + (qty * price)) / new_qty
            conn.execute(
                "UPDATE positions SET qty = ?, avg_cost = ?, updated_at = ? "
                "WHERE symbol = ?",
                (new_qty, new_avg, now_iso, symbol),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO positions (symbol, qty, avg_cost, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (symbol, qty, price, now_iso),
            )

    @staticmethod
    def _upsert_position_sell(
        conn: sqlite3.Connection,
        symbol: str,
        qty: float,
    ) -> None:
        """Reduce position quantity after a SELL fill.

        If the resulting quantity is ≤ 0, the position row is deleted.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        row = conn.execute(
            "SELECT qty FROM positions WHERE symbol = ?",
            (symbol,),
        ).fetchone()

        if row:
            new_qty = float(row["qty"]) - qty
            if new_qty <= 1e-8:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                conn.execute(
                    "UPDATE positions SET qty = ?, updated_at = ? WHERE symbol = ?",
                    (new_qty, now_iso, symbol),
                )
        else:
            logger.warning("SELL %s but no position found — ignoring", symbol)
