"""
execution/trade_logger.py — Structured Trade & Performance Logger
==================================================================

Provides auditable, JSON-structured logging for every trade execution
and daily performance snapshot.  All files are stored under the project
``logs/`` directory tree:

::

    logs/
    ├── trades/
    │   ├── 2026-05-26.json
    │   └── ...
    └── performance/
        ├── 2026-05-26.json
        └── ...

Design Notes
------------
- One JSON file per calendar day.  Trade logs *append* entries (the file
  contains a JSON array that grows with each trade).
- Performance snapshots are overwritten once per day (last write wins).
- All ``datetime`` objects are serialised to ISO-8601 strings.
- Directories are created lazily on first write.

Author : Nexus Quant OS — Execution Engineering Division
"""

from __future__ import annotations

import fcntl
import json
import logging
from dataclasses import asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any

from nexus_quant_os.execution.broker_base import (
    AccountSnapshot,
    OrderResult,
    Position,
)

logger = logging.getLogger("nexus_quant_os.execution.trade_logger")

# Project root: three levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _json_serialiser(obj: Any) -> str:
    """Custom JSON serialiser for ``datetime`` and ``date`` objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# CORE CLASS
# ═════════════════════════════════════════════════════════════════════

class TradeLogger:
    """Structured, file-based logger for trade executions and snapshots.

    Parameters
    ----------
    log_root : Path | str | None
        Root directory for log output.  Defaults to
        ``<project_root>/logs``.

    Usage
    -----
    >>> tl = TradeLogger()
    >>> tl.log_trade(order_result, reasoning={"signal": "MoE bullish"})
    >>> tl.log_daily_performance(account_snap, positions)
    >>> recent = tl.get_trade_history(days=7)
    """

    def __init__(self, log_root: Path | str | None = None) -> None:
        self._log_root = Path(log_root) if log_root else _PROJECT_ROOT / "logs"
        self._trades_dir = self._log_root / "trades"
        self._perf_dir = self._log_root / "performance"

        # Lazy-create on first access
        _ensure_dir(self._trades_dir)
        _ensure_dir(self._perf_dir)

        logger.info("TradeLogger initialised — root=%s", self._log_root)

    # ── Trade logging ─────────────────────────────────────────────

    def log_trade(
        self,
        result: OrderResult,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        """Append a single trade record to today's trade log.

        Parameters
        ----------
        result : OrderResult
            The executed order result.
        reasoning : dict, optional
            Arbitrary key-value metadata describing *why* this trade
            was placed (e.g. model signals, firewall tier, etc.).
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._trades_dir / f"{today_str}.json"

        record: dict[str, Any] = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "order": asdict(result),
            "reasoning": reasoning or {},
        }

        # Read existing entries (if any) and append — with file locking
        if filepath.exists():
            with open(filepath, 'r+', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    try:
                        existing = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            "Corrupt trade log %s — starting fresh", filepath,
                        )
                        existing = []
                    existing.append(record)
                    f.seek(0)
                    f.truncate()
                    json.dump(existing, f, indent=2, default=_json_serialiser)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump([record], f, indent=2, default=_json_serialiser)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        logger.debug(
            "Logged trade: %s %s %.2f @ %.4f → %s",
            result.side, result.symbol, result.qty,
            result.filled_price, filepath.name,
        )

    # ── Performance snapshot ──────────────────────────────────────

    def log_daily_performance(
        self,
        snapshot: AccountSnapshot,
        positions: list[Position],
    ) -> None:
        """Write (or overwrite) today's daily performance snapshot.

        Parameters
        ----------
        snapshot : AccountSnapshot
            Account-level equity/cash snapshot.
        positions : list[Position]
            All currently held positions.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._perf_dir / f"{today_str}.json"

        record: dict[str, Any] = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "account": asdict(snapshot),
            "positions": [asdict(p) for p in positions],
            "summary": {
                "total_equity": snapshot.equity,
                "total_cash": snapshot.cash,
                "num_positions": len(positions),
                "total_unrealized_pl": sum(p.unrealized_pl for p in positions),
            },
        }

        filepath.write_text(
            json.dumps(record, indent=2, default=_json_serialiser),
            encoding="utf-8",
        )
        logger.info(
            "Daily performance snapshot → %s  equity=%.2f  positions=%d",
            filepath.name, snapshot.equity, len(positions),
        )

    # ── History retrieval ─────────────────────────────────────────

    def get_trade_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Read trade records from the last *N* calendar days.

        Parameters
        ----------
        days : int
            Number of past calendar days to scan (default 30).

        Returns
        -------
        list[dict]
            Flat list of trade records across all matching days,
            sorted chronologically (oldest first).
        """
        all_records: list[dict[str, Any]] = []
        today = datetime.now(timezone.utc).date()

        for offset in range(days):
            target_date = today - timedelta(days=offset)
            filepath = self._trades_dir / f"{target_date.isoformat()}.json"
            if not filepath.exists():
                continue
            try:
                day_records = json.loads(filepath.read_text(encoding="utf-8"))
                if isinstance(day_records, list):
                    all_records.extend(day_records)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping corrupt log %s: %s", filepath, exc)

        # Return oldest-first
        all_records.reverse()

        logger.debug(
            "Retrieved %d trade records from last %d days",
            len(all_records), days,
        )
        return all_records

    def get_performance_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Read daily performance snapshots from the last *N* days.

        .. deprecated::
            This method is not called by the API server.
            Performance history is served from SimulatedBroker's SQLite store.
            Retained for potential future use or manual debugging.

        Parameters
        ----------
        days : int
            Number of past calendar days to scan (default 30).

        Returns
        -------
        list[dict]
            List of daily snapshots sorted chronologically.
        """
        snapshots: list[dict[str, Any]] = []
        today = datetime.now(timezone.utc).date()

        for offset in range(days):
            target_date = today - timedelta(days=offset)
            filepath = self._perf_dir / f"{target_date.isoformat()}.json"
            if not filepath.exists():
                continue
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                snapshots.append(data)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Skipping corrupt snapshot %s: %s", filepath, exc)

        # Oldest first
        snapshots.reverse()

        logger.debug(
            "Retrieved %d performance snapshots from last %d days",
            len(snapshots), days,
        )
        return snapshots
