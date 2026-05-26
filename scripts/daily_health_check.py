#!/usr/bin/env python3
"""
scripts/daily_health_check.py — Nexus Quant OS Daily Health Check
===================================================================

Standalone health-check script that validates system readiness:
  1. Data freshness   — can yfinance fetch recent market data?
  2. Checkpoint exists — is there a trained MoE Router .pt file?
  3. DB integrity      — does paper_trading.db exist with tables?

Usage:
    cd /Users/coolguy/developer/nexus_quant_os
    PYTHONPATH=. python scripts/daily_health_check.py

Output:
    - JSON summary to stdout
    - JSON log file written to logs/health/YYYY-MM-DD.json

Author : Nexus Quant OS — Operations Division
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus_quant_os.health_check")

# ═════════════════════════════════════════════════════════════════════
# Project Paths
# ═════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_DIR = _PROJECT_ROOT / "nexus_quant_os" / "models" / "checkpoints"
_DB_PATH = _PROJECT_ROOT / "paper_trading.db"
_LOG_DIR = _PROJECT_ROOT / "logs" / "health"


# ═════════════════════════════════════════════════════════════════════
# Check 1: Data Freshness
# ═════════════════════════════════════════════════════════════════════


def check_data_freshness() -> dict[str, Any]:
    """Verify that yfinance can fetch recent market data.

    Attempts to download the last 5 trading days of SPY close
    prices. Passes if the most recent data point is within 5
    calendar days of today (accounts for weekends/holidays).

    Returns
    -------
    dict
        ``{"check": "data_freshness", "status": "PASS"|"FAIL",
          "details": ..., "timestamp": ...}``
    """
    result: dict[str, Any] = {
        "check": "data_freshness",
        "status": "UNKNOWN",
        "details": "",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        import yfinance as yf

        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="5d")

        if hist.empty:
            result["status"] = "FAIL"
            result["details"] = "yfinance returned empty DataFrame for SPY"
            return result

        latest_date = hist.index[-1].date()
        days_stale = (date.today() - latest_date).days

        if days_stale <= 5:
            result["status"] = "PASS"
            result["details"] = (
                f"Latest data: {latest_date.isoformat()} "
                f"({days_stale} day(s) ago)"
            )
        else:
            result["status"] = "FAIL"
            result["details"] = (
                f"Data is stale: latest={latest_date.isoformat()} "
                f"({days_stale} days ago, threshold=5)"
            )

    except ImportError:
        result["status"] = "FAIL"
        result["details"] = "yfinance is not installed"
    except Exception as exc:
        result["status"] = "FAIL"
        result["details"] = f"yfinance error: {exc}"

    return result


# ═════════════════════════════════════════════════════════════════════
# Check 2: Model Checkpoint
# ═════════════════════════════════════════════════════════════════════


def check_checkpoint_exists() -> dict[str, Any]:
    """Verify that at least one MoE Router checkpoint (.pt) exists.

    Scans ``nexus_quant_os/models/checkpoints/`` for files matching
    the pattern ``moe_router_*.pt``.

    Returns
    -------
    dict
        ``{"check": "checkpoint_exists", "status": "PASS"|"FAIL",
          "details": ..., "timestamp": ...}``
    """
    result: dict[str, Any] = {
        "check": "checkpoint_exists",
        "status": "UNKNOWN",
        "details": "",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        if not _CHECKPOINT_DIR.exists():
            result["status"] = "FAIL"
            result["details"] = (
                f"Checkpoint directory does not exist: {_CHECKPOINT_DIR}"
            )
            return result

        checkpoints = sorted(_CHECKPOINT_DIR.glob("moe_router_*.pt"))

        if checkpoints:
            latest = checkpoints[-1]
            size_mb = latest.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(latest.stat().st_mtime)
            result["status"] = "PASS"
            result["details"] = (
                f"Found {len(checkpoints)} checkpoint(s). "
                f"Latest: {latest.name} ({size_mb:.1f} MB, "
                f"modified {mtime.isoformat()})"
            )
        else:
            result["status"] = "FAIL"
            result["details"] = (
                f"No moe_router_*.pt files found in {_CHECKPOINT_DIR}"
            )

    except Exception as exc:
        result["status"] = "FAIL"
        result["details"] = f"Checkpoint check error: {exc}"

    return result


# ═════════════════════════════════════════════════════════════════════
# Check 3: Database Integrity
# ═════════════════════════════════════════════════════════════════════


def check_db_integrity() -> dict[str, Any]:
    """Verify that the paper-trading SQLite database exists and
    contains the expected tables.

    Checks for the existence of ``paper_trading.db`` and queries
    ``sqlite_master`` for table names.

    Returns
    -------
    dict
        ``{"check": "db_integrity", "status": "PASS"|"FAIL",
          "details": ..., "timestamp": ...}``
    """
    result: dict[str, Any] = {
        "check": "db_integrity",
        "status": "UNKNOWN",
        "details": "",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        if not _DB_PATH.exists():
            result["status"] = "FAIL"
            result["details"] = (
                f"Database file not found: {_DB_PATH}"
            )
            return result

        conn = sqlite3.connect(str(_DB_PATH))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        if tables:
            result["status"] = "PASS"
            result["details"] = (
                f"Database exists with {len(tables)} table(s): "
                f"{', '.join(sorted(tables))}"
            )
        else:
            result["status"] = "FAIL"
            result["details"] = (
                f"Database exists but contains no tables: {_DB_PATH}"
            )

    except Exception as exc:
        result["status"] = "FAIL"
        result["details"] = f"Database check error: {exc}"

    return result


# ═════════════════════════════════════════════════════════════════════
# Run All Checks
# ═════════════════════════════════════════════════════════════════════


def run_all_checks() -> dict[str, Any]:
    """Execute all health checks and return a structured report.

    Returns
    -------
    dict
        ``{"run_timestamp": ..., "overall_status": "PASS"|"FAIL",
          "checks": [...], "summary": {"passed": N, "failed": N}}``
    """
    checks = [
        check_data_freshness(),
        check_checkpoint_exists(),
        check_db_integrity(),
    ]

    n_passed = sum(1 for c in checks if c["status"] == "PASS")
    n_failed = sum(1 for c in checks if c["status"] != "PASS")

    overall = "PASS" if n_failed == 0 else "FAIL"

    return {
        "run_timestamp": datetime.now().isoformat(),
        "overall_status": overall,
        "checks": checks,
        "summary": {
            "passed": n_passed,
            "failed": n_failed,
            "total": len(checks),
        },
    }


# ═════════════════════════════════════════════════════════════════════
# Entry Point
# ═════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    results = run_all_checks()

    # ── Print JSON to stdout ──────────────────────────────────────
    output = json.dumps(results, indent=2, ensure_ascii=False)
    print(output)

    # ── Write to log file ─────────────────────────────────────────
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / f"{date.today().isoformat()}.json"
    log_file.write_text(output, encoding="utf-8")

    logger.info("Health check complete → %s", log_file)
    logger.info("Overall status: %s", results["overall_status"])

    # Exit with non-zero code if any check failed
    sys.exit(0 if results["overall_status"] == "PASS" else 1)
