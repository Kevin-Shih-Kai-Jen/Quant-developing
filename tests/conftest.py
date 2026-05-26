"""
tests/conftest.py — Shared Pytest Fixtures for Nexus Quant OS
==============================================================

Provides reusable fixtures for database setup, sample market data,
and portfolio weights used across the test suite.

Author : Nexus Quant OS — Quality Assurance Division
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# ═════════════════════════════════════════════════════════════════════
# Database Fixtures
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_db(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary SQLite database for SimulatedBroker tests.

    Yields the path to a freshly created SQLite file. The database
    and its parent directory are cleaned up automatically when the
    test finishes (handled by ``tmp_path``).

    Yields
    ------
    str
        Absolute path to the temporary ``.db`` file.
    """
    db_path = tmp_path / "test_paper_trading.db"
    # Pre-create so the file exists before SimulatedBroker touches it
    conn = sqlite3.connect(str(db_path))
    conn.close()
    yield str(db_path)
    # Cleanup: tmp_path fixture handles directory removal automatically.
    # Explicitly close any lingering connections by removing the file.
    if db_path.exists():
        db_path.unlink()


# ═════════════════════════════════════════════════════════════════════
# Sample Data Fixtures
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_weights() -> dict[str, float]:
    """Return a representative target-weight dictionary.

    These weights approximate a diversified multi-asset portfolio
    with satellite stock positions (AVGO, NVDA).

    Returns
    -------
    dict[str, float]
        Ticker → target weight, summing to 1.0.
    """
    return {
        "SPY": 0.30,
        "QQQ": 0.20,
        "GLD": 0.15,
        "TLT": 0.10,
        "IWM": 0.10,
        "SHY": 0.10,
        "AVGO": 0.03,
        "NVDA": 0.02,
    }


@pytest.fixture
def sample_prices() -> dict[str, float]:
    """Return representative per-share prices for the asset universe.

    These prices are realistic mid-2024 approximations and are used
    to compute share quantities and portfolio values in broker tests.

    Returns
    -------
    dict[str, float]
        Ticker → price per share (USD).
    """
    return {
        "SPY": 450.0,
        "QQQ": 380.0,
        "GLD": 220.0,
        "TLT": 95.0,
        "IWM": 200.0,
        "SHY": 83.0,
        "AVGO": 170.0,
        "NVDA": 130.0,
    }
