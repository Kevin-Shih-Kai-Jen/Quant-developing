"""
tests/test_advisor.py — Unit tests for the AI Financial Advisor tools
=====================================================================

Tests the advisor's read-only DB access, tool context memory, and
market data fetching.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from nexus_quant_os.advisor.tools import (
    get_current_portfolio,
    get_market_snapshot,
    record_personal_context,
)

# ═════════════════════════════════════════════════════════════════════
# 1. Mocking ToolContext
# ═════════════════════════════════════════════════════════════════════

class MockToolContext:
    def __init__(self):
        self.state = {}
        
    def get_state(self, key: str, default=None):
        return self.state.get(key, default)
        
    def set_state(self, key: str, value):
        self.state[key] = value

# ═════════════════════════════════════════════════════════════════════
# 2. Portfolio Tool Tests
# ═════════════════════════════════════════════════════════════════════

def test_get_portfolio_no_db(mocker, tmp_path):
    """If DB doesn't exist, should return an error JSON."""
    fake_db = tmp_path / "non_existent.db"
    mocker.patch("nexus_quant_os.advisor.tools._DB_PATH", fake_db)
    
    result_str = get_current_portfolio()
    result = json.loads(result_str)
    
    assert "error" in result
    assert "No paper trading database found" in result["error"]

def test_get_portfolio_with_db(mocker, tmp_path):
    """If DB exists, should safely read cash and positions via read-only connection."""
    fake_db = tmp_path / "paper_trading.db"
    mocker.patch("nexus_quant_os.advisor.tools._DB_PATH", fake_db)
    
    # Setup mock DB schema and data
    with sqlite3.connect(fake_db) as conn:
        conn.execute("CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL, initial_capital REAL, updated_at TEXT)")
        conn.execute("CREATE TABLE positions (symbol TEXT PRIMARY KEY, qty REAL, avg_cost REAL, updated_at TEXT)")
        
        conn.execute("INSERT INTO account (id, cash, initial_capital, updated_at) VALUES (1, 50000.0, 100000.0, '2026-05-26')")
        conn.execute("INSERT INTO positions (symbol, qty, avg_cost, updated_at) VALUES ('SPY', 10.0, 450.0, '2026-05-26')")
        conn.execute("INSERT INTO positions (symbol, qty, avg_cost, updated_at) VALUES ('AAPL', 50.0, 150.0, '2026-05-26')")
        
    result_str = get_current_portfolio()
    result = json.loads(result_str)
    
    assert "cash" in result
    assert result["cash"] == 50000.0
    
    assert "positions" in result
    assert len(result["positions"]) == 2
    symbols = [p["symbol"] for p in result["positions"]]
    assert "SPY" in symbols
    assert "AAPL" in symbols

# ═════════════════════════════════════════════════════════════════════
# 3. Context Memory Test
# ═════════════════════════════════════════════════════════════════════

def test_record_personal_context():
    """Should append details to the appropriate context type in state."""
    ctx = MockToolContext()
    
    msg1 = record_personal_context("LifeEvent", "Going to military for 1 month.", ctx)
    assert "Successfully recorded" in msg1
    assert "LifeEvent" in ctx.state["personal_context"]
    assert len(ctx.state["personal_context"]["LifeEvent"]) == 1
    
    # Add a second one
    record_personal_context("LifeEvent", "Will return in August.", ctx)
    assert len(ctx.state["personal_context"]["LifeEvent"]) == 2

# ═════════════════════════════════════════════════════════════════════
# 4. Market Snapshot Test
# ═════════════════════════════════════════════════════════════════════

def test_get_market_snapshot(mocker):
    """Should call yfinance.download and format the output as JSON."""
    import pandas as pd
    
    # Mock yfinance to return a simple DataFrame
    df = pd.DataFrame(
        {
            ("Close", "SPY"): [450.0],
            ("Close", "QQQ"): [380.0],
        }
    )
    mocker.patch("nexus_quant_os.advisor.tools.yfinance.download", return_value=df)
    
    result_str = get_market_snapshot(["SPY", "QQQ"])
    result = json.loads(result_str)
    
    assert "SPY" in result
    assert result["SPY"] == 450.0
    assert "QQQ" in result
    assert result["QQQ"] == 380.0
