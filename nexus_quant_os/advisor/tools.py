"""
nexus_quant_os/advisor/tools.py — Custom Tools for the AI Advisor Agent
=======================================================================

These tools allow the Antigravity SDK agent to read the local portfolio state,
record long-term personal context into its conversation memory, and fetch
real-time market prices.

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import yfinance

try:
    from google.antigravity import ToolContext
except ImportError:
    # SDK not installed — define a minimal protocol for type hints / tests
    from typing import Any
    class ToolContext:  # type: ignore[no-redef]
        """Stub for when google.antigravity SDK is not installed."""
        def get_state(self, key: str, default: Any = None) -> Any: ...
        def set_state(self, key: str, value: Any) -> None: ...

logger = logging.getLogger("nexus_quant_os.advisor.tools")

# Project root: three levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _PROJECT_ROOT / "data" / "paper_trading.db"


def get_current_portfolio() -> str:
    """Gets the current quantitative portfolio status and cash balance from the local database.

    Use this tool whenever the user asks about their current holdings, account balance,
    or quantitative trading status.

    Returns:
        A JSON string containing cash balance, total equity (if snapshot exists),
        and a list of current positions.
    """
    if not _DB_PATH.exists():
        return json.dumps({"error": "No paper trading database found. Portfolio is empty or uninitialized."})

    try:
        # STRICT READ-ONLY CONNECTION
        # Using a URI with mode=ro ensures we cannot accidentally modify data.
        db_uri = f"file:{_DB_PATH}?mode=ro"
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get cash
            cash_row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            cash = cash_row["cash"] if cash_row else 0.0
            
            # Get positions
            pos_cursor = conn.execute("SELECT symbol, qty, avg_cost FROM positions WHERE qty > 0")
            positions = [dict(row) for row in pos_cursor.fetchall()]
            
        return json.dumps({
            "cash": cash,
            "positions": positions,
            "note": "This is the primary quantitative account. It does NOT include any external accounts."
        }, indent=2)
    except sqlite3.Error as e:
        logger.error("Failed to read portfolio from DB: %s", e)
        return json.dumps({"error": f"Database read error: {e}"})


def record_personal_context(context_type: str, details: str, ctx: ToolContext) -> str:
    """Records the user's personal or financial situation so you can remember it for future advice.

    Use this tool whenever the user shares personal constraints, goals, or life events
    (e.g., "I'm going to the military for a month", "I want to save for a house", 
    "My risk tolerance has decreased").

    Args:
        context_type: A short category name (e.g., 'LifeEvent', 'RiskTolerance', 'Goal').
        details: A detailed description of the user's situation.
        ctx: The tool context (injected automatically).

    Returns:
        A confirmation message.
    """
    # Retrieve current personal context state
    personal_context = ctx.get_state("personal_context", {})
    
    # Update state
    if context_type not in personal_context:
        personal_context[context_type] = []
    
    personal_context[context_type].append(details)
    ctx.set_state("personal_context", personal_context)
    
    return f"Successfully recorded {context_type}: {details}."


def get_market_snapshot(tickers: list[str]) -> str:
    """Fetches real-time or latest available market prices and volumes for the given tickers.

    Use this tool when you need current market context for a specific asset before giving advice.

    Args:
        tickers: A list of stock ticker symbols (e.g., ["AAPL", "MSFT", "SPY"]).

    Returns:
        A JSON string containing the latest close prices.
    """
    if not tickers:
        return json.dumps({"error": "No tickers provided."})
    
    try:
        tickers_str = " ".join(tickers)
        df = yfinance.download(tickers_str, period="1d", progress=False, threads=False)
        
        if df.empty:
            return json.dumps({"error": f"Could not fetch data for {tickers_str}"})
            
        prices = {}
        if len(tickers) == 1:
            sym = tickers[0]
            if "Close" in df.columns:
                val = df["Close"].iloc[-1]
                prices[sym] = float(val.item() if hasattr(val, "item") else val)
        else:
            for sym in tickers:
                try:
                    col = ("Close", sym)
                    if col in df.columns:
                        val = df[col].iloc[-1]
                        prices[sym] = float(val.item() if hasattr(val, "item") else val)
                except (KeyError, IndexError):
                    pass
                    
        return json.dumps(prices, indent=2)
    except Exception as e:
        logger.error("yfinance download failed in advisor tool: %s", e)
        return json.dumps({"error": f"Failed to fetch market data: {e}"})
