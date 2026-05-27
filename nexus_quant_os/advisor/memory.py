"""
advisor/memory.py — SQLite Long-Term Memory for AI Advisor
===========================================================

Persistent storage for user profile, conversation summaries, and
portfolio snapshots.  Used by the GeminiAdvisor to maintain context
across sessions.

Tables
------
user_profile          : Key-value store for personal data
conversation_summary  : Compressed conversation summaries
portfolio_snapshots   : Holdings snapshots from screenshots/Moomoo

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus_quant_os.advisor.memory")


class AdvisorMemory:
    """SQLite-backed long-term memory for the financial advisor.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.  Use ``':memory:'`` for testing.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key       TEXT PRIMARY KEY,
                value     TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summary (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                role       TEXT NOT NULL,
                summary    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                source     TEXT NOT NULL,
                holdings   TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── User Profile ──────────────────────────────────────────────

    def set_profile(self, key: str, value: str) -> None:
        """Set a user profile key-value pair."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO user_profile (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (key, value, now),
        )
        self._conn.commit()
        logger.info("Profile updated: %s = %s", key, value[:50])

    def get_profile(self, key: str) -> str | None:
        """Get a user profile value by key."""
        row = self._conn.execute(
            "SELECT value FROM user_profile WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_all_profiles(self) -> dict[str, str]:
        """Get all user profile key-value pairs."""
        rows = self._conn.execute(
            "SELECT key, value FROM user_profile ORDER BY key"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_profile(self, key: str) -> None:
        """Delete a user profile key."""
        self._conn.execute("DELETE FROM user_profile WHERE key = ?", (key,))
        self._conn.commit()

    # ── Conversation Summaries ────────────────────────────────────

    def add_conversation(self, role: str, summary: str) -> None:
        """Add a conversation turn summary."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO conversation_summary (timestamp, role, summary) "
            "VALUES (?, ?, ?)",
            (now, role, summary),
        )
        self._conn.commit()

    def get_recent_conversations(self, limit: int = 20) -> list[dict]:
        """Get the most recent conversation summaries."""
        rows = self._conn.execute(
            "SELECT role, summary, timestamp FROM conversation_summary "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_conversations(self) -> None:
        """Clear all conversation history."""
        self._conn.execute("DELETE FROM conversation_summary")
        self._conn.commit()
        logger.info("Conversation history cleared")

    # ── Portfolio Snapshots ───────────────────────────────────────

    def save_snapshot(
        self, source: str, holdings: list[dict],
    ) -> None:
        """Save a portfolio snapshot.

        Parameters
        ----------
        source : str
            Origin: 'screenshot', 'manual', or 'moomoo'.
        holdings : list[dict]
            List of dicts with keys: symbol, qty, avg_cost, market_value.
        """
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO portfolio_snapshots (timestamp, source, holdings) "
            "VALUES (?, ?, ?)",
            (now, source, json.dumps(holdings, ensure_ascii=False)),
        )
        self._conn.commit()
        logger.info("Portfolio snapshot saved: %s, %d positions", source, len(holdings))

    def get_latest_snapshot(self) -> dict | None:
        """Get the most recent portfolio snapshot."""
        row = self._conn.execute(
            "SELECT timestamp, source, holdings FROM portfolio_snapshots "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return {
                "timestamp": row["timestamp"],
                "source": row["source"],
                "holdings": json.loads(row["holdings"]),
            }
        return None

    # ── Context Builder ───────────────────────────────────────────

    def build_memory_context(self) -> str:
        """Build a context string from all stored memory for the system prompt.

        Returns
        -------
        str
            Formatted memory context string.
        """
        sections = []

        # User profile
        profiles = self.get_all_profiles()
        if profiles:
            lines = [f"  • {k}: {v}" for k, v in profiles.items()]
            sections.append("【用戶個人資料】\n" + "\n".join(lines))

        # Latest snapshot
        snapshot = self.get_latest_snapshot()
        if snapshot:
            holdings = snapshot["holdings"]
            lines = [
                f"  來源: {snapshot['source']}  時間: {snapshot['timestamp']}"
            ]
            for h in holdings:
                sym = h.get("symbol", "?")
                qty = h.get("qty", 0)
                cost = h.get("avg_cost", 0)
                lines.append(f"  • {sym}: {qty} 股 @ ${cost:.2f}")
            sections.append("【最近持倉記錄】\n" + "\n".join(lines))

        # Recent conversations
        convos = self.get_recent_conversations(limit=10)
        if convos:
            lines = []
            for c in convos:
                role = "用戶" if c["role"] == "user" else "顧問"
                lines.append(f"  [{role}] {c['summary'][:100]}")
            sections.append("【近期對話摘要】\n" + "\n".join(lines))

        if not sections:
            return "（尚無記憶資料）"

        return "\n\n".join(sections)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
