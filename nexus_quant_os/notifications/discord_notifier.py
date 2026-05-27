"""
notifications/discord_notifier.py — Discord Webhook Integration
================================================================

Sends structured trade reports and alerts to a Discord channel via
Webhook.  Uses only ``requests`` (already a project dependency) —
no discord.py or bot token required.

Data-flow position::

    run_moomoo_trade.py  →  DiscordNotifier.send_daily_report()
                         →  Discord Channel (Embed)

Author : Nexus Quant OS — Notifications Division
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger("nexus_quant_os.notifications.discord")

# ═════════════════════════════════════════════════════════════════════
# Data Models
# ═════════════════════════════════════════════════════════════════════


@dataclass
class TradeRecord:
    """Single executed trade."""
    symbol: str
    side: str       # "BUY" | "SELL"
    qty: float
    price: float
    status: str     # "FILLED" | "REJECTED" | "CANCELLED"
    order_id: str = ""


@dataclass
class TradeReport:
    """Complete daily pipeline execution report."""

    # Pipeline metadata
    timestamp: str = ""
    data_source: str = "REAL"
    n_price_rows: int = 0
    n_macro_rows: int = 0
    n_aligned_rows: int = 0
    n_features: int = 0

    # Model
    checkpoint_name: str = ""
    router_mode: str = ""      # "CHECKPOINT" | "RANDOM_FALLBACK"

    # Risk firewall
    risk_tier: str = "NORMAL"
    scale_factor: float = 1.0
    regime: str = ""
    hmm_danger: float = 0.0
    ood_score: float = 0.0

    # Portfolio
    optimizer_mode: str = ""
    target_weights: dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0
    cash_pct: float = 0.0

    # Execution
    pre_equity: float = 0.0
    pre_cash: float = 0.0
    post_equity: float = 0.0
    post_cash: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    n_filled: int = 0
    n_rejected: int = 0

    # Errors
    error: str | None = None
    traceback_str: str | None = None


# ═════════════════════════════════════════════════════════════════════
# Risk Tier → Embed Colour
# ═════════════════════════════════════════════════════════════════════

_TIER_COLOURS = {
    "NORMAL":    0x2ecc71,   # 🟢 Green
    "CAUTION":   0xf39c12,   # 🟡 Yellow
    "WARNING":   0xe67e22,   # 🟠 Orange
    "EMERGENCY": 0xe74c3c,   # 🔴 Red
}

_TIER_EMOJI = {
    "NORMAL":    "🟢",
    "CAUTION":   "🟡",
    "WARNING":   "🟠",
    "EMERGENCY": "🔴",
}


# ═════════════════════════════════════════════════════════════════════
# Discord Notifier
# ═════════════════════════════════════════════════════════════════════


class DiscordNotifier:
    """Send structured messages to Discord via Webhook.

    Parameters
    ----------
    webhook_url : str | None
        Discord Webhook URL.  If ``None`` or empty, reads from
        ``DISCORD_WEBHOOK_URL`` environment variable.
        If neither is set, all send methods become silent no-ops.

    username : str
        Display name shown in Discord.  Default ``'Nexus Quant OS'``.

    timeout : float
        HTTP request timeout in seconds.  Default ``10.0``.
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        username: str = "Nexus Quant OS",
        timeout: float = 10.0,
    ) -> None:
        self._url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self._username = username
        self._timeout = timeout

        if not self._url:
            logger.warning(
                "No Discord Webhook URL configured — notifications disabled. "
                "Set DISCORD_WEBHOOK_URL in .env to enable."
            )

    @property
    def is_configured(self) -> bool:
        """Return True if a webhook URL is set."""
        return bool(self._url)

    # ── Low-level sender ──────────────────────────────────────────

    def _send(self, payload: dict[str, Any]) -> bool:
        """POST a payload to the Discord Webhook.

        Returns True on success, False on failure (never raises).
        """
        if not self.is_configured:
            return False

        payload.setdefault("username", self._username)

        try:
            resp = requests.post(
                self._url,
                json=payload,
                timeout=self._timeout,
            )
            if resp.status_code in (200, 204):
                logger.info("Discord notification sent (HTTP %d)", resp.status_code)
                return True
            else:
                logger.error(
                    "Discord webhook failed: HTTP %d — %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        except requests.RequestException as exc:
            logger.error("Discord webhook request failed: %s", exc)
            return False

    # ── Daily Trade Report ────────────────────────────────────────

    def send_daily_report(self, report: TradeReport) -> bool:
        """Send a rich Embed with the daily pipeline + trade results.

        Parameters
        ----------
        report : TradeReport
            Structured pipeline execution results.

        Returns
        -------
        bool
            True if the message was sent successfully.
        """
        tier = report.risk_tier.upper()
        colour = _TIER_COLOURS.get(tier, 0x95a5a6)
        emoji = _TIER_EMOJI.get(tier, "⚪")

        embed: dict[str, Any] = {
            "title": f"🧠 Nexus Quant OS — Daily Report",
            "color": colour,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "SIMULATE ONLY • Not Financial Advice"},
            "fields": [],
        }

        # ── Description ──────────────────────────────────────────
        embed["description"] = (
            f"**{report.timestamp}** • {report.data_source}\n"
            f"Router: `{report.router_mode}` • "
            f"Checkpoint: `{report.checkpoint_name or 'N/A'}`"
        )

        # ── Risk Section ─────────────────────────────────────────
        embed["fields"].append({
            "name": f"{emoji} Risk Assessment",
            "value": (
                f"**Tier**: {tier}\n"
                f"**Regime**: {report.regime}\n"
                f"**Scale**: {report.scale_factor:.2f}\n"
                f"**HMM Danger**: {report.hmm_danger:.3f} • "
                f"**OOD**: {report.ood_score:.3f}"
            ),
            "inline": False,
        })

        # ── Target Allocation ────────────────────────────────────
        if report.target_weights:
            alloc_lines = []
            for sym, w in sorted(
                report.target_weights.items(), key=lambda x: -x[1],
            ):
                alloc_lines.append(
                    f"`{sym:<6}` {w:>6.1%}  •  "
                    f"${w * report.pre_equity:>10,.0f}"
                )
            alloc_text = "\n".join(alloc_lines)
            if report.cash_pct > 0.1:
                alloc_text += f"\n`{'CASH':<6}` {report.cash_pct/100:>6.1%}"
            embed["fields"].append({
                "name": "📊 Target Allocation",
                "value": alloc_text[:1024],
                "inline": False,
            })

        # ── Trade Execution ──────────────────────────────────────
        if report.trades:
            trade_lines = []
            for t in report.trades:
                icon = "✅" if t.status == "FILLED" else "❌"
                trade_lines.append(
                    f"{icon} **{t.side}** `{t.symbol}` x{t.qty:.0f} "
                    f"@ ${t.price:,.2f}"
                )
            embed["fields"].append({
                "name": f"📝 Trades ({report.n_filled} filled, "
                        f"{report.n_rejected} rejected)",
                "value": "\n".join(trade_lines)[:1024],
                "inline": False,
            })
        else:
            embed["fields"].append({
                "name": "📝 Trades",
                "value": "✅ Portfolio aligned — no trades needed",
                "inline": False,
            })

        # ── Account Summary ──────────────────────────────────────
        embed["fields"].append({
            "name": "💰 Account",
            "value": (
                f"**Equity**: ${report.post_equity:,.2f}\n"
                f"**Cash**: ${report.post_cash:,.2f}"
            ),
            "inline": True,
        })

        embed["fields"].append({
            "name": "📈 Pipeline Stats",
            "value": (
                f"Prices: {report.n_price_rows:,}\n"
                f"Aligned: {report.n_aligned_rows:,}\n"
                f"Features: {report.n_features}"
            ),
            "inline": True,
        })

        return self._send({"embeds": [embed]})

    # ── Alert ─────────────────────────────────────────────────────

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
    ) -> bool:
        """Send a standalone alert notification.

        Parameters
        ----------
        title : str
            Alert title.
        message : str
            Alert body (supports Discord markdown).
        severity : str
            ``'info'``, ``'warning'``, ``'error'``, or ``'critical'``.

        Returns
        -------
        bool
            True if sent.
        """
        colour_map = {
            "info":     0x3498db,   # Blue
            "warning":  0xf39c12,   # Yellow
            "error":    0xe74c3c,   # Red
            "critical": 0x8e44ad,   # Purple
        }
        emoji_map = {
            "info":     "ℹ️",
            "warning":  "⚠️",
            "error":    "🚨",
            "critical": "🆘",
        }

        embed = {
            "title": f"{emoji_map.get(severity, '⚠️')} {title}",
            "description": message[:4096],
            "color": colour_map.get(severity, 0xf39c12),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"Nexus Quant OS • {severity.upper()}"},
        }

        return self._send({"embeds": [embed]})

    # ── Error Report ──────────────────────────────────────────────

    def send_error(self, report: TradeReport) -> bool:
        """Send an error report when the pipeline crashes.

        Parameters
        ----------
        report : TradeReport
            Report with ``error`` and ``traceback_str`` populated.

        Returns
        -------
        bool
            True if sent.
        """
        tb = report.traceback_str or "No traceback available"
        # Discord code block max ~2000 chars
        if len(tb) > 1800:
            tb = "..." + tb[-1800:]

        return self.send_alert(
            title="Pipeline Execution Failed",
            message=(
                f"**Error**: `{report.error}`\n\n"
                f"**Timestamp**: {report.timestamp}\n\n"
                f"```\n{tb}\n```"
            ),
            severity="error",
        )

    # ── Test Message ──────────────────────────────────────────────

    def send_test(self) -> bool:
        """Send a test message to verify webhook connectivity.

        Returns
        -------
        bool
            True if the webhook is reachable and accepted the message.
        """
        embed = {
            "title": "✅ Nexus Quant OS — Webhook Test",
            "description": (
                "Discord 通知已成功連接！\n\n"
                "你會在以下時機收到通知：\n"
                "• 📊 每日開盤前 — 模型推論 + 自動下單報告\n"
                "• 📈 每日收盤後 — 持倉績效回報\n"
                "• 🚨 緊急告警 — 防火牆偵測到極端風險\n"
                "• ❌ 錯誤告警 — Pipeline 執行失敗"
            ),
            "color": 0x2ecc71,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Nexus Quant OS • SIMULATE ONLY"},
        }
        return self._send({"embeds": [embed]})

    # ── Health Check ──────────────────────────────────────────────

    def send_health_check(
        self,
        equity: float,
        cash: float,
        n_positions: int,
        positions_summary: str = "",
    ) -> bool:
        """Send a post-market health check / portfolio snapshot.

        Parameters
        ----------
        equity : float
            Current total equity.
        cash : float
            Available cash.
        n_positions : int
            Number of open positions.
        positions_summary : str
            Formatted position details string.

        Returns
        -------
        bool
            True if sent.
        """
        embed: dict[str, Any] = {
            "title": "📈 Post-Market Portfolio Snapshot",
            "color": 0x3498db,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Nexus Quant OS • SIMULATE"},
            "fields": [
                {
                    "name": "💰 Account",
                    "value": (
                        f"**Equity**: ${equity:,.2f}\n"
                        f"**Cash**: ${cash:,.2f}\n"
                        f"**Positions**: {n_positions}"
                    ),
                    "inline": False,
                },
            ],
        }
        if positions_summary:
            embed["fields"].append({
                "name": "📋 Holdings",
                "value": positions_summary[:1024],
                "inline": False,
            })
        return self._send({"embeds": [embed]})
