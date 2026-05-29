#!/usr/bin/env python3
"""
post_market_check.py — Post-Market Health Check + Discord Notification
======================================================================

Queries the Moomoo SIMULATE account and sends a portfolio snapshot
to Discord.  Called by the postmarket launchd schedule.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from nexus_quant_os.notifications.discord_notifier import DiscordNotifier


def main() -> None:
    notifier = DiscordNotifier()

    if not notifier.is_configured:
        print("Discord webhook not configured — skipping")
        return

    try:
        from nexus_quant_os.execution.futu_broker import FutuBroker

        broker = FutuBroker()
        acct = broker.get_account()
        positions = broker.get_positions()

        # Sanity check: warn if equity is suspiciously zero despite connection success
        if acct.equity <= 0 and len(positions) > 0:
            print(f"⚠️ Warning: equity=${acct.equity:.2f} but {len(positions)} positions exist — data inconsistency")

        pos_lines = []
        for p in positions:
            pnl_icon = "📈" if p.unrealized_pl >= 0 else "📉"
            pos_lines.append(
                f"{pnl_icon} `{p.symbol:<6}` x{p.qty:.0f}  "
                f"Cost ${p.avg_cost:.2f}  MktVal ${p.market_value:,.0f}  "
                f"P&L ${p.unrealized_pl:+,.2f}"
            )

        notifier.send_health_check(
            equity=acct.equity,
            cash=acct.cash,
            n_positions=len(positions),
            positions_summary="\n".join(pos_lines) if pos_lines else "No positions",
        )
        print(f"✅ Post-market snapshot sent to Discord")
        print(f"   Equity: ${acct.equity:,.2f}  Cash: ${acct.cash:,.2f}  Positions: {len(positions)}")

    except (ConnectionError, ImportError) as e:
        notifier.send_alert(
            title="Post-Market Check — OpenD Offline",
            message=f"Cannot connect to FutuOpenD: `{e}`\n\nMoomoo OpenD may not be running.",
            severity="warning",
        )
        print(f"⚠️ OpenD not available: {e}")

    except RuntimeError as e:
        # get_account() now raises RuntimeError on API failures
        notifier.send_alert(
            title="Post-Market Check — API Error",
            message=f"Account query failed: `{e}`\n\nMoomoo OpenD connected but returned an error.",
            severity="error",
        )
        print(f"❌ API error: {e}")

    except Exception as e:
        notifier.send_alert(
            title="Post-Market Check Failed",
            message=f"```\n{e}\n```",
            severity="error",
        )
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
