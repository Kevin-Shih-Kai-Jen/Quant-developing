"""
advisor/pipeline_bridge.py — Bridge to Nexus Quant OS Pipeline & Moomoo
========================================================================

Allows the AI Advisor to query live pipeline status, model weights,
risk assessment, and Moomoo account state without re-running the
full pipeline.

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus_quant_os.advisor.pipeline_bridge")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineBridge:
    """Read-only bridge to Nexus Quant OS subsystems.

    All methods are best-effort and return human-readable dicts.
    If a subsystem is unavailable, a helpful error message is returned
    instead of raising an exception.
    """

    # ── Moomoo Account ────────────────────────────────────────────

    @staticmethod
    def get_account_status() -> dict[str, Any]:
        """Query Moomoo SIMULATE account for equity, cash, positions.

        Returns
        -------
        dict
            Keys: equity, cash, buying_power, positions (list), error (if any).
        """
        try:
            from nexus_quant_os.execution.futu_broker import FutuBroker

            broker = FutuBroker(host="127.0.0.1", port=11111)
            account = broker.get_account()
            positions = broker.get_positions()

            pos_list = []
            for p in positions:
                pos_list.append({
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_cost": round(p.avg_cost, 2),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pl": round(p.unrealized_pl, 2),
                })

            return {
                "equity": round(account.equity, 2),
                "cash": round(account.cash, 2),
                "buying_power": round(account.buying_power, 2),
                "n_positions": len(positions),
                "positions": pos_list,
                "environment": "SIMULATE",
            }

        except (ConnectionError, ImportError) as e:
            return {"error": f"Moomoo OpenD 未連線: {e}"}
        except Exception as e:
            return {"error": f"帳戶查詢失敗: {e}"}

    # ── Market Prices ─────────────────────────────────────────────

    @staticmethod
    def get_market_prices(tickers: list[str]) -> dict[str, Any]:
        """Fetch latest prices for given tickers via yfinance.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols (e.g., ["SPY", "QQQ", "NVDA"]).

        Returns
        -------
        dict
            Mapping of ticker → latest close price.
        """
        try:
            import yfinance as yf

            prices = {}
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="2d")
                    if not hist.empty:
                        prices[ticker] = round(float(hist["Close"].iloc[-1]), 2)
                except Exception:
                    pass

            return prices if prices else {"error": "無法取得價格資料"}

        except ImportError:
            return {"error": "yfinance 未安裝"}

    # ── Latest Model Checkpoint Info ──────────────────────────────

    @staticmethod
    def get_checkpoint_info() -> dict[str, Any]:
        """Find and describe the latest MoE checkpoint.

        Returns
        -------
        dict
            Keys: name, path, size_mb, modified.
        """
        ckpt_dir = _PROJECT_ROOT / "nexus_quant_os" / "models" / "checkpoints"
        if not ckpt_dir.exists():
            return {"error": "Checkpoint 目錄不存在"}

        pt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
        if not pt_files:
            return {"error": "沒有找到任何 checkpoint (.pt)"}

        latest = pt_files[-1]
        stat = latest.stat()
        return {
            "name": latest.name,
            "path": str(latest),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "modified": str(
                __import__("datetime").datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
            ),
        }

    # ── Combined Status ───────────────────────────────────────────

    def get_full_status(self) -> str:
        """Get a formatted string of all system status for the advisor.

        Returns
        -------
        str
            Human-readable multi-line status report.
        """
        lines = ["═══ Nexus Quant OS 系統狀態 ═══\n"]

        # Moomoo account
        acct = self.get_account_status()
        if "error" in acct:
            lines.append(f"💼 Moomoo 帳戶: ⚠️ {acct['error']}")
        else:
            lines.append(f"💼 Moomoo 帳戶 ({acct['environment']})")
            lines.append(f"  總資產: ${acct['equity']:,.2f}")
            lines.append(f"  現金:   ${acct['cash']:,.2f}")
            lines.append(f"  持倉:   {acct['n_positions']} 檔")
            if acct["positions"]:
                for p in acct["positions"]:
                    pnl_icon = "📈" if p["unrealized_pl"] >= 0 else "📉"
                    lines.append(
                        f"  {pnl_icon} {p['symbol']:<6} x{p['qty']:.0f}  "
                        f"成本 ${p['avg_cost']:.2f}  "
                        f"P&L ${p['unrealized_pl']:+,.2f}"
                    )

        # Checkpoint
        ckpt = self.get_checkpoint_info()
        lines.append("")
        if "error" in ckpt:
            lines.append(f"🧠 模型: ⚠️ {ckpt['error']}")
        else:
            lines.append(f"🧠 模型 Checkpoint")
            lines.append(f"  名稱: {ckpt['name']}")
            lines.append(f"  大小: {ckpt['size_mb']} MB")
            lines.append(f"  更新: {ckpt['modified']}")

        # Asset universe
        try:
            from main import ASSET_UNIVERSE
            lines.append(f"\n📊 投資組合: {', '.join(ASSET_UNIVERSE)}")
        except ImportError:
            pass

        return "\n".join(lines)
