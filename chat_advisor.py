#!/usr/bin/env python3
"""
chat_advisor.py — AI 理財顧問 CLI
===================================

Interactive terminal interface for the Nexus Quant OS AI Financial Advisor.

Usage::

    # Load env and run
    set -a && source .env && set +a
    PYTHONPATH=. python3 chat_advisor.py

Commands:
    /image <path> [prompt]  — Analyze a broker screenshot
    /status                 — Show system & account status
    /portfolio              — Show current holdings
    /profile                — View stored personal profile
    /clear                  — Clear conversation history
    /help                   — Show this help
    /quit                   — Exit

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import os
import sys
import readline  # enables arrow keys and history in input()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexus_quant_os.advisor.advisor_v2 import GeminiAdvisor


def print_banner() -> None:
    """Print the startup banner."""
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║   🤖 Nexus Quant OS — AI 理財顧問               ║")
    print("  ║   Powered by Gemini 2.5 Flash                    ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()
    print("  📝 指令:")
    print("     /image <path> [問題]  — 辨識券商截圖")
    print("     /status               — 系統 + 帳戶狀態")
    print("     /portfolio            — 查看持倉")
    print("     /profile              — 個人資料")
    print("     /clear                — 清除對話")
    print("     /help                 — 顯示幫助")
    print("     /quit                 — 離開")
    print()
    print("  💬 直接輸入問題開始對話！")
    print("  ─" * 28)
    print()


def main() -> None:
    """Main interactive loop."""
    print_banner()

    try:
        advisor = GeminiAdvisor()
    except ValueError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    print("  ✅ 顧問已就緒\n")

    while True:
        try:
            user_input = input("  You: ").strip()

            if not user_input:
                continue

            # ── Slash Commands ────────────────────────────────
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                advisor.close()
                print("\n  👋 再見！你的個人資料已保存。")
                break

            if user_input.lower() == "/help":
                print_banner()
                continue

            if user_input.lower() == "/status":
                print(f"\n  {advisor.handle_status()}\n")
                continue

            if user_input.lower() == "/profile":
                print(f"\n  {advisor.handle_profile()}\n")
                continue

            if user_input.lower() == "/portfolio":
                print(f"\n  {advisor.handle_portfolio()}\n")
                continue

            if user_input.lower() == "/clear":
                print(f"\n  {advisor.handle_clear()}\n")
                continue

            # ── Image command ─────────────────────────────────
            if user_input.startswith("/image "):
                parts = user_input.split(" ", 2)
                if len(parts) < 2:
                    print("  ❌ 用法: /image <path> [問題]")
                    continue

                img_path = parts[1]
                prompt = parts[2] if len(parts) > 2 else ""

                if not os.path.exists(img_path):
                    print(f"  ❌ 找不到圖片: {img_path}")
                    continue

                print(f"  📷 分析圖片: {img_path}")
                print("\n  🤖 Nexus: ", end="", flush=True)
                reply = advisor.chat(prompt, image_path=img_path)
                print(reply)
                print()
                continue

            # ── Normal chat ───────────────────────────────────
            print("\n  🤖 Nexus: ", end="", flush=True)
            reply = advisor.chat(user_input)
            print(reply)
            print()

        except KeyboardInterrupt:
            advisor.close()
            print("\n\n  👋 再見！")
            break
        except EOFError:
            advisor.close()
            print("\n  👋 再見！")
            break


if __name__ == "__main__":
    main()
