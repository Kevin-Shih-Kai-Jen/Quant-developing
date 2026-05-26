"""
nexus_quant_os/advisor/cli.py — Conversational CLI for Nexus Advisor
====================================================================

Interactive command-line interface to chat with the AI Financial Advisor.

Usage:
    python -m nexus_quant_os.advisor.cli

Features:
- Type normally to chat.
- Type `/image /path/to/img.png <optional prompt>` to upload an image.
- Type `quit` or `exit` to stop.
- Session persists automatically.

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from google.antigravity.types import Image

from nexus_quant_os.advisor.agent import NexusAdvisor

# Project root: three levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_DIR = _PROJECT_ROOT / "data" / "advisor_memory"
_SESSION_FILE = _MEMORY_DIR / "last_session_id.txt"


def _get_last_session_id() -> str | None:
    """Reads the last conversation ID if it exists."""
    if _SESSION_FILE.exists():
        try:
            return _SESSION_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def _save_session_id(session_id: str) -> None:
    """Saves the conversation ID for future resumption."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(session_id, encoding="utf-8")


async def main_loop() -> None:
    """Main interactive chat loop."""
    print("=" * 60)
    print("🧠 Nexus Quant OS — AI Advisor Interactive Console")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the session.")
    print("To upload an image: /image /path/to/img.png [Optional question]")
    print("-" * 60)

    # Initialize agent (resuming if possible)
    last_id = _get_last_session_id()
    if last_id:
        print(f"[System] Resuming previous session: {last_id[:8]}...")
    else:
        print("[System] Starting new session...")
        
    advisor = NexusAdvisor(conversation_id=last_id)

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit"]:
                print("[System] Exiting... session saved.")
                break

            # Handle multimodal input
            payload: str | list = user_input
            if user_input.startswith("/image "):
                # parse format: "/image /path/to/image.png what do you see?"
                parts = user_input.split(" ", 2)
                if len(parts) >= 2:
                    img_path = parts[1]
                    prompt = parts[2] if len(parts) == 3 else "Please analyze this image."
                    
                    if not os.path.exists(img_path):
                        print(f"[Error] Image not found: {img_path}")
                        continue
                        
                    try:
                        img_obj = Image.from_file(img_path)
                        payload = [prompt, img_obj]
                        print(f"[System] Uploading image: {img_path}...")
                    except Exception as e:
                        print(f"[Error] Failed to load image: {e}")
                        continue

            print("\nNexus: ", end="", flush=True)
            
            # Stream response
            async for chunk in advisor.chat_stream(payload):
                print(chunk, end="", flush=True)
                
            print() # Newline after response completes
            
            # Save the session ID in case the user quits forcefully later
            if hasattr(advisor, "conversation_id") and advisor.conversation_id:
                _save_session_id(advisor.conversation_id)

        except KeyboardInterrupt:
            print("\n[System] Exiting... session saved.")
            break
        except Exception as e:
            print(f"\n[System Error] {e}")


if __name__ == "__main__":
    asyncio.run(main_loop())
