"""
advisor/advisor_v2.py — AI Financial Advisor (Gemini 2.5 Flash)
================================================================

Standalone conversational financial advisor using Google's Gemini 2.5 Flash
via the ``google-genai`` SDK.  Supports:

- Natural language chat in Chinese
- Broker screenshot recognition (multimodal image input)
- Long-term memory via SQLite
- Live Moomoo account & pipeline status queries
- Context-aware recommendations (military service, travel, etc.)

Data-flow position::

    chat_advisor.py (CLI)
        → GeminiAdvisor.chat()
            → Gemini 2.5 Flash API
            ← Response
        → Memory.save()
        → Display to user

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from nexus_quant_os.advisor.memory import AdvisorMemory
from nexus_quant_os.advisor.pipeline_bridge import PipelineBridge

logger = logging.getLogger("nexus_quant_os.advisor.advisor_v2")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _PROJECT_ROOT / "data" / "advisor_memory" / "advisor.db"

# ═════════════════════════════════════════════════════════════════════
# System Prompt
# ═════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_TEMPLATE = """\
你是 **Nexus**，Nexus Quant OS 量化交易系統的個人化 AI 理財顧問。

## 身分與風格
- 你是一位專業、冷靜、善於分析的量化金融顧問
- 使用繁體中文回答，語氣親切但專業
- 適時使用 emoji 讓回答更生動
- 回答要有結構（使用列表、分段）
- 重要數字要精確，不要模糊帶過

## 你的能力
1. **投資組合分析**：分析用戶的持倉，計算配置比例，評估風險
2. **資產配置建議**：根據個人狀況（風險偏好、資金、時間限制）推薦配置
3. **券商截圖辨識**：用戶上傳截圖時，辨識持倉（股票代碼、數量、成本、損益）
4. **市場觀察**：提供對 SPY/QQQ/TLT/GLD/NVDA/AVGO/IWM 的分析
5. **情境適配**：根據特殊情況（當兵、出國、考試）調整建議

## 投資組合標的
Nexus Quant OS 目前管理的資產：AVGO, GLD, IWM, NVDA, QQQ, SPY, TLT（7 檔美股 ETF + 個股）

## 重要限制
- ⚠️ 這是模擬交易帳戶（SIMULATE），不是真實資金
- ⚠️ 你的建議不構成正式投資建議
- ⚠️ 不要推薦上述 7 檔以外的個別股票（可以討論但不推薦買入）
- ⚠️ 永遠優先考慮風險管理和回撤保護

## 用戶記憶
以下是你記得的用戶相關資訊：

{memory_context}

## 系統狀態
{system_status}

## 回答指引
- 如果用戶分享個人資訊（風險偏好、資金、限制），要主動說「我已記住」
- 如果用戶問「目前推薦什麼」，參考系統狀態中的模型配置
- 如果用戶上傳截圖，先解析所有持倉，再給分析
- 給配置建議時要附上百分比和理由
- 不確定的事情要誠實說不確定
"""


class GeminiAdvisor:
    """Conversational AI financial advisor powered by Gemini 2.5 Flash.

    Parameters
    ----------
    api_key : str | None
        Google AI Studio API key.  Falls back to ``GEMINI_API_KEY`` env var.
    db_path : str | Path
        Path to the SQLite memory database.
    model_name : str
        Gemini model to use.  Default ``'gemini-2.5-flash'``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        db_path: str | Path | None = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY 未設定。請在 .env 加入：\n"
                "GEMINI_API_KEY=your_key_here\n"
                "前往 https://aistudio.google.com/apikey 免費申請"
            )

        self._client = genai.Client(api_key=key)
        self._model = model_name
        self._memory = AdvisorMemory(db_path or _DEFAULT_DB)
        self._bridge = PipelineBridge()
        self._history: list[types.Content] = []

        logger.info("GeminiAdvisor initialized: model=%s", model_name)

    # ── Chat ──────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        image_path: str | None = None,
    ) -> str:
        """Send a message to the advisor and get a response.

        Parameters
        ----------
        message : str
            User's text message.
        image_path : str | None
            Optional path to an image (broker screenshot, etc.).

        Returns
        -------
        str
            Advisor's response text.
        """
        # Build content parts
        parts: list[Any] = []

        if image_path:
            try:
                img_path = Path(image_path)
                if not img_path.exists():
                    return f"❌ 找不到圖片: {image_path}"

                # Read image bytes and determine mime type
                img_bytes = img_path.read_bytes()
                suffix = img_path.suffix.lower()
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }
                mime = mime_map.get(suffix, "image/png")

                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                if message:
                    parts.append(types.Part.from_text(text=message))
                else:
                    parts.append(types.Part.from_text(
                        text="請分析這張券商截圖，列出所有持倉（股票代碼、數量、成本、損益）"
                    ))
            except Exception as e:
                return f"❌ 圖片載入失敗: {e}"
        else:
            parts.append(types.Part.from_text(text=message))

        # Add user turn to history
        user_content = types.Content(role="user", parts=parts)
        self._history.append(user_content)

        # Keep history manageable (last 20 turns)
        if len(self._history) > 40:
            self._history = self._history[-40:]

        # Build system prompt with memory + status
        system_prompt = self._build_system_prompt()

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._history,  # type: ignore[arg-type]
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )

            reply = response.text or "(無回應)"

            # Add assistant turn to history
            self._history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reply)],
                )
            )

            # Save conversation summaries (truncated)
            self._memory.add_conversation("user", message[:200])
            self._memory.add_conversation("assistant", reply[:200])

            # Auto-detect profile updates from the conversation
            self._auto_extract_profile(message, reply)

            return reply

        except Exception as e:
            logger.error("Gemini API error: %s", e)
            # Remove the failed user turn
            self._history.pop()
            return f"❌ API 錯誤: {e}"

    # ── System Prompt Builder ─────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt with injected memory and status."""
        memory_ctx = self._memory.build_memory_context()

        # Get system status (best-effort, don't fail)
        try:
            status = self._bridge.get_full_status()
        except Exception:
            status = "（系統狀態暫時無法取得）"

        return _SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_ctx,
            system_status=status,
        )

    # ── Auto Profile Extraction ───────────────────────────────────

    def _auto_extract_profile(self, user_msg: str, reply: str) -> None:
        """Try to extract profile updates from the conversation.

        Simple keyword-based extraction — doesn't need LLM.
        """
        msg = user_msg.lower()

        # Risk tolerance
        if any(kw in msg for kw in ["風險偏好", "風險承受", "保守", "積極", "穩健"]):
            if "保守" in msg:
                self._memory.set_profile("風險偏好", "保守型")
            elif "積極" in msg or "激進" in msg:
                self._memory.set_profile("風險偏好", "積極型")
            elif "穩健" in msg or "中等" in msg:
                self._memory.set_profile("風險偏好", "穩健型")

        # Investment amount
        for unit, multiplier in [("萬", 10000), ("千", 1000), ("百萬", 1000000)]:
            if unit in msg and any(c.isdigit() for c in msg):
                import re
                match = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}", msg)
                if match:
                    amount = float(match.group(1)) * multiplier
                    if "台幣" in msg or "TWD" in msg.upper():
                        self._memory.set_profile("投資金額", f"NT${amount:,.0f}")
                    elif "美金" in msg or "USD" in msg.upper() or "$" in msg:
                        self._memory.set_profile("投資金額", f"US${amount:,.0f}")
                    else:
                        self._memory.set_profile("投資金額", f"NT${amount:,.0f}")
                    break

        # Military service
        if any(kw in msg for kw in ["當兵", "服役", "軍事"]):
            self._memory.set_profile("特殊限制", "即將服兵役，一段時間無法操作")

        # Occupation / student
        if any(kw in msg for kw in ["學生", "大學", "研究所"]):
            self._memory.set_profile("身份", "學生")
        elif any(kw in msg for kw in ["工程師", "上班族"]):
            self._memory.set_profile("身份", "上班族")

    # ── Slash Commands ────────────────────────────────────────────

    def handle_status(self) -> str:
        """Handle /status command — return system status."""
        return self._bridge.get_full_status()

    def handle_profile(self) -> str:
        """Handle /profile command — show stored profile."""
        profiles = self._memory.get_all_profiles()
        if not profiles:
            return "📝 尚未儲存任何個人資料。在對話中告訴我你的情況，我會自動記住。"
        lines = ["📝 已儲存的個人資料：\n"]
        for k, v in profiles.items():
            lines.append(f"  • {k}: {v}")
        return "\n".join(lines)

    def handle_portfolio(self) -> str:
        """Handle /portfolio command — show latest snapshot + Moomoo."""
        sections = []

        # Moomoo live status
        acct = self._bridge.get_account_status()
        if "error" not in acct:
            lines = [
                "💼 Moomoo 模擬帳戶：",
                f"  總資產: ${acct['equity']:,.2f}",
                f"  現金: ${acct['cash']:,.2f}",
            ]
            if acct["positions"]:
                for p in acct["positions"]:
                    icon = "📈" if p["unrealized_pl"] >= 0 else "📉"
                    lines.append(
                        f"  {icon} {p['symbol']:<6} x{p['qty']:.0f}  "
                        f"成本 ${p['avg_cost']:.2f}  "
                        f"P&L ${p['unrealized_pl']:+,.2f}"
                    )
            else:
                lines.append("  （目前無持倉）")
            sections.append("\n".join(lines))
        else:
            sections.append(f"💼 Moomoo: ⚠️ {acct['error']}")

        # Memory snapshot
        snap = self._memory.get_latest_snapshot()
        if snap:
            lines = [f"\n📋 上次記錄的持倉 ({snap['source']}, {snap['timestamp'][:10]})："]
            for h in snap["holdings"]:
                lines.append(f"  • {h.get('symbol','?')}: {h.get('qty',0)} 股 @ ${h.get('avg_cost',0):.2f}")
            sections.append("\n".join(lines))

        return "\n".join(sections) if sections else "（無持倉資料）"

    def handle_clear(self) -> str:
        """Handle /clear command — clear conversation history."""
        self._history.clear()
        self._memory.clear_conversations()
        return "🗑️ 對話歷史已清除。記憶庫中的個人資料保留不動。"

    def close(self) -> None:
        """Clean up resources."""
        self._memory.close()
