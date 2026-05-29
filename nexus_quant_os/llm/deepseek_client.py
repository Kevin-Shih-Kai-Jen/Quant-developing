"""
llm/deepseek_client.py — DeepSeek Local LLM Sentiment Client
==============================================================

透過 Ollama HTTP API 連接本地 DeepSeek 模型，對新聞標題進行
情緒分析，回傳結構化的情緒分數與推理說明。

設計原則：
    1. 結構化 JSON prompt — 要求模型回傳 sentiment_score 與 reason
    2. 重試機制 — 最多 3 次重試，含指數退避
    3. 優雅降級 — 模型不可用時回傳中性分數 0.0
    4. PiT 合規 — 僅分析傳入的標題，不引入未來資訊

Author : Nexus Quant OS — LLM Engineering Division
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

import requests

logger = logging.getLogger("nexus_quant_os.llm.deepseek_client")

# ─────────────────────────────────────────────────────────────────────
# 情緒分析 Prompt 模板
# ─────────────────────────────────────────────────────────────────────

_SENTIMENT_PROMPT_TEMPLATE = """\
You are a quantitative finance sentiment analyst.
Analyze the following financial news headlines and provide an overall market sentiment score.

Headlines:
{headlines}

Respond with ONLY a valid JSON object in this exact format (no markdown, no explanation outside the JSON):
{{
  "sentiment_score": <float between -1.0 (extremely bearish) and 1.0 (extremely bullish)>,
  "reason": "<brief one-sentence explanation>"
}}
"""

# ─────────────────────────────────────────────────────────────────────
# 預設回傳值（中性）
# ─────────────────────────────────────────────────────────────────────

_NEUTRAL_FALLBACK: dict[str, Any] = {
    "sentiment_score": 0.0,
    "reason": "Fallback: model unavailable or returned invalid response",
}


class DeepSeekClient:
    """透過 Ollama HTTP API 呼叫本地 DeepSeek 模型進行情緒分析。

    Usage
    -----
    >>> client = DeepSeekClient(model_name='deepseek-r1:8b')
    >>> result = client.analyze_sentiment(["Fed holds rates steady"])
    >>> print(result['sentiment_score'])
    0.15
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """初始化 DeepSeek 客戶端。

        Parameters
        ----------
        model_name : str
            Ollama 中已部署的模型名稱。
        base_url : str
            Ollama 伺服器位址。
        timeout : float
            HTTP 請求逾時秒數。
        max_retries : int
            最大重試次數。
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._generate_url = f"{self.base_url}/api/generate"

        logger.info(
            "DeepSeekClient 初始化完成 | model=%s  base_url=%s  "
            "timeout=%.0fs  retries=%d",
            self.model_name, self.base_url, self.timeout, self.max_retries,
        )

    def analyze_sentiment(self, headlines: list[str]) -> dict[str, Any]:
        """傳送新聞標題至 DeepSeek 並取得情緒分數。

        Parameters
        ----------
        headlines : list[str]
            新聞標題列表。

        Returns
        -------
        dict
            包含 ``sentiment_score`` (float, -1.0~1.0) 與
            ``reason`` (str) 的字典。
        """
        if not headlines:
            logger.warning("收到空白標題列表，回傳中性分數。")
            return dict(_NEUTRAL_FALLBACK)

        numbered = "\n".join(
            f"  {i + 1}. {h}" for i, h in enumerate(headlines)
        )
        prompt = _SENTIMENT_PROMPT_TEMPLATE.format(headlines=numbered)

        raw_text = self._call_ollama_with_retry(prompt)
        if raw_text is None:
            return dict(_NEUTRAL_FALLBACK)

        return self._parse_response(raw_text)

    def _call_ollama_with_retry(self, prompt: str) -> str | None:
        """對 Ollama /api/generate 端點發送請求，含指數退避重試。"""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Ollama 請求 (attempt %d/%d) → %s",
                    attempt, self.max_retries, self._generate_url,
                )
                resp = requests.post(
                    self._generate_url,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                response_text: str = data.get("response", "")
                if not response_text.strip():
                    logger.warning("Ollama 回傳空白回應 (attempt %d)。", attempt)
                    continue
                logger.debug("Ollama 回應 (attempt %d): %s", attempt, response_text[:200])
                return response_text

            except requests.exceptions.Timeout:
                logger.warning(
                    "Ollama 請求逾時 (attempt %d/%d)。", attempt, self.max_retries,
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Ollama 連線失敗 (attempt %d/%d) — 伺服器可能未啟動。",
                    attempt, self.max_retries,
                )
            except requests.exceptions.HTTPError as e:
                logger.warning(
                    "Ollama HTTP 錯誤 (attempt %d/%d): %s", attempt, self.max_retries, e,
                )
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Ollama 回應解析失敗 (attempt %d/%d): %s", attempt, self.max_retries, e,
                )

            if attempt < self.max_retries:
                backoff = 2 ** (attempt - 1)
                logger.info("重試前等待 %d 秒...", backoff)
                time.sleep(backoff)

        logger.error("Ollama 全部 %d 次重試均失敗，回傳中性分數。", self.max_retries)
        return None

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """從模型回應中提取 JSON 並驗證格式。"""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            return self._validate_and_clamp(parsed)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return self._validate_and_clamp(parsed)
            except json.JSONDecodeError:
                pass

        logger.warning("無法從 DeepSeek 回應中解析 JSON，回傳中性分數。")
        return dict(_NEUTRAL_FALLBACK)

    @staticmethod
    def _validate_and_clamp(parsed: dict[str, Any]) -> dict[str, Any]:
        """驗證解析結果並將分數限制在 [-1.0, 1.0] 範圍內。"""
        score = parsed.get("sentiment_score")
        reason = parsed.get("reason", "No reason provided")

        if score is None:
            return dict(_NEUTRAL_FALLBACK)

        try:
            score = float(score)
        except (TypeError, ValueError):
            return dict(_NEUTRAL_FALLBACK)
        if math.isnan(score) or math.isinf(score):
            logger.warning("LLM 回傳 NaN/Inf 分數，使用中性值。")
            return dict(_NEUTRAL_FALLBACK)

        clamped = max(-1.0, min(1.0, score))
        return {"sentiment_score": clamped, "reason": str(reason)}
