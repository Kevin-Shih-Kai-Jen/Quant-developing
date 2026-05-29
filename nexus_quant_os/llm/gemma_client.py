"""
llm/gemma_client.py — Google Gemma 4 API Sentiment Client
==========================================================

透過 Google AI Studio REST API 呼叫 Gemma 4 模型，作為
Gemini 2.0 Flash 的備用情緒分析引擎。

設計原則：
    1. 統一介面 — analyze_sentiment() 回傳格式與 GeminiClient/DeepSeekClient 一致
    2. 共用 API Key — 與 Gemini 使用相同的 GEMINI_API_KEY
    3. 速率限制 — 遵守免費額度，必要時自動等待
    4. 優雅降級 — API 不可用時回傳中性分數

Author : Nexus Quant OS — LLM Engineering Division
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger("nexus_quant_os.llm.gemma_client")



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

_NEUTRAL_FALLBACK: dict[str, Any] = {
    "sentiment_score": 0.0,
    "reason": "Fallback: Gemma API unavailable or returned invalid response",
}


class GemmaClient:
    """透過 Google AI Studio REST API 呼叫 Gemma 4 模型進行情緒分析。

    使用與 Gemini 相同的 API Key（GEMINI_API_KEY），作為備用模型。

    Usage
    -----
    >>> client = GemmaClient(api_key="your-key")
    >>> result = client.analyze_sentiment(["NVDA beats earnings"])
    >>> print(result['sentiment_score'])
    0.68
    """

    _MIN_REQUEST_INTERVAL: float = 0.5  # Ollama 本地無需嚴格限速

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemma4:e4b",
        timeout: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time: float = 0.0

        logger.info(
            "GemmaClient 初始化完成 | model=%s  timeout=%.0fs",
            self.model, self.timeout,
        )

    def analyze_sentiment(self, headlines: list[str]) -> dict[str, Any]:
        """傳送新聞標題至 Gemma 4 並取得情緒分數。"""

        if not headlines:
            return dict(_NEUTRAL_FALLBACK)

        numbered = "\n".join(
            f"  {i + 1}. {h}" for i, h in enumerate(headlines)
        )
        prompt = _SENTIMENT_PROMPT_TEMPLATE.format(headlines=numbered)

        raw_text = self._call_gemma_with_retry(prompt)
        if raw_text is None:
            return dict(_NEUTRAL_FALLBACK)

        return self._parse_response(raw_text)

    def _enforce_rate_limit(self) -> None:
        """確保相鄰請求間隔不低於 _MIN_REQUEST_INTERVAL 秒。"""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._MIN_REQUEST_INTERVAL:
            wait_time = self._MIN_REQUEST_INTERVAL - elapsed
            time.sleep(wait_time)
        self._last_request_time = time.monotonic()

    def _call_gemma_with_retry(self, prompt: str) -> str | None:
        """對本地 Ollama 端點發送請求，含指數退避重試。"""
        url = "http://localhost:11434/api/generate"
        
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                self._enforce_rate_limit()
                resp = requests.post(
                    url, json=payload, timeout=self.timeout,
                )

                if resp.status_code == 429:
                    logger.warning("Ollama API 配額超限或繁忙 (attempt %d)。", attempt)
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                if not text:
                    continue
                return text

            except requests.exceptions.Timeout:
                logger.warning("Ollama 請求逾時 (attempt %d)。", attempt)
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama 連線失敗 (attempt %d)。請確認 Ollama 已經在 localhost:11434 啟動！", attempt)
            except requests.exceptions.HTTPError as e:
                logger.warning("Ollama HTTP 錯誤 (attempt %d): %s", attempt, e)
            except (ValueError, KeyError) as e:
                logger.warning("Ollama 解析失敗 (attempt %d): %s", attempt, e)

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        logger.error("Gemma (Ollama) 全部重試失敗，回傳中性分數。")
        return None

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """從模型回應中提取 JSON 並驗證格式。"""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return self._validate_and_clamp(json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
        if match:
            try:
                return self._validate_and_clamp(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        logger.warning("無法從 Gemma 回應中解析 JSON。")
        return dict(_NEUTRAL_FALLBACK)

    @staticmethod
    def _validate_and_clamp(parsed: dict[str, Any]) -> dict[str, Any]:
        """驗證並限制分數至 [-1.0, 1.0]。"""
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
        return {"sentiment_score": max(-1.0, min(1.0, score)), "reason": str(reason)}
