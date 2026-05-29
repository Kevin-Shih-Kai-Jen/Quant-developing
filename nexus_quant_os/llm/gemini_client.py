"""
llm/gemini_client.py — Google Gemini API Sentiment Client
==========================================================

透過 Google AI Studio REST API 呼叫 Gemini 模型，對新聞標題
進行情緒分析。與 DeepSeekClient 共用相同的輸出介面。

設計原則：
    1. 統一介面 — analyze_sentiment() 回傳格式與 DeepSeekClient 一致
    2. 速率限制 — 遵守 15 requests/minute 限制，必要時自動等待
    3. API Key 管理 — 支援參數傳入或環境變數 GEMINI_API_KEY
    4. 優雅降級 — API Key 未設定或配額耗盡時回傳中性分數

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

logger = logging.getLogger("nexus_quant_os.llm.gemini_client")

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

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
    "reason": "Fallback: Gemini API unavailable or returned invalid response",
}


class GeminiClient:
    """透過 Google AI Studio REST API 呼叫 Gemini 模型進行情緒分析。

    Usage
    -----
    >>> client = GeminiClient(api_key="your-key", model="gemini-2.0-flash")
    >>> result = client.analyze_sentiment(["NVDA beats earnings"])
    >>> print(result['sentiment_score'])
    0.72
    """

    _MIN_REQUEST_INTERVAL: float = 4.0  # 15 req/min → ~4s interval

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time: float = 0.0

        if not self.api_key:
            logger.warning(
                "Gemini API Key 未設定！所有分析請求將回傳中性分數。"
            )
        else:
            logger.info(
                "GeminiClient 初始化完成 | model=%s  timeout=%.0fs",
                self.model, self.timeout,
            )

    def analyze_sentiment(self, headlines: list[str]) -> dict[str, Any]:
        """傳送新聞標題至 Gemini 並取得情緒分數。"""
        if not self.api_key:
            return dict(_NEUTRAL_FALLBACK)
        if not headlines:
            return dict(_NEUTRAL_FALLBACK)

        numbered = "\n".join(
            f"  {i + 1}. {h}" for i, h in enumerate(headlines)
        )
        prompt = _SENTIMENT_PROMPT_TEMPLATE.format(headlines=numbered)

        raw_text = self._call_gemini_with_retry(prompt)
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

    def _call_gemini_with_retry(self, prompt: str) -> str | None:
        """對 Gemini generateContent 端點發送請求，含指數退避重試。"""
        url = f"{_GEMINI_API_BASE}/{self.model}:generateContent"
        params = {"key": self.api_key}
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256,
            },
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                self._enforce_rate_limit()
                resp = requests.post(
                    url, params=params, json=payload, timeout=self.timeout,
                )

                if resp.status_code == 429:
                    logger.warning("Gemini API 配額超限 (attempt %d)。", attempt)
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = self._extract_text(data)
                if not text:
                    continue
                return text

            except requests.exceptions.Timeout:
                logger.warning("Gemini 請求逾時 (attempt %d)。", attempt)
            except requests.exceptions.ConnectionError:
                logger.warning("Gemini 連線失敗 (attempt %d)。", attempt)
            except requests.exceptions.HTTPError as e:
                error_msg = str(e).replace(self.api_key, '***API_KEY***') if self.api_key else str(e)
                logger.warning("Gemini HTTP 錯誤 (attempt %d): %s", attempt, error_msg)
            except (ValueError, KeyError) as e:
                logger.warning("Gemini 解析失敗 (attempt %d): %s", attempt, e)

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        logger.error("Gemini 全部重試失敗，回傳中性分數。")
        return None

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        """從 Gemini 回應結構提取生成文字。"""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else None
        except (IndexError, AttributeError):
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

        logger.warning("無法從 Gemini 回應中解析 JSON。")
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
        if math.isnan(score):
            logger.warning("LLM 回傳 NaN 分數，使用中性值。")
            return dict(_NEUTRAL_FALLBACK)
        return {"sentiment_score": max(-1.0, min(1.0, score)), "reason": str(reason)}
