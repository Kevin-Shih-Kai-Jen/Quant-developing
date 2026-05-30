"""
alpha_hunter/ai_analyst.py — LLM 深度分析

使用 Gemini Flash 分析公司的 MD&A、風險因子、催化劑。
"""

from __future__ import annotations

import json
import logging
import os
import math
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

from .models import AIAnalysis
from ._constants import (
    LLM_MAX_INPUT_CHARS, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_MAX_RETRIES, AI_BULLISH_THRESHOLD,
)

logger = logging.getLogger(__name__)


class AIAnalyst:
    """LLM 深度分析師。"""

    _ANALYSIS_PROMPT = '''You are a senior equity research analyst. Analyze the following company filing excerpts for {ticker}.

PROVIDE YOUR ANALYSIS AS A JSON OBJECT with exactly these fields:
{{
  "management_tone": <float -1 to 1, -1=very_negative, 0=neutral, 1=very_positive>,
  "ai_score": <float -1 to 1, overall outlook>,
  "confidence": <float 0 to 1>,
  "summary": "<200 words max, key findings>",
  "key_risks": ["<risk1>", "<risk2>", ...],
  "growth_catalysts": ["<catalyst1>", "<catalyst2>", ...],
  "has_new_product": <bool>,
  "has_ma_activity": <bool>,
  "has_market_expansion": <bool>,
  "has_cost_restructuring": <bool>,
  "has_regulatory_risk": <bool>
}}

Output ONLY valid JSON. No markdown, no explanation.

--- MD&A EXCERPT ---
{mda_text}

--- RISK FACTORS EXCERPT ---
{risk_text}

--- RECENT NEWS ---
{news_text}'''

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._session = requests.Session()

        if not self._api_key:
            logger.warning("AIAnalyst: No API key provided. Analysis will return neutral.")

    def analyze(
        self,
        ticker: str,
        mda_text: str = "",
        risk_text: str = "",
        recent_news: list[str] | None = None,
    ) -> AIAnalysis:
        """分析一家公司。"""
        ticker = ticker.upper()
        default_analysis = AIAnalysis(ticker=ticker)
        
        if not self._api_key:
            return default_analysis
            
        mda_trunc = mda_text[:LLM_MAX_INPUT_CHARS]
        risk_trunc = risk_text[:LLM_MAX_INPUT_CHARS]
        news_text = "\n".join(recent_news) if recent_news else "None"
        news_trunc = news_text[:LLM_MAX_INPUT_CHARS]

        prompt = self._ANALYSIS_PROMPT.format(
            ticker=ticker,
            mda_text=mda_trunc,
            risk_text=risk_trunc,
            news_text=news_trunc
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": LLM_TEMPERATURE}
        }

        result_json = None
        for attempt in range(self._max_retries):
            # 簡單的 local rate limit (Gemini API calls should be spaced out)
            with self._rate_lock:
                now = time.monotonic()
                if now - self._last_request_time < 4.0:
                    time.sleep(4.0 - (now - self._last_request_time))
                self._last_request_time = time.monotonic()

            try:
                resp = self._session.post(url, json=payload, timeout=self._timeout)
                if resp.status_code == 429:
                    wait = min(30, 4 ** (attempt + 1))
                    logger.warning("Gemini 429 rate limit, backing off %ds (attempt %d)", wait, attempt+1)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                text = text.strip()
                if text.startswith("```json"): text = text[7:]
                if text.startswith("```"): text = text[3:]
                if text.endswith("```"): text = text[:-3]
                text = text.strip()
                result_json = json.loads(text)
                break
            except requests.RequestException as e:
                logger.warning("AIAnalyst request failed (attempt %d): %s", attempt+1, e)
                time.sleep(2 ** attempt)
            except (KeyError, IndexError, ValueError) as e:
                logger.warning("AIAnalyst parse failed: %s", e)
                break
                
        if not result_json:
            return default_analysis

        # 驗證數值範圍
        def safe_float(val, min_val, max_val, default):
            try:
                v = float(val)
                if math.isnan(v) or math.isinf(v):
                    return default
                return max(min_val, min(max_val, v))
            except (TypeError, ValueError):
                return default

        def safe_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return False

        def safe_list(val, max_items=5, max_chars=200):
            if not isinstance(val, list):
                return []
            return [str(item)[:max_chars] for item in val[:max_items]]

        return AIAnalysis(
            ticker=ticker,
            management_tone=safe_float(result_json.get("management_tone"), -1.0, 1.0, 0.0),
            ai_score=safe_float(result_json.get("ai_score"), -1.0, 1.0, 0.0),
            confidence=safe_float(result_json.get("confidence"), 0.0, 1.0, 0.0),
            summary=str(result_json.get("summary", ""))[:200],
            key_risks=safe_list(result_json.get("key_risks")),
            growth_catalysts=safe_list(result_json.get("growth_catalysts")),
            has_new_product=safe_bool(result_json.get("has_new_product")),
            has_ma_activity=safe_bool(result_json.get("has_ma_activity")),
            has_market_expansion=safe_bool(result_json.get("has_market_expansion")),
            has_cost_restructuring=safe_bool(result_json.get("has_cost_restructuring")),
            has_regulatory_risk=safe_bool(result_json.get("has_regulatory_risk"))
        )
