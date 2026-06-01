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
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from .models import AIAnalysis
from ._constants import (
    LLM_MAX_INPUT_CHARS, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_MAX_RETRIES, AI_BULLISH_THRESHOLD,
    AI_CACHE_DIR, AI_ANALYSIS_TTL_DAYS,
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
        self._cache_dir = AI_CACHE_DIR
        self._cache_ttl = timedelta(days=AI_ANALYSIS_TTL_DAYS)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

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
        
        # ── 快取檢查 ──
        cache_file = self._cache_dir / f"{ticker}.json"
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(
                    cache_file.stat().st_mtime, tz=timezone.utc
                )
                age = datetime.now(timezone.utc) - mtime
                if timedelta(0) <= age < self._cache_ttl:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    return AIAnalysis(
                        ticker=ticker,
                        management_tone=cached.get("management_tone", 0.0),
                        ai_score=cached.get("ai_score", 0.0),
                        confidence=cached.get("confidence", 0.0),
                        summary=str(cached.get("summary", "")),
                        key_risks=cached.get("key_risks", []),
                        growth_catalysts=cached.get("growth_catalysts", []),
                        has_new_product=cached.get("has_new_product", False),
                        has_ma_activity=cached.get("has_ma_activity", False),
                        has_market_expansion=cached.get("has_market_expansion", False),
                        has_cost_restructuring=cached.get("has_cost_restructuring", False),
                        has_regulatory_risk=cached.get("has_regulatory_risk", False),
                    )
            except Exception as e:
                logger.warning("AI cache read failed for %s: %s", ticker, e)
                try:
                    cache_file.unlink()
                except Exception:
                    pass

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
                    if attempt == 0:
                        # 第一次 429：短暫等待後重試一次
                        logger.warning("Gemini 429, short backoff 5s (attempt 1)")
                        time.sleep(5)
                        continue
                    else:
                        # 第二次 429：放棄，回傳 neutral（不值得再等）
                        logger.warning("Gemini 429 persists, skipping AI for %s", ticker)
                        break
                resp.raise_for_status()
                data = resp.json()
                
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                text = text.strip()
                # 防呆：用 Regex 從 ```json...``` 或 ```...``` 中提取 JSON
                import re as _re
                md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
                if md_match:
                    text = md_match.group(1).strip()
                else:
                    # 沒有 Markdown 包裹，嘗試找第一個 { 到最後一個 }
                    brace_start = text.find('{')
                    brace_end = text.rfind('}')
                    if brace_start != -1 and brace_end > brace_start:
                        text = text[brace_start:brace_end + 1]
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

        analysis = AIAnalysis(
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

        # Edge Case #47: 地緣政治脫敏
        geopolitical_keywords = ["軍演", "共軍", "台海危機", "兩岸緊張"]
        if news_text and any(kw in news_text for kw in geopolitical_keywords):
            if analysis.ai_score < 0:
                logger.info("Edge Case #47: 偵測到地緣政治關鍵字，衰減負面 ai_score %f -> %f", analysis.ai_score, analysis.ai_score * 0.2)
                analysis.ai_score *= 0.2
            if analysis.management_tone < 0:
                analysis.management_tone *= 0.2

        # ── 成功後寫入快取 ──
        try:
            cache_data = {
                "ticker": ticker,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "management_tone": analysis.management_tone,
                "ai_score": analysis.ai_score,
                "confidence": analysis.confidence,
                "summary": analysis.summary,
                "key_risks": analysis.key_risks,
                "growth_catalysts": analysis.growth_catalysts,
                "has_new_product": analysis.has_new_product,
                "has_ma_activity": analysis.has_ma_activity,
                "has_market_expansion": analysis.has_market_expansion,
                "has_cost_restructuring": analysis.has_cost_restructuring,
                "has_regulatory_risk": analysis.has_regulatory_risk,
            }
            import uuid
            tmp = cache_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            tmp.replace(cache_file)
        except Exception as e:
            logger.warning("AI cache write failed for %s: %s", ticker, e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        return analysis
