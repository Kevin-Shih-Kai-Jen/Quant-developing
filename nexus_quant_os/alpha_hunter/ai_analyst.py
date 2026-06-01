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
{tw_context}

--- MD&A EXCERPT (OR ATTACHED PDF DATA) ---
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
            
        from .ticker_resolver import TickerResolver
        market = TickerResolver.detect_market(ticker)
        
        tw_context = ""
        if market == "TW":
            from .tw_news_fetcher import TWNewsFetcher
            tw_context = (
                "\n" + TWNewsFetcher.TW_MARKET_CONTEXT + "\n"
                "CRITICAL OVERRIDE: 分析台灣股票時，若遇到『長老吃筍』，請翻譯為『政府八大行庫停損』；"
                "『亮燈』或『亮紅燈』在股價或業績語境中代表『漲停板 Maximum Bullish』，切勿當作負面辭彙。\n"
            )

        news_text = "\n".join(recent_news) if recent_news else "None"
        news_trunc = news_text[:LLM_MAX_INPUT_CHARS]

        # ── 檢查是否為 PDF 檔案 (多模態支援) ──
        is_pdf = isinstance(mda_text, str) and mda_text.lower().endswith(".pdf") and os.path.exists(mda_text)
        
        if is_pdf:
            mda_trunc = f"Attached PDF document: {os.path.basename(mda_text)}"
            risk_trunc = risk_text[:LLM_MAX_INPUT_CHARS]
        else:
            mda_trunc = mda_text[:LLM_MAX_INPUT_CHARS]
            risk_trunc = risk_text[:LLM_MAX_INPUT_CHARS]

        prompt = self._ANALYSIS_PROMPT.format(
            ticker=ticker,
            tw_context=tw_context,
            mda_text=mda_trunc,
            risk_text=risk_trunc,
            news_text=news_trunc
        )

        result_json = None
        
        if is_pdf:
            # ── 多模態 PDF 解析流程 (使用 google.generativeai SDK) ──
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                # 使用 Gemini 1.5 Pro (Multimodal)
                model_obj = genai.GenerativeModel("gemini-1.5-pro")
                
                with self._rate_lock:
                    now = time.monotonic()
                    if now - self._last_request_time < 4.0:
                        time.sleep(4.0 - (now - self._last_request_time))
                    self._last_request_time = time.monotonic()

                logger.info("Uploading PDF to Gemini Vision: %s", mda_text)
                pdf_file = genai.upload_file(path=mda_text)
                
                # 提示詞中加入特別指示
                multimodal_prompt = (
                    f"{prompt}\n\n這是一份法說會簡報 PDF，請直接讀取裡面的圖表與文字，並總結出："
                    "1. 下季營收指引、2. 毛利率預測、3. 資本支出。"
                )
                
                resp = model_obj.generate_content([multimodal_prompt, pdf_file])
                text = resp.text.strip()
                
                # 分析完畢後刪除雲端檔案避免佔用配額
                try:
                    pdf_file.delete()
                except Exception as e:
                    logger.warning("Failed to delete PDF from Gemini: %s", e)
                
                # 解析 JSON
                import re as _re
                md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
                if md_match:
                    text = md_match.group(1).strip()
                else:
                    brace_start = text.find('{')
                    brace_end = text.rfind('}')
                    if brace_start != -1 and brace_end > brace_start:
                        text = text[brace_start:brace_end + 1]
                result_json = json.loads(text)
                
            except Exception as e:
                logger.warning("Gemini Multimodal Vision failed for %s: %s", ticker, e)
                # Fallback 到沒有 PDF 的狀態
                pass
                
        if not is_pdf or not result_json:
            # ── 純文字解析流程 (使用 REST API) ──
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": LLM_TEMPERATURE}
            }

            for attempt in range(self._max_retries):
                with self._rate_lock:
                    now = time.monotonic()
                    if now - self._last_request_time < 4.0:
                        time.sleep(4.0 - (now - self._last_request_time))
                    self._last_request_time = time.monotonic()

                try:
                    resp = self._session.post(url, json=payload, timeout=self._timeout)
                    if resp.status_code == 429:
                        if attempt == 0:
                            logger.warning("Gemini 429, short backoff 5s (attempt 1)")
                            time.sleep(5)
                            continue
                        else:
                            logger.warning("Gemini 429 persists, skipping AI for %s", ticker)
                            break
                    resp.raise_for_status()
                    data = resp.json()
                    
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    text = text.strip()
                    import re as _re
                    md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
                    if md_match:
                        text = md_match.group(1).strip()
                    else:
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
