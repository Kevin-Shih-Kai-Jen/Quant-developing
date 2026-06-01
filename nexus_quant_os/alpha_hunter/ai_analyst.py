"""
alpha_hunter/ai_analyst.py — LLM 深度分析 (The Abyss Edition)

【最高機密升級七：LLM 認知不確定性量化 (Self-Consistency Ensemble)】
使用 Gemini Flash 進行 5 次平行採樣，計算資訊熵。高熵（幻覺分歧）直接丟棄。
"""

from __future__ import annotations

import json
import logging
import os
import math
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import requests

from .models import AIAnalysis
from ._constants import (
    LLM_MAX_INPUT_CHARS, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_MAX_RETRIES, AI_BULLISH_THRESHOLD,
    AI_CACHE_DIR, AI_ANALYSIS_TTL_DAYS,
)

logger = logging.getLogger(__name__)


class AIAnalyst:
    """LLM 深度分析師 - 搭載深淵級不確定性防禦。"""

    _ANALYSIS_PROMPT = '''You are a senior equity research analyst. Analyze the following company filing excerpts for {ticker}.

PROVIDE YOUR ANALYSIS AS A JSON OBJECT with exactly these fields:
{{
  "management_tone": <float -1 to 1, -1=very_negative, 0=neutral, 1=very_positive>,
  "ai_score": <float -1 to 1, overall outlook>,
  "confidence": <float 0 to 1>,
  "summary": "<200 words max, key findings>",
  "key_risks": ["<risk1>", "<risk2>"],
  "growth_catalysts": ["<catalyst1>", "<catalyst2>"],
  "has_new_product": <bool>,
  "has_ma_activity": <bool>,
  "has_market_expansion": <bool>,
  "has_cost_restructuring": <bool>,
  "has_regulatory_risk": <bool>
}}

Output ONLY valid JSON. No markdown, no explanation.
{tw_context}
{temporal_anchoring}

--- MD&A EXCERPT ---
{mda_text}

--- RISK FACTORS ---
{risk_text}

--- RECENT NEWS ---
{news_text}'''

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
        n_ensemble: int = 5,  # 深淵級升級：自洽性抽樣次數
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._n_ensemble = n_ensemble
        self._rate_lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._cache_dir = AI_CACHE_DIR
        self._cache_ttl = timedelta(days=AI_ANALYSIS_TTL_DAYS)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if not self._api_key:
            logger.warning("AIAnalyst: No API key provided.")

    def _mask_entity(self, text: str, ticker: str) -> str:
        """【升級一：實體盲化協議】"""
        if not text:
            return text
        masked_text = text.replace(ticker, "[Company_A]")
        import re
        match = re.search(r'^(\d+)', ticker)
        if match:
            digits = match.group(1)
            masked_text = masked_text.replace(digits, "[Company_A]")
        return masked_text

    def _call_llm_once(self, prompt: str, is_pdf: bool, mda_text: str) -> Optional[Dict[str, Any]]:
        """執行單次 LLM 請求"""
        if not self._api_key:
            return None

        result_json = None
        if is_pdf:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                model_obj = genai.GenerativeModel("gemini-1.5-pro")
                
                with self._rate_lock:
                    now = time.monotonic()
                    if now - self._last_request_time < 4.0:
                        time.sleep(4.0 - (now - self._last_request_time))
                    self._last_request_time = time.monotonic()

                pdf_file = genai.upload_file(path=mda_text)
                resp = model_obj.generate_content([prompt, pdf_file], generation_config=genai.types.GenerationConfig(temperature=0.7))
                text = resp.text.strip()
                try:
                    pdf_file.delete()
                except Exception:
                    pass
                
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
                logger.warning("Gemini Vision failed: %s", e)
                
        if not is_pdf or not result_json:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
            # 【升級七】強制 Temperature=0.7 引入擾動
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7}
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
                        time.sleep(5)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
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
                except Exception as e:
                    logger.warning("API attempt %d failed: %s", attempt+1, e)
                    time.sleep(2 ** attempt)

        return result_json

    def analyze(
        self,
        ticker: str,
        mda_text: str = "",
        risk_text: str = "",
        recent_news: list[str] | None = None,
        backtest_date: str = "",
    ) -> AIAnalysis:
        ticker = ticker.upper()
        default_analysis = AIAnalysis(ticker=ticker)
        
        # 快取檢查...
        cache_file = self._cache_dir / f"{ticker}_analysis_{backtest_date or 'latest'}.json"
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
                age = datetime.now(timezone.utc) - mtime
                if age < self._cache_ttl:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    return AIAnalysis(**cached)
            except Exception:
                pass

        if not self._api_key:
            return default_analysis
            
        from .ticker_resolver import TickerResolver
        market = TickerResolver.detect_market(ticker)
        tw_context = ""
        if market == "TW":
            tw_context = "CRITICAL OVERRIDE: 亮燈=漲停板, 長老吃筍=八大行庫停損。\n"

        temporal_anchoring = f"\nCRITICAL: You are operating in {backtest_date}. DO NOT USE FUTURE KNOWLEDGE.\n" if backtest_date else ""

        is_pdf = isinstance(mda_text, str) and mda_text.lower().endswith(".pdf") and os.path.exists(mda_text)
        
        mda_trunc = f"Attached PDF: {os.path.basename(mda_text)}" if is_pdf else mda_text[:LLM_MAX_INPUT_CHARS]
        risk_trunc = risk_text[:LLM_MAX_INPUT_CHARS]
        news_trunc = ("\n".join(recent_news) if recent_news else "None")[:LLM_MAX_INPUT_CHARS]
            
        mda_trunc = self._mask_entity(mda_trunc, ticker)
        risk_trunc = self._mask_entity(risk_trunc, ticker)
        news_trunc = self._mask_entity(news_trunc, ticker)

        prompt = self._ANALYSIS_PROMPT.format(
            ticker="[Company_A]",
            temporal_anchoring=temporal_anchoring,
            tw_context=tw_context,
            mda_text=mda_trunc,
            risk_text=risk_trunc,
            news_text=news_trunc
        )

        # 【升級七：自洽性抽樣與溫度擾動】
        results = []
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_ensemble) as executor:
            futures = [executor.submit(self._call_llm_once, prompt, is_pdf, mda_text) for _ in range(self._n_ensemble)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        if not results:
            return default_analysis

        # 計算資訊熵 (Entropy Calculation)
        # 對布林值特徵計算多數決與熵
        features = ["has_new_product", "has_ma_activity", "has_market_expansion", "has_cost_restructuring", "has_regulatory_risk"]
        entropies = []
        final_features = {}
        
        for feat in features:
            trues = sum(1 for r in results if r.get(feat, False))
            p = trues / len(results)
            final_features[feat] = (p >= 0.5) # 多數決
            # 二元熵
            if p == 0 or p == 1:
                e = 0.0
            else:
                e = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
            entropies.append(e)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        
        # 平均數值分數
        avg_ai_score = sum(r.get("ai_score", 0.0) for r in results) / len(results)
        avg_tone = sum(r.get("management_tone", 0.0) for r in results) / len(results)

        # 毒藥檢測：如果平均資訊熵大於 0.8 (代表分歧極大)
        if avg_entropy > 0.8:
            logger.warning(f"High Entropy detected for {ticker}: {avg_entropy:.2f}. Forcing AI score to 0 (Neutral).")
            avg_ai_score = 0.0
            avg_tone = 0.0

        analysis = AIAnalysis(
            ticker=ticker,
            management_tone=avg_tone,
            ai_score=avg_ai_score,
            confidence=1.0 - avg_entropy, # 信心度反比於熵
            summary=results[0].get("summary", "")[:200], # 取第一個 summary
            key_risks=results[0].get("key_risks", [])[:5],
            growth_catalysts=results[0].get("growth_catalysts", [])[:5],
            **final_features
        )

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(analysis.dict(), f)
        except Exception:
            pass

        return analysis

    def invalidate_cache(self, ticker: str):
        import glob
        pattern = os.path.join(self._cache_dir, f"{ticker}_analysis_*.json")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass
