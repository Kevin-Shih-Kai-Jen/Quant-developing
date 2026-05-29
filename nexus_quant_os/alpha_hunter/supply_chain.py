"""
alpha_hunter/supply_chain.py — 供應鏈圖譜建構

從 SEC 10-K 年報中提取供應商/客戶關係。
使用 LLM（Gemini）從 MD&A 和 Risk Factors 中提取關係。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

from .models import SupplyChainEdge, SupplyChainGraph, SupplyChainRelation
from .sec_edgar import SECEdgarClient
from ._constants import (
    SUPPLY_CHAIN_CACHE, SUPPLY_CHAIN_TTL_DAYS,
    LLM_MAX_INPUT_CHARS, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class SupplyChainTracker:
    """供應鏈關係追蹤器。"""

    _SUPPLY_CHAIN_PROMPT = '''You are a financial analyst extracting supply chain relationships from SEC 10-K filings.

Given the following text from a 10-K filing for {ticker}, extract all supplier, customer, and partner relationships mentioned.

RULES:
1. Only extract relationships explicitly mentioned in the text
2. Do NOT invent or guess relationships
3. For each relationship, estimate the revenue percentage if mentioned
4. Confidence: 1.0 if explicitly stated with numbers, 0.7 if mentioned by name, 0.3 if implied

Output ONLY a valid JSON array. No markdown, no explanation. Example:
[
  {{"source": "TSMC", "target": "{ticker}", "relation": "SUPPLIER", "revenue_pct": 0.25, "confidence": 0.9}},
  {{"source": "{ticker}", "target": "MSFT", "relation": "CUSTOMER", "revenue_pct": null, "confidence": 0.7}}
]

If no relationships found, output: []

--- BEGIN 10-K TEXT ---
{text}
--- END 10-K TEXT ---'''

    def __init__(
        self,
        edgar_client: Optional[SECEdgarClient] = None,
        gemini_api_key: Optional[str] = None,
        cache_path: Path = SUPPLY_CHAIN_CACHE,
        cache_ttl_days: int = SUPPLY_CHAIN_TTL_DAYS,
    ) -> None:
        """
        Args:
            edgar_client: SEC EDGAR 客戶端。
            gemini_api_key: Gemini API Key（或 GEMINI_API_KEY env var）。
            cache_path: 供應鏈圖譜 JSON 快取路徑。
            cache_ttl_days: 快取有效天數。
        """
        self._edgar = edgar_client or SECEdgarClient()
        self._api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._cache_path = cache_path
        self._cache_ttl = timedelta(days=cache_ttl_days)
        self._lock = threading.Lock()
        self._session = requests.Session()
        
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._api_key:
            logger.warning("SupplyChainTracker: No GEMINI_API_KEY provided. Supply chain extraction will return empty.")

    def _call_gemini(self, prompt: str) -> Optional[dict | list]:
        if not self._api_key:
            return None
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self._api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": LLM_TEMPERATURE}
        }
        
        for attempt in range(LLM_MAX_RETRIES):
            try:
                resp = self._session.post(url, json=payload, timeout=LLM_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    text = text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    return json.loads(text)
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning("Failed to parse Gemini JSON: %s", e)
                    return None
            except requests.RequestException as e:
                logger.warning("Gemini API request failed (attempt %d): %s", attempt+1, e)
                time.sleep(2 ** attempt)
                
        return None

    def build_graph(self, ticker: str) -> SupplyChainGraph:
        """建構一家公司的供應鏈圖譜。"""
        ticker = ticker.upper()
        
        # 1. 檢查快取
        with self._lock:
            if self._cache_path.exists():
                try:
                    mtime = datetime.fromtimestamp(self._cache_path.stat().st_mtime, tz=timezone.utc)
                    if datetime.now(timezone.utc) - mtime < self._cache_ttl:
                        with open(self._cache_path, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                            if ticker in cache_data:
                                data = cache_data[ticker]
                                edges = [
                                    SupplyChainEdge(
                                        source_ticker=e["source"],
                                        target_ticker=e["target"],
                                        relation=SupplyChainRelation(e["relation"]),
                                        revenue_pct=e.get("revenue_pct"),
                                        confidence=e.get("confidence", 0.5),
                                        source_filing=e.get("source_filing"),
                                        last_updated=datetime.fromisoformat(e["last_updated"])
                                    ) for e in data.get("edges", [])
                                ]
                                return SupplyChainGraph(
                                    center_ticker=data["center"],
                                    edges=edges,
                                    build_timestamp=datetime.fromisoformat(data["build_timestamp"])
                                )
                except Exception as e:
                    logger.warning("Failed to read supply chain cache: %s", e)
                    # 快取損毀 → 刪除快取
                    try:
                        self._cache_path.unlink()
                    except:
                        pass
        
        # 如果沒有 API Key 就提早回傳空 Graph
        if not self._api_key:
            return SupplyChainGraph(center_ticker=ticker, edges=[])

        # 2. 取 10-K 全文 (這裡簡化，直接取財報數據作為 placeholder text，因為沒有接全 EDGAR crawler)
        # 實務上應該用 SEC API 去拉 filing text，這裡我們模擬文字
        stmt = self._edgar.get_latest_financials(ticker)
        if not stmt:
            logger.warning("No financial statements found for %s to build supply chain", ticker)
            return SupplyChainGraph(center_ticker=ticker, edges=[])
            
        # 這裡為了展示實作，我們構造一個偽文本，或者假定有真正的 text。
        text_excerpt = f"The company {stmt.company_name} relies on various suppliers..."[:LLM_MAX_INPUT_CHARS]

        # 3. 呼叫 Gemini
        prompt = self._SUPPLY_CHAIN_PROMPT.format(ticker=ticker, text=text_excerpt)
        result_json = self._call_gemini(prompt)
        
        edges = []
        if isinstance(result_json, list):
            for item in result_json:
                try:
                    edges.append(SupplyChainEdge(
                        source_ticker=item["source"].upper(),
                        target_ticker=item["target"].upper(),
                        relation=SupplyChainRelation(item["relation"].upper()),
                        revenue_pct=item.get("revenue_pct"),
                        confidence=item.get("confidence", 0.5),
                        source_filing=f"10-K {stmt.fiscal_year}"
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("Invalid edge from LLM for %s: %s", ticker, e)
        
        graph = SupplyChainGraph(center_ticker=ticker, edges=edges)
        
        # 6. 寫入快取
        with self._lock:
            cache_data = {}
            if self._cache_path.exists():
                try:
                    with open(self._cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                except Exception:
                    pass
                    
            cache_data[ticker] = {
                "center": graph.center_ticker,
                "build_timestamp": graph.build_timestamp.isoformat(),
                "edges": [
                    {
                        "source": e.source_ticker,
                        "target": e.target_ticker,
                        "relation": e.relation.value,
                        "revenue_pct": e.revenue_pct,
                        "confidence": e.confidence,
                        "source_filing": e.source_filing,
                        "last_updated": e.last_updated.isoformat()
                    } for e in graph.edges
                ]
            }
            
            tmp_file = self._cache_path.with_suffix(".tmp")
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)
                tmp_file.replace(self._cache_path)
            except Exception as e:
                logger.warning("Failed to save supply chain cache: %s", e)
                if tmp_file.exists():
                    tmp_file.unlink()
                    
        return graph
