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
import re
from collections import deque
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

from .models import SupplyChainEdge, SupplyChainGraph, SupplyChainRelation, RecursiveSupplyChainGraph
from .sec_edgar import SECEdgarClient
from ._constants import (
    SUPPLY_CHAIN_CACHE, SUPPLY_CHAIN_TTL_DAYS,
    LLM_MAX_INPUT_CHARS, LLM_TEMPERATURE, LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    SUPPLY_CHAIN_NODES_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_MAX_INPUT_CHARS,
)

logger = logging.getLogger(__name__)


class SupplyChainTracker:
    """供應鏈關係追蹤器。"""

    _RECURSIVE_SUPPLY_CHAIN_PROMPT = """You are a financial analyst. Identify supply chain relationships for the company described below.

RULES:
1. Extract relationships from the provided text AND your general knowledge of this company.
2. Prioritize well-known, publicly verifiable supplier/customer/competitor relationships.
3. Return at most 10 relationships.
4. Each ticker must be a real US stock ticker (e.g., AAPL, TSM) or Taiwan stock ticker (e.g., 2330.TW, 2454.TW). Do NOT use tickers from other markets.
5. Confidence: 1.0 if stated with numbers, 0.7 if mentioned by name, 0.3 if implied.

OUTPUT FORMAT: A valid JSON array. No markdown, no explanation, no preamble.

=== EXAMPLE 1 (NVDA) ===
[
  {{"source": "TSM", "target": "NVDA", "relation": "SUPPLIER", "revenue_pct": 0.25, "confidence": 0.9}},
  {{"source": "NVDA", "target": "MSFT", "relation": "CUSTOMER", "revenue_pct": null, "confidence": 0.7}},
  {{"source": "AMD", "target": "NVDA", "relation": "COMPETITOR", "revenue_pct": null, "confidence": 0.8}}
]

=== EXAMPLE 2 (No relationships found) ===
[]

=== EXAMPLE 3 (TSMC with Taiwan tickers) ===
[
  {{"source": "ASML", "target": "TSM", "relation": "SUPPLIER", "revenue_pct": 0.15, "confidence": 0.8}},
  {{"source": "TSM", "target": "AAPL", "relation": "CUSTOMER", "revenue_pct": 0.25, "confidence": 0.9}},
  {{"source": "2454.TW", "target": "TSM", "relation": "SUPPLIER", "revenue_pct": 0.05, "confidence": 0.6}}
]

=== NOW ANALYZE {ticker} ===
--- BEGIN 10-K TEXT ---
{text}
--- END 10-K TEXT ---"""

    US_TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')
    TW_TICKER_PATTERN = re.compile(r'^\d{4}(\.TW|\.TWO)?$')

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

    def _parse_json_from_text(self, text: str) -> Optional[list|dict]:
        text = text.strip()
        import re as _re
        md_match = _re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, _re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()
        else:
            brace_start = text.find('{')
            bracket_start = text.find('[')
            start = -1
            if brace_start != -1 and bracket_start != -1:
                start = min(brace_start, bracket_start)
            elif brace_start != -1:
                start = brace_start
            elif bracket_start != -1:
                start = bracket_start
            if start != -1:
                end = max(text.rfind('}'), text.rfind(']'))
                if end > start:
                    text = text[start:end + 1]
        try:
            return json.loads(text)
        except (KeyError, IndexError, ValueError) as e:
            logger.warning("Failed to parse JSON: %s", e)
            return None

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
                    return self._parse_json_from_text(text)
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning("Failed to parse Gemini JSON: %s", e)
                    return None
            except requests.RequestException as e:
                is_rate_limit = isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429
                if is_rate_limit:
                    retry_after = int(e.response.headers.get("Retry-After", 5 * (attempt + 1)))
                    wait_time = min(retry_after, 30)
                    logger.warning("Gemini 429 rate limited (attempt %d), waiting %ds", attempt+1, wait_time)
                else:
                    wait_time = 2 ** attempt
                    logger.warning("Gemini API request failed (attempt %d): %s", attempt+1, e)
                time.sleep(wait_time)
                
        return None

    def build_graph(self, ticker: str, filing_text: str = "") -> SupplyChainGraph:
        """建構一家公司的供應鏈圖譜。"""
        ticker = ticker.upper()
        
        # 1. 檢查快取
        with self._lock:
            if self._cache_path.exists():
                try:
                    mtime = datetime.fromtimestamp(self._cache_path.stat().st_mtime, tz=timezone.utc)
                    age = datetime.now(timezone.utc) - mtime
                    if timedelta(0) <= age < self._cache_ttl:
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
                    except Exception:
                        pass
        
        # 如果沒有 API Key 就提早回傳空 Graph
        if not self._api_key:
            return SupplyChainGraph(center_ticker=ticker, edges=[])

        # 2. 取 10-K 全文
        if not filing_text:
            company_name = ticker
            try:
                stmt = self._edgar.get_latest_financials(ticker)
                if stmt:
                    company_name = stmt.company_name
            except Exception:
                pass
            filing_text = f"The company {company_name} ({ticker}) - no filing text available."
            
        text_excerpt = filing_text[:LLM_MAX_INPUT_CHARS]

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
                        source_filing="10-K"
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


    def _check_ollama(self) -> bool:
        """啟動時探測 Ollama 是否在線且有模型可用。"""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                available = any(m["name"] == OLLAMA_MODEL for m in models)
                if available:
                    logger.info("Ollama available with model %s", OLLAMA_MODEL)
                    return True
                else:
                    logger.warning("Ollama online but model %s not found", OLLAMA_MODEL)
                    return False
            return False
        except Exception:
            logger.warning("Ollama not available at %s", OLLAMA_BASE_URL)
            return False

    def _call_ollama(self, prompt: str) -> Optional[list]:
        """呼叫本地 Ollama Gemma4，含 Structured Output。"""
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 2048,
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            text = resp.json()["response"]
            parsed = self._parse_json_from_text(text)
            # Handle both list and dict responses
            # Ollama often wraps results: {"relationships": [...]}
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    parsed = None
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
            logger.warning("Ollama returned empty edges for prompt, falling back")
            return None
        except Exception as e:
            logger.warning("Ollama call failed: %s", e)
            return None

    def _is_expandable_ticker(self, ticker: str) -> bool:
        """判斷這個 ticker 是否可以繼續遞迴展開。"""
        ticker = ticker.upper().strip()
        if self.US_TICKER_PATTERN.match(ticker):
            return True
        if self.TW_TICKER_PATTERN.match(ticker):
            return True
        return False

    def _ticker_to_filename(self, ticker: str) -> str:
        return ticker.replace(".", "_") + ".json"

    def _read_node_cache(self, ticker: str) -> Optional[list[SupplyChainEdge]]:
        cache_file = SUPPLY_CHAIN_NODES_DIR / self._ticker_to_filename(ticker)
        if not cache_file.exists():
            return None
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(timezone.utc) - mtime
        if not (timedelta(0) <= age < self._cache_ttl):
            return None  # 過期
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return [
                SupplyChainEdge(
                    source_ticker=e["source_ticker"],
                    target_ticker=e["target_ticker"],
                    relation=SupplyChainRelation(e["relation"]),
                    revenue_pct=e.get("revenue_pct"),
                    confidence=e.get("confidence", 0.5),
                    depth=e.get("depth", 0),
                    source_filing=e.get("source_filing"),
                    last_updated=datetime.fromisoformat(e["last_updated"]) if "last_updated" in e else datetime.now(timezone.utc)
                ) for e in data.get("edges", [])
            ]
        except Exception as e:
            logger.warning("Corrupt cache for %s: %s", ticker, e)
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None

    def _write_node_cache(self, ticker: str, edges: list[SupplyChainEdge], llm_source: str):
        SUPPLY_CHAIN_NODES_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = SUPPLY_CHAIN_NODES_DIR / self._ticker_to_filename(ticker)
        tmp_file = cache_file.with_suffix(".tmp")
        data = {
            "ticker": ticker,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "llm_source": llm_source,
            "edges": [
                {
                    "source_ticker": e.source_ticker,
                    "target_ticker": e.target_ticker,
                    "relation": e.relation.value,
                    "revenue_pct": e.revenue_pct,
                    "confidence": e.confidence,
                    "depth": e.depth,
                    "source_filing": e.source_filing,
                    "last_updated": e.last_updated.isoformat()
                } for e in edges
            ]
        }
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp_file.replace(cache_file)
        except Exception as e:
            logger.warning("Failed to cache %s: %s", ticker, e)
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def _get_filing_text_safe(self, ticker: str) -> str:
        """取得公司資訊，用於供應鏈提取。如果無法取得 10-K，用財務數據和通用知識替代。"""
        company_name = ticker
        try:
            stmt = self._edgar.get_latest_financials(ticker)
            if stmt:
                company_name = stmt.company_name
                # Build a rich context from available financial data
                parts = [
                    f"Company: {stmt.company_name} (Ticker: {ticker})",
                    f"Filing Date: {stmt.filing_date}" if stmt.filing_date else "",
                    f"Revenue: ${stmt.revenue:,.0f}" if stmt.revenue else "",
                    f"Revenue YoY Growth: {stmt.revenue_yoy_growth:.1%}" if stmt.revenue_yoy_growth else "",
                    f"Gross Margin: {stmt.gross_margin:.1%}" if stmt.gross_margin else "",
                    f"Operating Margin: {stmt.operating_margin:.1%}" if stmt.operating_margin else "",
                    f"Net Income: ${stmt.net_income:,.0f}" if stmt.net_income else "",
                    f"Market Cap: ${stmt.market_cap:,.0f}" if stmt.market_cap else "",
                    f"P/E Ratio: {stmt.pe_ratio:.1f}" if stmt.pe_ratio else "",
                    f"Debt to Equity: {stmt.debt_to_equity:.2f}" if stmt.debt_to_equity else "",
                ]
                financial_context = "\n".join(p for p in parts if p)
                return (
                    f"{financial_context}\n\n"
                    f"Based on your knowledge of {company_name} ({ticker}), identify its key "
                    f"suppliers, customers, competitors, and partners. Focus on publicly traded "
                    f"companies with US or Taiwan stock tickers."
                )
        except Exception:
            pass
        return (
            f"The company {company_name} ({ticker}). "
            f"Based on your knowledge of {company_name}, identify its key "
            f"suppliers, customers, competitors, and partners. Focus on publicly traded "
            f"companies with US or Taiwan stock tickers."
        )

    def _extract_edges_gemini(self, ticker: str, filing_text: str) -> list[SupplyChainEdge]:
        text_excerpt = filing_text[:LLM_MAX_INPUT_CHARS]
        prompt = self._RECURSIVE_SUPPLY_CHAIN_PROMPT.format(ticker=ticker, text=text_excerpt)
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
                        source_filing="10-K" if filing_text != f"The company {ticker} ({ticker}) - no filing text available." else "general_knowledge"
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("Invalid edge from LLM for %s: %s", ticker, e)
        return edges

    def _extract_edges_ollama(self, ticker: str, filing_text: str) -> Optional[list[SupplyChainEdge]]:
        text_excerpt = filing_text[:OLLAMA_MAX_INPUT_CHARS]
        prompt = self._RECURSIVE_SUPPLY_CHAIN_PROMPT.format(ticker=ticker, text=text_excerpt)
        result_json = self._call_ollama(prompt)
        if result_json is None:
            return None
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
                        source_filing="10-K" if filing_text != f"The company {ticker} ({ticker}) - no filing text available." else "general_knowledge"
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("Invalid edge from LLM for %s: %s", ticker, e)
        return edges

    def _deduplicate_edges(self, edges: list[SupplyChainEdge]) -> list[SupplyChainEdge]:
        unique_edges = {}
        for edge in edges:
            # key: tuple of (min, max, relation)
            k = (min(edge.source_ticker, edge.target_ticker), max(edge.source_ticker, edge.target_ticker), edge.relation.value)
            if k not in unique_edges or edge.confidence > unique_edges[k].confidence:
                unique_edges[k] = edge
        return list(unique_edges.values())

    def build_recursive_graph(
        self,
        center: str,
        max_depth: int = 3,
        max_api_calls: int = 30,
        min_confidence: float = 0.5,
        progress_callback = None,
    ) -> RecursiveSupplyChainGraph:
        
        center = center.upper().strip()
        visited: set[str] = set()
        all_edges: list[SupplyChainEdge] = []
        node_depths: dict[str, int] = {center: 0}
        llm_sources: dict[str, str] = {}
        queue: deque[tuple[str, int]] = deque([(center, 0)])
        api_calls_used = 0

        ollama_ok = self._check_ollama()
        if not ollama_ok:
            max_api_calls = min(max_api_calls, 15)
            if progress_callback:
                progress_callback({"type": "progress", "message": "⚠️ 本地模型不可用，改用雲端 API（已自動限制呼叫次數）", "visited": 0, "queue": 1, "api_used": 0})

        while queue and api_calls_used < max_api_calls:
            ticker, depth = queue.popleft()

            if ticker in visited:
                continue
            visited.add(ticker)

            cached_edges = self._read_node_cache(ticker)
            if cached_edges is not None:
                edges = cached_edges
                llm_sources[ticker] = "cache"
                if progress_callback:
                    progress_callback({"type": "progress", "message": f"💾 {ticker} 命中快取", "visited": len(visited), "queue": len(queue), "api_used": api_calls_used})
            else:
                filing_text = self._get_filing_text_safe(ticker)

                if progress_callback:
                    model_msg = "雲端 API" if depth == 0 or not ollama_ok else "本地模型"
                    progress_callback({"type": "progress", "message": f"🔍 展開 {ticker} (Layer {depth}, {model_msg})...", "visited": len(visited), "queue": len(queue), "api_used": api_calls_used})

                if depth == 0 or not ollama_ok:
                    # Prefer Gemini for root node or when Ollama unavailable
                    edges = self._extract_edges_gemini(ticker, filing_text)
                    llm_sources[ticker] = "gemini"
                    # Fallback to Ollama if Gemini failed (e.g. 429 rate limit)
                    if not edges and ollama_ok:
                        if progress_callback:
                            progress_callback({"type": "progress", "message": f"⚠️ 雲端 API 限速，改用本地模型分析 {ticker}...", "visited": len(visited), "queue": len(queue), "api_used": api_calls_used})
                        edges = self._extract_edges_ollama(ticker, filing_text)
                        if edges:
                            llm_sources[ticker] = "ollama_fallback"
                else:
                    # Prefer Ollama for deeper nodes
                    edges = self._extract_edges_ollama(ticker, filing_text)
                    if edges is not None:
                        llm_sources[ticker] = "ollama"
                    else:
                        edges = self._extract_edges_gemini(ticker, filing_text)
                        llm_sources[ticker] = "gemini_fallback"

                edges = edges or []
                api_calls_used += 1

                self._write_node_cache(ticker, edges, llm_sources[ticker])

            edges = [e for e in edges if e.confidence >= min_confidence]
            edges = sorted(edges, key=lambda e: e.confidence, reverse=True)[:10]

            for e in edges:
                e.depth = depth
            all_edges.extend(edges)

            if progress_callback:
                progress_callback({"type": "node_done", "ticker": ticker, "depth": depth, "edges": len(edges), "llm": llm_sources[ticker]})

            if depth < max_depth:
                for edge in edges:
                    for t in [edge.source_ticker, edge.target_ticker]:
                        if t not in visited and t not in node_depths:
                            if self._is_expandable_ticker(t):
                                node_depths[t] = depth + 1
                                queue.append((t, depth + 1))
                            else:
                                pass # 保留邊但不繼續遞迴

        unique_edges = self._deduplicate_edges(all_edges)

        return RecursiveSupplyChainGraph(
            center_ticker=center,
            edges=unique_edges,
            max_depth=max_depth,
            node_depths=node_depths,
            visited_tickers=visited,
            api_calls_used=api_calls_used,
            llm_source=llm_sources,
        )
