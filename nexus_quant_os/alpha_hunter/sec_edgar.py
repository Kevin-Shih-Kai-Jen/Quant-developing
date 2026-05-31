"""
alpha_hunter/sec_edgar.py — SEC EDGAR XBRL 資料抓取器

限制：
    - SEC 要求 User-Agent header 包含聯絡資訊
    - Rate limit: 10 req/sec（用 _enforce_rate_limit 控制）
    - 所有 CIK 必須是 10 位數字符串（左補零）

錯誤處理：
    - HTTP 404 → 回傳 None（公司不存在或無 XBRL）
    - HTTP 429 → sleep 60 秒後重試
    - 解析失敗 → log warning + 回傳 None
    - 不要 raise exception（呼叫者不需要 try/except）
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

from .models import FinancialStatement
from ._constants import (
    EDGAR_BASE_URL, EDGAR_COMPANY_URL, EDGAR_XBRL_URL,
    EDGAR_USER_AGENT, EDGAR_REQUEST_INTERVAL,
    EDGAR_CACHE_DIR, EDGAR_CACHE_TTL_DAYS,
)

logger = logging.getLogger(__name__)


class SECEdgarClient:
    """SEC EDGAR XBRL API 客戶端。

    用法：
        client = SECEdgarClient()
        stmt = client.get_latest_financials("NVDA")
        if stmt is not None:
            print(stmt.revenue)
    """

    def __init__(
        self,
        user_agent: str = EDGAR_USER_AGENT,
        cache_dir: Path = EDGAR_CACHE_DIR,
        cache_ttl_days: int = EDGAR_CACHE_TTL_DAYS,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            user_agent: SEC 要求的 User-Agent 字串。
                        格式: "AppName contact@email.com"
            cache_dir:  XBRL JSON 快取目錄。
            cache_ttl_days: 快取有效天數。
            timeout:    HTTP 請求超時秒數。
            max_retries: 失敗重試次數。
        """
        self._user_agent = user_agent
        self._cache_dir = cache_dir
        self._cache_ttl = timedelta(days=cache_ttl_days)
        self._timeout = timeout
        self._max_retries = max_retries
        self._last_request_time: float = 0.0
        self._rate_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self._user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })

        # 建立快取目錄
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入 ticker to CIK 映射表
        self._ticker_to_cik: dict[str, str] = {}
        self._load_tickers()

    def _load_tickers(self) -> None:
        """從 SEC 載入 ticker 對應 CIK 映射表並快取"""
        cache_file = self._cache_dir / "company_tickers.json"
        
        # 檢查快取
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            if datetime.now(timezone.utc) - mtime < timedelta(days=7):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data.values():
                        self._ticker_to_cik[item["ticker"].upper()] = str(item["cik_str"]).zfill(10)
                    return
                except Exception as e:
                    logger.warning("Failed to load tickers cache: %s", e)
        
        # 從網路下載
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get_json(url)
        if data:
            for item in data.values():
                self._ticker_to_cik[item["ticker"].upper()] = str(item["cik_str"]).zfill(10)
            # 寫入快取
            tmp_file = cache_file.with_suffix(".tmp")
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                tmp_file.replace(cache_file)
            except Exception as e:
                logger.warning("Failed to save tickers cache: %s", e)
                if tmp_file.exists():
                    tmp_file.unlink()

    # ── Public API ────────────────────────────────────────

    def resolve_cik(self, ticker: str) -> Optional[str]:
        """將 ticker 轉為 10 位 CIK 字串。"""
        ticker = ticker.upper()
        if ticker in self._ticker_to_cik:
            return self._ticker_to_cik[ticker]
        
        # Fallback to EFTS API search
        url = f'https://efts.sec.gov/LATEST/search-index?q="{ticker}"&forms=10-K'
        data = self._get_json(url)
        if data and "hits" in data and "hits" in data["hits"]:
            hits = data["hits"]["hits"]
            if hits:
                entity_id = hits[0].get("_id", "")
                if ":" in entity_id:
                    cik = entity_id.split(":")[0].zfill(10)
                    self._ticker_to_cik[ticker] = cik
                    return cik
        
        logger.warning("CIK not found for ticker: %s", ticker)
        return None

    def get_latest_financials(self, ticker: str) -> Optional[FinancialStatement]:
        """取得一家公司最新一季的財務報表。"""
        history = self.get_financials_history(ticker, n_quarters=1)
        if history:
            return history[0]
        return None

    def get_financials_history(
        self,
        ticker: str,
        n_quarters: int = 8,
    ) -> list[FinancialStatement]:
        """取得最近 N 季的財報歷史。"""
        ticker = ticker.upper()
        cik = self.resolve_cik(ticker)
        if not cik:
            return []

        cache_file = self._cache_dir / f"{cik}.json"
        data = None
        
        # 檢查快取
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            age = datetime.now(timezone.utc) - mtime
            if timedelta(0) <= age < self._cache_ttl:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning("Failed to read cache for %s: %s", ticker, e)
        
        # 下載
        if data is None:
            url = EDGAR_XBRL_URL.format(cik=cik)
            data = self._get_json(url)
            if data is None:
                return []
            
            # 寫入快取
            tmp_file = cache_file.with_suffix(".tmp")
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                tmp_file.replace(cache_file)
            except Exception as e:
                logger.warning("Failed to write cache for %s: %s", ticker, e)
                if tmp_file.exists():
                    tmp_file.unlink()

        try:
            statements = self._parse_xbrl_facts(data, cik, ticker)
            if not statements:
                logger.warning("No valid FinancialStatement could be parsed for %s", ticker)
                return []
            return statements[:n_quarters]
        except Exception as e:
            logger.exception("Error parsing XBRL facts for %s: %s", ticker, e)
            return []

    def get_filing_text(self, ticker: str, filing_type: str = "10-K",
                        sections: list[str] | None = None) -> dict[str, str]:
        """
        取得一家公司最新 Filing 的指定 section 文本。
        """
        import re

        cik = self.resolve_cik(ticker)
        if not cik:
            return {}

        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        subs_data = self._get_json(submissions_url)
        if not subs_data:
            return {}

        recent = subs_data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        target_idx = None
        for i, form in enumerate(forms):
            if form == filing_type:
                target_idx = i
                break

        if target_idx is None:
            logger.warning("No %s found for %s", filing_type, ticker)
            return {}

        accession = accessions[target_idx].replace("-", "")
        primary_doc = primary_docs[target_idx]

        cik_int = cik.lstrip("0") or "0"
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary_doc}"

        cache_file = self._cache_dir / f"{cik}_{filing_type}.txt"
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            age = datetime.now(timezone.utc) - mtime
            if timedelta(0) <= age < self._cache_ttl:
                try:
                    text = cache_file.read_text(encoding="utf-8")
                    return self._extract_sections(text, sections)
                except Exception:
                    pass  # 快取損壞，重新下載

        self._enforce_rate_limit()
        try:
            resp = self._session.get(filing_url, timeout=self._timeout)
            resp.raise_for_status()
            html = resp.text

            MAX_HTML_CHARS = 2_000_000
            if len(html) > MAX_HTML_CHARS:
                logger.warning("Filing HTML for %s is %d chars, truncating to %d",
                               ticker, len(html), MAX_HTML_CHARS)
                html = html[:MAX_HTML_CHARS]

        except requests.RequestException as e:
            logger.warning("Filing download failed for %s: %s", ticker, e)
            return {}

        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        import uuid
        tmp = cache_file.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(cache_file)
        except Exception as e:
            logger.warning("Failed to cache filing text for %s: %s", ticker, e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        return self._extract_sections(text, sections)

    def _extract_sections(self, full_text: str,
                          sections: list[str] | None = None) -> dict[str, str]:
        """從 10-K 純文字中，用關鍵字模糊匹配抽取指定章節。"""
        from ._constants import LLM_MAX_INPUT_CHARS
        import re
        
        if sections is None:
            sections = ["mda", "risk"]

        safe_text = full_text.lower()
        safe_text = safe_text.replace("\u2018", "").replace("\u2019", "")
        safe_text = safe_text.replace("\u201c", "").replace("\u201d", "")
        safe_text = safe_text.replace("'", "").replace("'", "")
        safe_text = re.sub(r'\s+', ' ', safe_text)

        TOC_SKIP = 5000 if len(safe_text) > 10000 else 0

        SECTION_MARKERS = {
            "mda": [
                "management's discussion and analysis",
                "management discussion and analysis",
                "md&a",
            ],
            "risk": [
                "risk factors",
            ],
            "business": [
                "description of business",
                "business overview",
            ],
        }

        MAX_SECTION_CHARS = LLM_MAX_INPUT_CHARS
        result = {}

        for section_key in sections:
            markers = SECTION_MARKERS.get(section_key, [])
            clean_markers = [m.replace("'", "").replace("'", "") for m in markers]
            best_start = -1

            for marker in clean_markers:
                idx = safe_text.find(marker, TOC_SKIP)
                if idx != -1:
                    best_start = idx
                    break

            if best_start == -1:
                for marker in clean_markers:
                    idx = safe_text.find(marker)
                    if idx != -1:
                        best_start = idx
                        break

            if best_start == -1:
                result[section_key] = ""
                continue

            chunk = full_text[best_start : best_start + MAX_SECTION_CHARS]
            result[section_key] = chunk

        return result

    # ── Private Methods ───────────────────────────────────

    def _enforce_rate_limit(self) -> None:
        """確保相鄰請求間隔不低於 EDGAR_REQUEST_INTERVAL。"""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < EDGAR_REQUEST_INTERVAL:
                time.sleep(EDGAR_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.monotonic()

    def _get_json(self, url: str) -> Optional[dict]:
        """帶有 rate limit + retry 的 GET 請求。"""
        for attempt in range(self._max_retries):
            self._enforce_rate_limit()
            try:
                resp = self._session.get(url, timeout=self._timeout)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    logger.warning("SEC EDGAR rate limit hit, sleeping 60s")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning("EDGAR request failed (attempt %d): %s", attempt+1, e)
                time.sleep(2 ** attempt)  # 指數退避
                
        logger.error("EDGAR request failed after %d retries: %s", self._max_retries, url)
        return None

    def _parse_xbrl_facts(
        self,
        data: dict,
        cik: str,
        ticker: str,
    ) -> list[FinancialStatement]:
        """解析 XBRL companyfacts JSON 為 FinancialStatement 列表。"""
        if "facts" not in data:
            return []
            
        facts = data["facts"]
        gaap = facts.get("us-gaap", {})
        
        # 收集所有的 periods
        periods: dict[str, dict] = {}  # key: end_date (e.g. "2024-01-28"), value: fields
        
        def extract_fact(tag: str, target_field: str):
            if tag not in gaap:
                return
            units = gaap[tag].get("units", {})
            # Bug #2/#7 fix: support USD, shares, USD/shares, pure
            items = []
            for unit_key in ("USD", "shares", "USD/shares", "pure"):
                if unit_key in units:
                    items = units[unit_key]
                    break
            if not items:
                return

            for item in items:
                form = item.get("form")
                if form not in ("10-K", "10-Q", "20-F", "6-K"):
                    continue
                end_date = item.get("end")
                if not end_date:
                    continue

                try:
                    val = float(item.get("val", 0))
                except (ValueError, TypeError):
                    continue

                start_date = item.get("start")  # None for instant (balance sheet)

                if end_date not in periods:
                    periods[end_date] = {
                        "filing_date": item.get("filed"),
                        "fy": item.get("fy"),
                        "fp": item.get("fp", ""),
                        "form": form,
                    }

                # Bug #1 fix: For period-based items (income/cashflow with start+end),
                # keep only the shortest duration (= single quarter, not YTD).
                # For instant items (balance sheet, no start), always overwrite.
                if start_date is not None:
                    existing_start = periods[end_date].get(f"_start_{target_field}")
                    if existing_start is not None:
                        # Compare durations: shorter = more granular (quarterly)
                        try:
                            new_duration = (datetime.strptime(end_date, "%Y-%m-%d") -
                                           datetime.strptime(start_date, "%Y-%m-%d")).days
                            old_duration = (datetime.strptime(end_date, "%Y-%m-%d") -
                                           datetime.strptime(existing_start, "%Y-%m-%d")).days
                            if new_duration >= old_duration:
                                continue  # Existing value is shorter (more granular), keep it
                        except ValueError:
                            continue
                    periods[end_date][f"_start_{target_field}"] = start_date

                periods[end_date][target_field] = val

        # 映射表
        extract_fact("Revenues", "revenue")
        extract_fact("SalesRevenueNet", "revenue")
        extract_fact("RevenueFromContractWithCustomerExcludingAssessedTax", "revenue")
        extract_fact("GrossProfit", "gross_profit")
        extract_fact("OperatingIncomeLoss", "operating_income")
        extract_fact("NetIncomeLoss", "net_income")
        extract_fact("EarningsPerShareDiluted", "eps_diluted")
        extract_fact("Assets", "total_assets")
        extract_fact("Liabilities", "total_liabilities")
        extract_fact("StockholdersEquity", "total_equity")
        extract_fact("CashAndCashEquivalentsAtCarryingValue", "cash_and_equivalents")
        extract_fact("LongTermDebt", "long_term_debt")
        extract_fact("ShortTermBorrowings", "short_term_debt")
        extract_fact("NetCashProvidedByOperatingActivities", "operating_cash_flow")
        extract_fact("PaymentsToAcquirePropertyPlantAndEquipment", "capex")

        # 組合 Statements
        statements = []
        for end_date, fields in periods.items():
            try:
                period_end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                filing_date_str = fields.get("filing_date")
                if filing_date_str:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                else:
                    filing_date = period_end

                fy = int(fields.get("fy", period_end.year))
                fp = str(fields.get("fp", ""))
                quarter = 4
                if fp.startswith("Q"):
                    try:
                        quarter = int(fp[1])
                    except ValueError:
                        pass
                elif fp == "FY":
                    # This is an annual (10-K full-year) period.
                    # Check if the revenue duration is > 100 days → annual, skip it
                    rev_start = fields.get("_start_revenue")
                    if rev_start is not None:
                        try:
                            dur = (period_end - datetime.strptime(rev_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days
                            if dur > 100:
                                continue  # Skip annual periods — we only want quarterly
                        except ValueError:
                            pass
                
                # 計算欄位
                revenue = fields.get("revenue")
                gross_profit = fields.get("gross_profit")
                operating_income = fields.get("operating_income")
                total_equity = fields.get("total_equity")
                
                gross_margin = None
                if gross_profit is not None and revenue is not None and abs(revenue) > 1e-6:
                    gross_margin = gross_profit / revenue
                    
                operating_margin = None
                if operating_income is not None and revenue is not None and abs(revenue) > 1e-6:
                    operating_margin = operating_income / revenue
                
                long_term_debt = fields.get("long_term_debt")
                short_term_debt = fields.get("short_term_debt")
                total_debt = None
                if long_term_debt is not None or short_term_debt is not None:
                    total_debt = (long_term_debt or 0.0) + (short_term_debt or 0.0)
                    
                debt_to_equity = None
                if total_debt is not None and total_equity is not None and total_equity > 1e-6:
                    debt_to_equity = total_debt / total_equity
                    
                operating_cash_flow = fields.get("operating_cash_flow")
                capex = fields.get("capex")
                fcf = None
                if operating_cash_flow is not None and capex is not None:
                    fcf = operating_cash_flow - abs(capex)

                stmt = FinancialStatement(
                    ticker=ticker,
                    cik=cik,
                    company_name=data.get("entityName", ticker),
                    filing_date=filing_date,
                    period_end=period_end,
                    fiscal_year=fy,
                    fiscal_quarter=quarter,
                    revenue=revenue,
                    gross_profit=gross_profit,
                    gross_margin=gross_margin,
                    operating_income=operating_income,
                    operating_margin=operating_margin,
                    net_income=fields.get("net_income"),
                    eps_diluted=fields.get("eps_diluted"),
                    total_assets=fields.get("total_assets"),
                    total_liabilities=fields.get("total_liabilities"),
                    total_equity=total_equity,
                    cash_and_equivalents=fields.get("cash_and_equivalents"),
                    total_debt=total_debt,
                    debt_to_equity=debt_to_equity,
                    operating_cash_flow=operating_cash_flow,
                    free_cash_flow=fcf,
                    capex=capex,
                )
                statements.append(stmt)
            except Exception as e:
                logger.warning("Failed to parse period %s for %s: %s", end_date, ticker, e)
                continue

        # 按期間結束日倒序排列
        statements.sort(key=lambda s: s.period_end, reverse=True)
        
        # 計算 YoY growth
        for i, stmt in enumerate(statements):
            if stmt.revenue is None:
                continue
            # 尋找前一年的同期
            target_year = stmt.fiscal_year - 1
            target_quarter = stmt.fiscal_quarter
            for prev_stmt in statements[i+1:]:
                if prev_stmt.fiscal_year == target_year and prev_stmt.fiscal_quarter == target_quarter:
                    if prev_stmt.revenue is not None and abs(prev_stmt.revenue) > 1e-6:
                        stmt.revenue_yoy_growth = (stmt.revenue - prev_stmt.revenue) / abs(prev_stmt.revenue)
                    break

        return statements
