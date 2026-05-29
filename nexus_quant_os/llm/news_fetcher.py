"""
llm/news_fetcher.py — 即時財經新聞 RSS 抓取引擎
==================================================

從多個免費 RSS 來源抓取最新財經新聞標題，供 SentimentAggregator
進行即時情緒分析。

設計原則：
    1. 多來源冗餘 — Yahoo Finance + Google News + MarketWatch
    2. 優雅降級 — 單一來源失敗時自動切換至可用來源
    3. 去重 + 時效性 — 只回傳最近 48 小時內的獨立標題（可透過 max_age_hours 調整）
    4. 無需 API Key — 全部使用公開 RSS feeds
    5. 快取 — 每 30 分鐘更新一次，避免過度請求

Author : Nexus Quant OS — Data Engineering Division
"""

from __future__ import annotations

import logging
import time
import re
from email.utils import parsedate_to_datetime
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

logger = logging.getLogger("nexus_quant_os.llm.news_fetcher")


# ═══════════════════════════════════════════════════════════════════════
# RSS 來源配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RSSSource:
    """單一 RSS 來源配置。"""
    name: str
    url: str
    priority: int = 0  # 數字越小，優先級越高


# 免費公開的財經 RSS 來源（不需要 API Key）
DEFAULT_RSS_SOURCES: list[RSSSource] = [
    RSSSource(
        name="Yahoo Finance — Market News",
        url="https://finance.yahoo.com/news/rssindex",
        priority=0,
    ),
    RSSSource(
        name="Yahoo Finance — Top Stories",
        url="https://finance.yahoo.com/rss/topstories",
        priority=1,
    ),
    RSSSource(
        name="Google News — Business",
        url="https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        priority=2,
    ),
    RSSSource(
        name="MarketWatch — Top Stories",
        url="https://feeds.marketwatch.com/marketwatch/topstories/",
        priority=3,
    ),
    RSSSource(
        name="MarketWatch — Market Pulse",
        url="https://feeds.marketwatch.com/marketwatch/marketpulse/",
        priority=4,
    ),
    RSSSource(
        name="CNBC — Top News",
        url="https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        priority=5,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# 核心抓取器
# ═══════════════════════════════════════════════════════════════════════

class NewsFetcher:
    """即時財經新聞 RSS 抓取器。

    從多個免費來源抓取最新財經標題，去重後回傳乾淨的標題列表。
    內建 30 分鐘快取，避免過度請求。

    Usage
    -----
    >>> fetcher = NewsFetcher()
    >>> headlines = fetcher.fetch_latest(max_headlines=15)
    >>> print(headlines)
    ['NVDA surges on AI demand', 'Fed holds rates steady', ...]
    """

    def __init__(
        self,
        sources: list[RSSSource] | None = None,
        cache_ttl_seconds: int = 1800,  # 30 分鐘快取
        request_timeout: float = 10.0,
        max_age_hours: int = 48,  # 只取最近 48 小時的新聞
    ) -> None:
        self.sources = sorted(
            sources or DEFAULT_RSS_SOURCES,
            key=lambda s: s.priority,
        )
        self.cache_ttl = cache_ttl_seconds
        self.timeout = request_timeout
        self.max_age = timedelta(hours=max_age_hours)

        self._cache: list[str] = []
        self._cache_time: float = 0.0

        logger.info(
            "NewsFetcher 初始化 | %d 個來源 | cache_ttl=%ds | max_age=%dh",
            len(self.sources), self.cache_ttl, max_age_hours,
        )

    def fetch_latest(self, max_headlines: int = 20) -> list[str]:
        """抓取最新財經標題。

        Parameters
        ----------
        max_headlines : int
            回傳的最大標題數量。

        Returns
        -------
        list[str]
            去重、清理過的新聞標題列表（最新在前）。
        """
        # 使用快取（若仍有效）
        now = time.monotonic()
        if self._cache and (now - self._cache_time) < self.cache_ttl:
            logger.debug("使用快取的 %d 條標題", len(self._cache))
            return self._cache[:max_headlines]

        # 從所有來源抓取
        all_headlines: list[str] = []
        seen_hashes: set[str] = set()

        for source in self.sources:
            try:
                headlines = self._fetch_from_source(source)
                for h in headlines:
                    h_hash = hashlib.md5(h.lower().encode()).hexdigest()
                    if h_hash not in seen_hashes:
                        seen_hashes.add(h_hash)
                        all_headlines.append(h)
                logger.info(
                    "✅ %s: 抓到 %d 條標題",
                    source.name, len(headlines),
                )
            except Exception as e:
                logger.warning(
                    "❌ %s 抓取失敗（優雅降級）: %s",
                    source.name, e,
                )

        if not all_headlines:
            logger.warning("所有 RSS 來源均失敗，回傳備用標題。")
            all_headlines = [
                "Markets trading mixed amid uncertainty",
                "Investors await economic data releases",
            ]

        # 更新快取
        self._cache = all_headlines
        self._cache_time = now

        result = self._diversity_sample(all_headlines, max_headlines)
        logger.info(
            "新聞抓取完成 | 共 %d 條獨立標題（多樣性抽樣取 %d 條）",
            len(all_headlines), len(result),
        )
        return result

    def _diversity_sample(self, headlines: list[str], max_headlines: int) -> list[str]:
        """多樣性抽樣：基於關鍵字分群，平均抽取不同領域的新聞以降低同質性。"""
        if len(headlines) <= max_headlines:
            return headlines
            
        category_patterns = {
            "MACRO": re.compile(r'\b(fed|rate|rates|inflation|cpi|pce|payroll|unemployment|economy|central bank|powell|gdp|recession)\b', re.IGNORECASE),
            "TECH": re.compile(r'\b(ai|apple|nvidia|microsoft|google|meta|amazon|semiconductor|chip|chips|tsmc|amd|intel|tech|cyber)\b', re.IGNORECASE),
            "MARKETS": re.compile(r'\b(stock|stocks|rally|plunge|drop|soar|s&p|nasdaq|dow|bull|bear|wall street|futures|market|markets|earnings)\b', re.IGNORECASE),
            "BONDS_COMMODITIES": re.compile(r'\b(bond|bonds|yield|yields|treasury|gold|oil|crude|energy|crypto|bitcoin|btc|ethereum)\b', re.IGNORECASE),
        }
        
        buckets = {cat: [] for cat in category_patterns.keys()}
        buckets["OTHERS"] = []
        
        for h in headlines:
            assigned = False
            for cat, pattern in category_patterns.items():
                if pattern.search(h):
                    buckets[cat].append(h)
                    assigned = True
                    break
            if not assigned:
                buckets["OTHERS"].append(h)
                
        # 平均從各個 Bucket 抽樣
        result = []
        bucket_keys = list(buckets.keys())
        
        while len(result) < max_headlines:
            added_this_round = False
            for key in bucket_keys:
                if buckets[key] and len(result) < max_headlines:
                    result.append(buckets[key].pop(0))
                    added_this_round = True
            
            # 如果所有 bucket 都抽光了，強制結束（防呆）
            if not added_this_round:
                break
                
        return result

    def _fetch_from_source(self, source: RSSSource) -> list[str]:
        """從單一 RSS 來源抓取標題。

        使用純 XML 解析（不依賴 feedparser），
        減少 Docker 映像的額外依賴。
        """
        resp = requests.get(
            source.url,
            timeout=self.timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NexusQuantOS/2.0; "
                    "+https://github.com/nexus-quant-os)"
                ),
            },
        )
        resp.raise_for_status()
        xml_text = resp.text

        # 拆分為 <item> 區塊，逐一處理（支援 max_age 過濾）
        items = re.findall(r"<item[^>]*>(.*?)</item>", xml_text, re.DOTALL)

        now_utc = datetime.now(timezone.utc)
        headlines: list[str] = []
        for item_xml in items:
            # max_age 過濾：解析 <pubDate>，跳過過時條目
            try:
                pub_match = re.search(
                    r"<pubDate[^>]*>(.*?)</pubDate>", item_xml, re.DOTALL,
                )
                if pub_match and pub_match.group(1).strip():
                    pub_dt = parsedate_to_datetime(pub_match.group(1).strip())
                    if now_utc - pub_dt > self.max_age:
                        continue  # skip stale
            except Exception:
                pass  # can't parse date, assume fresh

            # 提取 <title>
            title_match = re.search(
                r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                item_xml,
                re.DOTALL,
            )
            if not title_match:
                continue

            cleaned = self._clean_title(title_match.group(1).strip())
            if cleaned and len(cleaned) > 10:  # 過濾太短的標題
                headlines.append(cleaned)

        return headlines

    @staticmethod
    def _clean_title(title: str) -> str:
        """清理 HTML entities 和多餘空白。"""
        # 常見 HTML entities
        replacements = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&apos;": "'",
            "&#x27;": "'",
            "&#x2F;": "/",
        }
        for old, new in replacements.items():
            title = title.replace(old, new)

        # 移除 HTML 標籤殘留
        title = re.sub(r"<[^>]+>", "", title)

        # 多餘空白
        title = re.sub(r"\s+", " ", title).strip()

        return title


# ═══════════════════════════════════════════════════════════════════════
# 快速測試
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = NewsFetcher()
    headlines = fetcher.fetch_latest(max_headlines=15)

    print(f"\n{'='*72}")
    print(f"  最新財經新聞標題 ({len(headlines)} 條)")
    print(f"{'='*72}\n")
    for i, h in enumerate(headlines, 1):
        print(f"  {i:2d}. {h}")
    print()
