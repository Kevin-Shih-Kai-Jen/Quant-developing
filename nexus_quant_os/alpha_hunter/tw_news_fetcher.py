"""
alpha_hunter/tw_news_fetcher.py — 台股新聞抓取器

抓取台股相關新聞供 LLM 進行情緒分析。
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TWNewsFetcher:
    """台股新聞抓取器"""

    # ── Edge Case #6: 台股紅漲綠跌 LLM 在地化字典 ──
    # 當 AI Analyst 分析台股新聞時，必須將此字典注入 System Prompt，
    # 否則 LLM 會用美股邏輯把「紅色=危險=跌」解讀反了。
    TW_MARKET_CONTEXT = """
You are analyzing Taiwan stock market (台股) news. Apply these CRITICAL rules:

COLOR SEMANTICS (opposite of US/EU):
- RED (紅) = UP / Bullish / 漲停. "亮紅燈" in economic context = overheating (bearish).
- GREEN (綠) = DOWN / Bearish / 跌停.

INSTITUTIONAL FLOW TERMS:
- 外資買超 / 外資連買 = Foreign institutional NET BUY → Bullish momentum
- 三大法人買超 = All three institutional investors net buy → Strong bullish
- 投信買超 = Domestic mutual fund net buy → Mid-term bullish
- 主力出貨 = Market maker SELLING / distributing → Bearish (NOT shipping goods)
- 土洋對作 = Domestic vs foreign institutions trading opposite directions

TAIWAN SLANG:
- 吃筍 = Stop-loss / losing money → Bearish (NOT eating bamboo)
- 長老 / 八大行庫 = Government-linked funds → Smart money signal
- 軋空 = Short squeeze → Extreme bullish short-term
- 殺融資 = Forced margin call selling → Bearish cascade
- 庫存去化 = Inventory digestion → Bullish (cycle bottoming out)
- 法說會 = Investor conference / earnings call
"""

    @classmethod
    def get_news(cls, ticker: str, limit: int = 10) -> list[dict]:
        """取得近期新聞。

        資料來源優先順序：
            1. FinMind TaiwanStockNews API
            2. Yahoo Finance RSS (fallback)
        """
        from .ticker_resolver import TickerResolver

        stock_id = TickerResolver.to_canonical(ticker).replace(".TW", "")
        news_items = []

        # ── 方法 1：FinMind 新聞 API ──
        try:
            from datetime import datetime, timedelta
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {
                "dataset": "TaiwanStockNews",
                "data_id": stock_id,
                "start_date": start,
            }
            import requests
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("msg") == "success" and data.get("data"):
                for item in data["data"][:limit]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "content": item.get("description", item.get("content", "")),
                        "date": item.get("date", ""),
                        "source": item.get("source", "FinMind"),
                    })
        except Exception as e:
            logger.warning("FinMind news fetch failed for %s: %s", ticker, e)

        # ── 方法 2：Yahoo Finance RSS (fallback) ──
        if not news_items:
            try:
                yf_ticker = TickerResolver.to_yfinance(ticker)
                rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yf_ticker}&region=TW&lang=zh-TW"
                import requests
                resp = requests.get(rss_url, timeout=10)
                resp.raise_for_status()

                from xml.etree import ElementTree
                root = ElementTree.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    news_items.append({
                        "title": item.findtext("title", ""),
                        "content": item.findtext("description", ""),
                        "date": item.findtext("pubDate", ""),
                        "source": "Yahoo Finance",
                    })
            except Exception as e:
                logger.warning("Yahoo RSS fallback failed for %s: %s", ticker, e)

        return cls.filter_news(news_items)

    @classmethod
    def filter_news(cls, news_items: list[dict]) -> list[dict]:
        """過濾與標記新聞意圖 (Edge Case #43: 現金減資 vs 虧損減資)。"""
        filtered = []
        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", "")
            
            # 檢查減資意圖
            text_to_check = title + " " + content
            if "減資" in text_to_check:
                if "彌補虧損" in text_to_check:
                    item["risk_label"] = "RISK_CAPITAL_REDUCTION_DEFICIT"
                    item["risk_level"] = "CRITICAL"
                    logger.warning("Edge Case #43: 偵測到虧損減資意圖: %s", title)
                else:
                    item["risk_label"] = "CASH_CAPITAL_REDUCTION"
                    item["risk_level"] = "INFO"
                    
            filtered.append(item)
        return filtered

    @classmethod
    def build_llm_prompt(cls, ticker: str, news_items: list[dict]) -> str:
        """建構台股專用的 LLM 分析 Prompt（含在地化字典）。

        用途：ai_analyst.py 分析台股時，用此方法取代通用 Prompt。
        """
        from .ticker_resolver import TickerResolver
        market = TickerResolver.detect_market(ticker)

        news_text = "\n".join(
            f"- [{item.get('date', '')}] {item.get('title', '')}: {item.get('content', '')[:200]}"
            for item in news_items
        )

        if market == "TW":
            return f"{cls.TW_MARKET_CONTEXT}\n\n--- RECENT NEWS ---\n{news_text}"
        else:
            return f"--- RECENT NEWS ---\n{news_text}"
