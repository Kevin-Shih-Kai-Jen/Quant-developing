"""
alpha_hunter/mops_scraper.py — 公開資訊觀測站 (MOPS) 爬蟲

負責爬取與解析重大訊息及法說會資訊。
防禦 Edge Cases:
    - B2: 極易遭到 IP 封鎖。實作 Exponential Backoff 與隨機延遲。
"""

import logging
import random
import time
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

import requests

logger = logging.getLogger(__name__)


class MOPSScraper:
    """公開資訊觀測站爬蟲"""

    BASE_URL = "https://mops.twse.com.tw/mops/web/ajax_t05st01"

    def __init__(self, max_retries: int = 3):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml",
            }
        )
        self._max_retries = max_retries

    def get_company_announcements(
        self, stock_id: str, year: int, month: int
    ) -> list[dict]:
        """爬取重大訊息。

        為避免頻繁請求被封鎖，實作 Exponential Backoff。
        """
        payload = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "keyword4": "",
            "code1": "",
            "TYPEK2": "",
            "checkbtn": "",
            "queryName": "co_id",
            "inpuType": "co_id",
            "TYPEK": "all",
            "co_id": stock_id,
            "year": str(year - 1911),  # 民國年
            "month": str(month).zfill(2),
        }

        for attempt in range(self._max_retries):
            # 隨機延遲 1~3 秒
            time.sleep(random.uniform(1.0, 3.0))

            try:
                resp = self._session.post(self.BASE_URL, data=payload, timeout=15)
                resp.raise_for_status()

                # ── 編碼偵測 ──
                # MOPS 回傳的內容可能是 Big5 或 UTF-8
                content_type = resp.headers.get("Content-Type", "")
                if "big5" in content_type.lower():
                    resp.encoding = "big5"
                else:
                    resp.encoding = "utf-8"

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")

                # MOPS 重大訊息的 HTML 結構：<table class="hasBorder"> 中的 <tr>
                results = []
                table = soup.find("table", class_="hasBorder")
                if table is None:
                    return []

                rows = table.find_all("tr")[1:]  # 跳過 header row
                for tr in rows:
                    cols = tr.find_all("td")
                    if len(cols) < 4:
                        continue
                    results.append({
                        "date": cols[0].get_text(strip=True),
                        "time": cols[1].get_text(strip=True),
                        "subject": cols[2].get_text(strip=True),
                        "content": cols[3].get_text(strip=True)[:500],  # 截斷避免過長
                    })
                return results
            except requests.RequestException as e:
                logger.warning("MOPS request failed (attempt %d): %s", attempt + 1, e)
                # Exponential backoff
                time.sleep(2**attempt)

        return []

    def get_investor_conference_text(self, stock_id: str) -> Optional[str]:
        """爬取最新一場法說會的摘要文字。

        資料來源：MOPS 法說會專區
        URL: https://mops.twse.com.tw/mops/web/t100sb12

        Returns:
            法說會摘要文字（純文字），若無資料回傳 None
        """
        url = "https://mops.twse.com.tw/mops/web/ajax_t100sb12"
        roc_year = datetime.now().year - 1911

        payload = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "TYPEK": "all",
            "co_id": stock_id,
            "year": str(roc_year),
        }

        time.sleep(random.uniform(2.0, 5.0))  # 反爬蟲延遲

        try:
            resp = self._session.post(url, data=payload, timeout=15)
            resp.raise_for_status()

            # 編碼偵測
            if "big5" in resp.headers.get("Content-Type", "").lower():
                resp.encoding = "big5"
            else:
                resp.encoding = "utf-8"

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")

            # 提取法說會簡報文字
            text_blocks = soup.find_all("td")
            full_text = " ".join(td.get_text(strip=True) for td in text_blocks)

            if len(full_text) < 50:
                logger.info("法說會文字過短（%d 字），可能是掃描 PDF，無法提取。", len(full_text))
                return None

            return full_text[:8000]  # 截斷至 LLM 輸入上限

        except Exception as e:
            logger.warning("MOPS 法說會爬取失敗 [%s]: %s", stock_id, e)
            return None

    def get_cb_balance_and_premium(self, stock_id: str) -> tuple[float, bool]:
        """Edge Case #44: 可轉債 (CB) 養套殺防禦。

        資料來源：FinMind「TaiwanStockConvertibleBond」
        Returns:
            (premium_ratio, balance_dropped):
                premium_ratio: 溢價率（小數，如 0.25 = 25%）
                balance_dropped: 近 30 天餘額是否下降超過 20%
        """
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            params = {
                "dataset": "TaiwanStockConvertibleBond",
                "data_id": stock_id,
                "start_date": start,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("msg") != "success" or not data.get("data"):
                return 0.0, False

            df = pd.DataFrame(data["data"])
            if df.empty or "ConvertibleBondBalance" not in df.columns:
                return 0.0, False

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # 計算溢價率
            latest = df.iloc[-1]
            premium = float(latest.get("PremiumRate", 0)) / 100  # 百分比→小數

            # 計算餘額是否急降
            if len(df) >= 2:
                balance_30d_ago = df.iloc[max(0, len(df)-30)]["ConvertibleBondBalance"]
                balance_now = latest["ConvertibleBondBalance"]
                if balance_30d_ago > 0:
                    drop_pct = (balance_30d_ago - balance_now) / balance_30d_ago
                    balance_dropped = drop_pct > 0.20
                else:
                    balance_dropped = False
            else:
                balance_dropped = False

            return premium, balance_dropped

        except Exception as e:
            logger.warning("CB data fetch failed for %s: %s", stock_id, e)
            return 0.0, False
