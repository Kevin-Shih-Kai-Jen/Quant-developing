"""
alpha_hunter/tw_macro.py — 台灣總經指標管線

提供 6 維台灣總經特徵，用於訓練 TW_HMM 風控模型。
每個指標都必須遵守 Point-in-Time (PiT) 原則：
    - 資料的 actionable_date = release_date（公告日），不是統計期間結束日
    - 例：4 月份 CPI 在 5/5 公告，actionable_date = 5/5

資料來源：
    1. 景氣對策信號 → 國發會 OpenAPI
    2. 台灣 CPI → 主計總處 / FinMind「TaiwanConsumerPriceIndex」
    3. 台灣失業率 → FinMind「TaiwanUnemploymentRate」
    4. 央行重貼現率 → FinMind「TaiwanInterestRate」
    5. 台債殖利率利差 (10Y-2Y) → FinMind「TaiwanGovernmentBondYield」
    6. 工業生產指數 → FinMind「TaiwanIndustrialProductionIndex」

⚠️ 若 FinMind API 無法取得，全部 fallback 回傳空 DataFrame，不得 raise。
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


class TWMacroClient:
    """台灣總經指標客戶端。"""

    def __init__(self, api_token: Optional[str] = None):
        self._token = api_token
        self._session = requests.Session()

    def _fetch(self, dataset: str, start_date: str) -> pd.DataFrame:
        """向 FinMind 請求總經資料。"""
        params = {"dataset": dataset, "start_date": start_date}
        if self._token:
            params["token"] = self._token
        try:
            resp = self._session.get(FINMIND_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("msg") == "success" and data.get("data"):
                return pd.DataFrame(data["data"])
        except Exception as e:
            logger.warning("FinMind macro fetch failed [%s]: %s", dataset, e)
        return pd.DataFrame()

    # ── 指標 1：景氣對策信號 ──
    def get_business_indicator(self) -> pd.DataFrame:
        """取得景氣對策信號分數與燈號。

        回傳 DataFrame columns: [date, score, signal, release_date]
        signal: 紅燈=過熱, 黃紅燈=趨熱, 綠燈=穩定, 黃藍燈=趨冷, 藍燈=衰退
        score: 9(最冷)~45(最熱)
        """
        df = self._fetch("TaiwanBusinessIndicator", "2020-01-01")
        if df.empty:
            return pd.DataFrame(columns=["date", "score", "signal", "release_date"])
        df["date"] = pd.to_datetime(df["date"])
        # FinMind 欄位名稱可能是 score / TaiwanBusinessIndicator
        # 做安全的欄位重命名
        col_map = {}
        for col in df.columns:
            if "score" in col.lower() or "indicator" in col.lower():
                col_map[col] = "score"
            if "signal" in col.lower() or "light" in col.lower():
                col_map[col] = "signal"
        if col_map:
            df = df.rename(columns=col_map)
        if "release_date" not in df.columns:
            # PiT: 景氣燈號通常延遲 1 個月公告
            df["release_date"] = df["date"] + pd.DateOffset(months=1, days=27)
        return df

    # ── 指標 2：台灣 CPI ──
    def get_cpi(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """取得台灣消費者物價指數 (CPI) 年增率。"""
        df = self._fetch("TaiwanConsumerPriceIndex", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "cpi_yoy"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 指標 3：台灣失業率 ──
    def get_unemployment(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """取得台灣失業率。"""
        df = self._fetch("TaiwanUnemploymentRate", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "unemployment_rate"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 指標 4：央行重貼現率 ──
    def get_interest_rate(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """取得台灣央行重貼現率。"""
        df = self._fetch("TaiwanInterestRate", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "rediscount_rate"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 指標 5：台債殖利率 ──
    def get_bond_yield(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """取得台灣公債殖利率，可計算 10Y-2Y 利差。"""
        df = self._fetch("TaiwanGovernmentBondYield", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "yield_10y", "yield_2y", "spread"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 指標 6：工業生產指數 ──
    def get_industrial_production(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """取得台灣工業生產指數。"""
        df = self._fetch("TaiwanIndustrialProductionIndex", start_date)
        if df.empty:
            return pd.DataFrame(columns=["date", "production_index"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── 綜合：一次取得全部 6 維特徵 ──
    def get_all_features(self, start_date: str = "2020-01-01") -> pd.DataFrame:
        """合併全部 6 項指標為單一 DataFrame。

        用途：訓練 TW_HMM 時直接呼叫此方法。
        回傳 columns: [date, ndc_score, cpi_yoy, unemployment,
                        rediscount_rate, yield_spread, production_idx]
        """
        dfs = {
            "ndc": self.get_business_indicator(),
            "cpi": self.get_cpi(start_date),
            "unemp": self.get_unemployment(start_date),
            "rate": self.get_interest_rate(start_date),
            "bond": self.get_bond_yield(start_date),
            "prod": self.get_industrial_production(start_date),
        }
        # 以月為單位 merge（每個指標都是月頻率）
        # 若任何指標取不到，整體仍可運作（用 NaN 填充）
        result = pd.DataFrame({"date": pd.date_range(start_date, periods=60, freq="MS")})
        for name, df in dfs.items():
            if not df.empty and "date" in df.columns:
                result = pd.merge_asof(
                    result.sort_values("date"),
                    df.sort_values("date"),
                    on="date",
                    direction="backward",
                )
        return result
