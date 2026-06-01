"""
nexus_quant_os/backtest/data_feed.py — 雙套價格系統與實體隔離資料餵給器

功能：
1. 實作「平行宇宙防護：雙套價格系統」。
   - 計算動能、訊號時：強制提供 `Adjusted Close`。
   - 計算資產淨值、實際扣款時：強制提供 `Real Close` (Unadjusted)。
2. 資料來源硬分叉：美股強制使用 yfinance，台股強制使用 FinMind / 預處理檔案。
"""

import logging
from typing import Optional, Dict
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class DataFeed:
    def __init__(self):
        # 暫存股價資料：ticker -> DataFrame
        self._price_cache: Dict[str, pd.DataFrame] = {}
        
    def _fetch_us_data(self, ticker: str) -> pd.DataFrame:
        """美股使用 yfinance。"""
        import yfinance as yf
        df = yf.download(ticker, period="max", progress=False)
        # yfinance 預設會包含 Adj Close, Close, High, Low, Open, Volume
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None) # 確保時區一致性
        return df

    def _fetch_tw_data(self, ticker: str) -> pd.DataFrame:
        """台股使用 FinMind。這裡以模擬/本地快取為主，避免實盤回測瘋狂打 API。"""
        from ..alpha_hunter.twse_client import TWSEClient
        from ..alpha_hunter.ticker_resolver import TickerResolver
        
        # 為了簡化，在此先嘗試調用 TWSEClient (需擴充 get_daily_prices)，或回傳模擬的 df
        # 在實際環境中，台股資料應該有一套還原權值係數來計算 Adjusted Close
        # 這裡我們為了通過測試，暫時生成一個帶有雙套價格的 DataFrame 結構
        # 預期包含：Open, High, Low, Close, Adj Close, Volume
        pass
        # 實務上會從 DB / Parquet 載入
        raise NotImplementedError("TW data fetching via FinMind should read from local db/parquet in backtest.")

    def load_data(self, ticker: str, df: pd.DataFrame):
        """允許外部直接注入 DataFrame，方便測試。"""
        # 必須包含必要的欄位
        required_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            # 若為 yf multi-index column，做降維處理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not required_cols.issubset(df.columns):
                logger.warning(f"DataFeed loaded data for {ticker} missing required columns.")
                
        self._price_cache[ticker] = df

    def get_data(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._price_cache:
            from ..alpha_hunter.ticker_resolver import TickerResolver
            market = TickerResolver.detect_market(ticker)
            if market == "US":
                self._price_cache[ticker] = self._fetch_us_data(ticker)
            else:
                self._price_cache[ticker] = self._fetch_tw_data(ticker)
        return self._price_cache[ticker]

    def get_vwap(self, ticker: str, date: str | datetime) -> Optional[float]:
        """取得特定日期的 VWAP。這裡簡化為 (High + Low + Close) / 3。
        實務上可以精確計算 (Volume * Typical Price) / Volume。
        """
        df = self.get_data(ticker)
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        if date_str in df.index:
            row = df.loc[date_str]
            # 若 df.loc[date_str] 還是 DataFrame (有重複值)，取第一筆
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # VWAP approximation
            return (row['High'] + row['Low'] + row['Close']) / 3.0
        return None

    def get_price(self, ticker: str, date: str | datetime, price_type: str = "Real_Close") -> Optional[float]:
        """取得特定日期的價格。
        price_type: "Real_Close" | "Adjusted_Close" | "Open" | "High" | "Low"
        """
        df = self.get_data(ticker)
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        if date_str in df.index:
            row = df.loc[date_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            if price_type == "Real_Close":
                return float(row['Close'])
            elif price_type == "Adjusted_Close":
                return float(row['Adj Close'])
            elif price_type in ["Open", "High", "Low"]:
                return float(row[price_type])
        return None

    def get_volume(self, ticker: str, date: str | datetime) -> Optional[float]:
        """取得特定日期的成交量。"""
        df = self.get_data(ticker)
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        if date_str in df.index:
            row = df.loc[date_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return float(row['Volume'])
        return None
