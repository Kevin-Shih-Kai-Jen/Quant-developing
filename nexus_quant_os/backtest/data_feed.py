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
        
    def _fetch_tw_data(self, ticker: str) -> pd.DataFrame:
        """台股使用 FinMind (此處改為讀取本地 Parquet)。"""
        return self._fetch_from_parquet(ticker)

    def _fetch_us_data(self, ticker: str) -> pd.DataFrame:
        """美股使用 yfinance (此處亦改為讀取本地 Parquet)。"""
        return self._fetch_from_parquet(ticker)

    def _fetch_from_parquet(self, ticker: str) -> pd.DataFrame:
        import os
        from pathlib import Path
        import polars as pl
        base_dir = Path(__file__).resolve().parent.parent
        file_path = base_dir / "data_lake" / "parquet" / f"{ticker}.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        
        
        # 1. 延遲加載 (Lazy Load)
        lf = pl.scan_parquet(file_path)
        
        # 2. 收集並轉為 Pandas DataFrame
        df = lf.collect().to_pandas()
        
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 在這裡我們先不直接 reindex，而是讓 get_data 或 get_price 處理停牌
        return df

    def load_data(self, ticker: str, df: pd.DataFrame):
        """允許外部直接注入 DataFrame，方便測試。"""
        # 必須包含必要的欄位
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'} # Adj Close is now optional due to Phase 14
        if not required_cols.issubset(df.columns):
            # 若為 yf multi-index column，做降維處理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if not required_cols.issubset(df.columns):
                raise ValueError(f"DataFeed loaded data for {ticker} is missing required columns: {required_cols - set(df.columns)}")
                
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

    def _get_row(self, ticker: str, date_str: str) -> Optional[pd.Series]:
        df = self.get_data(ticker)
        if df.empty:
            return None
            
        # 停牌幽靈防禦：若當天無報價 (停牌/下市)，強制沿用前一個交易日的 Close，但 Volume 設為 0
        if date_str in df.index:
            row = df.loc[date_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row
        else:
            # 尋找前一個有效交易日
            past_dates = df.index[df.index <= date_str]
            if len(past_dates) > 0:
                last_date = past_dates[-1]
                row = df.loc[last_date].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0].copy()
                # 強制將 Volume 設為 0
                row['Volume'] = 0.0
                return row
        return None

    def get_vwap(self, ticker: str, date: str | datetime) -> Optional[float]:
        """取得特定日期的 VWAP。"""
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        row = self._get_row(ticker, date_str)
        if row is not None:
            return (row['High'] + row['Low'] + row['Close']) / 3.0
        return None

    def get_price(self, ticker: str, date: str | datetime, price_type: str = "Real_Close") -> Optional[float]:
        """取得特定日期的價格。"""
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        row = self._get_row(ticker, date_str)
        if row is not None:
            if price_type == "Real_Close":
                return float(row['Close'])
            elif price_type == "Adjusted_Close":
                # 有些 parquet 沒有 Adj Close，fallback 到 Close
                return float(row.get('Adj Close', row['Close']))
            elif price_type in ["Open", "High", "Low"]:
                return float(row[price_type])
        return None

    def get_volume(self, ticker: str, date: str | datetime) -> Optional[float]:
        """取得特定日期的成交量。"""
        date_str = str(date)[:10] if isinstance(date, datetime) else date
        row = self._get_row(ticker, date_str)
        if row is not None:
            return float(row['Volume'])
        return None
