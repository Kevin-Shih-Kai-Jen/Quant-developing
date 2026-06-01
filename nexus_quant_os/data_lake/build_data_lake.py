"""
nexus_quant_os/data_lake/build_data_lake.py — 離線數據湖建造器 (基於 Polars)

功能：
1. 下載全市場/核心宇宙過去 10 年 (2014-2024) 的日線資料。
2. 以 Parquet 格式離線存儲，供回測引擎讀取 (徹底避免 API 呼叫)。
3. 實作「動態流動性閥門 (Liquidity Valve)」：過去 20 日均成交額 < 5000 萬台幣的殭屍股直接拋棄。
"""

import os
import logging
from typing import List
from datetime import datetime
import pandas as pd
import polars as pl
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用常數定義閥門與日期
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_LAKE_DIR = str(BASE_DIR / "parquet")
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
LIQUIDITY_THRESHOLD_TWD = 50_000_000  # 5000 萬台幣

def build_data_lake(universe: List[str]):
    os.makedirs(DATA_LAKE_DIR, exist_ok=True)
    logger.info(f"Starting Data Lake Build for {len(universe)} tickers...")
    
    # yf.download 可以一次載入多檔，但為了好存成單一 parquet 與避免 OOM，我們分批或單獨抓取
    for ticker in universe:
        logger.info(f"Fetching data for {ticker}...")
        try:
            # yfinance return pandas DataFrame
            # 為求簡化，此處針對台股亦暫用 yf，實務上需接 FinMind 取得正確還原權值
            pdf = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            
            if pdf.empty:
                logger.warning(f"No data found for {ticker}. Skipping.")
                continue
                
            # 展平 MultiIndex 列名 (yf 特性)
            if isinstance(pdf.columns, pd.MultiIndex):
                pdf.columns = pdf.columns.get_level_values(0)
            
            # 清理欄位
            pdf = pdf.reset_index()
            # yfinance returns index name 'Date' or 'Datetime'
            date_col = 'Date' if 'Date' in pdf.columns else 'Datetime' if 'Datetime' in pdf.columns else pdf.columns[0]
            pdf.rename(columns={date_col: "date"}, inplace=True)
            # 強制將時區移除以確保 Polars 相容性
            pdf['date'] = pd.to_datetime(pdf['date']).dt.tz_localize(None)

            # 轉換為 Polars DataFrame 以進行高效運算與儲存
            df = pl.from_pandas(pdf)
            
            # Liquidity Valve: 計算過去 20 日平均成交額
            # Volume 單位：yfinance 的台股 Volume 通常是股數。
            # 每日成交額 = Close * Volume
            df = df.with_columns(
                (pl.col("Close") * pl.col("Volume")).alias("Daily_Turnover")
            )
            
            df = df.with_columns(
                pl.col("Daily_Turnover").rolling_mean(window_size=20).alias("MA20_Turnover")
            )
            
            # 檢查最近一筆有效的 20 日均量
            # 為了避免初次上市股被誤殺，我們只要有超過閥值就算通過
            max_turnover = df.select(pl.col("MA20_Turnover").max()).item()
            
            # 匯率簡易轉換 (若為美股)
            threshold = LIQUIDITY_THRESHOLD_TWD
            if not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
                # 美股，閥值換算為 USD (假設 1 USD = 30 TWD)
                threshold = LIQUIDITY_THRESHOLD_TWD / 30.0
                
            if max_turnover is None or max_turnover < threshold:
                logger.warning(f"Liquidity Valve Dropped {ticker}: Max 20MA Turnover {max_turnover} < {threshold}")
                continue
                
            # 存入 Parquet
            file_path = os.path.join(DATA_LAKE_DIR, f"{ticker}.parquet")
            df.write_parquet(file_path)
            logger.info(f"Saved {ticker} to Data Lake (Rows: {len(df)})")
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")

if __name__ == "__main__":
    # 測試宇宙：包含高流動性與可能被剔除的冷門股
    test_universe = ["2330.TW", "NVDA", "3231.TW", "3017.TW", "0050.TW", "1449.TW"] # 1449(佳和) 可能是冷門或流動性邊緣
    build_data_lake(test_universe)
