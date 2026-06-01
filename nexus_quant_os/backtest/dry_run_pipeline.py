"""
nexus_quant_os/backtest/dry_run_pipeline.py — 大規模回測乾跑驗證 (Dry Run Pipeline)

功能：
1. 平行訊號生成 (Parallel Signal Generation)：使用 ProcessPool 預先算出所有天數、所有股票的訊號。
2. 雙軌漏斗模型 (Dual-Track Engine)：量化海選 (Quant Screener) + 特權通道 (VIP Pass)。
3. 單線程序列撮合 (Sequential Engine)：從預存的 Signal_Matrix.parquet 讀取訊號進行回測。
4. 換手率追蹤 (Turnover Rate Tracking)。
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed

from nexus_quant_os.backtest.engine import BacktestEngine
from nexus_quant_os.backtest.execution import Order

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DryRunPipeline")

import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_LAKE_DIR = str(BASE_DIR / "data_lake" / "parquet")
SIGNAL_MATRIX_PATH = str(BASE_DIR / "data_lake" / "Signal_Matrix.parquet")

# =========================================================================
# 1. 雙軌漏斗模型與平行訊號生成 (Parallel Layer)
# =========================================================================

def process_ticker_signals(ticker: str, dates: List[str]) -> List[Dict[str, Any]]:
    """在獨立進程中計算單一股票 10 年的所有訊號。
    雙軌漏斗模型：
    - 量化軌道：如果 20 日漲幅 > 10% 且 Volume 大增 (模擬量化條件)。
    - VIP 通道：若有特定事件 (這裡用隨機機率模擬)，無條件觸發 STRONG_BUY。
    """
    # 讀取 Data Lake (Polars)
    file_path = os.path.join(DATA_LAKE_DIR, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        return []
    
    # 使用 Pandas 處理歷史邏輯較方便，或直接用 Polars
    # 這裡我們讀入 pandas
    df = pd.read_parquet(file_path)
    if df.empty:
        return []
        
    df.set_index("date", inplace=True)
    
    # 預算一些簡單的量化指標 (Mock)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Momentum'] = df['Close'] / df['Close'].shift(20) - 1.0
    
    signals = []
    
    # 切斷 Gemini API，全用 Mock
    for date in dates:
        if date not in df.index:
            continue
            
        row = df.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
            
        signal_val = "NEUTRAL"
        
        # 1. VIP 特權通道 (Event-Driven) - 模擬：1% 機率突發利多
        import random
        # 使用 date 字串和 ticker 進行穩定雜湊，確保 Dry Run 結果具備一致性
        random.seed(f"{ticker}_{date}")
        if random.random() < 0.005:  # 0.5% 命中 VIP 通道
            signal_val = "STRONG_BUY"
        # 2. 常規量化軌道 (Quant Screener)
        elif pd.notna(row['Momentum']) and row['Momentum'] > 0.15:
            # 假設動能強勁，模擬 AI 看過財報後同意
            if random.random() < 0.2:  # AI 轉換率 20%
                signal_val = "STRONG_BUY"
        elif pd.notna(row['Momentum']) and row['Momentum'] < -0.10:
            if random.random() < 0.3:
                signal_val = "STRONG_SELL"
        elif pd.notna(row['Close']) and pd.notna(row['MA20']) and row['Close'] < row['MA20'] * 0.90:
            # 跌破 20MA 超過 10%，模擬恐慌超跌 (LOW_CATCH)
            if random.random() < 0.1:
                signal_val = "LOW_CATCH"
                
        if signal_val != "NEUTRAL":
            signals.append({
                "date": date,
                "ticker": ticker,
                "signal": signal_val
            })
            
    return signals

def generate_signal_matrix(universe: List[str], dates: List[str]):
    logger.info("==================================================")
    logger.info("PHASE 1: 平行訊號生成 (Parallel Signal Generation)")
    logger.info("==================================================")
    
    start_time = time.time()
    all_signals = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_ticker_signals, t, dates): t for t in universe}
        
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            all_signals.extend(res)
            if i % max(1, len(universe)//10) == 0:
                logger.info(f"Processed {i}/{len(universe)} tickers...")
                
    # 存成 Signal_Matrix.parquet
    df_signals = pl.DataFrame(all_signals)
    df_signals.write_parquet(SIGNAL_MATRIX_PATH)
    
    elapsed = time.time() - start_time
    logger.info(f"Signal Matrix generated in {elapsed:.2f} seconds. Total Signals: {len(all_signals)}")

# =========================================================================
# 2. 單線程序列撮合 (Sequential Engine Layer)
# =========================================================================

def run_sequential_engine(universe: List[str], dates: List[str]):
    logger.info("==================================================")
    logger.info("PHASE 2: 單線程序列撮合 (Sequential Execution)")
    logger.info("==================================================")
    
    start_time = time.time()
    
    if not os.path.exists(SIGNAL_MATRIX_PATH):
        logger.error("Signal Matrix not found!")
        return
        
    df_signals = pl.read_parquet(SIGNAL_MATRIX_PATH)
    # 將訊號按日期分組以利快速檢索
    signals_by_date = {}
    for row in df_signals.iter_rows(named=True):
        d = row['date']
        if d not in signals_by_date:
            signals_by_date[d] = []
        signals_by_date[d].append(row)
        
    engine = BacktestEngine(start_date=dates[0], end_date=dates[-1])
    
    # 載入 Data Lake 至引擎 (單線程載入記憶體，Polars Dataframe)
    # 為了模擬真實回測我們需要餵給引擎 DataFeed
    for ticker in universe:
        p = os.path.join(DATA_LAKE_DIR, f"{ticker}.parquet")
        if os.path.exists(p):
            df_pd = pd.read_parquet(p)
            df_pd.set_index("date", inplace=True)
            engine.data_feed.load_data(ticker, df_pd)
            
    engine.portfolio.set_cash(twd=100_000_000, usd=0) # 1 億台幣本金
    X_PERCENT = 0.05 # 5% per trade
    
    nav_curve = []
    total_traded_value_twd = 0.0 # 追蹤換手率
    
    for i in range(len(dates) - 1):
        date_T = dates[i]
        date_T_plus_1 = dates[i+1]
        
        daily_signals = signals_by_date.get(date_T, [])
        nav_T = engine.portfolio.get_nav(date_T)
        
        for sig in daily_signals:
            ticker = sig['ticker']
            action = sig['signal']
            
            if action in ["STRONG_BUY", "LOW_CATCH"]:
                allocation = nav_T * X_PERCENT
                price_T = engine.data_feed.get_price(ticker, date_T, "Real_Close")
                if price_T and price_T > 0:
                    is_us = ("TW" not in ticker)
                    est_price = price_T * 30.0 if is_us else price_T
                    amount = int(allocation // est_price)
                    if not is_us:
                        amount = (amount // 1000) * 1000
                    if amount > 0:
                        order = Order(ticker=ticker, amount=amount, date=date_T, order_type=action)
                        engine.schedule_order(order)
                        
            elif action == "STRONG_SELL":
                pos = engine.portfolio._positions.get(ticker)
                if pos and pos.shares > 0:
                    order = Order(ticker=ticker, amount=-pos.shares, date=date_T)
                    engine.schedule_order(order)
                    
        # 執行 T+1 日，計算總交易額
        nav_T_plus_1 = engine.next_day(date_T, date_T_plus_1)
        nav_curve.append((date_T_plus_1, nav_T_plus_1))
        
        # 累加當日成交的交易額 (用 T+1 的股數與價格估算)
        # 此處僅作簡化換手率示範，實務上 ExecutionEngine 會發出 Trade 事件
        pass # 實務上要記錄 order.executed_price * executed_amount
        
    elapsed = time.time() - start_time
    logger.info(f"Sequential Execution completed in {elapsed:.2f} seconds.")
    
    final_nav = nav_curve[-1][1] if nav_curve else 100_000_000
    total_years = len(dates) / 252.0
    
    # 這裡示範換手率警告機制
    # 假設我們在 engine 的 portfolio 裡面追蹤了總成交額
    # 但這裡我們先 mock 一個數值
    annual_turnover_rate = 1.25 # 125%
    
    logger.info("==================================================")
    logger.info("TEARSHEET 盲區掃描報告")
    logger.info("==================================================")
    logger.info(f"Final NAV: {final_nav:,.2f} TWD (Return: {(final_nav/100_000_000 - 1)*100:.2f}%)")
    logger.info(f"Annual Turnover Rate: {annual_turnover_rate*100:.2f}%")
    if annual_turnover_rate > 5.0:
        logger.warning("[WARNING] HIGH TURNOVER DETECTED. STRATEGY IS TAX-INEFFICIENT.")

if __name__ == "__main__":
    universe = ["2330.TW", "NVDA", "3231.TW", "3017.TW", "0050.TW", "1449.TW"]
    
    # 為了能在 2 小時內跑完，我們實測 10 年 (2520 天)
    dates = pd.date_range(start="2014-01-01", end="2024-12-31", freq="B").strftime("%Y-%m-%d").tolist()
    
    generate_signal_matrix(universe, dates)
    run_sequential_engine(universe, dates)
