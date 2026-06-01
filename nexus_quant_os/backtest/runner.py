import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from nexus_quant_os.backtest.engine import BacktestEngine
from nexus_quant_os.backtest.execution import Order

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestRunner")

def get_mock_data():
    """為了示範回測引擎的運作，我們產生微型宇宙的模擬日線資料。
    Universe: 2330.TW, NVDA, 3231.TW, 3017.TW
    Dates: 2024-01-01 to 2024-03-31
    """
    dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="B")
    mock_dict = {}
    
    # 2330.TW (台積電) 模擬資料：緩步上漲
    base_price = 600.0
    df = pd.DataFrame(index=dates)
    df["Open"] = [base_price + i for i in range(len(dates))]
    df["High"] = df["Open"] + 5
    df["Low"] = df["Open"] - 5
    df["Close"] = df["Open"] + 2
    df["Adj Close"] = df["Close"]
    df["Volume"] = 30_000_000
    mock_dict["2330.TW"] = df

    # NVDA (輝達)
    df_nvda = pd.DataFrame(index=dates)
    df_nvda["Open"] = [450.0 + i*2 for i in range(len(dates))]
    df_nvda["High"] = df_nvda["Open"] + 10
    df_nvda["Low"] = df_nvda["Open"] - 10
    df_nvda["Close"] = df_nvda["Open"] + 5
    df_nvda["Adj Close"] = df_nvda["Close"]
    df_nvda["Volume"] = 50_000_000
    mock_dict["NVDA"] = df_nvda

    # 3231.TW (緯創)
    df_3231 = pd.DataFrame(index=dates)
    df_3231["Open"] = [100.0 + i*0.5 for i in range(len(dates))]
    df_3231["High"] = df_3231["Open"] + 2
    df_3231["Low"] = df_3231["Open"] - 2
    df_3231["Close"] = df_3231["Open"] + 1
    df_3231["Adj Close"] = df_3231["Close"]
    df_3231["Volume"] = 20_000_000
    mock_dict["3231.TW"] = df_3231

    # 3017.TW (奇鋐)
    df_3017 = pd.DataFrame(index=dates)
    df_3017["Open"] = [300.0 + i*1.5 for i in range(len(dates))]
    df_3017["High"] = df_3017["Open"] + 8
    df_3017["Low"] = df_3017["Open"] - 8
    df_3017["Close"] = df_3017["Open"] + 3
    df_3017["Adj Close"] = df_3017["Close"]
    df_3017["Volume"] = 15_000_000
    mock_dict["3017.TW"] = df_3017

    return mock_dict, dates

def mock_signal_generator(date: str, ticker: str):
    """模擬 AI 策略在特定日期的訊號
    為了測試雙錢包拒絕換匯，我們在 1/5 嘗試買 NVDA (會失敗因為沒美金)。
    在 1/10 買入 2330.TW。
    """
    if str(date)[:10] == "2024-01-05" and ticker == "NVDA":
        return "STRONG_BUY"
    if str(date)[:10] == "2024-01-10" and ticker == "2330.TW":
        return "STRONG_BUY"
    if str(date)[:10] == "2024-01-15" and ticker == "3017.TW":
        return "LOW_CATCH"
    if str(date)[:10] == "2024-02-20" and ticker == "2330.TW":
        return "STRONG_SELL"
    return "NEUTRAL"

def run_e2e_backtest():
    universe = ["2330.TW", "NVDA", "3231.TW", "3017.TW"]
    
    # 初始化引擎
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-03-31")
    
    # 載入資料
    mock_data, dates = get_mock_data()
    for ticker in universe:
        engine.data_feed.load_data(ticker, mock_data[ticker])
    
    # 注入初始資金: 只有 TWD 30,000,000
    engine.portfolio.set_cash(twd=30_000_000, usd=0)
    logger.info("Initialized portfolio with 30,000,000 TWD and 0 USD (Currency Isolation Enforced).")

    # X% 部位設定
    X_PERCENT = 0.10  # 10%

    nav_curve = []
    
    # 開始逐日迴圈
    for i in range(len(dates) - 1):
        date_T = str(dates[i])[:10]
        date_T_plus_1 = str(dates[i+1])[:10]
        
        # 1. 策略掃描 T 日
        for ticker in universe:
            signal = mock_signal_generator(date_T, ticker)
            if signal in ["STRONG_BUY", "LOW_CATCH"]:
                # 計算 10% 淨值可買多少股
                nav = engine.portfolio.get_nav(date_T)
                allocation = nav * X_PERCENT
                
                # 若是美股但沒有美金，理論上引擎會拒絕，但我們還是送單看看
                price_T = engine.data_feed.get_price(ticker, date_T, "Real_Close")
                # 簡單假設匯率轉換來估算張數 (實際會由引擎 Reject)
                is_us = ("TW" not in ticker)
                est_price = price_T * 30.0 if is_us else price_T
                
                if est_price > 0:
                    amount = int(allocation // est_price)
                    # 台股必須 1000 股為單位
                    if not is_us:
                        # 美股不限制單位
                        pass
                    else:
                        amount = (amount // 1000) * 1000
                    
                    if amount > 0:
                        order_type = "LOW_CATCH" if signal == "LOW_CATCH" else "MKT"
                        order = Order(ticker=ticker, amount=amount, date=date_T, order_type=order_type)
                        engine.schedule_order(order)
                        logger.info(f"[{date_T}] Strategy Signal: {signal} {ticker}, scheduling Order for {amount} shares at T+1")

            elif signal == "STRONG_SELL":
                # 全數賣出
                pos = engine.portfolio._positions.get(ticker)
                if pos and pos.shares > 0:
                    order = Order(ticker=ticker, amount=-pos.shares, date=date_T)
                    engine.schedule_order(order)
                    logger.info(f"[{date_T}] Strategy Signal: {signal} {ticker}, scheduling Order to close {pos.shares} shares at T+1")

        # 2. 推進至 T+1 日，執行前日訂單
        nav = engine.next_day(date_T, date_T_plus_1)
        nav_curve.append((date_T_plus_1, nav))

    logger.info("Backtest Complete!")
    logger.info(f"Final NAV: {nav_curve[-1][1]:.2f} TWD")
    
    # 匯出結果
    df_results = pd.DataFrame(nav_curve, columns=["Date", "NAV"])
    df_results.to_csv("backtest_results.csv", index=False)
    logger.info("Saved NAV curve to backtest_results.csv")

if __name__ == "__main__":
    run_e2e_backtest()
