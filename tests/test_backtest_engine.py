import pytest
import pandas as pd
from nexus_quant_os.backtest.engine import BacktestEngine
from nexus_quant_os.backtest.execution import Order
from nexus_quant_os.backtest.portfolio import InsufficientFundsException

@pytest.fixture
def mock_data():
    """模擬 T 日與 T+1 日的市場資料"""
    # 2330.TW: Normal liquidity, no lock limit
    df_2330 = pd.DataFrame({
        "Open": [100.0, 105.0],
        "High": [102.0, 108.0],
        "Low": [99.0, 104.0],
        "Close": [101.0, 106.0],
        "Adj Close": [100.0, 105.0],
        "Volume": [10_000_000, 12_000_000]
    }, index=["2024-01-01", "2024-01-02"])
    
    # 飆股.TW: Limit lock on T+1 (High == Low)
    df_lock = pd.DataFrame({
        "Open": [50.0, 55.0],
        "High": [52.0, 55.0],  # T+1 High == Low
        "Low": [49.0, 55.0],
        "Close": [51.0, 55.0],
        "Adj Close": [51.0, 55.0],
        "Volume": [100_000, 5_000]
    }, index=["2024-01-01", "2024-01-02"])

    # NVDA: US market
    df_nvda = pd.DataFrame({
        "Open": [400.0, 410.0],
        "High": [405.0, 415.0],
        "Low": [395.0, 408.0],
        "Close": [400.0, 412.0],
        "Adj Close": [400.0, 412.0],
        "Volume": [50_000_000, 60_000_000]
    }, index=["2024-01-01", "2024-01-02"])
    
    return {
        "2330.TW": df_2330,
        "飆股.TW": df_lock,
        "NVDA": df_nvda
    }

def test_currency_isolation(mock_data):
    """1. 拒絕自動換匯測試"""
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-01-02")
    engine.data_feed.load_data("NVDA", mock_data["NVDA"])
    
    # 只給台幣
    engine.portfolio.set_cash(twd=100_000_000, usd=0)
    
    order = Order(ticker="NVDA", amount=1000, date="2024-01-01")
    result = engine.execution.execute(order, "2024-01-02")
    
    assert result.status == "REJECTED_INSUFFICIENT_FUNDS"

def test_execution_microstructure(mock_data):
    """2. VWAP 執行機制與漲停拒絕測試"""
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-01-02")
    engine.data_feed.load_data("2330.TW", mock_data["2330.TW"])
    engine.data_feed.load_data("飆股.TW", mock_data["飆股.TW"])
    
    engine.portfolio.set_cash(twd=10_000_000, usd=0)
    
    # 正常成交：驗證成交價是否精準等於 T+1 日的 VWAP
    order = Order(ticker="2330.TW", amount=1000, date="2024-01-01")
    result = engine.execution.execute(order, "2024-01-02")
    
    expected_vwap = (108.0 + 104.0 + 106.0) / 3.0
    assert result.status == "FILLED"
    assert abs(result.executed_price - expected_vwap) < 1e-5
    
    # 一字漲停：驗證系統是否拒絕成交
    order_limit_up = Order(ticker="飆股.TW", amount=1000, date="2024-01-01")
    result_lock = engine.execution.execute(order_limit_up, "2024-01-02")
    assert result_lock.status == "REJECTED_LIMIT_LOCK"

def test_dual_price_system():
    """3. 雙套價格邏輯防禦"""
    # 這裡主要是概念驗證，確保 signal generator (概念上) 使用 Adj_Close，
    # 而 execution / portfolio 使用 Real_Close
    
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-01-02")
    assert engine.portfolio.uses("Real_Close")

def test_tw_sell_tax(mock_data):
    """驗證賣出台股時扣除 0.3% 證交稅"""
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-01-02")
    engine.data_feed.load_data("2330.TW", mock_data["2330.TW"])
    engine.portfolio.set_cash(twd=10_000_000, usd=0)
    
    # 直接塞給 portfolio 部位
    engine.portfolio.update_position("2330.TW", 1000, 100.0)
    
    # 賣出 1000 股
    order = Order(ticker="2330.TW", amount=-1000, date="2024-01-01")
    res = engine.execution.execute(order, "2024-01-02")
    
    assert res.status == "FILLED"
    
    expected_vwap = (108.0 + 104.0 + 106.0) / 3.0
    revenue = 1000 * expected_vwap
    commission = revenue * 0.001425
    sell_tax = revenue * 0.003
    expected_cash = 10_000_000 + revenue - commission - sell_tax
    
    assert abs(engine.portfolio.get_cash("TWD") - expected_cash) < 1e-4

def test_volume_participation_cap(mock_data):
    """驗證 MAX_PARTICIPATION_RATE = 0.1"""
    engine = BacktestEngine(start_date="2024-01-01", end_date="2024-01-02")
    engine.data_feed.load_data("2330.TW", mock_data["2330.TW"])
    engine.portfolio.set_cash(twd=1_000_000_000, usd=0)
    
    # T+1 volume 是 12_000_000，10% 是 1_200_000
    # 我們試圖買 2_000_000 股
    order = Order(ticker="2330.TW", amount=2_000_000, date="2024-01-01")
    res = engine.execution.execute(order, "2024-01-02")
    
    assert res.status == "FILLED"
    assert res.executed_amount == 1_200_000
