import pytest
from unittest.mock import MagicMock
from nexus_quant_os.backtest.engine import BacktestEngine
from nexus_quant_os.backtest.execution import Order

def test_engine_next_day_advances_portfolio_time():
    """Test 12-1: engine.py must call advance_day()"""
    engine = BacktestEngine(start_date="2020-01-01", end_date="2020-01-10")
    engine.portfolio = MagicMock()
    engine.execution = MagicMock()
    
    engine.next_day("2020-01-01", "2020-01-02")
    
    # Portfolio advance_day should be called with next_date
    engine.portfolio.advance_day.assert_called_once_with("2020-01-02")

def test_engine_rejected_orders_logging(caplog):
    """Test 12-2: Rejected orders should be logged as warnings, not silent pass."""
    engine = BacktestEngine(start_date="2020-01-01", end_date="2020-01-10")
    engine.portfolio = MagicMock()
    
    mock_res = MagicMock()
    mock_res.status = "REJECTED_NO_LIQUIDITY"
    engine.execution = MagicMock()
    engine.execution.execute.return_value = mock_res
    
    order = Order(ticker="2330.TW", amount=1000, date="2020-01-01")
    engine.schedule_order(order)
    
    import logging
    with caplog.at_level(logging.WARNING):
        engine.next_day("2020-01-01", "2020-01-02")
        
    assert "REJECTED" in caplog.text

