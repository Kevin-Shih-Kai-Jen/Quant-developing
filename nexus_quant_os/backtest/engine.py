"""
nexus_quant_os/backtest/engine.py — 回測引擎主迴圈

功能：
1. 日級 (Daily) 事件迴圈。
2. 在 T 日處理訊號，在 T+1 日執行訂單 (強制 VWAP)。
3. PiT (Point-in-Time) 嚴格控制時間推移。
"""

import logging
from typing import List
from .data_feed import DataFeed
from .portfolio import PortfolioManager
from .execution import ExecutionEngine, Order

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        self.data_feed = DataFeed()
        self.portfolio = PortfolioManager(self.data_feed)
        self.execution = ExecutionEngine(self.data_feed, self.portfolio)
        self._pending_orders: List[Order] = []

    def schedule_order(self, order: Order):
        """策略在 T 日呼叫此方法送單。"""
        self._pending_orders.append(order)

    def next_day(self, current_date: str, next_date: str):
        """
        時間推移到 T+1 日。
        1. 執行昨日的 pending_orders。
        2. 處理除權息事件。
        """
        logger.info(f"Advancing time from {current_date} to {next_date}")
        
        # 推進 T+2 結算
        self.portfolio.advance_day(next_date)
        
        # 執行訂單
        unfilled = []
        for order in self._pending_orders:
            # order.date 是 T 日，現在我們用 next_date (T+1) 的資料執行
            res = self.execution.execute(order, next_date)
            logger.info(f"Order {order.ticker} {order.amount} -> {res.status}")
            if res.status in ["REJECTED_NO_LIQUIDITY", "PENDING", "REJECTED_NO_FUNDS"]:
                logger.warning(f"[WARNING] Order {order.ticker} {order.amount} was {res.status}. Reason: {getattr(res, 'reason', 'N/A')}")
        self._pending_orders.clear()

        # 處理 corporate actions 暫略... 實務上要比對 dividend 日期
        
        # Log NAV
        nav = self.portfolio.get_nav(next_date)
        logger.info(f"[{next_date}] NAV (TWD equiv): {nav}")
        return nav
