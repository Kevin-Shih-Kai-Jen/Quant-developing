"""
nexus_quant_os/main_runner.py — 大規模回測神經中樞 (The Abyss Runner)

負責縫合所有的深淵級協議，並透過 EventBus 驅動整個模擬過程。
"""

import logging
from typing import List, Dict
import random

# Core Infrastructure
from nexus_quant_os.core.event_bus import EventBus
from nexus_quant_os.core.models import AlphaSignalEvent, OrderEvent

# Features & AI
from nexus_quant_os.data_lake.feature_store import feature_store
from nexus_quant_os.alpha_hunter.ai_analyst import AIAnalyst
from nexus_quant_os.alpha_hunter.orthogonizer import AlphaOrthogonizer

# Portfolio & Execution
from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer
from nexus_quant_os.portfolio.convexity import ConvexityHedger
from nexus_quant_os.backtest.portfolio import PortfolioManager
from nexus_quant_os.backtest.execution import ExecutionEngine, Order
from nexus_quant_os.backtest.tearsheet import CPCVTearsheet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MainRunner")

class DummyDataFeed:
    """測試用的假報價源"""
    def __init__(self):
        self.prices = {}
        
    def get_price(self, ticker, date, price_type="Close"):
        if price_type == "High": return 105.0
        if price_type == "Low": return 95.0
        return self.prices.get(ticker, 100.0)
        
    def get_volume(self, ticker, date):
        return 5_000_000 # 5千張
        
    def get_vwap(self, ticker, date):
        return self.prices.get(ticker, 100.0)

class AbyssRunner:
    def __init__(self):
        self.data_feed = DummyDataFeed()
        self.portfolio_manager = PortfolioManager(self.data_feed)
        self.portfolio_manager.set_cash(twd=30_000_000, usd=0.0)
        self.execution_engine = ExecutionEngine(self.data_feed, self.portfolio_manager)
        
        self.optimizer = PortfolioOptimizer(max_sector_exposure=0.3)
        self.convexity_hedger = ConvexityHedger(target_allocation=0.015)
        self.orthogonizer = AlphaOrthogonizer()
        
        self.current_date = "2020-01-01"
        self._setup_event_bus()
        
    def _setup_event_bus(self):
        """【防呆關鍵】：所有模組互相不知道對方，全靠 EventBus 溝通"""
        EventBus.clear()
        
        # 1. AI 產生訊號 -> 觸發 Portfolio 計算權重
        EventBus.subscribe("SIGNAL_GENERATED", self._on_signal_generated)
        
        # 2. Portfolio 產生訂單 -> 觸發 Execution 執行
        EventBus.subscribe("ORDER_CREATED", self._on_order_created)
        
    def _on_signal_generated(self, event: AlphaSignalEvent):
        """接收訊號後，進入 HRP 與行業中立配置"""
        logger.info(f"Received Signal for {event.ticker}: {event.action} (AI Score: {event.ai_score:.2f})")
        
        # 模擬 HRP 分配 (單檔測試，直接給權重)
        target_weight = 0.1 if event.action == "STRONG_BUY" else 0.0
        
        # 發佈訂單事件
        order = OrderEvent(
            order_id=f"ORD_{event.ticker}_{self.current_date}",
            perm_id=event.perm_id,
            ticker=event.ticker,
            date=self.current_date,
            amount=10000 if event.action == "STRONG_BUY" else -10000,
            order_type="VWAP",
            target_weight=target_weight
        )
        EventBus.publish("ORDER_CREATED", order)
        
    def _on_order_created(self, event: OrderEvent):
        """接收訂單後，進入 Execution Engine"""
        logger.info(f"Received Order: {event.ticker} {event.amount} shares")
        # 轉換為 ExecutionEngine 吃得下的 Order 物件
        exec_order = Order(ticker=event.ticker, amount=event.amount, date=event.date)
        
        # 執行 (模擬 T+1 執行)
        result = self.execution_engine.execute(exec_order, t_plus_1_date=self.current_date)
        logger.info(f"Execution Result: {result.status} @ {result.executed_price}")

    def run_smoke_test(self):
        """
        執行 2020 年上半年縮水版壓力測試。
        驗證 Convexity Hedger, OFI, AI Entropy 等。
        """
        logger.info("Starting Abyss Protocol Smoke Test (2020-01 to 2020-06)...")
        
        timeline = [
            ("2020-01-15", 0.0),    # 平靜期
            ("2020-02-20", -0.01),  # 開始下跌
            ("2020-03-19", -0.06),  # 【黑天鵝】台股跌停
            ("2020-04-10", 0.02)    # 反彈
        ]
        
        nav_history = []
        payoff_history = []
        
        for date_str, market_drop in timeline:
            self.current_date = date_str
            logger.info(f"\n{'='*40}\n[DATE: {date_str}] Market Drop: {market_drop:.1%}")
            
            # 1. 尾部凸性避險 (Tail-Risk Hedging)
            nav = self.portfolio_manager.get_nav(self.current_date)
            # 扣血/建倉
            bleed = self.convexity_hedger.allocate_hedge(nav)
            if bleed > 0:
                self.portfolio_manager.deduct_cash("TWD", bleed)
            
            # 結算
            payoff = self.convexity_hedger.process_market_crash(market_drop)
            payoff_history.append(payoff)
            if payoff > 0:
                self.portfolio_manager.add_cash("TWD", payoff)
                
            # 2. 模擬生成 AI 訊號
            if market_drop < -0.04:
                action = "LOW_CATCH"
                score = 0.9
            else:
                action = "STRONG_BUY"
                score = 0.6
                
            # 將訊號發佈到 EventBus
            signal = AlphaSignalEvent(
                perm_id="P_2330",
                ticker="2330.TW",
                date=self.current_date,
                action=action,
                ai_score=score
            )
            EventBus.publish("SIGNAL_GENERATED", signal)
            
            # 結算當日 NAV
            final_nav = self.portfolio_manager.get_nav(self.current_date)
            nav_history.append(final_nav)
            logger.info(f"[END OF DAY {date_str}] Portfolio NAV: {final_nav:,.2f} TWD")
            
        logger.info("\nGenerating Tearsheet...")
        tearsheet = CPCVTearsheet(daily_navs=nav_history, daily_convexity_payoffs=payoff_history)
        tearsheet.generate_report()
            
if __name__ == "__main__":
    runner = AbyssRunner()
    runner.run_smoke_test()
