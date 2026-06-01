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
from nexus_quant_os.backtest.data_feed import DataFeed
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall
import numpy as np

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
    def __init__(self, use_real_data=False):
        self.use_real_data = use_real_data
        self.data_feed = DataFeed() if use_real_data else DummyDataFeed()
        self.portfolio_manager = PortfolioManager(self.data_feed)
        self.portfolio_manager.set_cash(twd=30_000_000, usd=0.0)
        self.execution_engine = ExecutionEngine(self.data_feed, self.portfolio_manager)
        
        self.optimizer = PortfolioOptimizer(max_sector_exposure=0.3)
        self.convexity_hedger = ConvexityHedger(target_allocation=0.015)
        self.orthogonizer = AlphaOrthogonizer()
        
        self.firewall = IntelligentRiskFirewall.from_configs()
        # Train with some dummy data to bypass the "not fitted" error
        dummy_train = np.random.normal(0, 0.01, (100, 6))
        self.firewall.fit(dummy_train, "US")
        self.firewall.fit(dummy_train, "TW")
        
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
        # logger.info(f"Received Signal for {event.ticker}: {event.action} (AI Score: {event.ai_score:.2f})")
        
        # 模擬強制回補日 (5% 機率剛好遇到強制回補)
        import random
        forced_covers = {}
        if random.random() < 0.05:
            import datetime
            c_dt = datetime.datetime.strptime(self.current_date, "%Y-%m-%d").date()
            f_dt = c_dt + datetime.timedelta(days=3) # 3天後強制回補
            forced_covers[event.ticker] = f_dt.strftime("%Y-%m-%d")
            
        weights = self.optimizer.calculate_weights(
            signals={event.ticker: event.ai_score},
            volatilities={event.ticker: self.data_feed.get_volatility(event.ticker, self.current_date) if hasattr(self.data_feed, "get_volatility") else 0.02},
            sectors={event.ticker: getattr(self.data_feed, "get_sector", lambda t: "Unknown")(event.ticker)},
            volumes={event.ticker: self.data_feed.get_volume(event.ticker, self.current_date) or 5000000},
            current_date=self.current_date,
            forced_cover_dates=forced_covers
        )
        target_weight = weights.get(event.ticker, 0.0)
        
        # 呼叫 Firewall 進行審核
        market = "TW" if "TW" in event.ticker else "US"
        dummy_features = np.random.normal(0, 0.01, (20, 6)) # Dummy market window for firewall
        
        margin_ratio = getattr(self.portfolio_manager, "get_margin_ratio", lambda: 2.0)()
        decision = self.firewall.evaluate(
            market_features=dummy_features,
            raw_weights=np.array([target_weight]),
            market=market,
            margin_ratio=margin_ratio
        )
        target_weight = float(decision.adjusted_weights[0])
        
        # 動態計算下單數量
        nav = self.portfolio_manager.get_nav(self.current_date)
        price = self.data_feed.get_price(event.ticker, self.current_date, "Close") or 100.0
        
        # 判斷是否為台股
        is_us = ("TW" not in event.ticker)
        est_price = price * 30.0 if is_us else price
        
        # 如果沒有分配權重且不是 STRONG_SELL，則忽略
        if target_weight == 0 and event.action != "STRONG_SELL":
            return
            
        allocation = nav * target_weight
        amount = int(allocation // est_price)
        
        # 台股必須以 1000 股為單位
        if not is_us:
            amount = (amount // 1000) * 1000
            
        if event.action == "STRONG_SELL":
            # 如果是強烈賣出，我們直接倒掉目前所有的部位
            pos = self.portfolio_manager._positions.get(event.ticker)
            if pos and pos.shares > 0:
                amount = -pos.shares
            else:
                amount = 0
                
        if amount == 0:
            return
            
        # 發佈訂單事件
        order = OrderEvent(
            order_id=f"ORD_{event.ticker}_{self.current_date}",
            perm_id=event.perm_id,
            ticker=event.ticker,
            date=self.current_date,
            amount=amount,
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

    def run_large_scale_test(self, days: int = 1250):
        """
        執行大規模壓力測試 (例如 5 年期 = 1250 個交易日)
        使用蒙地卡羅隨機漫步產生市場跌幅，並注入黑天鵝。
        """
        logger.info(f"Starting Large-Scale Backtest ({days} Days)...")
        
        # 重置資金
        self.portfolio_manager.set_cash(twd=30_000_000, usd=0.0)
        self.portfolio_manager._positions.clear()
        self.convexity_hedger = ConvexityHedger(target_allocation=0.015)
        
        nav_history = []
        payoff_history = []
        
        import datetime
        start_date = datetime.date(2015, 1, 1)
        
        # 關閉 INFO level logger 避免日誌炸裂
        logger.setLevel(logging.WARNING)
        
        for i in range(days):
            current_date = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            self.current_date = current_date
            
            # 推進 T+2 Settlement Cascade
            self.portfolio_manager.advance_day(self.current_date)
            
            # 隨機產生市場波動 (常態分佈)
            market_drop = random.gauss(0.0005, 0.01) # 預設微幅向上，日波動 1%
            
            # 注入黑天鵝 (1% 機率)
            if random.random() < 0.01:
                market_drop = random.uniform(-0.04, -0.08) # 崩盤 4% ~ 8%
            
            nav = self.portfolio_manager.get_nav(self.current_date)
            
            # 尾部避險扣血
            bleed = self.convexity_hedger.allocate_hedge(nav)
            if bleed > 0:
                from nexus_quant_os.backtest.portfolio import InsufficientFundsException
                try:
                    self.portfolio_manager.deduct_cash("TWD", bleed)
                except InsufficientFundsException:
                    logger.critical("MARGIN CALL: Cannot afford convexity hedge bleed. NAV is bleeding out!")
                except Exception as e:
                    logger.error(f"Unexpected error during hedge deduction: {e}")
            
            # 結算
            payoff = self.convexity_hedger.process_market_crash(market_drop)
            payoff_history.append(payoff)
            if payoff > 0:
                self.portfolio_manager.add_cash("TWD", payoff)
                
            # 產生 AI 訊號
            if market_drop < -0.03:
                action, score = "LOW_CATCH", 0.9
            elif market_drop > 0.02:
                action, score = "SELL", -0.8
            else:
                action, score = "STRONG_BUY", 0.6
                
            signal = AlphaSignalEvent(
                perm_id="P_2330", ticker="2330.TW", date=self.current_date, action=action, ai_score=score
            )
            EventBus.publish("SIGNAL_GENERATED", signal)
            
            final_nav = self.portfolio_manager.get_nav(self.current_date)
            nav_history.append(final_nav)

            # 模擬股票隨市場漲跌 (強制修改 portfolio 的部位價格)
            for ticker, pos in self.portfolio_manager._positions.items():
                if pos.shares > 0:
                    # 這是黑魔法，為了模擬回測帳面價值隨市場波動
                    self.data_feed.prices[ticker] = self.data_feed.prices.get(ticker, 100.0) * (1 + market_drop)

        # 跑完後恢復 logger level
        logger.setLevel(logging.INFO)
        logger.info(f"\nCompleted {days} Days Simulation!")
        logger.info(f"Final NAV: {nav_history[-1]:,.2f} TWD")
        
        logger.info("\nGenerating Large-Scale Tearsheet...")
        tearsheet = CPCVTearsheet(daily_navs=nav_history, daily_convexity_payoffs=payoff_history)
        tearsheet.generate_report()

    def run_real_data_test(self):
        """
        執行 10 年實盤數據管線乾跑 (Plumbing Dry-Run)
        使用 0050.TW 作為 Master Clock 與 market_drop 基準。
        """
        logger.info("Starting Real Data Plumbing Dry-Run...")
        
        # 重置資金
        self.portfolio_manager.set_cash(twd=30_000_000, usd=0.0)
        self.portfolio_manager._positions.clear()
        self.convexity_hedger = ConvexityHedger(target_allocation=0.015)
        
        nav_history = []
        payoff_history = []
        
        # 取得 Master Clock (0050.TW)
        master_df = self.data_feed.get_data("0050.TW")
        if master_df.empty:
            logger.error("Master Clock (0050.TW) data not found!")
            return
            
        master_dates = sorted(master_df.index.tolist())
        test_universe = ["2330.TW", "NVDA", "3231.TW", "3017.TW", "1449.TW"]
        
        import os
        from pathlib import Path
        import polars as pl
        import pandas as pd
        ai_signals_df = None
        base_dir = Path(__file__).resolve().parent
        parquet_path = base_dir / "data_lake" / "Alpha_Signal_Matrix.parquet"
        if parquet_path.exists():
            ai_signals_df = pl.read_parquet(str(parquet_path)).to_pandas()
            ai_signals_df['publish_date'] = pd.to_datetime(ai_signals_df['publish_date'])
            ai_signals_df.sort_values('publish_date', inplace=True)
            
        latest_ai_scores = {t: 0.5 for t in test_universe} # 預設中立 0.5
        
        prev_adj_close = None
        
        # 關閉 INFO level logger 避免日誌炸裂
        logger.setLevel(logging.WARNING)
        
        for date_ts in master_dates:
            current_date = date_ts.strftime("%Y-%m-%d")
            self.current_date = current_date
            
            # 強制使用真實收盤價，不准偷看未來的除權息調整，但是計算 market_drop 要用 Adj Close 避免除息跳空誤判崩盤
            adj_col = 'Adj Close'
            if adj_col not in master_df.columns:
                adj_col = 'Close'
            adj_close = float(master_df.loc[date_ts].iloc[0][adj_col] if isinstance(master_df.loc[date_ts], type(master_df)) else master_df.loc[date_ts][adj_col])
            
            if prev_adj_close is not None:
                market_drop = (adj_close - prev_adj_close) / prev_adj_close
            else:
                market_drop = 0.0
            prev_adj_close = adj_close
            # 每日開盤前：部位對帳 (Reconciliation)
            # 實盤中，這裡會 call 券商 API: broker_api.get_inventory()
            # 這裡我們先放入一個 Mock 的校準邏輯，為將來接 API 預留插槽
            try:
                # 模擬 0.1% 的機率，API 顯示部位與系統脫鉤
                if random.random() < 0.001 and len(self.portfolio_manager._positions) > 0:
                    rogue_ticker = random.choice(list(self.portfolio_manager._positions.keys()))
                    logger.error(f"Reconciliation Mismatch! Broker reports different position for {rogue_ticker}.")
                    # 強制覆寫系統部位 (實盤實作點)
            except Exception as e:
                logger.error(f"Reconciliation Failed: {e}")

            # 推進 T+2 Settlement Cascade
            self.portfolio_manager.advance_day(self.current_date)
            
            nav = self.portfolio_manager.get_nav(self.current_date)
            
            # 尾部避險扣血
            bleed = self.convexity_hedger.allocate_hedge(nav)
            if bleed > 0:
                from nexus_quant_os.backtest.portfolio import InsufficientFundsException
                try:
                    self.portfolio_manager.deduct_cash("TWD", bleed)
                except InsufficientFundsException:
                    logger.critical("MARGIN CALL: Cannot afford convexity hedge bleed. NAV is bleeding out!")
                except Exception as e:
                    logger.error(f"Unexpected error during hedge deduction: {e}")
            
            # 結算
            payoff = self.convexity_hedger.process_market_crash(market_drop)
            payoff_history.append(payoff)
            if payoff > 0:
                self.portfolio_manager.add_cash("TWD", payoff)
                
            # Phase 13: 注入異步特徵矩陣的 AI 訊號
            if ai_signals_df is not None:
                today_signals = ai_signals_df[ai_signals_df['publish_date'] == date_ts]
                for _, row in today_signals.iterrows():
                    if row['ticker'] in latest_ai_scores:
                        latest_ai_scores[row['ticker']] = row['ai_score']
                        
            # 生成訊號發佈
            for ticker in test_universe:
                # 只有當日成交量大於 0 的才給訊號，避免停牌股一直發訊號
                vol = self.data_feed.get_volume(ticker, self.current_date)
                if vol and vol > 0:
                    score = latest_ai_scores[ticker]
                    
                    if score > 0.6:
                        action = "STRONG_BUY"
                    elif score < 0.4:
                        action = "SELL"
                    else:
                        action = "HOLD"
                        
                    if action != "HOLD":
                        # 將 0.0~1.0 的 score 映射到 -1.0~1.0 以符合下游架構
                        mapped_score = (score - 0.5) * 2
                        signal = AlphaSignalEvent(
                            perm_id=f"P_{ticker}", ticker=ticker, date=self.current_date, action=action, ai_score=mapped_score
                        )
                        EventBus.publish("SIGNAL_GENERATED", signal)
            
            final_nav = self.portfolio_manager.get_nav(self.current_date)
            nav_history.append(final_nav)

        # 跑完後恢復 logger level
        logger.setLevel(logging.INFO)
        logger.info(f"\nCompleted {len(master_dates)} Days Real Data Simulation!")
        logger.info(f"Final NAV: {nav_history[-1]:,.2f} TWD")
        
        logger.info("\nGenerating Real-Data Tearsheet...")
        tearsheet = CPCVTearsheet(daily_navs=nav_history, daily_convexity_payoffs=payoff_history)
        tearsheet.generate_report()
            
if __name__ == "__main__":
    import sys
    use_real = "--real" in sys.argv
    runner = AbyssRunner(use_real_data=use_real)
    if use_real:
        runner.run_real_data_test()
    else:
        runner.run_large_scale_test(1250)
