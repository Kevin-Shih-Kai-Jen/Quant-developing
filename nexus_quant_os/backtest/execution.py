"""
nexus_quant_os/backtest/execution.py — 交易微結構模擬與訂單執行

功能：
1. 交易成本模型：台股賣出強制扣除 0.3% 證交稅。
2. 假單規避與低接模型：預設使用 T+1 VWAP 成交。斷頭低接使用 (Open+Low)/2。
3. 一字漲跌停拒絕：Limit-Up/Down Lock Reject。
4. 流動性作弊防禦：Volume Participation Cap (10%)。
5. 499 張天花板防禦：Order Slicer。
"""

import logging
import math
from typing import List, Optional
from dataclasses import dataclass
from .portfolio import PortfolioManager, InsufficientFundsException

logger = logging.getLogger(__name__)

TW_SELL_TAX = 0.003
MAX_PARTICIPATION_RATE = 0.1

@dataclass
class Order:
    ticker: str
    amount: int  # 股數 (台股 1 張 = 1000 股)
    date: str    # 送單日期 T 日
    order_type: str = "MKT"  # "MKT", "LOW_CATCH"
    status: str = "PENDING"
    executed_price: float = 0.0
    executed_amount: int = 0

class ExecutionEngine:
    def __init__(self, data_feed, portfolio: PortfolioManager):
        self._data_feed = data_feed
        self._portfolio = portfolio

    def check_ofi(self, ticker: str, date: str, is_buy: bool) -> bool:
        """
        【升級九：毒性訂單流與逆向選擇防禦 (OFI)】
        檢查短期訂單流失衡。若買方想買，但市場充斥主動賣單 (Hit Bid)，則判定為毒性訂單流。
        """
        # 實務上需讀取 L2 Tick Data 計算 OFI = 買單變化 - 賣單變化
        # 這裡作為防呆架構展示，先預設無毒。
        is_toxic = False
        if is_toxic:
            logger.warning(f"Toxic Order Flow detected for {ticker}. Order suspended.")
            return False
        return True

    def execute(self, order: Order, t_plus_1_date: str) -> Order:
        """
        在 T+1 日執行 T 日送出的訂單。
        """
        ticker = order.ticker
        from ..alpha_hunter.ticker_resolver import TickerResolver
        market = TickerResolver.detect_market(ticker)
        currency = "USD" if market == "US" else "TWD"

        # 取得 T+1 日的價格資訊
        open_p = self._data_feed.get_price(ticker, t_plus_1_date, "Open")
        high_p = self._data_feed.get_price(ticker, t_plus_1_date, "High")
        low_p = self._data_feed.get_price(ticker, t_plus_1_date, "Low")
        volume = self._data_feed.get_volume(ticker, t_plus_1_date)

        if open_p is None or high_p is None or low_p is None or volume is None:
            order.status = "REJECTED_NO_DATA"
            return order

        # 1. 漲跌停鎖死拒絕 (Limit Lock Reject)
        if high_p == low_p:
            order.status = "REJECTED_LIMIT_LOCK"
            logger.warning(f"Order REJECTED_LIMIT_LOCK: {ticker} on {t_plus_1_date} (High == Low).")
            return order

        # 2. 決定執行價格
        if order.order_type == "LOW_CATCH":
            # 斷頭低接
            exec_price = (open_p + low_p) / 2.0
        else:
            # 預設 MKT -> VWAP
            exec_price = self._data_feed.get_vwap(ticker, t_plus_1_date)
            if exec_price is None:
                order.status = "REJECTED_NO_VWAP"
                return order

        # 3. 流動性參與率防護 (Volume Participation Cap)
        # 台股 volume 通常是「股」數或「張」數？yf 是股數，FinMind 也是股數。
        max_allowed_shares = int(volume * MAX_PARTICIPATION_RATE)
        
        # 決定方向
        is_buy = order.amount > 0
        target_shares = abs(order.amount)

        # 3.5 【升級九：毒性訂單流攔截】
        if not self.check_ofi(ticker, t_plus_1_date, is_buy):
            order.status = "SUSPENDED_TOXIC_FLOW"
            return order

        # 4. 499 張天花板自動拆單邏輯 (如果是台股，且超過 499 張 = 499,000 股)
        if market == "TW" and target_shares > 499_000:
            logger.info(f"OrderSlicer activated for {ticker}: target {target_shares} > 499,000 shares.")
            # 這裡簡化實作：如果總量超過 499 張，實際上我們還是會受限於 max_allowed_shares
            # Order Slicer 的本質是「拆單送出避免被擋」，在日級回測中，我們視為「只要符合 participation rate 就能逐步成交」
            pass

        # 應用流動性上限
        if target_shares > max_allowed_shares:
            logger.warning(f"Participation Cap Hit for {ticker}. Target: {target_shares}, Max: {max_allowed_shares}")
            actual_shares = max_allowed_shares
        else:
            actual_shares = target_shares
            
        if actual_shares <= 0:
            order.status = "REJECTED_NO_LIQUIDITY"
            return order

        # 5. 【升級三：平方根市場衝擊模型】(Square-Root Impact)
        # Slippage = Base_Fee + sigma * sqrt(Order_Size / Daily_Volume)
        base_fee = 0.001425 if market == "TW" else 0.0
        # 為了簡化，此處假設日波動率 sigma 為 2% (0.02)。實務上應從 DataLake 取真實 volatility。
        sigma = 0.02
        
        # 計算參與率的平方根
        participation_ratio = actual_shares / max(volume, 1)
        impact_penalty = sigma * math.sqrt(participation_ratio)
        total_slippage_rate = base_fee + impact_penalty
        
        if is_buy:
            # 買入時，滑價讓成本變高
            final_exec_price = exec_price * (1 + total_slippage_rate)
            cost = actual_shares * final_exec_price
            try:
                self._portfolio.deduct_cash(currency, cost)
            except InsufficientFundsException:
                order.status = "REJECTED_INSUFFICIENT_FUNDS"
                return order
                
            self._portfolio.update_position(ticker, actual_shares, final_exec_price)
        else:
            # 賣出時，滑價與證交稅讓收入變少
            final_exec_price = exec_price * (1 - total_slippage_rate)
            revenue = actual_shares * final_exec_price
            sell_tax = revenue * TW_SELL_TAX if market == "TW" else 0.0
            net_revenue = revenue - sell_tax
            
            try:
                self._portfolio.update_position(ticker, -actual_shares, final_exec_price)
                self._portfolio.add_cash(currency, net_revenue)
            except ValueError:
                # 放空被拒絕
                order.status = "REJECTED_SHORT_FORBIDDEN"
                return order

        order.status = "FILLED"
        order.executed_price = final_exec_price
        order.executed_amount = actual_shares if is_buy else -actual_shares
        return order
