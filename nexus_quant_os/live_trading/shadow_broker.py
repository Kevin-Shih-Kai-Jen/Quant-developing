"""
nexus_quant_os/live_trading/shadow_broker.py — 影子對抗測試券商介面

這是在真正接上實盤 API 之前的演習沙盒環境。
繼承自 BrokerBase，提供 100% 介面兼容的虛擬券商實作。
負責攔截訂單、模擬滑價、維護虛擬帳戶淨值與持倉狀態。
"""

import logging
import random
import uuid
from datetime import datetime
from typing import Dict, Any, List

from nexus_quant_os.execution.broker_base import (
    BrokerBase,
    AccountSnapshot,
    Position,
    OrderIntent,
    OrderResult,
)

logger = logging.getLogger("ShadowBroker")


class ShadowBroker(BrokerBase):
    def __init__(self, initial_cash: float = 100_000.0):
        """
        初始化影子券商
        :param initial_cash: 模擬帳戶初始美金餘額
        """
        self._cash = initial_cash
        # key: symbol, value: (qty, avg_cost, current_price)
        self._positions: Dict[str, dict] = {}
        self.orders_history: List[OrderResult] = []

    def get_account(self) -> AccountSnapshot:
        """回傳模擬帳戶快照"""
        equity = self._cash
        for pos in self._positions.values():
            equity += pos["qty"] * pos["current_price"]

        return AccountSnapshot(
            equity=equity,
            cash=self._cash,
            buying_power=self._cash, # 不考慮保證金
            timestamp=datetime.utcnow()
        )

    def get_positions(self) -> list[Position]:
        """回傳虛擬持倉"""
        positions = []
        for symbol, data in self._positions.items():
            qty = data["qty"]
            if qty == 0:
                continue
            avg_cost = data["avg_cost"]
            current_price = data["current_price"]
            market_value = qty * current_price
            unrealized_pl = market_value - (qty * avg_cost)
            unrealized_pl_pct = (unrealized_pl / (qty * avg_cost)) if (qty * avg_cost) != 0 else 0.0

            positions.append(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pl=unrealized_pl,
                    unrealized_pl_pct=unrealized_pl_pct,
                )
            )
        return positions

    def reconcile(self, target_weights: dict[str, float]) -> list[OrderIntent]:
        """計算調倉意圖 (Reconciliation)"""
        # 簡單取 100 元作為虛擬股價進行差額計算，因為 ShadowBroker 沒有真實報價源
        account = self.get_account()
        total_equity = account.equity

        intents = []
        
        # 1. 取得現有權重
        current_weights = {}
        for pos in self.get_positions():
            current_weights[pos.symbol] = pos.market_value / total_equity if total_equity > 0 else 0.0

        # 2. 找出需要賣出的部位 (權重減少)
        for symbol, curr_w in current_weights.items():
            target_w = target_weights.get(symbol, 0.0)
            if target_w < curr_w - 0.005: # 0.5% buffer
                delta_w = curr_w - target_w
                delta_usd = total_equity * delta_w
                # 假設每股 100 鎂
                qty = int(delta_usd / 100.0)
                if qty > 0:
                    intents.append(OrderIntent(symbol=symbol, side="SELL", qty=qty, reason=f"Reduce weight by {delta_w:.2%}"))
                    
        # 3. 找出需要買入的部位 (權重增加)
        for symbol, target_w in target_weights.items():
            curr_w = current_weights.get(symbol, 0.0)
            if target_w > curr_w + 0.005:
                delta_w = target_w - curr_w
                delta_usd = total_equity * delta_w
                qty = int(delta_usd / 100.0)
                if qty > 0:
                    intents.append(OrderIntent(symbol=symbol, side="BUY", qty=qty, reason=f"Increase weight by {delta_w:.2%}"))

        return intents

    def execute(self, intents: list[OrderIntent]) -> list[OrderResult]:
        """模擬券商執行訂單"""
        results = []
        for intent in intents:
            order_id = f"SB_{uuid.uuid4().hex[:8]}"
            
            # 模擬滑價
            slippage = random.uniform(0.001, 0.003)
            base_price = 100.0 # 預設虛擬股價
            
            if intent.side == "BUY":
                fill_price = base_price * (1 + slippage)
                cost = fill_price * intent.qty
                if self._cash >= cost:
                    self._cash -= cost
                    status = "FILLED"
                else:
                    status = "REJECTED" # 現金不足
            else:
                fill_price = base_price * (1 - slippage)
                status = "FILLED"
                self._cash += fill_price * intent.qty

            if status == "FILLED":
                # 更新持倉
                pos = self._positions.get(intent.symbol, {"qty": 0, "avg_cost": 0.0, "current_price": fill_price})
                if intent.side == "BUY":
                    new_qty = pos["qty"] + intent.qty
                    new_cost = ((pos["qty"] * pos["avg_cost"]) + (intent.qty * fill_price)) / new_qty
                    self._positions[intent.symbol] = {"qty": new_qty, "avg_cost": new_cost, "current_price": fill_price}
                else:
                    new_qty = max(0, pos["qty"] - intent.qty)
                    self._positions[intent.symbol] = {"qty": new_qty, "avg_cost": pos["avg_cost"], "current_price": fill_price}

            result = OrderResult(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                filled_price=fill_price if status == "FILLED" else 0.0,
                commission=0.0,
                timestamp=datetime.utcnow(),
                order_id=order_id,
                status=status
            )
            results.append(result)
            self.orders_history.append(result)
            logger.info(f"Shadow Broker Execute: {result}")

        return results

    def is_market_open(self) -> bool:
        """演習模式隨時都是開盤"""
        return True

    def get_portfolio_value(self) -> float:
        """取得總淨值"""
        return self.get_account().equity
