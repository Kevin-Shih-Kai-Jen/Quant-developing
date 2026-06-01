"""
nexus_quant_os/live_trading/shadow_broker.py — 影子對抗測試券商介面

這是在真正接上永豐 Shioaji 或群益 API 之前的沙盒環境。
它會攔截來自 ExecutionEngine 的訂單，將其轉化為符合真實券商 API 的 JSON 格式，
並模擬回應 (包含滑價、部分成交、拒絕等)。
"""

import logging
import json
import random
from typing import Dict, Any

logger = logging.getLogger("ShadowBroker")

class ShadowBroker:
    def __init__(self):
        self.orders = {}
        self.trade_counter = 0

    def submit_order(self, ticker: str, action: str, quantity: int, order_type: str = "MKT") -> Dict[str, Any]:
        """
        模擬券商 API 下單口
        :param action: "Buy" or "Sell"
        :param quantity: 股數
        """
        self.trade_counter += 1
        order_id = f"SB_{self.trade_counter:08d}"
        
        # 轉換為永豐/群益類似的 JSON
        broker_json = {
            "Action": action.upper(),
            "Symbol": ticker.split('.')[0] if ".TW" in ticker else ticker,
            "Quantity": quantity,
            "PriceType": order_type,
            "TimeInForce": "ROD" if order_type != "MKT" else "IOC"
        }
        
        logger.info(f"Shadow Broker Received Order: {json.dumps(broker_json)}")
        
        # 模擬券商防呆機制：如果一次超過 499 張 (499,000 股)，直接退單
        if quantity > 499_000:
            logger.error(f"Broker Reject: Order size {quantity} exceeds 499 limit.")
            status = "Rejected"
            msg = "Exceeds 499 limit"
        else:
            status = "Accepted"
            msg = "Order successfully routed to exchange"
            
        response = {
            "OrderID": order_id,
            "Status": status,
            "Message": msg
        }
        
        self.orders[order_id] = broker_json
        return response
        
    def get_execution_report(self, order_id: str, current_price: float) -> Dict[str, Any]:
        """
        模擬券商回報成交結果 (Fill Report)
        """
        if order_id not in self.orders:
            return {"Status": "NotFound"}
            
        order = self.orders[order_id]
        
        # 模擬真實滑價 (0.1% ~ 0.3%)
        slippage = random.uniform(0.001, 0.003)
        if order["Action"] == "BUY":
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)
            
        report = {
            "OrderID": order_id,
            "Status": "Filled",
            "FillPrice": round(fill_price, 2),
            "FillQuantity": order["Quantity"]
        }
        
        logger.info(f"Shadow Broker Fill Report: {json.dumps(report)}")
        return report
