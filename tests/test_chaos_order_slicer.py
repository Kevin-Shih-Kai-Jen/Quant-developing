import unittest
from unittest.mock import MagicMock
import pandas as pd
from nexus_quant_os.alpha_hunter.order_slicer import SmartOrderSlicer
from nexus_quant_os.execution.broker_base import OrderIntent

# 確保 moomoo 模組被 mock
import sys
moomoo_mock = MagicMock()
moomoo_mock.RET_OK = 0
moomoo_mock.OrderType.NORMAL = 1
moomoo_mock.ModifyOrderOp.CANCEL = 2
moomoo_mock.TrdSide.BUY = 3
sys.modules['moomoo'] = moomoo_mock

class TestChaosOrderSlicer(unittest.TestCase):
    def setUp(self):
        self.broker = MagicMock()
        self.broker._TRD_ENV = "SIMULATE"
        self.broker._to_futu_code.return_value = "US.AAPL"
        
        self.ctx = MagicMock()
        self.broker._get_trade_ctx.return_value = self.ctx
        self.intent = OrderIntent("AAPL", "BUY", 2000, "Chaos Test")

    def test_p0_zero_division_crash(self):
        """故障注入：API 異常回傳股價為 0.0，驗證系統不會因為 ZeroDivisionError 崩潰"""
        self.broker._get_prices.return_value = {"AAPL": 0.0}
        
        # 執行拆單，不應噴出 Exception
        results = SmartOrderSlicer.execute_sliced_order(self.broker, self.intent)
        
        # 斷言防禦成功：安全拒絕，未引發 Crash
        self.assertEqual(results[0].status, "REJECTED")
        self.assertEqual(results[0].order_id, "NO_PRICE")

    def test_p0_phantom_fill_cancellation_failure(self):
        """故障注入：Timeout 取消時，API 斷線導致取消失敗"""
        self.broker._get_prices.return_value = {"AAPL": 150.0}
        self.ctx.place_order.return_value = (0, pd.DataFrame([{"order_id": "ORD_001"}]))
        
        # 一直未成交 (Timeout)
        self.ctx.order_list_query.return_value = (0, pd.DataFrame([{"order_status": "SUBMITTED"}]))
        
        # 故障注入：取消訂單也失敗 (模擬斷線, ret != RET_OK)
        self.ctx.modify_order.return_value = (-1, "Network Disconnect")
        
        # 設定 timeout 為 0.1 秒以快速觸發
        results = SmartOrderSlicer.execute_sliced_order(self.broker, self.intent, timeout_seconds=0.1)
        
        # 斷言防禦成功：狀態必須是 UNKNOWN_OR_PHANTOM，不能盲目相信已取消
        self.assertEqual(results[0].status, "UNKNOWN_OR_PHANTOM")

if __name__ == "__main__":
    unittest.main()
