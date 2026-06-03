import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from nexus_quant_os.portfolio.cvxpy_optimizer import PortfolioOptimizer
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall, RiskTier, FirewallDecision, MarketRegimeDetector, OODAnomalyDetector, RegimePrediction, AnomalyResult
from nexus_quant_os.execution.broker_base import OrderIntent, OrderResult
from nexus_quant_os.execution.futu_broker import FutuBroker

# Mocks
import sys
moomoo_mock = MagicMock()
moomoo_mock.RET_OK = 0
moomoo_mock.OrderType.NORMAL = 1
moomoo_mock.ModifyOrderOp.CANCEL = 2
moomoo_mock.TrdSide.BUY = 3
moomoo_mock.TrdEnv.SIMULATE = 4
sys.modules['moomoo'] = moomoo_mock

class TestNuclearArmor(unittest.TestCase):
    
    def test_optimizer_poison_data(self):
        """破口二：毒數據穿透 - 測試 Optimizer 收受 NaN 矩陣時是否觸發全現金防爆"""
        optimizer = PortfolioOptimizer(n_assets=3, asset_names=["AAPL", "MSFT", "SHY"])
        
        # 故意注入毒數據
        moe_weights = np.array([0.5, np.nan, 0.2])
        cov_matrix = np.array([[0.1, np.inf, 0], [np.nan, 0.2, 0], [0, 0, 0.05]])
        
        # 執行優化
        weights = optimizer.optimize(
            moe_weights=moe_weights,
            expert_utilisation=np.array([1.0, 1.0]),
            cov_matrix=cov_matrix
        )
        
        # 斷言：應該觸發 fallback, idx_shy = 2 (SHY) 拿到 1.0, 其它 0.0
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[2], 1.0)
        
    @patch('nexus_quant_os.portfolio.cvxpy_optimizer.compute_risk_parity_weights')
    @patch('cvxpy.Problem.solve')
    def test_optimizer_deadlock_timeout(self, mock_solve, mock_risk_parity):
        """破口二：無盡死鎖 - 測試 cvxpy 求解超時或崩潰時是否會被攔截並退回全現金"""
        # 模擬 C++ 底層崩潰或超時拋出異常
        mock_solve.side_effect = Exception("OSQP Backend crashed / Timed out")
        mock_risk_parity.side_effect = Exception("Risk Parity also crashed")
        
        optimizer = PortfolioOptimizer(n_assets=3, asset_names=["AAPL", "MSFT", "SHY"])
        moe_weights = np.array([0.4, 0.4, 0.2])
        cov_matrix = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.05]])
        
        # Expert 意見一致，迫使其進入 _constrained_mvo
        weights = optimizer.optimize(
            moe_weights=moe_weights,
            expert_utilisation=np.array([1.0, 1.0]), # dispersion = 0
            cov_matrix=cov_matrix
        )
        
        # 斷言：MVO 崩潰後，應該觸發 except，最終Fallback至全現金
        self.assertEqual(weights[2], 1.0) # SHY 權重為 1.0
        
    def test_firewall_override_prohibition(self):
        """破口一：職責越權 - 測試融資斷頭潮(Margin Call)是否會覆蓋 EMERGENCY"""
        mock_hmm = MagicMock(spec=MarketRegimeDetector)
        mock_hmm._is_fitted = True
        # 模擬核爆級崩盤
        mock_hmm.predict.return_value = RegimePrediction(
            most_likely_regime=1, regime_probabilities=np.array([0, 1, 0]),
            danger_probability=0.99, bear_probability=0.99, is_dangerous=True, regime_label="CRASH"
        )
        
        mock_ood = MagicMock(spec=OODAnomalyDetector)
        mock_ood._is_fitted = True
        mock_ood.detect.return_value = AnomalyResult(is_ood=True, ae_zscore=5.0, ae_reconstruction_mse=2.0, isolation_forest_score=-0.8, combined_anomaly_score=0.9)
        
        fw = IntelligentRiskFirewall(mock_hmm, mock_ood)
        fw._is_fitted = {'US': True, 'TW': True}
        
        # 測試：觸發 Margin Call 且為 TW 市場
        decision = fw.evaluate(
            market_features=np.zeros((5, 6)), 
            raw_weights=np.array([0.5, 0.5]), 
            market="TW", 
            margin_ratio=1.25 # < 1.30 觸發撿屍邏輯
        )
        
        # 斷言：絕對不可降級為 CAUTION，必須維持 EMERGENCY
        self.assertEqual(decision.risk_tier, RiskTier.EMERGENCY)
        
    def test_fat_finger_limit(self):
        """破口四：胖手指絕對上限 - 測試單筆委託超額是否會被強行拒絕"""
        broker = FutuBroker()
        broker._get_trade_ctx = MagicMock()
        broker._get_prices = MagicMock(return_value={"BRK.A": 650000.0}) # 股價 65 萬美金
        
        # 意圖購買 1 股
        intents = [OrderIntent("BRK.A", "BUY", 1, "Fat Finger Test")]
        
        results = broker.execute(intents)
        
        # 斷言：總價 65萬 > 10萬，應被攔截，狀態為 REJECTED，order_id 包含 FAT_FINGER
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "REJECTED")
        self.assertEqual(results[0].order_id, "FAT_FINGER_REJECTED")
        
    def test_cascading_failures_isolation(self):
        """破口三：連鎖崩潰 - 測試單一標的執行失敗不會中斷其他標的"""
        broker = FutuBroker()
        
        # 第一檔正常，第二檔異常(模擬異常價格以拋出 Exception)，第三檔正常
        broker._get_prices = MagicMock(return_value={
            "AAPL": 150.0,
            "BAD_STOCK": "THIS_IS_A_STRING_PRICE_TO_CAUSE_CRASH", 
            "MSFT": 300.0
        })
        
        intents = [
            OrderIntent("AAPL", "BUY", 10, "Test"),
            OrderIntent("BAD_STOCK", "BUY", 10, "Test"),
            OrderIntent("MSFT", "BUY", 10, "Test")
        ]
        
        # _to_futu_code will crash on BAD_STOCK because we will mock it to throw exception
        broker._to_futu_code = MagicMock(side_effect=lambda x: "US."+x if x != "BAD_STOCK" else int("CRASH"))
        
        results = broker.execute(intents)
        
        # 斷言：雖然 BAD_STOCK 崩潰，但 AAPL 和 MSFT 仍然要執行（且沒有被 CANCEL）
        # 因為我們 mock 了 _to_futu_code，BAD_STOCK 會觸發 Exception。
        executed_symbols = [r.symbol for r in results]
        
        self.assertIn("AAPL", executed_symbols)
        self.assertIn("BAD_STOCK", executed_symbols)
        self.assertIn("MSFT", executed_symbols)
        
        bad_result = next(r for r in results if r.symbol == "BAD_STOCK")
        self.assertEqual(bad_result.status, "CANCELLED")
        self.assertEqual(bad_result.order_id, "ERROR")

if __name__ == "__main__":
    unittest.main()
