import pytest
import numpy as np
from unittest.mock import MagicMock
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall, RiskTier, FirewallConfig
from nexus_quant_os.execution.broker_base import OrderIntent

# ── 🌪️ 第三象限：防火牆越權與裝睡測試 ──
def test_q3_firewall_rebellion():
    """【防線叛變】測試：核爆級崩盤合併融資斷頭時，防火牆不准解除最高警報！"""
    
    hmm_mock = MagicMock()
    # 模擬核爆級崩盤 (HMM Danger = 0.99)
    hmm_mock.predict.return_value = MagicMock(danger_probability=0.99, bear_probability=0.99, regime_label="CRASH")
    ood_mock = MagicMock()
    ood_mock.detect.return_value = MagicMock(combined_anomaly_score=0.1, is_ood=False)

    fw = IntelligentRiskFirewall(hmm_detector=hmm_mock, ood_detector=ood_mock, config=FirewallConfig())
    fw._is_fitted = {"TW": True}
    fw.hmm_detectors = {"TW": hmm_mock}
    fw.ood_detectors = {"TW": ood_mock}
    
    # 💣 發動攻擊：故意觸發維持率過低 (< 1.30)
    decision = fw.evaluate(np.zeros((1, 6)), np.array([0.5, 0.5]), market="TW", margin_ratio=1.20)
    
    # 🛡️ 裝甲斷言：EMERGENCY 就是 EMERGENCY，絕對不准變成 CAUTION 去接刀！
    assert decision.risk_tier == RiskTier.EMERGENCY, "🚨 致命錯誤：防火牆違抗軍令，在核爆時強行抄底！"
    assert decision.scale_factor <= 0.05, "🚨 致命錯誤：核爆時倉位未降至安全底線！"

# ── 🌪️ 第四象限：券商執行大災難測試 ──
def test_q4_fat_finger_and_isolation(mocker):
    """【胖手指與連鎖崩潰】測試：天價訂單與單點異常不可牽連全局"""
    from nexus_quant_os.execution.futu_broker import FutuBroker
    broker = FutuBroker(host="127.0.0.1", port=11111)
    
    mocker.patch.object(broker, '_get_prices', return_value={"AAPL": 150.0, "CRAZY": 500.0})
    
    # 模擬 SmartOrderSlicer 內部崩潰，會拋出 Exception
    # 這裡我們直接 mock execute_sliced_order 來模擬單檔股票處理時崩潰
    mocker.patch('nexus_quant_os.alpha_hunter.order_slicer.SmartOrderSlicer.execute_sliced_order', side_effect=Exception("Network Timeout"))
    
    intents = [
        OrderIntent(symbol="AAPL", side="BUY", qty=10, reason="Normal"),
        # 💣 攻擊 2：胖手指訂單：買入 CRAZY 1,000,000 股 (總價數億美金)
        OrderIntent(symbol="CRAZY", side="BUY", qty=1_000_000, reason="Bug")
    ]
    
    results = broker.execute(intents)
    
    # 🛡️ 裝甲斷言 1：單點隔離，AAPL 的失敗絕對不准影響後續迴圈！
    # 預期 results 會有兩筆，一筆 AAPL (ERROR), 一筆 CRAZY (FAT_FINGER)
    assert len(results) == 2, "🚨 致命錯誤：AAPL 的異常導致整個迴圈崩潰，後續委託被吃掉！"
    
    # 🛡️ 裝甲斷言 2：胖手指必須在本地被強制 REJECT
    crazy_result = next(r for r in results if r.symbol == "CRAZY")
    assert crazy_result.status == "REJECTED", "🚨 致命錯誤：胖手指防線被擊穿，天價訂單已送出！"
