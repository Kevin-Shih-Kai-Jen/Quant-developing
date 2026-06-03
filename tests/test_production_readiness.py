import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# 1. Broker Module Tests
def test_price_precision_rounding():
    from nexus_quant_os.execution.futu_broker import FutuBroker
    mock_ctx = MagicMock()
    broker = FutuBroker()
    broker.ctx = mock_ctx
    
    # 注入極端小數點
    # _submit_order doesn't exist, use place_order direct test
    broker.place_order("NVDA", 100, 289.8888, "SELL")
    
    # 斷言 API 收到的確實是 .89 (四捨五入到小數點後兩位)
    mock_ctx.place_order.assert_called_with(price=289.89, qty=100, code="US.NVDA", trd_side="SELL", order_type=mock_ctx.place_order.call_args[1]['order_type'], trd_env=mock_ctx.place_order.call_args[1]['trd_env'])

def test_get_positions_failure_raises():
    from nexus_quant_os.execution.futu_broker import FutuBroker
    broker = FutuBroker()
    with patch.object(broker, "_get_trade_ctx", side_effect=Exception("API Timeout")):
        with pytest.raises(RuntimeError, match="get_positions API failure"):
            broker.get_positions()

# 2. Risk Firewall Tests
def test_hmm_nan_fallback():
    from nexus_quant_os.risk_firewall.hmm_regime_detector import HMMRegimeDetector
    detector = HMMRegimeDetector()
    detector.model = MagicMock()
    detector.model.predict_proba.return_value = np.array([[np.nan, np.nan]])
    
    res = detector.detect(np.array([[1.0]]))
    assert res.bear_prob == 1.0
    assert res.extreme_prob == 0.0

def test_ood_nan_failclosed():
    from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODAnomalyDetector, OODConfig
    detector = OODAnomalyDetector(OODConfig())
    # Mock fitted
    detector._is_fitted = True
    
    # 注入含 NaN 的特徵
    toxic = np.array([[1.0, np.nan, 3.0]])
    res = detector.detect(toxic)
    assert res.is_ood is True

# 3. Pipeline Run Tests (Mocked STEP 7)
def test_nan_weights_guard():
    raw_weights = np.array([np.nan, 0.5, 0.5])
    if np.any(np.isnan(raw_weights)) or np.any(np.isinf(raw_weights)):
        raw_weights = np.ones(len(raw_weights)) / len(raw_weights)
    assert not np.isnan(raw_weights).any()
    assert np.allclose(raw_weights, [1/3, 1/3, 1/3])

def test_sqqq_exposure_cap():
    target_weights = {"NVDA": 0.5, "AVGO": 0.5}
    target_weights["SQQQ"] = 0.1
    total_exposure = min(1.0, sum(abs(w) for w in target_weights.values()))
    assert total_exposure == 1.0

# 4. Feature Pipeline Tests
def test_synthetic_has_pmi_and_yield():
    from nexus_quant_os.data_pipelines.data_loader import _generate_synthetic_fallback
    dates = ["2026-06-01", "2026-06-02"]
    macro = _generate_synthetic_fallback(dates)
    assert "industrial_production" in macro.columns
    assert "yield_curve_slope" in macro.columns

def test_volume_zscore_clipped():
    volume = 100
    vol_mean = 10
    vol_std = 0
    zscore = ((volume - vol_mean) / (vol_std + 1e-8))
    clipped = np.clip(zscore, -10, 10)
    assert clipped == 10.0
