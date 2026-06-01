import pytest
import numpy as np
from scripts.train_tw_hmm import TWMarketRegimeDetector, generate_synthetic_tw_data
from nexus_quant_os.risk_firewall.hmm_regime_detector import HMMConfig, MarketRegime

def test_hmm_label_stability():
    """
    Test that TWMarketRegimeDetector consistently maps:
    State 0 = EXTREME_SHOCK (Bear, lowest return)
    State 1 = BEAR_HIGH_VOL (Neutral, middle return)
    State 2 = BULL_LOW_VOL (Bull, highest return)
    """
    # 模擬 5 次訓練，確保每次都是穩定的
    for i in range(5):
        # 產生帶有隨機噪聲的資料，確保結果不被單次隨機種子綁死
        np.random.seed(i)
        features = generate_synthetic_tw_data()
        
        config = HMMConfig(n_regimes=3, n_iter=100, random_state=i)
        detector = TWMarketRegimeDetector(config)
        detector.fit(features)
        
        # 檢查 emission means (index 1 is return)
        means = detector._hmm.means_[:, 1]
        
        # 取得排序後的 states
        state_to_regime = detector._state_to_regime
        
        # 我們期望 state_to_regime 包含三個 keys
        assert len(state_to_regime) == 3
        
        # 找出 EXTREME_SHOCK 的 state
        bear_state = next(k for k, v in state_to_regime.items() if v == MarketRegime.EXTREME_SHOCK)
        neutral_state = next(k for k, v in state_to_regime.items() if v == MarketRegime.BEAR_HIGH_VOL)
        bull_state = next(k for k, v in state_to_regime.items() if v == MarketRegime.BULL_LOW_VOL)
        
        # 驗證 Bear return < Neutral return < Bull return
        assert means[bear_state] < means[neutral_state]
        assert means[neutral_state] < means[bull_state]
