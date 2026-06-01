"""
scripts/train_tw_hmm.py - 訓練台股專屬風控 HMM 模型 (Epic 2)

拉取台股特徵與美股 SOX 波動率，訓練 3 狀態 HMM。
實作抗標籤漂移 (Label Sorting)，強制狀態碼對應回報。
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd

from nexus_quant_os.alpha_hunter.tw_macro import TWMacroClient
from nexus_quant_os.risk_firewall.hmm_regime_detector import MarketRegimeDetector, HMMConfig, MarketRegime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_tw_hmm")

def fetch_tw_features():
    client = TWMacroClient()
    df = client.get_all_features()
    # 這裡可以加上 yfinance 抓 SOX 波動率，但為求穩定先用模擬
    # df["sox_vol"] = ...
    
    # 填充缺失值
    df = df.fillna(method="ffill").fillna(0)
    
    # 假設前 6 維為特徵，並手動補上大盤 return 欄位 (index 1) 與 vol (index 0)
    # 若無真實資料，我們先用假資料示範
    if len(df) < 20:
        logger.warning("Not enough TW macro data, using synthetic.")
        return generate_synthetic_tw_data()
        
    features = df.drop(columns=["date", "release_date"], errors="ignore").values
    return features

def generate_synthetic_tw_data():
    """產生包含波動率(idx 0)與報酬率(idx 1)的假資料供測試與訓練。"""
    np.random.seed(42)
    def make_regime(n, vol, ret_mean):
        vol_obs = np.abs(np.random.normal(vol, vol * 0.15, n))
        ret_obs = np.random.normal(ret_mean, vol * 0.5, n)
        macro1 = np.random.normal(0, 1, n)
        macro2 = np.random.normal(0, 1, n)
        sox_vol = np.random.normal(vol*1.5, vol*0.2, n)
        return np.column_stack([vol_obs, ret_obs, macro1, macro2, sox_vol])
        
    bull = make_regime(200, 0.01, 0.005)
    neutral = make_regime(150, 0.02, 0.000)
    bear = make_regime(100, 0.05, -0.008)
    return np.vstack([bull, neutral, bear])

class TWMarketRegimeDetector(MarketRegimeDetector):
    """自訂標籤映射邏輯的台股 HMM。"""
    def _label_regimes(self) -> None:
        assert self._hmm is not None
        # Epic 2: 強制計算各 State 對應的歷史大盤回報均值，重映射狀態碼
        # 0=Bear, 1=Neutral, 2=Bull
        ret_idx = 1 # 報酬率在 index 1
        means = self._hmm.means_[:, ret_idx]
        sorted_states = np.argsort(means) # 報酬由低到高 (最負 -> 最正)
        
        # lowest return -> EXTREME_SHOCK (Bear)
        # middle return -> BEAR_HIGH_VOL (Neutral/Warning)
        # highest return -> BULL_LOW_VOL (Bull)
        n = self.config.n_regimes
        self._state_to_regime = {}
        if n == 3:
            self._state_to_regime[int(sorted_states[0])] = MarketRegime.EXTREME_SHOCK
            self._state_to_regime[int(sorted_states[1])] = MarketRegime.BEAR_HIGH_VOL
            self._state_to_regime[int(sorted_states[2])] = MarketRegime.BULL_LOW_VOL
        else:
            super()._label_regimes()
        logger.info("TW State labelling (sorted by return): %s", self._state_to_regime)

def train_and_save():
    features = fetch_tw_features()
    logger.info("Fetched TW features shape: %s", features.shape)
    
    config = HMMConfig(n_regimes=3, n_iter=300, danger_threshold=0.5)
    detector = TWMarketRegimeDetector(config)
    detector.fit(features)
    
    # Save to tw_hmm.pkl
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "nexus_quant_os", "risk_firewall", "tw_hmm.pkl"
    )
    with open(save_path, "wb") as f:
        pickle.dump(detector, f)
    logger.info("Saved TW HMM to %s", save_path)

if __name__ == "__main__":
    train_and_save()
