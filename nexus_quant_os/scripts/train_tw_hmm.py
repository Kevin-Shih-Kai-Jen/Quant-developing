"""
nexus_quant_os/scripts/train_tw_hmm.py — 總經狀態隱馬可夫模型訓練器

【深淵死神級防禦 P0】
1. 實作 MacroSyncRollingFitter (總經發布日同步滾動訓練)。
2. 強制延遲 (Publication Lag)：模擬真實世界的 CPI/失業率發布遞延。
3. 標籤錨定 (Label Anchoring)：永遠將最高波動率的狀態定義為 State 0 (危機/熊市)。
"""

import logging
import numpy as np
import polars as pl
from typing import Dict, List, Any

# 假設已安裝 hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

logger = logging.getLogger("MacroHMM")

class MacroSyncRollingFitter:
    def __init__(self, n_states: int = 3, lag_days: int = 35):
        """
        :param n_states: 隱藏狀態數量 (例如: 0=危機, 1=震盪, 2=牛市)
        :param lag_days: 總經數據的發布延遲天數 (例如 1月份 CPI 通常 2月5日發布，延遲大約 35 天)
        """
        self.n_states = n_states
        self.lag_days = lag_days
        self.model = None
        # 訓練歷史紀錄，用於確保不使用未來資料
        self._last_trained_date = None

    def _anchor_labels(self, X: np.ndarray, hidden_states: np.ndarray):
        """
        標籤錨定 (Label Anchoring)
        避免 EM 演算法每次 random initialization 導致標籤亂跳 (Label Switching)。
        邏輯：計算每個狀態下，大盤波動率 (假設 X 的特徵 0 是大盤波動率)
        強制將波動率最大的狀態命名為 State 0。
        """
        # 如果 X 沒有資料或狀態只有一個
        if len(np.unique(hidden_states)) <= 1:
            return
            
        state_vols = {}
        for state in range(self.n_states):
            mask = (hidden_states == state)
            if np.sum(mask) > 0:
                # 假設 X[:, 0] 是波動率特徵
                state_vols[state] = np.mean(X[mask, 0])
            else:
                state_vols[state] = 0.0
                
        # 依照波動率由大到小排序狀態
        sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols[s], reverse=True)
        
        # 建立映射表: 舊狀態 -> 新狀態 (0, 1, 2)
        mapping = {old_s: new_s for new_s, old_s in enumerate(sorted_states)}
        
        # 交換 model.means_ 與 model.covars_
        if self.model is not None:
            new_means = np.copy(self.model.means_)
            new_covars = np.copy(self.model.covars_)
            new_transmat = np.copy(self.model.transmat_)
            new_startprob = np.copy(self.model.startprob_)
            
            for old_s, new_s in mapping.items():
                new_means[new_s] = self.model.means_[old_s]
                new_covars[new_s] = self.model.covars_[old_s]
                new_startprob[new_s] = self.model.startprob_[old_s]
                # transmat 需要交換行與列
                for old_s2, new_s2 in mapping.items():
                    new_transmat[new_s, new_s2] = self.model.transmat_[old_s, old_s2]
                    
            self.model.means_ = new_means
            self.model.covars_ = new_covars
            self.model.transmat_ = new_transmat
            self.model.startprob_ = new_startprob
            
        logger.info(f"HMM Labels Anchored. Mapping: {mapping}")

    def fit_predict_rolling(self, macro_df: pl.DataFrame, current_date: str) -> int:
        """
        在 current_date 當下進行滾動預測。
        1. 強制過濾掉 current_date - lag_days 以後的資料 (Publication Lag)。
        2. 每月只允許重新 fit 一次，減少無意義算力消耗與標籤震盪。
        3. 回傳今日的預測狀態。
        """
        if GaussianHMM is None:
            logger.warning("hmmlearn not installed. Returning default state 1.")
            return 1
            
        import datetime
        curr_dt = datetime.datetime.strptime(current_date, "%Y-%m-%d").date()
        cutoff_dt = curr_dt - datetime.timedelta(days=self.lag_days)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d")
        
        # 強制物理隔離：過濾出真實世界中已經公佈的資料
        historical_df = macro_df.filter(pl.col("date") <= cutoff_str)
        
        if historical_df.height < 100: # 資料太少不夠訓練
            return 1 # Default neutral state
            
        # 觸發器：如果是這個月第一次呼叫 (或是距離上次訓練超過 30 天)
        should_train = False
        if self._last_trained_date is None:
            should_train = True
        else:
            last_dt = datetime.datetime.strptime(self._last_trained_date, "%Y-%m-%d").date()
            if (curr_dt - last_dt).days >= 30:
                should_train = True
                
        # 假設 DataFrame 除了 date 外都是 features
        features = historical_df.select(pl.exclude("date")).to_numpy()
        
        if should_train:
            logger.info(f"[{current_date}] Macro-Synchronous Rolling Fit Triggered. Using data up to {cutoff_str}")
            self.model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=100)
            self.model.fit(features)
            
            # 預測歷史狀態，用於標籤錨定
            hidden_states = self.model.predict(features)
            self._anchor_labels(features, hidden_states)
            
            self._last_trained_date = current_date
            
        # 預測當前狀態 (基於最新的 features)
        latest_feature = features[-1:]
        current_state = self.model.predict(latest_feature)[0]
        return current_state
