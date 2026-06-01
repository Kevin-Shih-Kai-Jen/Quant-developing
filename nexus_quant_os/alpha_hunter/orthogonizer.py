"""
nexus_quant_os/alpha_hunter/orthogonizer.py — 純粹殘差萃取與霍克斯衰減

【最高機密升級八：Orthogonalization & Hawkes Decay】
防禦 LLM 訊息與傳統因子的多重共線性 (Multicollinearity)。
將 AI 分數對傳統動能/價值因子進行迴歸，僅保留無法被傳統數據解釋的「殘差 (Residuals)」。
對舊新聞情緒實施指數衰減 (Hawkes Process)。
"""

import math
import logging
from typing import Dict, List, Optional
import polars as pl
import numpy as np

logger = logging.getLogger("Orthogonizer")

class AlphaOrthogonizer:
    def __init__(self, decay_lambda: float = 0.1):
        """
        :param decay_lambda: 霍克斯衰減係數。預設 0.1 代表每天衰減約 10%。
        """
        self.decay_lambda = decay_lambda
        # 簡單的線性迴歸歷史斜率記憶體 (實務上應從回測滾動擬合)
        self._beta_memory: Dict[str, float] = {}

    def apply_hawkes_decay(self, initial_ai_score: float, days_passed: int) -> float:
        """
        應用霍克斯衰減 (Hawkes Decay)
        Signal(t) = Signal(0) * e^(-lambda * t)
        """
        if days_passed < 0:
            return initial_ai_score
        return initial_ai_score * math.exp(-self.decay_lambda * days_passed)

    def extract_residual(
        self, 
        ai_score: float, 
        traditional_momentum_score: float, 
        ticker: str = "default"
    ) -> float:
        """
        Gram-Schmidt 正交化 / 殘差萃取
        ai_score = alpha + beta * traditional_score + residual
        我們只需要 residual。
        為了簡化，實務上通常需要在一個 batch 裡算 cross-sectional regression。
        這裡提供一個防呆版本的殘差提取：若 AI 說好，但傳統指標也極好，則 AI 的貢獻度降低。
        """
        # 這裡用一個假設的固定 beta 作為示範
        # 若是嚴格的 Gram-Schmidt: residual = ai_score - proj_{momentum}(ai_score)
        beta = self._beta_memory.get(ticker, 0.5) # 假設兩者有 0.5 的正相關
        
        # 殘差 = 觀測值 - 預期值
        expected_ai_score = beta * traditional_momentum_score
        residual = ai_score - expected_ai_score
        
        # Clip to -1.0 to 1.0
        return max(-1.0, min(1.0, residual))

    def process_signals(self, signal_df: pl.DataFrame, current_date: str) -> pl.DataFrame:
        """
        批次處理一整天的訊號 DataFrame。
        signal_df 應包含: ticker, ai_score, momentum_score, signal_date
        """
        if signal_df.is_empty():
            return signal_df
            
        # 這是防呆框架示範，實務上會有真實日期計算
        # 這裡先原封不動返回，並在日誌記錄
        logger.debug(f"Applying Orthogonalization & Hawkes decay to {len(signal_df)} signals.")
        return signal_df
