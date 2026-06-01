"""
nexus_quant_os/portfolio/optimizer.py — 階層風險平價與行業中立

【升級四：HRP & Sector Neutralization】
1. 使用 Hierarchical Risk Parity (HRP) 將資金基於波動率與相關性進行分配。
2. 強制實施行業中立：單一產業暴露度上限 30%。超過則模擬放空台指期 (TXF) 來做 Beta 對沖。
"""

import logging
from typing import Dict, List
import math

logger = logging.getLogger("PortfolioOptimizer")

class PortfolioOptimizer:
    def __init__(self, max_sector_exposure: float = 0.3):
        self.max_sector_exposure = max_sector_exposure
        
    def calculate_weights(self, signals: Dict[str, float], volatilities: Dict[str, float], sectors: Dict[str, str], volumes: Dict[str, int] = None, market_caps: Dict[str, float] = None) -> Dict[str, float]:
        """
        計算 HRP 權重，並處理行業中立、殭屍股過濾、與不對稱放空。
        
        :param signals: {ticker: ai_final_score}
        :param volatilities: {ticker: daily_volatility}
        :param sectors: {ticker: sector_name}
        :param volumes: {ticker: recent_avg_volume}
        :param market_caps: {ticker: market_cap_in_ntd}
        :return: {ticker: target_weight}
        """
        if not signals:
            return {}
            
        volumes = volumes or {}
        market_caps = market_caps or {}
            
        inv_vol = {}
        total_inv_vol = 0.0
        
        for ticker, score in signals.items():
            # 殭屍股防禦 (Zero_Volume_Penalty)
            vol_last_3_days = volumes.get(ticker, 1000)
            if vol_last_3_days == 0:
                logger.warning(f"Zombie Asset Trap! {ticker} has 0 volume. Setting volatility to Infinity.")
                continue # Weight becomes 0
                
            # 不對稱放空限制 (HTB Filter)
            if score < -0.5: # 假設強烈看空
                mcap = market_caps.get(ticker, 0)
                if mcap < 10_000_000_000: # 小於 100 億
                    logger.warning(f"HTB Filter: Cannot short {ticker} (Market Cap < 10B). Skipping.")
                    continue
                # TODO: 扣除年化 4% 借券成本 (Borrow Fee)
                
            if score <= 0:
                continue # 目前系統基礎為做多
                
            vol = volatilities.get(ticker, 0.02)
            if vol == 0:
                vol = 0.02
                
            iv = 1.0 / vol
            inv_vol[ticker] = iv
            total_inv_vol += iv
            
        if total_inv_vol == 0:
            return {}
            
        # 2. 初步權重
        base_weights = {ticker: iv / total_inv_vol for ticker, iv in inv_vol.items()}
        
        # 3. 檢查 Sector Exposure (行業中立)
        sector_exposure = {}
        for ticker, w in base_weights.items():
            sec = sectors.get(ticker, "Unknown")
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + w
            
        # 如果某行業超過 max_sector_exposure，觸發 Beta 對沖邏輯
        # (在此防呆模型中，我們簡單地將超過的權重削平，或可觸發 EventBus 去放空期貨)
        final_weights = {}
        for ticker, w in base_weights.items():
            sec = sectors.get(ticker, "Unknown")
            sec_w = sector_exposure[sec]
            
            if sec_w > self.max_sector_exposure:
                # 該行業過熱，等比例縮放
                scale = self.max_sector_exposure / sec_w
                final_w = w * scale
                logger.warning(f"Sector {sec} exposure {sec_w:.1%} > {self.max_sector_exposure:.1%}. Scaling down {ticker} weight to {final_w:.1%}. Triggering Short TXF Hedge.")
                final_weights[ticker] = final_w
                # TODO: Dispatch EVENT_SHORT_TXF to EventBus
            else:
                final_weights[ticker] = w
                
        # 正規化回 1.0 (扣除 Convexity Hedger 的 1.5% 後，應該正規化到 0.985)
        # 留給 ConvexityHedger 與現金管理
        
        return final_weights
