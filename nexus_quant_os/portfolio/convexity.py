"""
nexus_quant_os/portfolio/convexity.py — 尾部凸性避險與末日流動性

【最高機密升級十：Tail-Risk Convexity Hedging】
強迫從總資金撥出 1.5% 定期買入台指選擇權 (TXO) 的極度價外賣權 (Deep OTM Puts)。
遇到 VIX 狂噴的黑天鵝日，產生 20~50 倍凸性爆發，注入龐大現金流。
"""

import logging

logger = logging.getLogger("ConvexityHedger")

class ConvexityHedger:
    def __init__(self, target_allocation: float = 0.015):
        """
        :param target_allocation: 撥出多少比例的 NAV 去買 Deep OTM Puts (預設 1.5%)
        """
        self.target_allocation = target_allocation
        self.current_hedge_value = 0.0
        self.hedge_cost_basis = 0.0

    def allocate_hedge(self, current_nav: float) -> float:
        """
        計算需要多少資金來建倉/維持 1.5% 的尾部避險部位。
        :return: 需扣除的現金 (Bleed)
        """
        target_hedge_value = current_nav * self.target_allocation
        if self.current_hedge_value < target_hedge_value:
            required_cash = target_hedge_value - self.current_hedge_value
            self.hedge_cost_basis += required_cash
            self.current_hedge_value = target_hedge_value
            logger.debug(f"Allocated {required_cash:,.2f} to Deep OTM Puts. Total hedge: {self.current_hedge_value:,.2f}")
            return required_cash
        return 0.0

    def process_market_crash(self, market_drop_percentage: float) -> float:
        """
        當大盤發生崩盤時，觸發凸性爆發 (Convexity Payoff)。
        回傳爆發後產生的現金流。
        market_drop_percentage 通常是負數 (e.g. -0.05 代表跌 5%)
        """
        drop = -market_drop_percentage if market_drop_percentage < 0 else 0.0
        
        if drop <= 0:
            # 平時緩步扣血 (時間價值流失)
            bleed = self.current_hedge_value * 0.05 # 每天流失 5% 權利金
            self.current_hedge_value -= bleed
            return 0.0
            
        # 假設跌幅超過 4%，啟動 20 倍到 50 倍的凸性爆發
        if drop > 0.04:
            multiplier = 20 + (drop - 0.04) * 1000 # 簡單的非線性乘數
            multiplier = min(50, multiplier)
            
            payoff = self.current_hedge_value * multiplier
            logger.warning(f"BLACK SWAN DETECTED! Market dropped {market_drop_percentage:.2%}. Convexity Payoff: {multiplier:.1f}x -> {payoff:,.2f} Cash Generated!")
            
            # 爆發後部位清空
            self.current_hedge_value = 0.0
            self.hedge_cost_basis = 0.0
            return payoff
            
        return 0.0
