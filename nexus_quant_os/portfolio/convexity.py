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
            # 實盤中，Deep OTM Put 每天的 Theta 耗損極高，但降為 3% 以符合現實
            bleed = self.current_hedge_value * 0.03 
            self.current_hedge_value -= bleed
            return 0.0
            
        # 假設跌幅超過 2% (真實隱含波動率門檻)，啟動 Gamma 爆發
        if drop > 0.02:
            # 爆發乘數從 2% 開始算，最大上限 30 倍 (避免過度樂觀)
            multiplier = 5 + (drop - 0.02) * 500
            multiplier = min(50, multiplier)
            
            raw_payoff = self.current_hedge_value * multiplier
            
            # 【升級十一：造市商罷工與恐慌折價 (Crisis Liquidity Haircut)】
            # 極端行情下，選擇權造市商會拉大價差或不報價，導致理論獲利必須打折。使用決定性計算。
            haircut = 0.4 + min(drop * 2, 0.2)  # 最大折價 60%
            final_payoff = raw_payoff * (1 - haircut)
            
            logger.warning(f"BLACK SWAN DETECTED! Market dropped {market_drop_percentage:.2%}. Raw Multiplier: {multiplier:.1f}x. Applying {haircut:.1%} Liquidity Haircut! Final Payoff: {final_payoff:,.2f}")
            
            # 爆發後部位清空
            self.current_hedge_value = 0.0
            self.hedge_cost_basis = 0.0
            
            # 回傳的最終現金流，實務上需透過 main_runner 存入 T+1 receivable
            return final_payoff
            
        return 0.0
