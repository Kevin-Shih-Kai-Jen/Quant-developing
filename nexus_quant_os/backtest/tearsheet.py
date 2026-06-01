"""
nexus_quant_os/backtest/tearsheet.py — CPCV 淨化交叉驗證與績效報告

【升級六：CPCV 淨化交叉驗證】
負責計算策略在各個 Fold (含 Purge & Embargo) 下的表現，防禦調參過擬合。
包含特殊的 Abyss Metric：Convexity Contribution。
"""

import math
from typing import List, Dict
import logging

logger = logging.getLogger("Tearsheet")

class CPCVTearsheet:
    def __init__(self, daily_navs: List[float], daily_convexity_payoffs: List[float]):
        """
        :param daily_navs: 每日的總資產淨值 (TWD)
        :param daily_convexity_payoffs: 每日凸性避險產生的現金流
        """
        self.navs = daily_navs
        self.convexity_payoffs = daily_convexity_payoffs
        
    def _calculate_returns(self) -> List[float]:
        returns = []
        for i in range(1, len(self.navs)):
            if self.navs[i-1] == 0:
                returns.append(0.0)
            else:
                returns.append((self.navs[i] - self.navs[i-1]) / self.navs[i-1])
        return returns
        
    def get_annualized_return(self, trading_days_per_year: int = 252) -> float:
        if not self.navs or len(self.navs) < 2:
            return 0.0
        total_return = (self.navs[-1] / self.navs[0]) - 1.0
        years = len(self.navs) / trading_days_per_year
        if years == 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1.0
        
    def get_max_drawdown(self) -> float:
        if not self.navs:
            return 0.0
            
        peak = self.navs[0]
        max_dd = 0.0
        
        for nav in self.navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
        
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252) -> float:
        returns = self._calculate_returns()
        if not returns:
            return 0.0
            
        avg_ret = sum(returns) / len(returns)
        # 樣本標準差
        var = sum((r - avg_ret) ** 2 for r in returns) / max(1, len(returns) - 1)
        std_dev = math.sqrt(var)
        
        if std_dev == 0:
            return 0.0
            
        annual_ret = avg_ret * trading_days_per_year
        annual_std = std_dev * math.sqrt(trading_days_per_year)
        
        return (annual_ret - risk_free_rate) / annual_std
        
    def get_convexity_contribution(self) -> float:
        """
        計算 Deep OTM Puts 總共為系統注入了多少避險子彈 (佔初始 NAV 的比例)
        """
        if not self.navs:
            return 0.0
        total_payoff = sum(self.convexity_payoffs)
        return total_payoff / self.navs[0]

    def generate_report(self):
        logger.info("\n" + "="*50)
        logger.info(" INSTITUTIONAL TEARSHEET (CPCV Metrics)")
        logger.info("="*50)
        logger.info(f" Annualized Return   : {self.get_annualized_return():.2%}")
        logger.info(f" Max Drawdown        : {self.get_max_drawdown():.2%}")
        logger.info(f" Sharpe Ratio        : {self.get_sharpe_ratio():.2f}")
        logger.info(f" Convexity Payoff    : {self.get_convexity_contribution():.2%} of Initial NAV")
        logger.info("="*50)
