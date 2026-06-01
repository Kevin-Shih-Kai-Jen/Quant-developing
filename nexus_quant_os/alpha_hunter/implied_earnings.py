"""
alpha_hunter/implied_earnings.py — 隱含季盈餘推估模組

利用台股高頻月營收數據推估季報 EPS，獲取財報空窗期的 Alpha。
防禦 Edge Cases:
    - B12: 股本膨脹陷阱。除權息或可轉債轉換會改變在外流通股數，
      不可直接沿用上季股本計算 EPS。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ImpliedEarningsEstimator:
    """隱含季盈餘推估器。"""

    @classmethod
    def estimate_current_quarter_eps(
        cls,
        ticker: str,
        recent_monthly_revenue: list[float],
        last_quarter_net_margin: float,
        outstanding_shares: int,
        is_q4: bool = False,
        historical_q4_net_margins: list[float] = None,
    ) -> Optional[float]:
        """推估當季 EPS。

        Args:
            ticker: 公司代碼
            recent_monthly_revenue: 當季已公布的月營收列表 (長度 1~3)
            last_quarter_net_margin: 上季淨利率
            outstanding_shares: 最新在外流通股數 (防禦股本膨脹陷阱)
            is_q4: 是否為推估 Q4 (1~3月空窗期)
            historical_q4_net_margins: 過去三年 Q4 淨利率列表，若為推估 Q4 則需提供

        Returns:
            推估的 EPS，若資料不足回傳 None
        """
        if not recent_monthly_revenue or outstanding_shares <= 0:
            return None

        # 推估整季營收：已公布營收 + (剩餘月份 * 過去三個月平均)
        months_reported = len(recent_monthly_revenue)
        if months_reported >= 3:
            estimated_quarter_rev = sum(recent_monthly_revenue[:3])
        else:
            avg_rev = sum(recent_monthly_revenue) / months_reported
            estimated_quarter_rev = sum(recent_monthly_revenue) + avg_rev * (
                3 - months_reported
            )

        # 決定淨利率保守估值
        conservative_net_margin = last_quarter_net_margin
        if is_q4 and historical_q4_net_margins:
            avg_q4_margin = sum(historical_q4_net_margins) / len(historical_q4_net_margins)
            conservative_net_margin = min(last_quarter_net_margin, avg_q4_margin)
            logger.info("[%s] Q4 EPS estimate uses conservative margin: %.4f (Q3: %.4f, HistQ4: %.4f)",
                        ticker, conservative_net_margin, last_quarter_net_margin, avg_q4_margin)

        # 簡單推估稅後淨利
        estimated_net_income = estimated_quarter_rev * conservative_net_margin

        # 防禦 B12：必須使用最新 outstanding_shares
        estimated_eps = estimated_net_income / outstanding_shares

        return estimated_eps
