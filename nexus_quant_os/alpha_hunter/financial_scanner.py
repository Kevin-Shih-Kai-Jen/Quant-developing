"""
alpha_hunter/financial_scanner.py — 基本面篩選器

從 SECEdgarClient 取得的 FinancialStatement 執行 5 項篩選條件。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import FinancialStatement, ScanResult, ValuationRating
from .sec_edgar import SECEdgarClient
from ._constants import (
    MIN_REVENUE_GROWTH_YOY, MIN_FCF_YIELD, MAX_DEBT_TO_EQUITY,
    VALUATION_PEER_MULTIPLIER, DEFAULT_SCAN_UNIVERSE,
)

logger = logging.getLogger(__name__)


class FinancialScanner:
    """基本面掃描器。

    用法：
        scanner = FinancialScanner()
        results = scanner.scan_universe()
        candidates = [r for r in results if r.is_candidate]
    """

    def __init__(
        self,
        edgar_client: Optional[SECEdgarClient] = None,
        universe: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            edgar_client: SEC EDGAR 客戶端。不傳就自動建立。
            universe: 掃描範圍 ticker 列表。不傳用 DEFAULT_SCAN_UNIVERSE。
        """
        self._edgar = edgar_client or SECEdgarClient()
        self._universe = universe or list(DEFAULT_SCAN_UNIVERSE)

    def scan_universe(self) -> list[ScanResult]:
        """掃描整個投資範圍。"""
        results = []
        for ticker in self._universe:
            try:
                result = self.scan_single(ticker)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning("Failed to scan %s: %s", ticker, e)
        
        if not results:
            logger.error("All tickers failed during financial scan.")
            return []

        results.sort(key=lambda r: r.pass_count, reverse=True)
        return results

    def scan_single(self, ticker: str) -> Optional[ScanResult]:
        """掃描單一公司。"""
        # Bug #6 fix: fetch 2 quarters in one call to avoid redundant download in _evaluate
        history = self._edgar.get_financials_history(ticker, n_quarters=2)
        if not history:
            return None

        return self._evaluate(history[0], history)

    def _evaluate(self, stmt: FinancialStatement, history: list[FinancialStatement] | None = None) -> ScanResult:
        """根據 5 項條件評估一家公司。"""
        result = ScanResult(
            ticker=stmt.ticker,
            company_name=stmt.company_name,
            latest_statement=stmt,
        )

        # 條件 1: 營收年增率
        if stmt.revenue_yoy_growth is not None and stmt.revenue_yoy_growth > MIN_REVENUE_GROWTH_YOY:
            result.passes_revenue_growth = True

        # 條件 2: 毛利率改善 (比較當季 vs 上季)
        if history is not None and len(history) >= 2:
            prev_stmt = history[1]
            if stmt.gross_margin is not None and prev_stmt.gross_margin is not None:
                if stmt.gross_margin > prev_stmt.gross_margin:
                    result.passes_margin_expansion = True
        elif stmt.gross_margin is not None and stmt.gross_margin > 0.3:
            # Fallback 簡化版: 若無歷史可比但毛利率 > 30% 算過
            result.passes_margin_expansion = True

        # 條件 3: 估值合理
        pe = stmt.pe_ratio
        if pe is not None and pe > 0:
            if pe < 30:
                result.passes_valuation = True

        # 條件 4: 自由現金流健康
        if stmt.free_cash_flow is not None and stmt.free_cash_flow > 0:
            if stmt.fcf_yield is not None and stmt.fcf_yield > MIN_FCF_YIELD:
                result.passes_cash_flow = True
            elif stmt.fcf_yield is None:
                # 簡化版: 若無市值無法算 yield，FCF > 0 也算過
                result.passes_cash_flow = True

        # 條件 5: 負債健康
        if stmt.debt_to_equity is None:
            result.passes_debt_health = True
        elif stmt.debt_to_equity < MAX_DEBT_TO_EQUITY:
            result.passes_debt_health = True

        # 估值等級判定
        if pe is None or pe < 0:
            result.valuation_rating = ValuationRating.FAIR
        elif pe < 10:
            result.valuation_rating = ValuationRating.DEEPLY_UNDERVALUED
        elif pe < 18:
            result.valuation_rating = ValuationRating.UNDERVALUED
        elif pe < 28:
            result.valuation_rating = ValuationRating.FAIR
        elif pe < 45:
            result.valuation_rating = ValuationRating.OVERVALUED
        else:
            result.valuation_rating = ValuationRating.EXTREMELY_OVERVALUED

        return result
