"""
alpha_hunter/financial_scanner.py — 基本面篩選器

從 SECEdgarClient 取得的 FinancialStatement 執行 5 項篩選條件。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import FinancialStatement, ScanResult, ValuationRating
from .interfaces import FinancialDataClient, MarketConfig, MARKET_CONFIGS
from .ticker_resolver import TickerResolver
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
        data_client: Optional[FinancialDataClient] = None,
        universe: Optional[list[str]] = None,
        market_config: Optional[MarketConfig] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            data_client: 市場資料客戶端。不傳就自動建立。
            universe: 掃描範圍 ticker 列表。不傳用 DEFAULT_SCAN_UNIVERSE。
        """
        if "edgar_client" in kwargs and data_client is None:
            data_client = kwargs["edgar_client"]

        if data_client is not None:
            self._client = data_client
        else:
            # 向後相容：不傳就用美股 SECEdgarClient
            from .sec_edgar import SECEdgarClient
            self._client = SECEdgarClient()
        self._market_config = market_config or MARKET_CONFIGS["US"]
        self._universe = universe or list(DEFAULT_SCAN_UNIVERSE)

    def scan_universe(self) -> list[ScanResult]:
        """掃描整個投資範圍（並行化）。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        def _scan_one(ticker: str) -> Optional[ScanResult]:
            try:
                return self.scan_single(ticker)
            except Exception as e:
                logger.warning("Failed to scan %s: %s", ticker, e)
                return None

        # SEC rate limit = 10 req/s，用 5 個 worker 搭配 0.12s 間隔是安全的
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_scan_one, t): t for t in self._universe}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        if not results:
            logger.error("All tickers failed during financial scan.")
            return []

        results.sort(key=lambda r: r.pass_count, reverse=True)
        return results

    def scan_single(self, ticker: str) -> Optional[ScanResult]:
        """掃描單一公司。"""
        history = self._client.get_financials_history(ticker, n_quarters=5)
        if not history:
            return None

        config = TickerResolver.get_market_config(ticker)
        return self._evaluate(history[0], history, config)

    def _evaluate(self, stmt: FinancialStatement, history: list[FinancialStatement] | None = None, config: Optional[MarketConfig] = None) -> ScanResult:
        """根據 5 項條件評估一家公司。"""
        if config is None:
            config = getattr(self, '_market_config', MARKET_CONFIGS["US"])

        eps_yoy = None
        if history and len(history) >= 5:
            yoy_stmt = history[4]
            if stmt.eps_diluted is not None and yoy_stmt.eps_diluted is not None and yoy_stmt.eps_diluted > 0:
                eps_yoy = (stmt.eps_diluted - yoy_stmt.eps_diluted) / yoy_stmt.eps_diluted

        result = ScanResult(
            ticker=stmt.ticker,
            company_name=stmt.company_name,
            latest_statement=stmt,
            eps_yoy=eps_yoy if eps_yoy is not None else 0.0
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
            if pe < config.valuation_pe_cap:
                result.passes_valuation = True

        # 條件 4: 自由現金流健康
        if stmt.free_cash_flow is not None and stmt.free_cash_flow > 0:
            if stmt.fcf_yield is not None and stmt.fcf_yield > MIN_FCF_YIELD:
                result.passes_cash_flow = True
            elif stmt.fcf_yield is None:
                # 簡化版: 若無市值無法算 yield，FCF > 0 也算過
                result.passes_cash_flow = True

        # 條件 5: 負債健康
        sector = getattr(stmt, 'sector', None)
        if sector and sector in config.debt_exempt_sectors:
            result.passes_debt_health = True
        elif stmt.debt_to_equity is None:
            result.passes_debt_health = True
        elif stmt.debt_to_equity < MAX_DEBT_TO_EQUITY:
            result.passes_debt_health = True

        # 估值等級判定
        tiers = config.valuation_tiers
        if pe is None or pe < 0:
            result.valuation_rating = ValuationRating.FAIR
        elif pe < tiers.get("DEEPLY_UNDERVALUED", 10):
            result.valuation_rating = ValuationRating.DEEPLY_UNDERVALUED
        elif pe < tiers.get("UNDERVALUED", 18):
            result.valuation_rating = ValuationRating.UNDERVALUED
        elif pe < tiers.get("FAIR", 28):
            result.valuation_rating = ValuationRating.FAIR
        elif pe < tiers.get("OVERVALUED", 45):
            result.valuation_rating = ValuationRating.OVERVALUED
        else:
            result.valuation_rating = ValuationRating.EXTREMELY_OVERVALUED

        # ── Epic 1: 本業純度檢驗 (Core Purity) ──
        if history and len(history) >= 5:
            # YoY 比較: 取 4 季前的 statement
            yoy_stmt = history[4]
            
            eps_yoy = None
            if stmt.eps_diluted is not None and yoy_stmt.eps_diluted is not None and yoy_stmt.eps_diluted > 0:
                eps_yoy = (stmt.eps_diluted - yoy_stmt.eps_diluted) / yoy_stmt.eps_diluted
                result.eps_yoy = eps_yoy  # Update result
                
            op_inc_yoy = None
            if stmt.operating_income is not None and yoy_stmt.operating_income is not None and yoy_stmt.operating_income > 0:
                op_inc_yoy = (stmt.operating_income - yoy_stmt.operating_income) / yoy_stmt.operating_income

            if eps_yoy is not None and op_inc_yoy is not None:
                if eps_yoy < 0 and op_inc_yoy > 0:
                    result.flags.append("FX_Hidden_Gem")
                elif eps_yoy > 0 and op_inc_yoy < 0:
                    result.flags.append("Earnings_Quality_Discount")

        # ── Epic 1: Q4 隱含季盈餘推估 ──
        from .implied_earnings import ImpliedEarningsEstimator
        if config.market == "TW" and getattr(self._client, "get_monthly_revenue", None):
            current_month = datetime.now(timezone.utc).month
            # 在 1~3 月期間，若最新財報為 Q3，則推估 Q4
            if current_month in (1, 2, 3) and stmt.fiscal_quarter == 3:
                try:
                    df_rev = self._client.get_monthly_revenue(ticker, months=3)
                    if not df_rev.empty:
                        recent_revs = df_rev["revenue"].tolist()
                        
                        hist_q4_margins = []
                        if history:
                            for h in history:
                                if h.fiscal_quarter == 4 and h.net_income is not None and h.revenue is not None and h.revenue > 0:
                                    hist_q4_margins.append(h.net_income / h.revenue)
                                if len(hist_q4_margins) >= 3:
                                    break
                                    
                        last_q_net_margin = stmt.net_income / stmt.revenue if stmt.revenue and stmt.net_income else None
                        # 台股股本 = total_equity / 10 只是粗略估算，實際最好使用 outstanding_shares
                        out_shares = stmt.total_equity / 10 if stmt.total_equity else None
                        
                        if last_q_net_margin is not None and out_shares is not None:
                            est_eps = ImpliedEarningsEstimator.estimate_current_quarter_eps(
                                ticker=ticker,
                                recent_monthly_revenue=recent_revs,
                                last_quarter_net_margin=last_q_net_margin,
                                outstanding_shares=int(out_shares),
                                is_q4=True,
                                historical_q4_net_margins=hist_q4_margins
                            )
                            if est_eps is not None:
                                result.estimated_eps = est_eps
                                result.is_estimate = True
                                logger.info("[%s] Implied Q4 EPS estimated: %.4f", ticker, est_eps)
                except Exception as e:
                    logger.warning("Failed to estimate Q4 EPS for %s: %s", ticker, e)

        return result
