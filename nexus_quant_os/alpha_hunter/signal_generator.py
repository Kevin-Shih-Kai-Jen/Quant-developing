"""
alpha_hunter/signal_generator.py — 最終信號整合

整合 Tier 1-3 的結果，加上技術面確認，輸出 AlphaSignal。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .models import (
    AlphaSignal, SignalStrength, ScanResult, SupplyChainGraph, AIAnalysis,
)
from .financial_scanner import FinancialScanner
from .supply_chain import SupplyChainTracker
from .ai_analyst import AIAnalyst
from ._constants import (
    AI_BULLISH_THRESHOLD, TECHNICAL_MA_PERIOD, DEFAULT_SCAN_UNIVERSE,
)

logger = logging.getLogger(__name__)


class AlphaSignalGenerator:
    """Alpha 信號整合生成器。"""

    def __init__(
        self,
        scanner: Optional[FinancialScanner] = None,
        tracker: Optional[SupplyChainTracker] = None,
        analyst: Optional[AIAnalyst] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self._scanner = scanner or FinancialScanner()
        self._tracker = tracker or SupplyChainTracker(gemini_api_key=gemini_api_key)
        self._analyst = analyst or AIAnalyst(api_key=gemini_api_key)

    def get_supply_chain(self, ticker: str) -> SupplyChainGraph:
        """取得一家公司的供應鏈圖譜（公開 API）。"""
        return self._tracker.build_graph(ticker.upper())

    def generate_signals(
        self,
        universe: list[str] | None = None,
        include_supply_chain: bool = True,
        include_ai_analysis: bool = True,
        progress_callback=None,
    ) -> list[AlphaSignal]:
        """對整個投資範圍生成信號。"""
        if universe is not None:
            # Bug #4 fix: use a local scanner copy instead of mutating shared state
            original_universe = self._scanner._universe
            self._scanner._universe = universe
            
        try:
            scan_results = self._scanner.scan_universe()
        finally:
            # Bug #4 fix: always restore original universe
            if universe is not None:
                self._scanner._universe = original_universe
        candidates = [r for r in scan_results if r.is_candidate]
        
        if progress_callback:
            progress_callback(f"Phase 1 complete: found {len(candidates)} candidates from {len(scan_results)} stocks.")
            
        # 並行處理候選股（AI 分析 + 技術面確認）
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _process_candidate(scan):
            try:
                return self._generate_from_scan(
                    scan,
                    include_supply_chain=include_supply_chain,
                    include_ai_analysis=include_ai_analysis,
                )
            except Exception as e:
                logger.warning("Failed to generate signal for %s: %s", scan.ticker, e)
                return None

        signals = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(_process_candidate, c): c for c in candidates}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    signals.append(result)
                if progress_callback:
                    progress_callback(f"Processed {i}/{len(candidates)} candidates")

        signals.sort(key=lambda s: s.composite_score, reverse=True)
        self._save_signal_history(signals)
        if progress_callback:
            progress_callback("Analysis complete.")
        return signals

    def generate_single(self, ticker: str) -> AlphaSignal:
        """對單一公司生成信號。"""
        scan = self._scanner.scan_single(ticker)
        if scan is None:
            # Bug #5 fix: create a minimal ScanResult with a stub FinancialStatement
            # instead of passing None for a non-Optional field
            from .models import FinancialStatement
            stub_stmt = FinancialStatement(
                ticker=ticker, cik="", company_name=ticker,
                filing_date=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
                fiscal_year=datetime.now(timezone.utc).year,
                fiscal_quarter=1,
            )
            scan = ScanResult(ticker=ticker, company_name=ticker, latest_statement=stub_stmt)
            
        return self._generate_from_scan(scan, include_supply_chain=True, include_ai_analysis=True)
        
    def _generate_from_scan(
        self, 
        scan: ScanResult, 
        include_supply_chain: bool, 
        include_ai_analysis: bool
    ) -> AlphaSignal:
        ticker = scan.ticker
        
        # 取得文本（供 AI 分析和供應鏈用）
        filing_texts = {}
        if include_ai_analysis or include_supply_chain:
            try:
                from .ticker_resolver import TickerResolver
                from .data_client_factory import DataClientFactory
                market_config = TickerResolver.get_market_config(ticker)
                filing_type = market_config.filing_types.get("annual", "10-K")
                client = DataClientFactory.get_client(ticker)
                
                filing_texts = client.get_filing_text(
                    ticker, filing_type=filing_type, sections=["mda", "risk"]
                )
                # 外國公司沒有 10-K，嘗試 20-F
                if not any(filing_texts.values()) and filing_type == "10-K":
                    filing_texts = client.get_filing_text(
                        ticker, filing_type="20-F", sections=["mda", "risk"]
                    )
            except Exception as e:
                logger.warning("Failed to get filing text for %s: %s", ticker, e)
                
        graph = None
        if include_supply_chain:
            filing_full_text = " ".join(filing_texts.values()) if filing_texts else ""
            graph = self._tracker.build_graph(ticker, filing_text=filing_full_text)
            
        analysis = None
        if include_ai_analysis:
            analysis = self._analyst.analyze(
                ticker, 
                mda_text=filing_texts.get("mda", ""), 
                risk_text=filing_texts.get("risk", "")
            )
            
        technical_ok = self._check_technical(ticker)
        
        signal = AlphaSignal(
            ticker=ticker,
            company_name=scan.company_name,
            signal_strength=SignalStrength.NEUTRAL,
            scan_result=scan,
            supply_chain=graph,
            ai_analysis=analysis,
        )
        
        # 判定各項確認
        signal.fundamental_pass = scan.is_candidate
        
        if graph and len(graph.edges) > 0:
            healthy_edges = sum(1 for e in graph.edges if e.confidence >= 0.5)
            signal.supply_chain_healthy = (healthy_edges > 0)
        else:
            signal.supply_chain_healthy = False
            
        if analysis:
            signal.ai_bullish = (analysis.ai_score > AI_BULLISH_THRESHOLD)
            
        signal.technical_confirm = technical_ok
        
        # 評分
        signal.composite_score = self._compute_composite_score(scan, graph, analysis, technical_ok)
        signal.signal_strength = self._determine_strength(signal)
        
        # Edge Case #44: 可轉債 (CB) 養套殺防禦
        try:
            from .mops_scraper import MOPSScraper
            cb_scraper = MOPSScraper()
            premium, balance_dropped = cb_scraper.get_cb_balance_and_premium(ticker)
            if premium > 0.20 and balance_dropped:
                logger.warning("Toxic_Dilution_Risk triggered for %s: CB Premium > 20%% and balance dropping. Forcing AVOID.", ticker)
                signal.signal_strength = SignalStrength.AVOID
                signal.composite_score = max(0.0, signal.composite_score - 0.5)
        except Exception as e:
            logger.warning("Failed to check CB balance for %s: %s", ticker, e)
            
        # 決策三 (動態預期跨欄): Sell-the-News 防禦機制
        # 若事前情緒極度樂觀 (AI Score > 0.8 模擬 Z-Score > 1.5)
        if analysis and analysis.ai_score > 0.8:
            if scan.latest_statement:
                # 這裡檢查 EPS YoY 或 Implied EPS YoY
                eps_yoy = getattr(scan, "eps_yoy", 0.0)
                if getattr(scan, "is_estimate", False):
                    estimated_eps = getattr(scan, "estimated_eps", None)
                    last_year_eps = getattr(scan, "last_year_eps", None)
                    if estimated_eps is not None and last_year_eps is not None and last_year_eps > 0:
                        eps_yoy = (estimated_eps - last_year_eps) / last_year_eps
                    elif estimated_eps is not None and getattr(scan, "eps_yoy", None) is not None:
                        eps_yoy = scan.eps_yoy # Fallback to existing eps_yoy if last_year_eps is missing
                
                # 若未達 30% 則觸發否決
                if eps_yoy < 0.30:
                    logger.warning("Sell-the-News triggered for %s: Overheated sentiment (%.2f) but EPS YoY %.1f%% < 30%%", 
                                   ticker, analysis.ai_score, eps_yoy * 100)
                    signal.signal_strength = SignalStrength.AVOID
                    signal.fundamental_pass = False
                    
        # 決策四 (低基期地雷 / Low-Base Effect Anomaly)
        if scan.latest_statement:
            last_eps = getattr(scan, "last_year_eps", 1.0)
            if last_eps <= 0 and getattr(scan, "eps_yoy", 0) > 1.0:
                logger.info(f"Low-Base Effect Normalizer activated for {ticker}: Last year EPS <= 0. Normalizing AI output.")
                if analysis:
                    analysis.ai_score = min(analysis.ai_score, 0.3) # 限制過度樂觀
        
        # 決策五 (AI 綠洗照妖鏡 / Buzzword Washing Deception)
        if analysis and analysis.ai_score > 0.4:
            catalysts_text = " ".join(analysis.growth_catalysts).upper()
            if "AI" in catalysts_text or "高階" in catalysts_text or "成長" in catalysts_text:
                # 取得 CAPEX YoY (如果 FeatureStore 沒有，不懲罰)
                capex_yoy = getattr(scan, "capex_yoy", None) 
                if capex_yoy is not None and capex_yoy <= 0:
                    logger.warning(f"LIAR_PENALTY triggered for {ticker}: AI buzzwords detected but CAPEX YoY is {capex_yoy:.1%}. Forcing score to 0.")
                    analysis.ai_score = 0.0
                    signal.ai_bullish = False
                    signal.composite_score = 0.0
                    signal.signal_strength = SignalStrength.AVOID
        
        # 決策二 (籌碼絕對否決): Smart Money Veto
        # 在 technical_ok 回傳 False 時觸發，或另外寫邏輯
        try:
            is_above_20ma, is_smart_money_veto = self._check_smart_money_and_technical(ticker)
            signal.technical_confirm = is_above_20ma
            if is_smart_money_veto:
                logger.warning("Smart Money Veto triggered for %s: Institutional 3-day net sell + below 20MA", ticker)
                signal.signal_strength = SignalStrength.AVOID
                signal.composite_score = max(0.0, signal.composite_score - 0.5)
        except Exception as e:
            logger.warning("Failed to check Smart Money Veto for %s: %s", ticker, e)

        return signal

    def _check_smart_money_and_technical(self, ticker: str) -> tuple[bool, bool]:
        """技術面確認與籌碼否決：
        1. 價格是否在 200MA 之上。
        2. 是否觸發 Smart Money Veto (外資/投信連 3 日賣超且跌破 20MA)。
        Returns: (is_above_200ma, is_veto)
        """
        try:
            from .ticker_resolver import TickerResolver
            yf_ticker = TickerResolver.to_yfinance(ticker)
            df = yf.download(yf_ticker, period="1y", progress=False)
            if df.empty or len(df) < TECHNICAL_MA_PERIOD:
                return False, False
                
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            close_prices = df[price_col]
            
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
                
            ma200 = close_prices.rolling(TECHNICAL_MA_PERIOD).mean().iloc[-1]
            ma20 = close_prices.rolling(20).mean().iloc[-1]
            current = close_prices.iloc[-1]
            
            is_above_200ma = bool(current > ma200)
            is_below_20ma = bool(current < ma20)
            
            is_veto = False
            if is_below_20ma:
                from .twse_client import TWSEClient
                twse = TWSEClient()
                flow_df = twse.get_institutional_flow(ticker, days=3)
                if not flow_df.empty and len(flow_df) >= 3:
                    # 連續三日賣超
                    is_veto = (flow_df["buy_sell"] < 0).all()
                    
            return is_above_200ma, is_veto
        except Exception as e:
            logger.warning("Technical/Veto check failed for %s: %s", ticker, e)
            return False, False

    def _check_technical(self, ticker: str) -> bool:
        # 保留舊方法以相容其他模組
        is_above, _ = self._check_smart_money_and_technical(ticker)
        return is_above

    def _compute_composite_score(
        self,
        scan: Optional[ScanResult],
        graph: Optional[SupplyChainGraph],
        analysis: Optional[AIAnalysis],
        technical: bool,
    ) -> float:
        """計算綜合評分 [0, 1]。"""
        score = 0.0

        # Tier 1 (35%)
        if scan is not None:
            score += 0.35 * (scan.pass_count / 5.0)

        # Tier 2 (15%)
        if graph is not None and len(graph.edges) > 0:
            healthy = sum(1 for e in graph.edges if e.confidence >= 0.5)
            score += 0.15 * min(healthy / 5.0, 1.0)
        else:
            score += 0.15 * 0.5  # 無資料 → 中性

        # Tier 3 (30%)
        if analysis is not None:
            ai_01 = (analysis.ai_score + 1.0) / 2.0
            score += 0.30 * ai_01 * analysis.confidence

        # Tier 4 (20%)
        if technical:
            score += 0.20

        return float(np.clip(score, 0.0, 1.0))

    def _determine_strength(self, signal: AlphaSignal) -> SignalStrength:
        """判定信號強度。"""
        count = signal.confirmation_count
        if count >= 3:
            return SignalStrength.STRONG_BUY
        elif count >= 2:
            return SignalStrength.BUY
        elif count >= 1:
            return SignalStrength.NEUTRAL
        else:
            return SignalStrength.AVOID

    def _save_signal_history(self, signals: list[AlphaSignal]) -> None:
        """將生成的信號清單寫入歷史記錄。"""
        import json
        from ._constants import SIGNALS_HISTORY_DIR
        
        SIGNALS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = SIGNALS_HISTORY_DIR / f"signals_{timestamp}.json"
        
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total": len(signals),
                "signals": [s.to_dict() for s in signals]
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved signal history to %s", filepath)
        except Exception as e:
            logger.error("Failed to save signal history: %s", e)
