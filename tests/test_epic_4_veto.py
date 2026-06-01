import pytest
import pandas as pd
from datetime import datetime, timezone
from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator
from nexus_quant_os.alpha_hunter.models import ScanResult, FinancialStatement, AIAnalysis, SignalStrength

def test_sell_the_news_veto():
    """Test Decision 3: Sell-the-News triggers 30% hurdle"""
    generator = AlphaSignalGenerator()
    
    # Mock data
    stmt = FinancialStatement(
        ticker="2330", cik="", company_name="TSMC",
        filing_date=datetime.now(timezone.utc), period_end=datetime.now(timezone.utc),
        fiscal_year=2024, fiscal_quarter=1
    )
    scan = ScanResult(ticker="2330", company_name="TSMC", latest_statement=stmt, eps_yoy=0.25)
    scan.passes_revenue_growth = True
    scan.passes_valuation = True
    scan.passes_margin_expansion = True
    
    # Test 1: Normal sentiment, shouldn't trigger veto
    # Mock analysis, supply chain tracker, technical check
    generator._analyst.analyze = lambda *args, **kwargs: AIAnalysis(ticker="2330", ai_score=0.5, confidence=0.8)
    generator._tracker.build_graph = lambda *args, **kwargs: None
    generator._check_smart_money_and_technical = lambda *args, **kwargs: (True, False)
    
    signal_normal = generator._generate_from_scan(scan, include_supply_chain=False, include_ai_analysis=True)
    
    # Test 2: Overheated sentiment, SHOULD trigger veto
    generator._analyst.analyze = lambda *args, **kwargs: AIAnalysis(ticker="2330", ai_score=0.9, confidence=0.8)
    signal_overheated = generator._generate_from_scan(scan, include_supply_chain=False, include_ai_analysis=True)
    
    # normal passes fundamental, overheated gets vetoed
    assert signal_normal.fundamental_pass is True
    assert signal_normal.signal_strength != SignalStrength.AVOID
    
    assert signal_overheated.fundamental_pass is False
    assert signal_overheated.signal_strength == SignalStrength.AVOID


def test_smart_money_veto():
    """Test Decision 2: Smart money veto triggers"""
    generator = AlphaSignalGenerator()
    
    stmt = FinancialStatement(
        ticker="2330", cik="", company_name="TSMC",
        filing_date=datetime.now(timezone.utc), period_end=datetime.now(timezone.utc),
        fiscal_year=2024, fiscal_quarter=1
    )
    scan = ScanResult(ticker="2330", company_name="TSMC", latest_statement=stmt, eps_yoy=0.50)
    scan.passes_revenue_growth = True
    scan.passes_valuation = True
    scan.passes_margin_expansion = True
    
    generator._analyst.analyze = lambda *args, **kwargs: AIAnalysis(ticker="2330", ai_score=0.5, confidence=0.8)
    generator._tracker.build_graph = lambda *args, **kwargs: None
    
    # Mock veto true
    generator._check_smart_money_and_technical = lambda *args, **kwargs: (False, True)
    
    signal = generator._generate_from_scan(scan, include_supply_chain=False, include_ai_analysis=True)
    
    assert signal.signal_strength == SignalStrength.AVOID
    assert signal.technical_confirm is False
