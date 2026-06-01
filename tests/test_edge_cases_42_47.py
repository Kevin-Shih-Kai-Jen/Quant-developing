import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from nexus_quant_os.alpha_hunter.models import ScanResult, FinancialStatement, SignalStrength, AIAnalysis
from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator
from nexus_quant_os.alpha_hunter.tw_news_fetcher import TWNewsFetcher
from nexus_quant_os.alpha_hunter.ai_analyst import AIAnalyst
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall, FirewallConfig, RiskTier
from nexus_quant_os.risk_firewall.hmm_regime_detector import MarketRegimeDetector, RegimePrediction
from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODAnomalyDetector, AnomalyResult

def test_edge_case_43_capital_reduction():
    news = [
        {"title": "台積電現金減資", "content": "退還股款"},
        {"title": "陽明海運減資彌補虧損", "content": "慘"},
    ]
    filtered = TWNewsFetcher.filter_news(news)
    assert filtered[0]["risk_label"] == "CASH_CAPITAL_REDUCTION"
    assert filtered[0]["risk_level"] == "INFO"
    assert filtered[1]["risk_label"] == "RISK_CAPITAL_REDUCTION_DEFICIT"
    assert filtered[1]["risk_level"] == "CRITICAL"

@patch("nexus_quant_os.alpha_hunter.mops_scraper.MOPSScraper.get_cb_balance_and_premium")
def test_edge_case_44_cb_trap(mock_cb):
    mock_cb.return_value = (0.25, True) # premium > 20%, balance dropped
    gen = AlphaSignalGenerator()
    stmt = FinancialStatement(ticker="2330", fiscal_year=2026, fiscal_quarter=1)
    scan = ScanResult(ticker="2330", company_name="Test", latest_statement=stmt)
    
    # We mock out ai and supply chain just to test the CB Trap
    with patch.object(gen, "_check_technical", return_value=True):
        signal = gen._generate_from_scan(scan, include_supply_chain=False, include_ai_analysis=False)
    
    assert signal.signal_strength == SignalStrength.AVOID

def test_edge_case_45_margin_call_cascade():
    hmm = MagicMock()
    ood = MagicMock()
    hmm.predict.return_value = RegimePrediction(
        danger_probability=0.9, 
        bear_probability=0.9, 
        regime_label="CRASH",
        most_likely_regime=1,
        regime_probabilities=np.array([0.1, 0.9]),
        is_dangerous=True
    )
    ood.detect.return_value = AnomalyResult(
        is_ood=True, 
        ae_zscore=5.0, 
        isolation_forest_score=0.9,
        ae_reconstruction_mse=0.1,
        combined_anomaly_score=1.0
    )
    
    fw = IntelligentRiskFirewall(hmm, ood)
    fw._is_fitted["TW"] = True
    fw.hmm_detectors["TW"] = hmm
    fw.ood_detectors["TW"] = ood
    
    # Normally this would be EMERGENCY due to extreme HMM danger
    decision = fw.evaluate(np.array([[1]]), np.array([1.0]), market="TW", margin_ratio=1.25)
    # The Margin Call Cascade logic overrides EMERGENCY to CAUTION for buying the dip
    assert decision.risk_tier == RiskTier.CAUTION
    assert "MARGIN CALL CASCADE - BUY THE DIP" in decision.veto_reason

def test_edge_case_47_geopolitical_desensitization():
    analyst = AIAnalyst(api_key="TEST")
    with patch.object(analyst._session, "post") as mock_post:
        # Mock Gemini response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"ai_score": -0.8, "management_tone": -0.5}'}]}}]
        }
        mock_post.return_value = mock_resp
        
        # With normal news, should be negative
        analysis_normal = analyst.analyze("2330", recent_news=["市場需求疲弱"])
        assert analysis_normal.ai_score == -0.8
        
        # With geopolitical keyword, should be attenuated by 80% (i.e. * 0.2)
        analysis_geo = analyst.analyze("2331", recent_news=["共軍宣佈環台軍演"])
        assert analysis_geo.ai_score == pytest.approx(-0.16)
        assert analysis_geo.management_tone == pytest.approx(-0.1)

