"""
tests/test_tw_market_e2e.py — 台股模組端到端整合測試

驗證台股相關模組的資料流動是否正確，確保各模組交互作用無誤。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from nexus_quant_os.alpha_hunter.models import FinancialStatement
from nexus_quant_os.alpha_hunter.twse_client import TWSEClient
from nexus_quant_os.alpha_hunter.tw_news_fetcher import TWNewsFetcher
from nexus_quant_os.alpha_hunter.entity_resolver import EntityResolver
from nexus_quant_os.alpha_hunter.implied_earnings import ImpliedEarningsEstimator
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall, FirewallConfig
from nexus_quant_os.risk_firewall.hmm_regime_detector import MarketRegimeDetector, HMMConfig
from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODAnomalyDetector, OODConfig

@pytest.fixture
def mock_twse_client():
    client = TWSEClient()
    stmt = FinancialStatement(
        ticker="2330.TW",
        market="TW",
        currency="TWD",
        stock_id="2330",
        filing_date=datetime.now(timezone.utc),
        period_end=datetime.now(timezone.utc),
        fiscal_year=2024,
        fiscal_quarter=1,
        revenue=1000000.0,
        gross_profit=530000.0,
        operating_income=450000.0,
        gross_margin=0.53,
        operating_margin=0.45,
    )
    client.get_latest_financials = MagicMock(return_value=stmt)
    client.get_monthly_revenue = MagicMock(return_value=pd.DataFrame({
        "date": ["2024-04-01", "2024-05-01"],
        "revenue": [300000.0, 350000.0]
    }))
    return client

def test_tw_e2e_pipeline(mock_twse_client):
    # 1. 實體對齊
    ticker = EntityResolver.resolve_entity("台積電")
    assert ticker == "2330.TW"
    
    # 2. 獲取基本面財報
    stmt = mock_twse_client.get_latest_financials(ticker)
    assert stmt.ticker == "2330.TW"
    assert stmt.gross_margin == 0.53
    
    # 3. 獲取高頻月營收與隱含盈餘估計
    rev_df = mock_twse_client.get_monthly_revenue(ticker, months=2)
    recent_revs = rev_df["revenue"].tolist()
    
    # 假設最新流通股數為 25,930,000,000 股
    eps_est = ImpliedEarningsEstimator.estimate_current_quarter_eps(
        ticker=ticker,
        recent_monthly_revenue=recent_revs,
        last_quarter_gross_margin=stmt.gross_margin,
        last_quarter_operating_margin=stmt.operating_margin,
        outstanding_shares=25930000000
    )
    
    assert eps_est is not None
    assert eps_est > 0.0
    
    # 4. 新聞情緒 (Mock)
    with patch.object(TWNewsFetcher, "get_news", return_value=[{"title": "台積電營收大增"}]):
        news = TWNewsFetcher.get_news(ticker)
        assert len(news) == 1
        
    # 5. 風控防火牆獨立訓練
    firewall = IntelligentRiskFirewall(
        hmm_detector=MarketRegimeDetector(HMMConfig(n_regimes=2, n_iter=10)),
        ood_detector=OODAnomalyDetector(OODConfig(ae_epochs=1)),
        config=FirewallConfig(smooth_blend=False)
    )
    
    # Fake TW features
    tw_features = np.random.randn(50, 6)
    firewall.fit(tw_features, market="TW")
    
    assert firewall._is_fitted["TW"] is True
    
    weights = np.array([0.5, 0.5])
    decision = firewall.evaluate(tw_features, weights, market="TW")
    
    assert decision.adjusted_weights is not None
    assert len(decision.adjusted_weights) == 2
