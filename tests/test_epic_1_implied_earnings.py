import pytest
from nexus_quant_os.alpha_hunter.implied_earnings import ImpliedEarningsEstimator

def test_implied_eps_q4_seasonality():
    """
    Test Q4 seasonality conservative min margin logic.
    If Q3 margin is 20% and historical Q4 is 5%, it should use 5%.
    """
    # given
    ticker = "2330.TW"
    recent_monthly_revenue = [1000.0, 1000.0, 1000.0] # Total 3000
    last_quarter_net_margin = 0.20 # 20%
    outstanding_shares = 1000
    historical_q4_net_margins = [0.05, 0.05, 0.05] # Avg 5%

    # when
    est_eps = ImpliedEarningsEstimator.estimate_current_quarter_eps(
        ticker=ticker,
        recent_monthly_revenue=recent_monthly_revenue,
        last_quarter_net_margin=last_quarter_net_margin,
        outstanding_shares=outstanding_shares,
        is_q4=True,
        historical_q4_net_margins=historical_q4_net_margins
    )

    # then
    # expected revenue = 3000
    # conservative margin = min(0.20, 0.05) = 0.05
    # net income = 3000 * 0.05 = 150
    # eps = 150 / 1000 = 0.15
    assert est_eps == pytest.approx(0.15)

def test_implied_eps_normal():
    """Test normal non-Q4 estimation."""
    ticker = "2330.TW"
    recent_monthly_revenue = [1000.0, 1000.0] # Total 2000, 2 months -> proj 3000
    last_quarter_net_margin = 0.20
    outstanding_shares = 1000
    
    est_eps = ImpliedEarningsEstimator.estimate_current_quarter_eps(
        ticker=ticker,
        recent_monthly_revenue=recent_monthly_revenue,
        last_quarter_net_margin=last_quarter_net_margin,
        outstanding_shares=outstanding_shares,
        is_q4=False
    )
    
    # expected revenue = 3000
    # margin = 0.20
    # net income = 600
    # eps = 0.60
    assert est_eps == pytest.approx(0.60)
