import pytest
from datetime import datetime, timezone
from nexus_quant_os.alpha_hunter.models import FinancialStatement, EnrichedNode
from nexus_quant_os.alpha_hunter.interfaces import FinancialDataClient, MARKET_CONFIGS
from nexus_quant_os.alpha_hunter.ticker_resolver import TickerResolver
from nexus_quant_os.alpha_hunter.data_client_factory import DataClientFactory
from nexus_quant_os.alpha_hunter.sec_edgar import SECEdgarClient

def test_financial_statement_backwards_compatibility():
    # Should be able to construct with kwargs even if cik is now optional and moved
    stmt = FinancialStatement(ticker="NVDA", cik="0001045810", company_name="Nvidia",
                              filing_date=datetime.now(timezone.utc),
                              period_end=datetime.now(timezone.utc),
                              fiscal_year=2024, fiscal_quarter=1)
    assert stmt.ticker == "NVDA"
    assert stmt.cik == "0001045810"
    assert stmt.market == "US"
    assert stmt.currency == "USD"

def test_financial_statement_tw_market():
    stmt = FinancialStatement(ticker="2330.TW", market="TW", currency="TWD")
    assert stmt.ticker == "2330.TW"
    assert stmt.cik is None
    assert stmt.market == "TW"
    assert stmt.currency == "TWD"

def test_enriched_node_extensions():
    node = EnrichedNode(ticker="2330.TW", depth=0)
    assert node.market == "US"  # default
    assert node.high_freq_catalyst_score == 0.0
    
    node2 = EnrichedNode(ticker="2330.TW", depth=0, market="TW", high_freq_catalyst_score=2.5)
    assert node2.market == "TW"
    assert node2.high_freq_catalyst_score == 2.5

def test_ticker_resolver_detect_market():
    assert TickerResolver.detect_market("NVDA") == "US"
    assert TickerResolver.detect_market("2330.TW") == "TW"
    assert TickerResolver.detect_market("2330") == "TW"
    assert TickerResolver.detect_market("TSM") == "US"
    assert TickerResolver.detect_market("5871-KY") == "TW"

def test_ticker_resolver_to_canonical():
    assert TickerResolver.to_canonical("TSM") == "2330.TW"
    assert TickerResolver.to_canonical("NVDA") == "NVDA"
    assert TickerResolver.to_canonical("2330.TW") == "2330.TW"
    assert TickerResolver.to_canonical("2330") == "2330.TW"

def test_ticker_resolver_to_yfinance():
    assert TickerResolver.to_yfinance("NVDA") == "NVDA"
    assert TickerResolver.to_yfinance("2330.TW") == "2330.TW"
    assert TickerResolver.to_yfinance("2330") == "2330.TW"
    assert TickerResolver.to_yfinance("TSM") == "TSM"

def test_ticker_resolver_is_derivative():
    assert TickerResolver.is_derivative("034567") == True
    assert TickerResolver.is_derivative("08213") == True
    assert TickerResolver.is_derivative("2330") == False
    assert TickerResolver.is_derivative("NVDA") == False

def test_data_client_factory_protocol():
    client = DataClientFactory.get_client("NVDA")
    assert isinstance(client, SECEdgarClient)
    assert isinstance(client, FinancialDataClient)
    
    with pytest.raises(NotImplementedError):
        DataClientFactory.get_client("2330.TW")
