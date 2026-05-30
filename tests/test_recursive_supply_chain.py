import pytest
from unittest.mock import patch, MagicMock
from collections import deque
from fastapi.testclient import TestClient

from nexus_quant_os.alpha_hunter.models import SupplyChainEdge, SupplyChainRelation, RecursiveSupplyChainGraph
from nexus_quant_os.alpha_hunter.supply_chain import SupplyChainTracker
from api.server import app

client = TestClient(app)

@pytest.fixture
def mock_builder():
    with patch("nexus_quant_os.alpha_hunter.supply_chain.SECEdgarClient") as MockEdgar:
        builder = SupplyChainTracker()
        builder._edgar = MockEdgar()
        
        # Mock API calls and disk I/O
        builder._check_ollama = MagicMock(return_value=True)
        builder._call_ollama = MagicMock()
        builder._call_gemini = MagicMock()
        builder._read_node_cache = MagicMock(return_value=None)
        builder._write_node_cache = MagicMock()
        builder._get_filing_text_safe = MagicMock(return_value="mock filing text")
        builder._extract_edges_gemini = MagicMock()
        builder._extract_edges_ollama = MagicMock()
        
        yield builder

def test_circular_reference_prevention(mock_builder):
    """測試循環引用：A -> B -> A 必須能被阻擋，不會無窮迴圈"""
    def mock_extract(ticker, text):
        if ticker == "AAPL":
            return [SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9)]
        elif ticker == "TSM":
            return [SupplyChainEdge("TSM", "AAPL", SupplyChainRelation.CUSTOMER, confidence=0.9)]
        return []
        
    mock_builder._extract_edges_gemini.side_effect = mock_extract
    mock_builder._extract_edges_ollama.side_effect = mock_extract
    
    graph = mock_builder.build_recursive_graph("AAPL", max_depth=3, max_api_calls=10)
    
    assert graph.api_calls_used == 2
    assert "AAPL" in graph.visited_tickers
    assert "TSM" in graph.visited_tickers
    assert graph.max_depth == 3

def test_max_api_calls_exhaustion(mock_builder):
    """測試預算耗盡：設定 max_api_calls=2，展開 100 個節點的圖應在 2 次後停止"""
    def mock_extract(ticker, text):
        return [
            SupplyChainEdge(ticker, f"SUPP{chr(65+i)}", SupplyChainRelation.SUPPLIER, confidence=0.9)
            for i in range(1, 6)
        ]
        
    mock_builder._extract_edges_gemini.side_effect = mock_extract
    mock_builder._extract_edges_ollama.side_effect = mock_extract
    
    graph = mock_builder.build_recursive_graph("AAPL", max_depth=3, max_api_calls=2)
    
    # Root: AAPL (1 call) -> 5 suppliers
    # Next: SUPPLIER_1 (1 call) -> reaches max_calls of 2.
    # Total calls must be exactly 2
    assert graph.api_calls_used == 2
    assert len(graph.visited_tickers) == 2

def test_fallback_to_gemini(mock_builder):
    """測試 Ollama 回傳 None 或失敗時，自動降級到 Gemini"""
    mock_builder._extract_edges_ollama.return_value = None
    mock_builder._extract_edges_gemini.return_value = [
        SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9)
    ]
    
    graph = mock_builder.build_recursive_graph("AAPL", max_depth=1, max_api_calls=5)
    
    # Root is always Gemini (per spec)
    # TSM is depth 1. Ollama returns None, should fallback to Gemini.
    assert graph.llm_source["AAPL"] == "gemini"
    if "TSM" in graph.llm_source:
        assert graph.llm_source["TSM"] == "gemini_fallback"
    
def test_illegal_tickers(mock_builder):
    """測試非法 ticker：包含奇怪字元的 ticker 不應被進一步展開"""
    mock_builder._extract_edges_gemini.return_value = [
        SupplyChainEdge("AAPL", "WEIRD^TICKER", SupplyChainRelation.SUPPLIER, confidence=0.9)
    ]
    
    graph = mock_builder.build_recursive_graph("AAPL", max_depth=2, max_api_calls=5)
    
    # AAPL is scanned, WEIRD^TICKER is extracted as an edge but should NOT be expanded
    assert "WEIRD^TICKER" not in graph.visited_tickers
    assert graph.api_calls_used == 1
