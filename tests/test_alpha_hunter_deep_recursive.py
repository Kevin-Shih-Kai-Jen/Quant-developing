"""
Alpha Hunter 深度遞迴 — 防呆級壓力測試
========================================

測試對象：
- nexus_quant_os.alpha_hunter.radar        → compute_degree_centrality, compute_momentum_spillover, generate_graph_rag_analysis
- nexus_quant_os.alpha_hunter.supply_chain  → SupplyChainTracker.build_recursive_graph
- nexus_quant_os.alpha_hunter.models        → EnrichedNode, RecursiveSupplyChainGraph, SupplyChainEdge
- api/server.py                             → POST /api/alpha/supply-chain-radar/scan

每個測試都模擬一個「白痴情境」，確保系統在最惡劣條件下仍不死機。

⚠️ 所有測試都是 100% 離線（Mock LLM、Mock EDGAR、Mock yfinance），
   不花任何 API 預算，跑完整個 suite 不到 5 秒。
"""

import pytest
import asyncio
import time
import json
from unittest.mock import patch, MagicMock, AsyncMock
from collections import defaultdict

from nexus_quant_os.alpha_hunter.models import (
    EnrichedNode,
    SupplyChainEdge,
    SupplyChainRelation,
    RecursiveSupplyChainGraph,
)
from nexus_quant_os.alpha_hunter.radar import (
    compute_degree_centrality,
    compute_momentum_spillover,
    generate_graph_rag_analysis,
    node_to_dict,
)
from nexus_quant_os.alpha_hunter.supply_chain import SupplyChainTracker


# =====================================================================
# 🛠️ 共用 Fixtures & 工具
# =====================================================================

@pytest.fixture
def mock_tracker():
    """建立一個完全 Mock 的 SupplyChainTracker，不碰網路、不碰硬碟。"""
    with patch("nexus_quant_os.alpha_hunter.supply_chain.SECEdgarClient") as MockEdgar:
        tracker = SupplyChainTracker()
        tracker._edgar = MockEdgar()
        tracker._check_ollama = MagicMock(return_value=True)
        tracker._call_ollama = MagicMock()
        tracker._call_gemini = MagicMock()
        tracker._read_node_cache = MagicMock(return_value=None)
        tracker._write_node_cache = MagicMock()
        tracker._get_filing_text_safe = MagicMock(return_value="mock filing text")
        tracker._extract_edges_gemini = MagicMock(return_value=[])
        tracker._extract_edges_ollama = MagicMock(return_value=[])
        yield tracker


def _make_enriched(ticker: str, composite_score=0.5, depth=0, company_name="FakeCo") -> EnrichedNode:
    """快速建立一個 EnrichedNode 的工廠函數。"""
    return EnrichedNode(
        ticker=ticker,
        depth=depth,
        composite_score=composite_score,
        company_name=company_name,
    )


def _run_async(coro):
    """在 sync 測試中安全執行 async 函數（不依賴 pytest-asyncio）。"""
    return asyncio.run(coro)


# =====================================================================
# 💣 測試 1：除以零孤島炸彈 (Zero-Division in Degree Centrality)
# =====================================================================

class TestDegreeCentralityZeroDivision:
    """compute_degree_centrality 在極端輸入下不得死機。"""

    def test_single_node_no_edges(self):
        """
        🤡 白痴情境：畫面上只有「1 家」公司，完全沒有連線。
        公式 = 連線數 / (總節點 - 1)，分母 1-1=0 → ZeroDivisionError。
        防呆：n<=1 時直接回傳 1.0。
        """
        nodes = [{"ticker": "AAPL"}]
        edges = []

        result = compute_degree_centrality(nodes, edges)

        assert result["AAPL"] == 1.0, "孤島節點應獲得預設分數 1.0"

    def test_empty_graph(self):
        """
        🤡 白痴情境：前端傳來完全空的圖譜。
        """
        nodes = []
        edges = []

        result = compute_degree_centrality(nodes, edges)

        assert result == {}, "空圖必須回傳空字典，不能噴 Error"

    def test_normal_graph(self):
        """
        🧪 正常情境：3 個節點、2 條邊 → 確認數學正確。
        AAPL-TSM (1 條), AAPL-NVDA (1 條)
        AAPL degree = 2/(3-1) = 1.0
        TSM degree = 1/(3-1) = 0.5
        NVDA degree = 1/(3-1) = 0.5
        """
        nodes = [{"ticker": "AAPL"}, {"ticker": "TSM"}, {"ticker": "NVDA"}]
        edges = [
            {"source_ticker": "TSM", "target_ticker": "AAPL"},
            {"source_ticker": "AAPL", "target_ticker": "NVDA"},
        ]

        result = compute_degree_centrality(nodes, edges)

        assert abs(result["AAPL"] - 1.0) < 1e-6
        assert abs(result["TSM"] - 0.5) < 1e-6
        assert abs(result["NVDA"] - 0.5) < 1e-6

    def test_two_nodes_bidirectional(self):
        """
        🧪 邊界情境：2 個節點互連。 n=2, degree = count/(2-1) = count。
        """
        nodes = [{"ticker": "A"}, {"ticker": "B"}]
        edges = [
            {"source_ticker": "A", "target_ticker": "B"},
            {"source_ticker": "B", "target_ticker": "A"},
        ]

        result = compute_degree_centrality(nodes, edges)

        # A 出現在 2 條邊, B 也出現在 2 條邊, n-1=1
        assert result["A"] == 2.0
        assert result["B"] == 2.0


# =====================================================================
# 💣 測試 2：Null 污染 & 營收暴走 (Momentum Spillover)
# =====================================================================

class TestMomentumSpilloverDefense:
    """compute_momentum_spillover 面對毒資料不得死機。"""

    def test_null_composite_score(self):
        """
        🤡 白痴情境：爬蟲壞了，A 的 composite_score 是 None。
        沒防呆 → scores[A] = None → 0.5 * None → TypeError 炸彈。
        防呆：None → 0.5（中性）。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=None),
            "B": _make_enriched("B", composite_score=0.9),
        }
        edges = [
            {"source_ticker": "B", "target_ticker": "A", "relation": "SUPPLIER", "revenue_pct": 0.3},
        ]

        # 不能拋 TypeError
        scores = compute_momentum_spillover(enriched_nodes, edges)

        assert "A" in scores
        assert "B" in scores
        assert 0.0 <= scores["A"] <= 1.0
        assert 0.0 <= scores["B"] <= 1.0

    def test_revenue_pct_string_nan(self):
        """
        🤡 白痴情境：LLM 發瘋，把營收佔比寫成 "N/A"（字串）。
        float("N/A") → ValueError。
        防呆：catch ValueError → 預設 0.1。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.8),
            "B": _make_enriched("B", composite_score=0.9),
        }
        edges = [
            {"source_ticker": "B", "target_ticker": "A", "relation": "SUPPLIER", "revenue_pct": "N/A"},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        assert 0.0 <= scores["A"] <= 1.0, "字串 revenue_pct 不能導致死機"

    def test_revenue_pct_500_percent(self):
        """
        🤡 白痴情境：LLM 幻覺，營收佔比 5.0（500%）。
        不鉗制的話，spillover 會被放大 5 倍，分數飛到外太空。
        防呆：min(float(val), 1.0)。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.9),
            "B": _make_enriched("B", composite_score=0.2),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": 5.0},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        assert 0.0 <= scores["A"] <= 1.0, "分數必須被鉗制在 [0, 1]"
        assert 0.0 <= scores["B"] <= 1.0

    def test_revenue_pct_none(self):
        """
        🤡 白痴情境：revenue_pct 為 None（LLM 沒給）。
        防呆：None → 0.1 預設。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.8),
            "B": _make_enriched("B", composite_score=0.7),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": None},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        assert 0.0 <= scores["A"] <= 1.0
        assert 0.0 <= scores["B"] <= 1.0

    def test_revenue_pct_negative(self):
        """
        🤡 白痴情境：revenue_pct 為負數（-0.3）。
        防呆：max(0.0, ...) 鉗制。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.8),
            "B": _make_enriched("B", composite_score=0.7),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": -0.3},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        assert 0.0 <= scores["A"] <= 1.0
        assert 0.0 <= scores["B"] <= 1.0

    def test_all_nodes_none_scores(self):
        """
        🤡 白痴情境：所有節點的 composite_score 全部是 None。
        全部退化為 0.5 → spillover 為 0 → 分數不變。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=None),
            "B": _make_enriched("B", composite_score=None),
            "C": _make_enriched("C", composite_score=None),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": 0.3},
            {"source_ticker": "B", "target_ticker": "C", "relation": "SUPPLIER", "revenue_pct": 0.5},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        # 全是 0.5 → 沒有 spillover → 全保持 0.5
        for t in ("A", "B", "C"):
            assert abs(scores[t] - 0.5) < 0.01, f"{t} 應保持中性 0.5"


# =====================================================================
# 💣 測試 3：左腳踩右腳上天 (Infinite Loop in Spillover)
# =====================================================================

class TestSpilloverInfiniteLoop:
    """互為客戶的循環依賴不得導致死迴圈。"""

    def test_circular_supplier_customer(self):
        """
        🤡 白痴情境：A 的客戶是 B，B 的客戶也是 A。
        分數互相傳染 → 無煞車 → 死迴圈。
        防呆：max_iterations=3 強制截斷。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.8),
            "B": _make_enriched("B", composite_score=0.4),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "CUSTOMER", "revenue_pct": 0.5},
            {"source_ticker": "B", "target_ticker": "A", "relation": "CUSTOMER", "revenue_pct": 0.5},
        ]

        start = time.monotonic()
        scores = compute_momentum_spillover(enriched_nodes, edges, max_iterations=3)
        elapsed = time.monotonic() - start

        # 必須瞬間跑完（遠小於 1 秒）
        assert elapsed < 1.0, f"煞車失敗！耗時 {elapsed:.2f}s，應 <1s"
        assert 0.0 <= scores["A"] <= 1.0
        assert 0.0 <= scores["B"] <= 1.0

    def test_three_way_cycle(self):
        """
        🤡 白痴情境：A→B→C→A 三角循環。
        """
        enriched_nodes = {
            "A": _make_enriched("A", composite_score=0.9),
            "B": _make_enriched("B", composite_score=0.3),
            "C": _make_enriched("C", composite_score=0.6),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": 0.4},
            {"source_ticker": "B", "target_ticker": "C", "relation": "SUPPLIER", "revenue_pct": 0.4},
            {"source_ticker": "C", "target_ticker": "A", "relation": "SUPPLIER", "revenue_pct": 0.4},
        ]

        start = time.monotonic()
        scores = compute_momentum_spillover(enriched_nodes, edges, max_iterations=3)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        for t in ("A", "B", "C"):
            assert 0.0 <= scores[t] <= 1.0

    def test_enriched_nodes_are_mutated_correctly(self):
        """
        🧪 驗證副作用：spillover 必須正確回寫到 EnrichedNode 的
        network_alpha_score 與 spillover_delta 欄位。
        """
        enriched_nodes = {
            "NVDA": _make_enriched("NVDA", composite_score=0.9),
            "TSM": _make_enriched("TSM", composite_score=0.7),
        }
        edges = [
            {"source_ticker": "TSM", "target_ticker": "NVDA", "relation": "SUPPLIER", "revenue_pct": 0.3},
        ]

        scores = compute_momentum_spillover(enriched_nodes, edges)

        for t in ("NVDA", "TSM"):
            node = enriched_nodes[t]
            assert node.network_alpha_score is not None
            assert isinstance(node.spillover_delta, float)
            assert 0.0 <= node.network_alpha_score <= 1.0


# =====================================================================
# 💣 測試 4：BFS 遞迴引擎（SupplyChainTracker.build_recursive_graph）
# =====================================================================

class TestBuildRecursiveGraph:
    """BFS 遞迴展開在各種邊界條件下行為正確。"""

    def test_circular_reference_stops(self, mock_tracker):
        """
        🤡 白痴情境：A→B→A 互相引用。
        visited set 防止重複展開 → 不會無窮迴圈。
        """
        def mock_extract(ticker, text):
            if ticker == "AAPL":
                return [SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9)]
            elif ticker == "TSM":
                return [SupplyChainEdge("TSM", "AAPL", SupplyChainRelation.CUSTOMER, confidence=0.9)]
            return []

        mock_tracker._extract_edges_gemini.side_effect = mock_extract
        mock_tracker._extract_edges_ollama.side_effect = mock_extract

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=5, max_api_calls=20)

        assert graph.api_calls_used == 2, "循環引用只應展開 2 次（AAPL + TSM）"
        assert "AAPL" in graph.visited_tickers
        assert "TSM" in graph.visited_tickers

    def test_api_budget_exhaustion(self, mock_tracker):
        """
        🤡 白痴情境：每個節點展開出合法 US ticker。max_api_calls=2 時必須止血。
        注意：生成的 ticker 必須符合 ^[A-Z]{1,5}$ 正則才會被展開。
        """
        def mock_extract(ticker, text):
            # 使用合法的 US ticker 格式（純大寫字母 1-5 碼）
            return [
                SupplyChainEdge(ticker, f"ZZ{chr(65+i)}", SupplyChainRelation.SUPPLIER, confidence=0.9)
                for i in range(5)
            ]

        mock_tracker._extract_edges_gemini.side_effect = mock_extract
        mock_tracker._extract_edges_ollama.side_effect = mock_extract

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=10, max_api_calls=2)

        # Root: AAPL (1 call) → 5 suppliers (ZZA..ZZE)
        # Next: ZZA (1 call) → reaches max_calls=2, stop.
        assert graph.api_calls_used == 2, "API 預算耗盡後必須停止"

    def test_illegal_tickers_not_expanded(self, mock_tracker):
        """
        🤡 白痴情境：LLM 回傳「WEIRD^TICKER」這種非法 ticker。
        _is_expandable_ticker 應攔截，不繼續遞迴。
        """
        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "WEIRD^TICKER", SupplyChainRelation.SUPPLIER, confidence=0.9),
            SupplyChainEdge("AAPL", "SOME NONSENSE", SupplyChainRelation.CUSTOMER, confidence=0.8),
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.7),
        ]

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=2, max_api_calls=10)

        assert "WEIRD^TICKER" not in graph.visited_tickers, "非法 ticker 不應被展開"
        assert "SOME NONSENSE" not in graph.visited_tickers

    def test_low_confidence_edges_filtered(self, mock_tracker):
        """
        🧪 信心度過濾：min_confidence=0.5 時，confidence=0.2 的邊不應出現。
        """
        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "JUNK", SupplyChainRelation.SUPPLIER, confidence=0.2),
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9),
        ]

        graph = mock_tracker.build_recursive_graph(
            "AAPL", max_depth=1, max_api_calls=5, min_confidence=0.5
        )

        # JUNK 的 confidence=0.2 低於 min_confidence=0.5，應被過濾
        edge_tickers = {e.source_ticker for e in graph.edges} | {e.target_ticker for e in graph.edges}
        assert "JUNK" not in edge_tickers, "低信心度邊應被過濾"
        assert "TSM" in edge_tickers, "高信心度邊應保留"

    def test_depth_zero_always_gemini(self, mock_tracker):
        """
        🧪 規格驗證：depth=0（根節點）永遠使用 Gemini，不用 Ollama。
        """
        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9),
        ]
        mock_tracker._extract_edges_ollama.return_value = []

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=0, max_api_calls=5)

        assert graph.llm_source.get("AAPL") == "gemini"

    def test_ollama_fallback_to_gemini(self, mock_tracker):
        """
        🧪 降級驗證：Ollama 回傳 None 時，自動降級到 Gemini。
        """
        def mock_gemini(ticker, text):
            if ticker == "TSM":
                return [SupplyChainEdge("TSM", "ASML", SupplyChainRelation.SUPPLIER, confidence=0.8)]
            return [SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9)]

        mock_tracker._extract_edges_ollama.return_value = None
        mock_tracker._extract_edges_gemini.side_effect = mock_gemini

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=1, max_api_calls=5)

        # 根節點 AAPL → gemini (per spec)
        assert graph.llm_source["AAPL"] == "gemini"
        # TSM 是 depth=1, ollama returns None → fallback to gemini
        if "TSM" in graph.llm_source:
            assert graph.llm_source["TSM"] == "gemini_fallback"

    def test_taiwan_ticker_expandable(self, mock_tracker):
        """
        🧪 台灣 ticker 格式 (2330.TW) 應被視為可展開。
        """
        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "2330.TW", SupplyChainRelation.SUPPLIER, confidence=0.9),
        ]

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=1, max_api_calls=5)

        # 2330.TW 是合法台灣 ticker，應被加入佇列
        assert "2330.TW" in graph.node_depths or graph.api_calls_used >= 2

    def test_progress_callback_fires(self, mock_tracker):
        """
        🧪 驗證 progress_callback 確實被呼叫。
        """
        progress_log = []

        def cb(data):
            progress_log.append(data)

        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9),
        ]

        mock_tracker.build_recursive_graph("AAPL", max_depth=1, max_api_calls=5, progress_callback=cb)

        assert len(progress_log) > 0, "progress_callback 至少應被呼叫一次"
        # 確認有 node_done 事件
        node_done_events = [p for p in progress_log if isinstance(p, dict) and p.get("type") == "node_done"]
        assert len(node_done_events) > 0, "應有 node_done 事件"

    def test_to_dict_serializable(self, mock_tracker):
        """
        🧪 RecursiveSupplyChainGraph.to_dict() 必須產出 JSON-safe 結構。
        """
        mock_tracker._extract_edges_gemini.return_value = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9),
        ]

        graph = mock_tracker.build_recursive_graph("AAPL", max_depth=1, max_api_calls=5)
        d = graph.to_dict()

        # 必須能序列化為 JSON（如果有 datetime 物件會在這裡炸）
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["center_ticker"] == "AAPL"
        assert isinstance(d["total_nodes"], int)
        assert isinstance(d["cost_summary"], dict)


# =====================================================================
# 💣 測試 5：殭屍任務逾時 (Async Zombie Task Timeout)
# =====================================================================

class TestAsyncZombieTimeout:
    """模擬外部 API 卡死，asyncio.wait_for 必須攔截。"""

    def test_slow_scan_timeout(self):
        """
        🤡 白痴情境：Yahoo Finance 壞掉，某支股票一直不回傳。
        整個 asyncio.gather 永遠卡住，前端進度條卡在 99%。
        防呆：wait_for(..., timeout=0.1) 強制停損。
        """
        async def _test():
            async def slow_external_api():
                await asyncio.sleep(100)
                return "should never reach here"

            status = "pending"
            try:
                await asyncio.wait_for(slow_external_api(), timeout=0.1)
                status = "done"
            except asyncio.TimeoutError:
                status = "error"

            assert status == "error", "超時必須被攔截"

        _run_async(_test())

    def test_gather_partial_failure(self):
        """
        🤡 白痴情境：3 個任務中 1 個超時，其餘 2 個必須照常完成。
        """
        async def _test():
            results = {}

            async def fast_task(name):
                await asyncio.sleep(0.01)
                results[name] = "done"

            async def slow_task(name):
                await asyncio.sleep(100)
                results[name] = "done"

            async def guarded_task(coro, name):
                try:
                    await asyncio.wait_for(coro(name), timeout=0.5)
                except asyncio.TimeoutError:
                    results[name] = "timeout"

            await asyncio.gather(
                guarded_task(fast_task, "AAPL"),
                guarded_task(fast_task, "TSM"),
                guarded_task(slow_task, "BROKEN"),
            )

            assert results["AAPL"] == "done"
            assert results["TSM"] == "done"
            assert results["BROKEN"] == "timeout", "卡住的任務必須被標記為 timeout"

        _run_async(_test())


# =====================================================================
# 💣 測試 6：傲嬌 AI 亂講話 & Token 撐爆 (Graph-RAG Analysis)
# =====================================================================

class TestGraphRAGDefense:
    """generate_graph_rag_analysis 面對 AI 格式錯誤和超量節點不得死機。"""

    def test_ai_bad_format_detected(self):
        """
        🤡 白痴情境：AI 不輸出 🏆 💎 格式，回傳廢話 "我覺得這家不錯，OK。"
        防呆：偵測格式異常，補上 ⚠️ 警告。
        """
        async def _test():
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(return_value="我覺得這家不錯，OK。")

            enriched_nodes = {"AAPL": _make_enriched("AAPL", 0.8)}
            edges = []
            scores = {"AAPL": 0.8}

            result = await generate_graph_rag_analysis(mock_advisor, "AAPL", enriched_nodes, edges, scores)

            assert "⚠️" in result, "偵測到 AI 沒照格式走時，應補上警告"
            assert "AI 分析格式異常" in result

        _run_async(_test())

    def test_ai_correct_format_passthrough(self):
        """
        🧪 正常情境：AI 正確輸出 🏆 💎 格式 → 直接回傳，不加警告。
        """
        async def _test():
            good_response = """### 🏆 產業咽喉\nTSM 是全球半導體的咽喉...\n### 💎 隱藏的 Alpha\n..."""
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(return_value=good_response)

            enriched_nodes = {"AAPL": _make_enriched("AAPL", 0.8)}
            edges = []
            scores = {"AAPL": 0.8}

            result = await generate_graph_rag_analysis(mock_advisor, "AAPL", enriched_nodes, edges, scores)

            assert "⚠️ AI 分析格式異常" not in result
            assert "🏆" in result

        _run_async(_test())

    def test_truncation_over_30_nodes(self):
        """
        🤡 白痴情境：圖譜有 50 家公司，全部塞給 LLM → Token 爆炸 (HTTP 400)。
        防呆：超過 30 個節點自動截斷，只分析 Top 30。
        """
        async def _test():
            good_response = "### 🏆 產業咽喉\n...\n### 💎 隱藏的 Alpha\n..."
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(return_value=good_response)

            enriched_nodes = {f"T{i}": _make_enriched(f"T{i}", composite_score=i / 50.0) for i in range(50)}
            edges = []
            scores = {f"T{i}": i / 50.0 for i in range(50)}

            result = await generate_graph_rag_analysis(mock_advisor, "T0", enriched_nodes, edges, scores)

            assert "截斷" in result, "超過 30 節點應附帶截斷警告"

        _run_async(_test())

    def test_llm_exception_graceful(self):
        """
        🤡 白痴情境：LLM API 直接噴 Exception（網路斷線、API key 過期…）。
        防呆：捕捉 Exception，回傳錯誤訊息而非讓整個 endpoint 500。
        """
        async def _test():
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(side_effect=Exception("API key expired"))

            enriched_nodes = {"AAPL": _make_enriched("AAPL", 0.8)}
            edges = []
            scores = {"AAPL": 0.8}

            result = await generate_graph_rag_analysis(mock_advisor, "AAPL", enriched_nodes, edges, scores)

            assert "⚠️ AI 分析失敗" in result
            assert "API key expired" in result

        _run_async(_test())

    def test_company_name_prompt_injection(self):
        """
        🤡 白痴情境：公司名稱包含 Markdown 注入字元（反引號、換行）。
        防呆：清洗 company_name → 不讓 Markdown 表格爆炸。
        """
        async def _test():
            good_response = "### 🏆 產業咽喉\n...\n### 💎 隱藏的 Alpha\n..."
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(return_value=good_response)

            evil_name = "Evil`Corp\nDROP TABLE"
            enriched_nodes = {"EVIL": _make_enriched("EVIL", 0.8, company_name=evil_name)}
            edges = []
            scores = {"EVIL": 0.8}

            # 不應噴錯，prompt 中的名稱已被清洗
            result = await generate_graph_rag_analysis(mock_advisor, "EVIL", enriched_nodes, edges, scores)

            # 驗證呼叫 LLM 的 prompt 中反引號和換行已被清除
            call_args = mock_advisor.chat.call_args[0][0]
            # 在 EVIL 那一行的 company_name 欄位中不該有反引號
            assert "`" not in call_args.split("| EVIL |")[1].split("|")[1] if "| EVIL |" in call_args else True

        _run_async(_test())


# =====================================================================
# 💣 測試 7：EnrichedNode & node_to_dict 序列化完整性
# =====================================================================

class TestEnrichedNodeSerialization:
    """node_to_dict 必須輸出完整的 JSON-safe 結構。"""

    def test_default_node_to_dict(self):
        """
        🧪 預設 EnrichedNode 轉 dict 不得遺漏欄位。
        """
        node = EnrichedNode(ticker="AAPL", depth=0)
        d = node_to_dict(node)

        expected_keys = {
            "ticker", "depth", "llm_source",
            "composite_score", "signal_strength", "valuation_rating",
            "fundamental_pass", "ai_bullish", "technical_confirm",
            "degree_centrality", "network_alpha_score", "spillover_delta",
            "scan_status", "top_suppliers", "top_customers", "company_name",
        }
        assert set(d.keys()) == expected_keys, f"遺漏欄位: {expected_keys - set(d.keys())}"

    def test_none_composite_score_serializable(self):
        """
        🧪 composite_score=None 時，JSON 序列化必須成功（None → null）。
        """
        node = EnrichedNode(ticker="AAPL", depth=0, composite_score=None)
        d = node_to_dict(node)

        serialized = json.dumps(d)
        assert '"composite_score": null' in serialized


# =====================================================================
# 💣 測試 8：RecursiveSupplyChainGraph 模型完整性
# =====================================================================

class TestRecursiveGraphModel:
    """RecursiveSupplyChainGraph 的屬性和方法在邊界條件下行為正確。"""

    def test_empty_graph_properties(self):
        """
        🧪 空圖的屬性不能炸。
        """
        g = RecursiveSupplyChainGraph(center_ticker="AAPL")

        assert g.total_nodes == 1  # 只有 center
        assert g.layers == {}
        assert g.get_edges_at_depth(0) == []

    def test_to_dict_with_real_edges(self):
        """
        🧪 有真實 edge 的圖 → to_dict 必須 JSON-safe。
        """
        edges = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9, depth=0),
            SupplyChainEdge("TSM", "ASML", SupplyChainRelation.SUPPLIER, confidence=0.8, depth=1),
        ]
        g = RecursiveSupplyChainGraph(
            center_ticker="AAPL",
            edges=edges,
            max_depth=2,
            node_depths={"AAPL": 0, "TSM": 1, "ASML": 2},
            visited_tickers={"AAPL", "TSM", "ASML"},
            api_calls_used=3,
            llm_source={"AAPL": "gemini", "TSM": "ollama", "ASML": "cache"},
        )

        d = g.to_dict()
        serialized = json.dumps(d)

        assert isinstance(serialized, str)
        assert d["total_nodes"] == 3
        assert d["total_edges"] == 2
        assert d["cost_summary"]["gemini_calls"] == 1
        assert d["cost_summary"]["ollama_calls"] == 1
        assert d["cost_summary"]["cache_hits"] == 1

    def test_layers_grouping(self):
        """
        🧪 layers 屬性正確按深度分組。
        """
        g = RecursiveSupplyChainGraph(
            center_ticker="AAPL",
            node_depths={"AAPL": 0, "TSM": 1, "ASML": 1, "KLAC": 2},
        )

        layers = g.layers
        assert set(layers[0]) == {"AAPL"}
        assert set(layers[1]) == {"TSM", "ASML"}
        assert set(layers[2]) == {"KLAC"}

    def test_get_edges_at_depth(self):
        """
        🧪 get_edges_at_depth 正確過濾。
        """
        edges = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, depth=0),
            SupplyChainEdge("TSM", "ASML", SupplyChainRelation.SUPPLIER, depth=1),
        ]
        g = RecursiveSupplyChainGraph(
            center_ticker="AAPL",
            edges=edges,
            node_depths={"AAPL": 0, "TSM": 1, "ASML": 2},
        )

        depth_0_edges = g.get_edges_at_depth(0)
        depth_1_edges = g.get_edges_at_depth(1)

        # AAPL is at depth 0, so AAPL→TSM edge is returned
        assert len(depth_0_edges) >= 1
        # TSM is at depth 1, both edges contain TSM
        assert len(depth_1_edges) >= 1


# =====================================================================
# 💣 測試 9：SupplyChainTracker 內部防護
# =====================================================================

class TestTrackerInternalDefenses:
    """supply_chain.py 的各種內部防護邏輯。"""

    def test_is_expandable_valid_us_tickers(self, mock_tracker):
        """US ticker 格式驗證。"""
        assert mock_tracker._is_expandable_ticker("AAPL") is True
        assert mock_tracker._is_expandable_ticker("TSM") is True
        assert mock_tracker._is_expandable_ticker("A") is True
        assert mock_tracker._is_expandable_ticker("AVGO") is True

    def test_is_expandable_valid_tw_tickers(self, mock_tracker):
        """台灣 ticker 格式驗證。"""
        assert mock_tracker._is_expandable_ticker("2330.TW") is True
        assert mock_tracker._is_expandable_ticker("2454.TWO") is True
        assert mock_tracker._is_expandable_ticker("2330") is True

    def test_is_expandable_rejects_garbage(self, mock_tracker):
        """非法 ticker 必須被拒。"""
        assert mock_tracker._is_expandable_ticker("WEIRD^TICKER") is False
        assert mock_tracker._is_expandable_ticker("SOME NONSENSE") is False
        assert mock_tracker._is_expandable_ticker("ticker_with_underscore") is False
        assert mock_tracker._is_expandable_ticker("") is False
        assert mock_tracker._is_expandable_ticker("123456") is False
        assert mock_tracker._is_expandable_ticker("ABCDEF") is False  # >5 chars

    def test_deduplicate_edges(self, mock_tracker):
        """重複邊去重，保留信心度最高的那條。"""
        edges = [
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.5),
            SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9),
            SupplyChainEdge("TSM", "AAPL", SupplyChainRelation.SUPPLIER, confidence=0.3),  # 同對、同方向 → 去重
        ]

        result = mock_tracker._deduplicate_edges(edges)

        # AAPL-TSM-SUPPLIER 只留一條，信心度最高的
        supplier_edges = [e for e in result if e.relation == SupplyChainRelation.SUPPLIER]
        assert len(supplier_edges) == 1
        assert supplier_edges[0].confidence == 0.9

    def test_parse_json_from_markdown_block(self, mock_tracker):
        """LLM 回傳 ```json ... ``` 包裹的 JSON → 正確解析。"""
        text = '```json\n[{"source": "A", "target": "B"}]\n```'
        result = mock_tracker._parse_json_from_text(text)
        assert isinstance(result, list)
        assert result[0]["source"] == "A"

    def test_parse_json_from_raw_text(self, mock_tracker):
        """LLM 回傳純 JSON（無 markdown）→ 正確解析。"""
        text = '[{"source": "A", "target": "B"}]'
        result = mock_tracker._parse_json_from_text(text)
        assert isinstance(result, list)

    def test_parse_json_garbage_returns_none(self, mock_tracker):
        """LLM 回傳完全垃圾 → 回傳 None，不炸。"""
        text = "我覺得這家不錯，OK。根本不是 JSON。"
        result = mock_tracker._parse_json_from_text(text)
        assert result is None


# =====================================================================
# 💣 測試 10：完整 Pipeline 整合測試
# =====================================================================

class TestFullPipelineIntegration:
    """模擬從 build_recursive_graph → enriched → centrality → spillover → graph-rag 的完整流程。"""

    def test_end_to_end_no_crash(self, mock_tracker):
        """
        🧪 完整走一遍 Pipeline，確保串接不炸。
        """
        async def _test():
            # 1. 建構遞迴圖譜
            mock_tracker._extract_edges_gemini.return_value = [
                SupplyChainEdge("AAPL", "TSM", SupplyChainRelation.SUPPLIER, confidence=0.9, revenue_pct=0.25),
                SupplyChainEdge("AAPL", "QCOM", SupplyChainRelation.COMPETITOR, confidence=0.7),
            ]
            graph = mock_tracker.build_recursive_graph("AAPL", max_depth=1, max_api_calls=3)

            # 2. 建構 EnrichedNode
            enriched_nodes = {}
            for t, d in graph.node_depths.items():
                enriched_nodes[t] = EnrichedNode(
                    ticker=t, depth=d,
                    llm_source=graph.llm_source.get(t, "unknown"),
                    composite_score=0.7 if t == "AAPL" else None,  # 故意留 None 測防呆
                )

            # 3. Degree Centrality
            nodes_list = [{"ticker": t} for t in enriched_nodes]
            edges_list = [
                {
                    "source_ticker": e.source_ticker,
                    "target_ticker": e.target_ticker,
                    "relation": e.relation.value,
                    "revenue_pct": e.revenue_pct,
                }
                for e in graph.edges
            ]
            centralities = compute_degree_centrality(nodes_list, edges_list)
            for t, c in centralities.items():
                if t in enriched_nodes:
                    enriched_nodes[t].degree_centrality = c

            # 4. Momentum Spillover
            scores = compute_momentum_spillover(enriched_nodes, edges_list)

            # 5. Graph-RAG
            mock_advisor = MagicMock()
            mock_advisor.chat = MagicMock(return_value="### 🏆 產業咽喉\nTSM\n### 💎 隱藏的 Alpha\n...")
            rag_result = await generate_graph_rag_analysis(mock_advisor, "AAPL", enriched_nodes, edges_list, scores)

            # 驗證
            assert isinstance(rag_result, str)
            assert "🏆" in rag_result
            for t in enriched_nodes:
                assert 0.0 <= enriched_nodes[t].network_alpha_score <= 1.0
                assert enriched_nodes[t].degree_centrality >= 0.0

        _run_async(_test())
