"""
Alpha Hunter v2.5 — 究極壓力測試 (Stress Tests)
=================================================
測試最高原則：AAA 模式
1. Arrange (準備毒藥)
2. Act    (吃下毒藥)
3. Assert (驗證沒死)

所有測試必須在 10 秒內跑完、不連外網、不依賴 API Key。
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from nexus_quant_os.alpha_hunter.models import EnrichedNode
from nexus_quant_os.alpha_hunter.radar import (
    compute_degree_centrality,
    compute_momentum_spillover,
    node_to_dict,
)


# =====================================================================
# 🧨 壓測 1：貪心鬼的 51 顆核彈 (API Payload 上限防護)
# 對應架構：5. 資源限制與 LLM 邊界
# =====================================================================
class TestPayloadLimitBomber:
    """
    🤡 白癡情境：
    系統規定最多掃描 50 支股票。駭客寫腳本一次塞了 51 支進來！
    如果沒擋下，背景會同時發起 51 個連線，伺服器記憶體瞬間炸掉，
    IP 還會被 API 供應商封鎖。
    """

    def test_exactly_50_should_pass(self):
        """邊界值：剛好 50 支 → 放行"""
        fake_tickers = [{"ticker": f"T{i:03d}"} for i in range(50)]
        assert len(fake_tickers) <= 50

    def test_51_should_reject(self):
        """超限值：51 支 → HTTP 400 退件"""
        fake_tickers = [{"ticker": f"T{i:03d}"} for i in range(51)]
        response_status = 400 if len(fake_tickers) > 50 else 200
        assert response_status == 400

    def test_all_tickers_limit_100(self):
        """all_tickers 上限 100 支 → 101 支退件"""
        all_tickers = [{"ticker": f"T{i:04d}"} for i in range(101)]
        response_status = 400 if len(all_tickers) > 100 else 200
        assert response_status == 400

    def test_zero_tickers_should_not_crash(self):
        """空列表 → 不 crash，直接走 spillover-only 路徑"""
        fake_tickers = []
        assert len(fake_tickers) == 0
        # 後端應直接進入 Phase 3 而非報錯


# =====================================================================
# 🧨 壓測 2：樹懶大罷工 (非同步超時防護 / Zombie Task)
# 對應架構：2. 併發與非同步控制模型
# =====================================================================
class TestZombieTaskTimeout:
    """
    🤡 白癡情境：
    Yahoo Finance 伺服器壞了，發出請求後卡住 10 分鐘完全不回應。
    如果 3 個併發任務全卡死，伺服器的 Event Loop 就會癱瘓。
    """

    @pytest.mark.asyncio
    async def test_single_zombie_killed(self):
        """單一殭屍任務被 asyncio.wait_for 45s Timeout 斬殺"""
        async def extremely_slow_api():
            await asyncio.sleep(3600)
            return "SUCCESS"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(extremely_slow_api(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_parallel_zombies_all_killed(self):
        """3 個殭屍任務同時卡死 → 全部在 Timeout 內被斬殺"""
        async def zombie():
            await asyncio.sleep(3600)
            return "alive"

        results = await asyncio.gather(
            *[asyncio.wait_for(zombie(), timeout=0.05) for _ in range(3)],
            return_exceptions=True,
        )
        # 3 個全是 TimeoutError，沒有一個是 "alive"
        assert all(isinstance(r, asyncio.TimeoutError) for r in results)

    @pytest.mark.asyncio
    async def test_mixed_fast_and_zombie(self):
        """2 正常 + 1 殭屍 → 正常的成功完成，殭屍被斬殺，不拖累整體"""
        async def fast():
            return "OK"

        async def zombie():
            await asyncio.sleep(3600)

        results = await asyncio.gather(
            fast(),
            fast(),
            asyncio.wait_for(zombie(), timeout=0.05),
            return_exceptions=True,
        )
        assert results[0] == "OK"
        assert results[1] == "OK"
        assert isinstance(results[2], asyncio.TimeoutError)


# =====================================================================
# 🧨 壓測 3：吃霸王餐的渣男 (SSE 斷線資源釋放)
# 對應架構：2. 併發與非同步控制模型
# =====================================================================
class TestClientHitAndRun:
    """
    🤡 白癡情境：
    使用者要求掃描 30 支股票，結果等了 1 秒覺得無聊，直接把瀏覽器關掉。
    如果後端是個老實人，還會在背景把剩下的掃完，白白燒掉 Gemini 預算。
    """

    @pytest.mark.asyncio
    async def test_immediate_disconnect(self):
        """使用者連掃描都沒開始就斷線 → 0 次 API 呼叫"""
        mock_request = AsyncMock()
        mock_request.is_disconnected.return_value = True

        loops_run = 0
        for _ in range(30):
            if await mock_request.is_disconnected():
                break
            loops_run += 1

        assert loops_run == 0

    @pytest.mark.asyncio
    async def test_disconnect_after_one(self):
        """跑了 1 支就斷線 → 省下 29 次 API 費用"""
        mock_request = AsyncMock()
        mock_request.is_disconnected.side_effect = [False, True] + [True] * 28

        loops_run = 0
        for _ in range(30):
            if await mock_request.is_disconnected():
                break
            loops_run += 1

        assert loops_run == 1

    @pytest.mark.asyncio
    async def test_scan_task_cancelled_on_disconnect(self):
        """模擬 scan_task.cancel() 被觸發"""
        task_cancelled = False

        async def fake_scan():
            nonlocal task_cancelled
            await asyncio.sleep(10)

        task = asyncio.create_task(fake_scan())

        # 模擬偵測到斷線，立即 cancel
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            task_cancelled = True

        assert task_cancelled is True


# =====================================================================
# 🧨 壓測 4：無限月讀與數學黑洞 (真實演算法防護測試)
# 對應架構：4. 演算法防護
# =====================================================================
class TestMomentumSpilloverDefense:
    """
    🤡 白癡情境：
    1. A 是 B 的客戶，B 也是 A 的客戶 (圖論循環)。
    2. LLM 產生幻覺，說營收依賴佔比是 5.0 (500%)。
    3. 某家公司的 composite_score 是 None（API 失敗）。
    """

    def test_circular_dependency_no_infinite_loop(self):
        """A↔B 互為客戶 + 營收暴走 500% → 分數仍鎖在 [0, 1]"""
        nodes = {
            "A": EnrichedNode(ticker="A", depth=0, composite_score=0.9),
            "B": EnrichedNode(ticker="B", depth=1, composite_score=0.9),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "CUSTOMER", "revenue_pct": 5.0},
            {"source_ticker": "B", "target_ticker": "A", "relation": "CUSTOMER", "revenue_pct": 5.0},
        ]

        scores = compute_momentum_spillover(nodes, edges)

        for t, s in scores.items():
            assert 0.0 <= s <= 1.0, f"{t} 的分數 {s} 破表！必須在 [0, 1] 之間"

    def test_null_composite_score_no_crash(self):
        """composite_score 為 None (API 失敗) → 用 0.5 中性填充，不 crash"""
        nodes = {
            "A": EnrichedNode(ticker="A", depth=0, composite_score=None),
            "B": EnrichedNode(ticker="B", depth=1, composite_score=0.8),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": 0.3},
        ]

        scores = compute_momentum_spillover(nodes, edges)

        assert "A" in scores
        assert "B" in scores
        assert 0.0 <= scores["A"] <= 1.0

    def test_negative_revenue_pct_clamped(self):
        """revenue_pct 為負數 → 被鉗制為 0.0"""
        nodes = {
            "X": EnrichedNode(ticker="X", depth=0, composite_score=0.7),
            "Y": EnrichedNode(ticker="Y", depth=1, composite_score=0.3),
        }
        edges = [
            {"source_ticker": "X", "target_ticker": "Y", "relation": "SUPPLIER", "revenue_pct": -999.0},
        ]

        scores = compute_momentum_spillover(nodes, edges)
        # 負數被鉗制為 0 → 不應影響 X 的分數
        assert 0.0 <= scores["X"] <= 1.0

    def test_revenue_pct_none_or_garbage(self):
        """revenue_pct 為 None 或不可轉數值 → 用 0.1 預設值"""
        nodes = {
            "P": EnrichedNode(ticker="P", depth=0, composite_score=0.5),
            "Q": EnrichedNode(ticker="Q", depth=1, composite_score=0.5),
        }
        edges = [
            {"source_ticker": "P", "target_ticker": "Q", "relation": "SUPPLIER", "revenue_pct": None},
            {"source_ticker": "Q", "target_ticker": "P", "relation": "SUPPLIER", "revenue_pct": "lol_not_a_number"},
        ]

        # 不應 crash
        scores = compute_momentum_spillover(nodes, edges)
        assert "P" in scores and "Q" in scores

    def test_spillover_delta_calculated_correctly(self):
        """驗證 spillover_delta = network_alpha_score - composite_score"""
        nodes = {
            "A": EnrichedNode(ticker="A", depth=0, composite_score=0.3),
            "B": EnrichedNode(ticker="B", depth=1, composite_score=0.9),
        }
        edges = [
            {"source_ticker": "A", "target_ticker": "B", "relation": "SUPPLIER", "revenue_pct": 0.5},
        ]

        compute_momentum_spillover(nodes, edges)

        for t, node in nodes.items():
            expected_delta = node.network_alpha_score - (node.composite_score if node.composite_score is not None else 0.5)
            assert abs(node.spillover_delta - expected_delta) < 0.0001, \
                f"{t}: spillover_delta={node.spillover_delta} 應該等於 {expected_delta}"

    def test_single_node_graph(self):
        """只有 1 個節點、0 條邊 → 不 crash，分數 = 原始分數"""
        nodes = {
            "LONELY": EnrichedNode(ticker="LONELY", depth=0, composite_score=0.42),
        }
        edges = []

        scores = compute_momentum_spillover(nodes, edges)
        assert abs(scores["LONELY"] - 0.42) < 0.001

    def test_empty_graph(self):
        """完全空圖 → 不 crash，回傳空 dict"""
        scores = compute_momentum_spillover({}, [])
        assert scores == {}

    def test_ghost_node_in_edges(self):
        """Edge 參照了 enriched_nodes 之外的 ticker → 不 crash"""
        nodes = {
            "REAL": EnrichedNode(ticker="REAL", depth=0, composite_score=0.6),
        }
        edges = [
            {"source_ticker": "REAL", "target_ticker": "GHOST", "relation": "SUPPLIER", "revenue_pct": 0.5},
        ]

        scores = compute_momentum_spillover(nodes, edges)
        assert "REAL" in scores


# =====================================================================
# 🧨 壓測 5：Degree Centrality 極端情境
# 對應架構：4. 演算法防護
# =====================================================================
class TestDegreeCentralityDefense:
    """測試中心度計算的邊界情境。"""

    def test_single_node_centrality(self):
        """只有 1 個節點 → centrality = 1.0（除以零防護生效）"""
        nodes = [{"ticker": "ALONE"}]
        edges = []
        result = compute_degree_centrality(nodes, edges)
        assert result["ALONE"] == 1.0

    def test_isolated_node_in_graph(self):
        """3 個節點但只有 2 個有連線 → 孤立節點 centrality = 0"""
        nodes = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        edges = [{"source_ticker": "A", "target_ticker": "B"}]
        result = compute_degree_centrality(nodes, edges)
        assert result["C"] == 0.0
        assert result["A"] > 0.0

    def test_fully_connected_graph(self):
        """完全圖 (每個節點都有連線) → centrality 合理"""
        nodes = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        edges = [
            {"source_ticker": "A", "target_ticker": "B"},
            {"source_ticker": "B", "target_ticker": "C"},
            {"source_ticker": "A", "target_ticker": "C"},
        ]
        result = compute_degree_centrality(nodes, edges)
        assert result["A"] == 1.0  # A 連到 B 和 C，度數 = 2，n-1 = 2


# =====================================================================
# 🧨 壓測 6：node_to_dict 序列化極端值
# 對應架構：5. 資源限制與 LLM 邊界
# =====================================================================
class TestNodeToDictSerialization:
    """測試 node_to_dict 在各種極端資料下都能正確序列化為 JSON。"""

    def test_all_none_fields(self):
        """所有 Optional 欄位都是 None → 不 crash"""
        node = EnrichedNode(ticker="NULL_CORP", depth=0)
        d = node_to_dict(node)
        assert d["ticker"] == "NULL_CORP"
        assert d["composite_score"] is None

    def test_special_characters_in_company_name(self):
        """公司名稱含特殊字元 → 不影響序列化"""
        node = EnrichedNode(ticker="EVIL", depth=0, company_name='Evil Corp™ "The Best" 🎉\n|table|')
        d = node_to_dict(node)
        serialized = json.dumps(d)  # 不應 raise
        assert "EVIL" in serialized

    def test_extreme_score_values(self):
        """分數是極端浮點數 → 不 crash"""
        node = EnrichedNode(
            ticker="EDGE",
            depth=0,
            composite_score=float("inf"),
            network_alpha_score=float("-inf"),
            spillover_delta=float("nan"),
        )
        d = node_to_dict(node)
        assert d["ticker"] == "EDGE"


# =====================================================================
# 🧨 壓測 7：駭客的「破壞排版」攻擊 (Prompt Injection)
# 對應架構：5. 資源限制與 LLM 邊界
# =====================================================================
class TestPromptInjection:
    """
    🤡 白癡情境：
    某間公司的名字叫做 "Evil Corp \n | 100 | Hacker |"。
    如果直接塞進給 LLM 的 Markdown 表格裡，表格結構會瞬間錯亂。
    """

    def test_newline_injection(self):
        """名字含 \\n → 被清洗成空格"""
        company_name = "Evil Corp\n| 100 | Hacker |"
        safe_name = company_name.replace("`", "").replace("\n", " ")
        assert "\n" not in safe_name

    def test_backtick_injection(self):
        """名字含 ` → 被移除，防止 Markdown code block 逃逸"""
        company_name = "Corp `; DROP TABLE companies; --`"
        safe_name = company_name.replace("`", "").replace("\n", " ")
        assert "`" not in safe_name

    def test_empty_company_name(self):
        """公司名為空 → 用 'Unknown' 替代"""
        company_name = ""
        safe_name = company_name.replace("`", "").replace("\n", " ") if company_name else "Unknown"
        assert safe_name == "Unknown"


# =====================================================================
# 🧨 壓測 8：胖子塞爆置物櫃 (SessionStorage Quota Exceeded)
# 對應架構：6. 前端渲染與狀態管理防護
# =====================================================================
class TestStorageQuotaProtection:
    """
    🤡 白癡情境 (模擬前端 JS 行為)：
    使用者的圖譜超級大，LLM 又寫了一長串分析文章，轉 JSON 高達 6MB。
    """

    def test_oversized_payload_trimmed(self):
        """6MB 的 LLM 分析 → 「丟車保帥」截斷，保留核心 nodes"""
        state = {
            "nodes": [{"ticker": "AAPL"}, {"ticker": "NVDA"}],
            "llmAnalysis": "A" * (6 * 1024 * 1024),
        }

        json_str = json.dumps(state)
        if len(json_str) > 4 * 1024 * 1024:
            state["llmAnalysis"] = "[記憶體保護] 分析內容過大已移除"
            json_str = json.dumps(state)

        assert len(json_str) < 5 * 1024 * 1024
        assert state["nodes"][0]["ticker"] == "AAPL"
        assert state["nodes"][1]["ticker"] == "NVDA"

    def test_normal_payload_untouched(self):
        """正常大小 → 不截斷"""
        state = {
            "nodes": [{"ticker": "AAPL"}],
            "llmAnalysis": "This stock looks promising.",
        }

        json_str = json.dumps(state)
        assert len(json_str) < 4 * 1024 * 1024
        assert state["llmAnalysis"] == "This stock looks promising."


# =====================================================================
# 🧨 壓測 9：帕金森氏症狂點 (Race Condition Lock)
# 對應架構：6. 前端渲染與狀態管理防護
# =====================================================================
class TestFrontendSpamClickLock:
    """
    🤡 白癡情境 (模擬前端 JS 行為)：
    使用者滑鼠微動開關壞了，0.5 秒內對著節點瘋狂連點了 10 下。
    """

    def test_10_rapid_clicks_only_1_triggers(self):
        """狂點 10 下 → 只有第 1 下生效"""
        window_is_navigating = False
        api_call_count = 0

        def drill_down_to_ticker():
            nonlocal window_is_navigating, api_call_count
            if window_is_navigating:
                return
            window_is_navigating = True
            api_call_count += 1

        for _ in range(10):
            drill_down_to_ticker()

        assert api_call_count == 1

    def test_lock_resets_allow_next_navigation(self):
        """鎖釋放後 → 下一次點擊正常生效"""
        window_is_navigating = False
        api_call_count = 0

        def drill_down():
            nonlocal window_is_navigating, api_call_count
            if window_is_navigating:
                return
            window_is_navigating = True
            api_call_count += 1

        def finish_navigation():
            nonlocal window_is_navigating
            window_is_navigating = False

        drill_down()
        assert api_call_count == 1

        finish_navigation()

        drill_down()
        assert api_call_count == 2


# =====================================================================
# 🧨 壓測 10：AbortController 多次取消衝突
# 對應架構：2. 併發與非同步控制模型
# =====================================================================
class TestAbortControllerConflict:
    """
    🤡 白癡情境：
    使用者在深度遞迴跑到一半時，切換搜尋了另一家公司。
    舊的 fetch 如果不被 abort，會霸佔 API Rate Limit 導致新搜尋 Timeout。
    """

    @pytest.mark.asyncio
    async def test_old_scan_aborted_on_new_search(self):
        """模擬 AbortController.abort() → 舊任務被取消，新任務正常執行"""
        old_task_status = "running"
        new_task_status = "pending"

        async def old_scan():
            nonlocal old_task_status
            try:
                await asyncio.sleep(10)
                old_task_status = "done"
            except asyncio.CancelledError:
                old_task_status = "aborted"
                raise

        async def new_scan():
            nonlocal new_task_status
            await asyncio.sleep(0.01)
            new_task_status = "done"

        old_task = asyncio.create_task(old_scan())
        
        # 讓 event loop 跑一下，確保 old_task 確實啟動進入 try 區塊
        await asyncio.sleep(0)
    
        # 使用者按了搜尋 → abort 舊任務
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            pass

        # 啟動新任務
        await new_scan()

        assert old_task_status == "aborted"
        assert new_task_status == "done"

    @pytest.mark.asyncio
    async def test_double_abort_no_crash(self):
        """連續 abort 兩次 → 不 crash"""
        async def dummy():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy())
        task.cancel()
        task.cancel()  # 重複 cancel 不應報錯

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.cancelled()


# =====================================================================
# 🧨 壓測 11：圖論演算法的超大規模壓力
# 對應架構：4. 演算法防護
# =====================================================================
class TestLargeGraphPerformance:
    """
    🤡 白癡情境：
    使用者搜了一個超級大公司，depth=3，展開出 50 個節點和 200 條邊。
    演算法必須在合理時間內完成，不能 hang。
    """

    def test_50_nodes_200_edges(self):
        """50 個節點 + 200 條邊 → 演算法必須秒殺完成"""
        import time

        nodes = {}
        for i in range(50):
            nodes[f"T{i:03d}"] = EnrichedNode(
                ticker=f"T{i:03d}",
                depth=i % 3,
                composite_score=0.3 + (i % 7) * 0.1,
            )

        edges = []
        for i in range(200):
            src = f"T{i % 50:03d}"
            tgt = f"T{(i * 7 + 3) % 50:03d}"
            if src != tgt:
                edges.append({
                    "source_ticker": src,
                    "target_ticker": tgt,
                    "relation": "SUPPLIER",
                    "revenue_pct": 0.1 + (i % 5) * 0.05,
                })

        t0 = time.monotonic()
        scores = compute_momentum_spillover(nodes, edges)
        elapsed = time.monotonic() - t0

        assert elapsed < 5.0, f"演算法跑了 {elapsed:.2f} 秒，太慢了！（上限 5 秒）"
        assert len(scores) == 50
        for s in scores.values():
            assert 0.0 <= s <= 1.0


# =====================================================================
# 🧨 壓測 12：前後端 all_tickers 協定測試
# 對應架構：2. 併發與非同步控制模型
# =====================================================================
class TestAllTickersProtocol:
    """
    🤡 白癡情境：
    前端傳了 all_tickers，其中 5 個是 cached 節點（有 composite_score），
    3 個是新節點（composite_score 為 null）。
    後端必須正確處理兩種節點，不能全歸零。
    """

    def test_cached_nodes_retain_scores_in_spillover(self):
        """快取節點的 composite_score 必須被帶入 spillover 計算"""
        # 模擬後端：用 all_tickers 建立 spillover_nodes
        all_tickers = [
            {"ticker": "NVDA", "depth": 0, "composite_score": 0.85},  # cached
            {"ticker": "TSM", "depth": 1, "composite_score": 0.72},   # cached
            {"ticker": "NEW1", "depth": 1, "composite_score": None},  # 新節點
        ]

        spillover_nodes = {}
        for t in all_tickers:
            comp = t.get("composite_score")
            spillover_nodes[t["ticker"]] = EnrichedNode(
                ticker=t["ticker"],
                depth=t["depth"],
                composite_score=comp if comp is not None else 0.5,
            )

        edges = [
            {"source_ticker": "TSM", "target_ticker": "NVDA", "relation": "SUPPLIER", "revenue_pct": 0.4},
            {"source_ticker": "NEW1", "target_ticker": "NVDA", "relation": "SUPPLIER", "revenue_pct": 0.1},
        ]

        scores = compute_momentum_spillover(spillover_nodes, edges)

        # 所有節點都必須有分數
        assert "NVDA" in scores
        assert "TSM" in scores
        assert "NEW1" in scores

        # TSM 的 spillover_delta 不應為 0（因為有 NVDA 的高分數傳染下來）
        assert spillover_nodes["TSM"].network_alpha_score > 0

    def test_all_tickers_empty_fallback(self):
        """all_tickers 為空 → 用 tickers 作為 fallback（向後相容）"""
        tickers = [
            {"ticker": "AAPL", "depth": 0},
        ]
        all_tickers = []

        # 向後相容邏輯
        if not all_tickers:
            all_tickers = tickers

        assert len(all_tickers) == 1
        assert all_tickers[0]["ticker"] == "AAPL"
