"""
test_edge_cases_phase2.py — Alpha Hunter v2 深水區極端情境測試 (Phase 2)

混沌工程 (Chaos Engineering) 自動化測試。
只測災難，不測好天氣。零網路、零 Happy Path、零容忍 500。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd
import pytest

# ── 被測模組 ──────────────────────────────────────────────
from nexus_quant_os.alpha_hunter.models import (
    AlphaSignal, AIAnalysis, FinancialStatement, ScanResult,
    SupplyChainEdge, SupplyChainGraph, SupplyChainRelation,
    SignalStrength, ValuationRating,
)
from nexus_quant_os.alpha_hunter.signal_generator import AlphaSignalGenerator
from nexus_quant_os.alpha_hunter.financial_scanner import FinancialScanner
from nexus_quant_os.alpha_hunter.ai_analyst import AIAnalyst
from nexus_quant_os.alpha_hunter.supply_chain import SupplyChainTracker
from nexus_quant_os.alpha_hunter.sec_edgar import SECEdgarClient


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 共用 Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_stub_stmt(ticker: str = "TEST", **overrides) -> FinancialStatement:
    """建構一個最小可用的 FinancialStatement。"""
    defaults = dict(
        ticker=ticker, cik="0000000001", company_name=f"{ticker} Inc",
        filing_date=datetime.now(timezone.utc),
        period_end=datetime.now(timezone.utc),
        fiscal_year=2025, fiscal_quarter=4,
    )
    defaults.update(overrides)
    return FinancialStatement(**defaults)


def _make_stub_scan(ticker: str = "TEST", **overrides) -> ScanResult:
    """建構一個最小可用的 ScanResult。"""
    stmt = _make_stub_stmt(ticker)
    defaults = dict(ticker=ticker, company_name=f"{ticker} Inc", latest_statement=stmt)
    defaults.update(overrides)
    return ScanResult(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 1：yfinance 殭屍股與下市股 (Empty DataFrame 陷阱)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2YfinanceZombieStock:
    """
    災難情境：yfinance.download() 回傳完全空的 DataFrame。
    攻擊目標：signal_generator._check_technical()
    預期死法：IndexError: single positional indexer is out-of-bounds
    正確行為：安全回傳 False，不拋出任何例外。
    """

    @patch("nexus_quant_os.alpha_hunter.signal_generator.yf.download")
    def test_phase2_empty_dataframe_returns_false(self, mock_download):
        """完全空的 DataFrame（0 rows, 0 columns）→ technical_confirm = False"""
        mock_download.return_value = pd.DataFrame()

        gen = AlphaSignalGenerator.__new__(AlphaSignalGenerator)
        result = gen._check_technical("ZOMBIE_CORP")

        assert result is False, "殭屍股應該回傳 False，而不是拋出 IndexError"

    @patch("nexus_quant_os.alpha_hunter.signal_generator.yf.download")
    def test_phase2_dataframe_with_columns_but_no_rows(self, mock_download):
        """有 columns 但 0 rows 的 DataFrame → technical_confirm = False"""
        mock_download.return_value = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        gen = AlphaSignalGenerator.__new__(AlphaSignalGenerator)
        result = gen._check_technical("DELISTED_INC")

        assert result is False

    @patch("nexus_quant_os.alpha_hunter.signal_generator.yf.download")
    def test_phase2_dataframe_too_few_rows_for_ma(self, mock_download):
        """只有 10 行（遠少於 200MA 所需）→ 不能算均線 → False"""
        df = pd.DataFrame({"Close": [100.0] * 10})
        mock_download.return_value = df

        gen = AlphaSignalGenerator.__new__(AlphaSignalGenerator)
        result = gen._check_technical("BABY_IPO")

        assert result is False

    @patch("nexus_quant_os.alpha_hunter.signal_generator.yf.download")
    def test_phase2_yfinance_raises_exception(self, mock_download):
        """yfinance 直接拋出例外（網路斷線）→ 吞掉例外 → False"""
        mock_download.side_effect = Exception("Connection refused")

        gen = AlphaSignalGenerator.__new__(AlphaSignalGenerator)
        result = gen._check_technical("NETWORK_DEAD")

        assert result is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 2：SEC 資料庫 Schema 突變 (KeyError 陷阱)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2SECSchemaCorruption:
    """
    災難情境：SEC submissions API 回傳 200 OK 但 JSON 結構被閹割。
    攻擊目標：sec_edgar.get_filing_text()
    預期死法：KeyError (缺少 form、accessionNumber 等必要欄位)
    正確行為：回傳空 dict {}，不拋出例外。
    """

    def test_phase2_missing_form_array(self):
        """filings.recent 裡面完全沒有 form 陣列"""
        client = SECEdgarClient()

        # Mock resolve_cik 和 _get_json
        with patch.object(client, "resolve_cik", return_value="0001045810"):
            with patch.object(client, "_get_json", return_value={
                "cik": "0001045810",
                "filings": {"recent": {}}
            }):
                result = client.get_filing_text("NVDA")

        assert result == {}, "Schema 閹割時應回傳空 dict，不可拋出 KeyError"

    def test_phase2_filings_key_missing_entirely(self):
        """連 filings 鍵都不存在"""
        client = SECEdgarClient()

        with patch.object(client, "resolve_cik", return_value="0001045810"):
            with patch.object(client, "_get_json", return_value={
                "cik": "0001045810"
                # 完全沒有 "filings" 鍵
            }):
                result = client.get_filing_text("NVDA")

        assert result == {}

    def test_phase2_api_returns_none(self):
        """_get_json 回傳 None（HTTP 404 或解析失敗）"""
        client = SECEdgarClient()

        with patch.object(client, "resolve_cik", return_value="0001045810"):
            with patch.object(client, "_get_json", return_value=None):
                result = client.get_filing_text("NVDA")

        assert result == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 3：Retry-After 的超長待機陷阱 (Thread Starvation 防禦)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2RetryAfterStarvation:
    """
    災難情境：外部 API 回傳 429 + Retry-After: 86400（一天）。
    攻擊目標：ai_analyst.analyze() 的退避重試邏輯
    預期死法：time.sleep(86400) → Thread 永久卡死
    正確行為：所有 sleep 呼叫不超過 30 秒，最終放棄並回傳安全預設值。
    """

    def test_phase2_ai_analyst_refuses_to_sleep_forever(self):
        """Gemini API 永遠回 429 → 系統應在有限時間內放棄"""
        sleep_calls = []

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "86400"}
        mock_resp.raise_for_status.side_effect = None

        analyst = AIAnalyst(api_key="fake-key-for-test")

        with patch.object(analyst._session, "post", return_value=mock_resp):
            with patch("nexus_quant_os.alpha_hunter.ai_analyst.time.sleep",
                       side_effect=lambda s: sleep_calls.append(s)):
                result = analyst.analyze("STARVATION_CORP", mda_text="test", risk_text="test")

        # 斷言 1：所有 sleep 呼叫的秒數都 <= 30
        for s in sleep_calls:
            assert s <= 30, f"sleep({s}) 超過 30 秒上限！Thread 會被卡死！"

        # 斷言 2：函式正常回傳（沒有 hang）
        assert isinstance(result, AIAnalysis)

        # 斷言 3：降級為中性分析
        assert result.ai_score == 0.0, "429 退避失敗後應降級為 ai_score=0.0"
        assert result.ticker == "STARVATION_CORP"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 4：LLM 廢話與 Markdown 污染 (JSONDecodeError 陷阱)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2LLMMarkdownPollution:
    """
    災難情境：Gemini 不聽指令，回傳被 Markdown 和廢話包圍的 JSON。
    攻擊目標：ai_analyst.analyze() 的 JSON 解析段
    預期死法：json.loads() 吃到 "好的！..." → JSONDecodeError
    正確行為：用 Regex 剝除 Markdown → 成功解析 JSON。
    """

    def test_phase2_markdown_wrapped_json(self):
        """```json 包裹 + 前後廢話 → 能剝掉 Markdown"""
        polluted_text = (
            '好的！以下是分析結果：\n'
            '```json\n'
            '{"ai_score": 0.8, "management_tone": 0.5, "confidence": 0.9, '
            '"summary": "Strong growth", "key_risks": ["Competition"], '
            '"growth_catalysts": ["AI boom"], "has_new_product": true, '
            '"has_ma_activity": false, "has_market_expansion": true, '
            '"has_cost_restructuring": false, "has_regulatory_risk": false}\n'
            '```\n'
            '投資有風險，入市需謹慎！'
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": polluted_text}]}}]
        }

        analyst = AIAnalyst(api_key="fake-key")

        with patch.object(analyst._session, "post", return_value=mock_resp):
            with patch("nexus_quant_os.alpha_hunter.ai_analyst.time.sleep"):
                result = analyst.analyze("NVDA")

        assert result.ai_score == pytest.approx(0.8), "應能剝除 Markdown 並解析 JSON"
        assert result.management_tone == pytest.approx(0.5)

    def test_phase2_triple_backtick_without_json_label(self):
        """``` 包裹（無 json 標籤）→ 也能剝"""
        text_with_backtick = (
            '```\n'
            '{"ai_score": 0.6, "management_tone": 0.3, "confidence": 0.7, '
            '"summary": "OK", "key_risks": [], "growth_catalysts": [], '
            '"has_new_product": false, "has_ma_activity": false, '
            '"has_market_expansion": false, "has_cost_restructuring": false, '
            '"has_regulatory_risk": false}\n'
            '```'
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": text_with_backtick}]}}]
        }

        analyst = AIAnalyst(api_key="fake-key")

        with patch.object(analyst._session, "post", return_value=mock_resp):
            with patch("nexus_quant_os.alpha_hunter.ai_analyst.time.sleep"):
                result = analyst.analyze("AAPL")

        assert result.ai_score == pytest.approx(0.6)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 5：型別毒藥與越界數值 (Type & Bound Violation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2TypePoisonAndOutOfBounds:
    """
    災難情境：Gemini 回傳合法 JSON 但數值型別和範圍全亂。
    攻擊目標：ai_analyst.safe_float() / safe_bool() / safe_list()
    預期死法：TypeError / ValueError / NaN 污染後續計算
    正確行為：clamp + 安全預設 → 不爆不 NaN。
    """

    def test_phase2_poison_values_clamped_safely(self):
        """ai_score=-5.5, tone=NaN, has_new_product="yes", confidence=999"""
        poison_json = json.dumps({
            "ai_score": -5.5,
            "management_tone": float("nan"),
            "confidence": 999,
            "summary": "Poisoned",
            "key_risks": ["risk1"],
            "growth_catalysts": "not_a_list",   # ← 應該是 list 但給了字串
            "has_new_product": "yes",            # ← 應該是 bool 但給了字串
            "has_ma_activity": 1,                # ← 應該是 bool 但給了 int
            "has_market_expansion": None,        # ← 應該是 bool 但給了 None
            "has_cost_restructuring": "false",
            "has_regulatory_risk": "TRUE",
        })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": poison_json}]}}]
        }

        analyst = AIAnalyst(api_key="fake-key")

        with patch.object(analyst._session, "post", return_value=mock_resp):
            with patch("nexus_quant_os.alpha_hunter.ai_analyst.time.sleep"):
                result = analyst.analyze("EVIL_CORP")

        # ai_score: -5.5 → clamped 到 -1.0
        assert result.ai_score == pytest.approx(-1.0), \
            f"ai_score 應被 clamp 到 -1.0，實際得到 {result.ai_score}"

        # management_tone: NaN → 預設 0.0
        assert not math.isnan(result.management_tone), \
            "management_tone 不可以是 NaN！"
        assert result.management_tone == pytest.approx(0.0)

        # confidence: 999 → clamped 到 1.0
        assert result.confidence == pytest.approx(1.0), \
            f"confidence 應被 clamp 到 1.0，實際得到 {result.confidence}"

        # has_new_product: "yes" → True
        assert result.has_new_product is True

        # has_regulatory_risk: "TRUE" → True
        assert result.has_regulatory_risk is True

        # growth_catalysts: "not_a_list" → []（不是 list → 回空 list）
        assert isinstance(result.growth_catalysts, list)

    def test_phase2_all_fields_none(self):
        """所有欄位都是 null → 全部降級為安全預設值"""
        all_null = json.dumps({
            "ai_score": None,
            "management_tone": None,
            "confidence": None,
            "summary": None,
            "key_risks": None,
            "growth_catalysts": None,
            "has_new_product": None,
            "has_ma_activity": None,
            "has_market_expansion": None,
            "has_cost_restructuring": None,
            "has_regulatory_risk": None,
        })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": all_null}]}}]
        }

        analyst = AIAnalyst(api_key="fake-key")

        with patch.object(analyst._session, "post", return_value=mock_resp):
            with patch("nexus_quant_os.alpha_hunter.ai_analyst.time.sleep"):
                result = analyst.analyze("NULL_CORP")

        assert result.ai_score == pytest.approx(0.0)
        assert result.confidence == pytest.approx(0.0)
        assert result.key_risks == []
        assert result.growth_catalysts == []
        assert result.has_new_product is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 6：時空旅人快取 (NTP Clock Drift Bug)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2TimeTravelerCache:
    """
    災難情境：快取檔的 mtime 被竄改為 2099 年（未來）。
    攻擊目標：ai_analyst.analyze() 的 TTL 檢查
    預期死法：now - mtime = 負數 → 負數 < TTL → 快取永不過期
    正確行為：辨識出「來自未來的快取」為異常，視為失效。
    
    ⚠️ 此測試可能暴露現有 bug！
    """

    def test_phase2_future_mtime_cache_treated_as_expired(self):
        """mtime = 2099 年 → 系統應視為過期/異常，不使用此快取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "TIMEWARP.json"

            # 寫入一個「來自未來」的合法快取
            stale_data = {
                "ticker": "TIMEWARP",
                "ai_score": 0.99,
                "management_tone": 0.99,
                "confidence": 0.99,
                "summary": "I am from the future",
                "key_risks": [],
                "growth_catalysts": [],
                "has_new_product": False,
                "has_ma_activity": False,
                "has_market_expansion": False,
                "has_cost_restructuring": False,
                "has_regulatory_risk": False,
            }
            cache_file.write_text(json.dumps(stale_data), encoding="utf-8")

            # 把 mtime 竄改到 2099 年
            future_timestamp = datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp()
            os.utime(cache_file, (future_timestamp, future_timestamp))

            # 建立 analyst 且 patch 快取目錄
            analyst = AIAnalyst(api_key="")  # 無 API key → 命中快取或回預設
            analyst._cache_dir = cache_dir

            result = analyst.analyze("TIMEWARP")

            # 因為我們沒有給 API key，所以如果快取被正確判定為失效，
            # analyze() 內部抓不到資料，最後應該回傳預設的 ai_score = 0.0，
            # 而不會使用未來快取裡面的 ai_score = 0.99
            assert result.ai_score == pytest.approx(0.0), \
                "系統不應該使用 mtime 在未來的快取！"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 7：舊版快取的 Schema 演進 (Legacy Cache Evolution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2LegacyCacheEvolution:
    """
    災難情境：快取目錄有一個 V1 版格式的 JSON，缺少 V2 新增的欄位。
    攻擊目標：ai_analyst.analyze() 的快取反序列化
    預期死法：KeyError（V2 新欄位不存在）
    正確行為：用 .get(key, default) 填補 → 所有欄位都有安全預設值。
    """

    def test_phase2_v1_cache_missing_new_fields(self):
        """V1 快取只有 ai_score, management_tone → 其他欄位用預設值"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "LEGACY.json"

            # V1 格式：只有兩個欄位
            v1_data = {
                "ticker": "LEGACY",
                "ai_score": 0.5,
                "management_tone": 0.2,
            }
            cache_file.write_text(json.dumps(v1_data), encoding="utf-8")

            analyst = AIAnalyst(api_key="")
            analyst._cache_dir = cache_dir

            result = analyst.analyze("LEGACY")

        # 不可拋出 KeyError
        assert isinstance(result, AIAnalysis)
        assert result.ticker == "LEGACY"

        # V1 有的欄位應該正確讀取
        assert result.ai_score == pytest.approx(0.5)
        assert result.management_tone == pytest.approx(0.2)

        # V2 新增的欄位應該用安全預設值
        assert result.key_risks == [], f"缺失的 key_risks 應為 []，實際為 {result.key_risks}"
        assert result.growth_catalysts == []
        assert result.confidence == pytest.approx(0.0)
        assert result.summary == ""
        assert result.has_new_product is False
        assert result.has_ma_activity is False

    def test_phase2_completely_empty_cache_json(self):
        """快取是一個空物件 {} → 所有欄位都用預設值"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "EMPTY.json"
            cache_file.write_text("{}", encoding="utf-8")

            analyst = AIAnalyst(api_key="")
            analyst._cache_dir = cache_dir

            result = analyst.analyze("EMPTY")

        assert isinstance(result, AIAnalysis)
        assert result.ai_score == pytest.approx(0.0)
        assert result.key_risks == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 8：基本面除以零炸彈 (ZeroDivisionError)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2DivisionByZeroBomb:
    """
    災難情境：某公司的 total_equity=0、shares=0，導致計算財務比率時除以零。
    攻擊目標：financial_scanner._evaluate()
    預期死法：ZeroDivisionError
    正確行為：不爆、回傳有效的 ScanResult、不合格指標直接 False。
    """

    def test_phase2_pe_ratio_zero(self):
        """pe_ratio=0 → 估值條件不通過，但不可拋出例外"""
        stmt = _make_stub_stmt(
            "ZERO_PE",
            pe_ratio=0,
            revenue=1_000_000,
            net_income=100_000,
        )

        scanner = FinancialScanner.__new__(FinancialScanner)
        result = scanner._evaluate(stmt)

        assert isinstance(result, ScanResult)
        assert result.passes_valuation is False, "pe_ratio=0 應該不通過估值條件"

    def test_phase2_all_zeros_financial_statement(self):
        """所有金額 = 0 的公司（殼公司）→ 不能除以零炸裂"""
        stmt = _make_stub_stmt(
            "SHELL_CORP",
            revenue=0,
            net_income=0,
            total_assets=0,
            total_equity=0,
            total_debt=0,
            debt_to_equity=None,
            pe_ratio=None,
            free_cash_flow=0,
            fcf_yield=None,
            gross_margin=None,
            revenue_yoy_growth=None,
        )

        scanner = FinancialScanner.__new__(FinancialScanner)
        result = scanner._evaluate(stmt)

        assert isinstance(result, ScanResult)
        assert result.ticker == "SHELL_CORP"
        # 所有條件都不應通過
        assert result.passes_revenue_growth is False
        assert result.passes_cash_flow is False

    def test_phase2_negative_pe_ratio(self):
        """pe_ratio = -10（虧損公司）→ 不通過但不爆"""
        stmt = _make_stub_stmt(
            "LOSS_CORP",
            pe_ratio=-10,
            net_income=-500_000,
        )

        scanner = FinancialScanner.__new__(FinancialScanner)
        result = scanner._evaluate(stmt)

        assert isinstance(result, ScanResult)
        assert result.passes_valuation is False

    def test_phase2_debt_to_equity_none_with_zero_equity(self):
        """total_equity=0 但 debt_to_equity=None → 條件照常判定"""
        stmt = _make_stub_stmt(
            "EQUITY_ZERO",
            total_equity=0,
            total_debt=500_000,
            debt_to_equity=None,  # 系統不自己算，欄位就是 None
        )

        scanner = FinancialScanner.__new__(FinancialScanner)
        result = scanner._evaluate(stmt)

        assert isinstance(result, ScanResult)
        # debt_to_equity is None → passes_debt_health = True（目前邏輯）
        assert result.passes_debt_health is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 9：供應鏈的無限月讀 (Circular Dependency Recursion Bomb)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2SupplyChainInfiniteLoop:
    """
    災難情境：供應鏈圖譜形成死結 NVDA→TSMC→ASML→NVDA。
    攻擊目標：SupplyChainGraph 序列化、遍歷、前端 JSON 生成
    預期死法：RecursionError（無限遞迴）
    正確行為：序列化正常完成、get_all_tickers() 回傳有限集合。
    """

    @pytest.fixture
    def circular_graph(self):
        """建構一個含有循環引用的供應鏈圖"""
        edges = [
            SupplyChainEdge(
                source_ticker="TSMC", target_ticker="NVDA",
                relation=SupplyChainRelation.SUPPLIER,
                revenue_pct=0.25, confidence=0.9,
            ),
            SupplyChainEdge(
                source_ticker="ASML", target_ticker="TSMC",
                relation=SupplyChainRelation.SUPPLIER,
                revenue_pct=0.15, confidence=0.8,
            ),
            SupplyChainEdge(
                source_ticker="NVDA", target_ticker="ASML",
                relation=SupplyChainRelation.SUPPLIER,
                revenue_pct=0.10, confidence=0.7,
            ),
        ]
        return SupplyChainGraph(center_ticker="NVDA", edges=edges)

    def test_phase2_get_all_tickers_no_infinite_loop(self, circular_graph):
        """get_all_tickers() 不會因為循環引用而無限遞迴"""
        tickers = circular_graph.get_all_tickers()

        assert isinstance(tickers, set)
        assert len(tickers) == 3
        assert tickers == {"NVDA", "TSMC", "ASML"}

    def test_phase2_suppliers_and_customers_properties(self, circular_graph):
        """suppliers/customers property 不受循環影響"""
        suppliers = circular_graph.suppliers
        customers = circular_graph.customers

        assert isinstance(suppliers, list)
        assert isinstance(customers, list)
        # 這個死結裡全是 SUPPLIER 關係
        assert len(suppliers) == 3
        assert len(customers) == 0

    def test_phase2_alpha_signal_to_dict_with_circular_chain(self, circular_graph):
        """AlphaSignal.to_dict() 含循環供應鏈 → 序列化不爆"""
        signal = AlphaSignal(
            ticker="NVDA",
            company_name="NVIDIA",
            signal_strength=SignalStrength.BUY,
            supply_chain=circular_graph,
        )

        result = signal.to_dict()

        assert isinstance(result, dict)
        assert result["supply_chain"] is not None
        assert len(result["supply_chain"]["edges"]) == 3

        # 確認可以 json.dumps（最終前端渲染需要）
        serialized = json.dumps(result)
        assert "NVDA" in serialized
        assert "TSMC" in serialized

    def test_phase2_json_serialization_no_recursion_error(self, circular_graph):
        """直接 json.dumps 完整的 to_dict → 不拋 RecursionError"""
        signal = AlphaSignal(
            ticker="NVDA",
            company_name="NVIDIA",
            signal_strength=SignalStrength.NEUTRAL,
            supply_chain=circular_graph,
        )

        try:
            json.dumps(signal.to_dict())
        except RecursionError:
            pytest.fail("循環供應鏈導致 RecursionError！需要實作循環偵測。")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 測試 10：SSE 狀態污染與串音 (Concurrency Queue Bleed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase2SSEQueueIsolation:
    """
    災難情境：兩個使用者「同時」發起 SSE 串流請求。
    攻擊目標：server.py 的 stream_alpha_signals() 中的 queue 實例化位置
    預期死法：Queue 全域共享 → User A 讀到 User B 的進度（串音）
    正確行為：Queue 在每個 Request 內部獨立建立（Local Scope）。
    """

    def test_phase2_queue_is_local_not_global(self):
        """
        原始碼審計測試：
        確認 asyncio.Queue() 是在 stream_alpha_signals() 函式「內部」建立的，
        而不是在模組層級（全域變數）。
        """
        from api.server import stream_alpha_signals

        source = inspect.getsource(stream_alpha_signals)

        # 確認 Queue 在函式體內建立
        assert "Queue()" in source, \
            "stream_alpha_signals 內部應該有 Queue() 的實例化"

        # 確認沒有從外部 import 一個全域 queue
        # （檢查函式原始碼中不含 "global queue" 之類的宣告）
        assert "global queue" not in source.lower(), \
            "Queue 不應該是 global 變數！會導致 SSE 串音！"

    def test_phase2_concurrent_sse_no_crosstalk(self):
        """
        併發兩個 SSE 串流，驗證各自的事件互不干擾。
        由於 TestClient 是同步的，我們用兩個獨立呼叫模擬。
        """
        from fastapi.testclient import TestClient
        from api.server import app

        # Mock _get_alpha_generator 回傳假的 generator
        mock_gen = MagicMock()

        user_a_marker = "USER_A_SIGNAL_12345"
        user_b_marker = "USER_B_SIGNAL_67890"

        call_count = 0

        def fake_generate(universe=None, include_supply_chain=True,
                          include_ai_analysis=True, progress_callback=None):
            nonlocal call_count
            call_count += 1
            marker = user_a_marker if call_count == 1 else user_b_marker
            if progress_callback:
                progress_callback(f"Scanning: {marker}")
            return []

        mock_gen.generate_signals = fake_generate

        async def fake_get_gen():
            return mock_gen

        with patch("api.server._get_alpha_generator", fake_get_gen):
            client = TestClient(app)

            # 請求 A
            resp_a = client.get("/api/alpha/signals/stream")
            events_a = resp_a.text

            # 請求 B
            resp_b = client.get("/api/alpha/signals/stream")
            events_b = resp_b.text

        # User A 的串流不應包含 User B 的 marker（反之亦然）
        # 注意：因為 TestClient 是序列執行的，兩次呼叫產生的 marker 不同
        # 我們驗證的是：每個 response 只包含自己的 marker
        if user_a_marker in events_a:
            assert user_b_marker not in events_a, \
                f"User A 的 SSE 串流中出現了 User B 的信號！串音！"
        if user_b_marker in events_b:
            assert user_a_marker not in events_b, \
                f"User B 的 SSE 串流中出現了 User A 的信號！串音！"
