"""
tests/test_edge_cases_30_41.py — 第 30~41 號邊界災難防禦測試

驗證所有深水區 Edge Cases 的防禦機制是否正確實作。
"""
import pytest
from datetime import datetime, timezone, timedelta

from nexus_quant_os.alpha_hunter.order_slicer import OrderSlicer
from nexus_quant_os.alpha_hunter._constants import TW_SELL_TAX, DEFAULT_TW_SCAN_UNIVERSE
from nexus_quant_os.alpha_hunter.entity_resolver import EntityResolver
from nexus_quant_os.alpha_hunter.ticker_resolver import TickerResolver
from nexus_quant_os.alpha_hunter.twse_client import TWSEClient


# ── Edge Case #30: 499 張天花板 ──

class TestOrderSlicer:
    def test_order_slicer_basic(self):
        """600 張 → [499, 101]（自動拆單）"""
        result = OrderSlicer.slice_order(600_000, market="TW")
        assert result == [499_000, 101_000]

    def test_order_slicer_exact_499(self):
        """剛好 499 張 → 不需拆單"""
        result = OrderSlicer.slice_order(499_000, market="TW")
        assert result == [499_000]

    def test_order_slicer_us_no_split(self):
        """美股不需切單"""
        result = OrderSlicer.slice_order(1_000_000, market="US")
        assert result == [1_000_000]

    def test_order_slicer_negative_shares(self):
        """負數股數 → ValueError"""
        with pytest.raises(ValueError):
            OrderSlicer.slice_order(-100, market="TW")

    def test_shares_to_lots(self):
        assert OrderSlicer.shares_to_lots(5000) == 5
        assert OrderSlicer.shares_to_lots(999) == 1

    def test_lots_to_shares(self):
        assert OrderSlicer.lots_to_shares(10) == 10_000


# ── Edge Case #31: 證券交易稅 ──

class TestTransactionTax:
    def test_tw_sell_tax_constant(self):
        """確認 TW_SELL_TAX = 0.003"""
        assert TW_SELL_TAX == 0.003


# ── Edge Case #34: 倖存者偏誤 ──

class TestSurvivorshipBias:
    def test_default_tw_scan_universe_exists(self):
        """DEFAULT_TW_SCAN_UNIVERSE 不為空且包含台積電"""
        assert len(DEFAULT_TW_SCAN_UNIVERSE) > 0
        assert "2330.TW" in DEFAULT_TW_SCAN_UNIVERSE


# ── Edge Case #36: NDA 代稱映射 ──

class TestNDAAliases:
    def test_entity_resolver_nda_aliases(self):
        """「北美大客戶」→ AAPL"""
        assert EntityResolver.resolve_entity("北美大客戶") == "AAPL"
        assert EntityResolver.resolve_entity("AI晶片霸主") == "NVDA"
        assert EntityResolver.resolve_entity("水果牌") == "AAPL"
        assert EntityResolver.resolve_entity("黃仁勳") == "NVDA"


# ── Edge Case #37: 集團張冠李戴 ──

class TestConglomerateDisambiguation:
    def test_entity_resolver_conglomerate(self):
        """緯創 + 伺服器上下文 → 6669.TW（緯穎）"""
        result = EntityResolver.resolve_conglomerate("緯創", "AI 伺服器大單")
        assert result == "6669.TW"

    def test_entity_resolver_conglomerate_default(self):
        """緯創 + 無特殊上下文 → 3231.TW（緯創本體）"""
        result = EntityResolver.resolve_conglomerate("緯創", "筆電代工訂單增加")
        assert result == "3231.TW"


# ── Edge Case #39: 英文代號台股 ──

class TestEnglishTickerNames:
    def test_ticker_resolver_english_names(self):
        """M31 → 6643.TW"""
        assert TickerResolver.detect_market("M31") == "TW"
        assert TickerResolver.to_canonical("M31") == "6643.TW"

    def test_ememory(self):
        """EMEMORY → 3529.TW"""
        assert TickerResolver.to_canonical("EMEMORY") == "3529.TW"


# ── Edge Case #32: 尾盤集合競價 ──

class TestCallAuction:
    def test_call_auction_period_true(self):
        """13:26 台北時間 → True"""
        import zoneinfo
        taipei_tz = zoneinfo.ZoneInfo("Asia/Taipei")
        dt = datetime(2024, 6, 1, 13, 26, 0, tzinfo=taipei_tz)
        assert TWSEClient.is_call_auction_period(dt) is True

    def test_call_auction_period_false(self):
        """13:00 台北時間 → False"""
        import zoneinfo
        taipei_tz = zoneinfo.ZoneInfo("Asia/Taipei")
        dt = datetime(2024, 6, 1, 13, 0, 0, tzinfo=taipei_tz)
        assert TWSEClient.is_call_auction_period(dt) is False


# ── Edge Case #40: 斷路器 ──

class TestCircuitBreaker:
    def test_circuit_breaker_flag(self):
        """TWSEClient 有 _circuit_open 屬性"""
        client = TWSEClient()
        assert hasattr(client, "_circuit_open")
        assert client._circuit_open is False
