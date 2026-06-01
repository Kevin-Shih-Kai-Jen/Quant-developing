"""
alpha_hunter/interfaces.py — 跨市場抽象介面

所有市場的資料客戶端（SEC EDGAR、TWSE）都必須實作這個 Protocol。
業務邏輯層（FinancialScanner、SignalGenerator）只依賴這個介面，
不依賴具體的 SECEdgarClient 或 TWSEClient。

設計原則：
    1. Protocol 不需要繼承，只需要方法簽名匹配（鴨子型別）。
    2. 台股專有方法（如 get_monthly_revenue）不放在共用介面裡，
       由呼叫方用 hasattr() 或 isinstance() 檢查。
    3. 所有方法都回傳 None 代表「取不到資料」，不拋例外。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .models import FinancialStatement


@runtime_checkable
class FinancialDataClient(Protocol):
    """所有市場資料客戶端的共用介面。

    SECEdgarClient 和未來的 TWSEClient 都必須有這些方法。
    使用 @runtime_checkable 讓 isinstance() 可以在執行時檢查。

    用法範例：
        client: FinancialDataClient = DataClientFactory.get_client("2330.TW")
        stmt = client.get_latest_financials("2330.TW")
    """

    def get_latest_financials(self, ticker: str) -> Optional[FinancialStatement]:
        """取得最新一季財報。找不到回傳 None。"""
        ...

    def get_financials_history(
        self, ticker: str, n_quarters: int = 8
    ) -> list[FinancialStatement]:
        """取得最近 N 季的財報。找不到回傳空 list。"""
        ...

    def get_filing_text(
        self, ticker: str, filing_type: str = "10-K",
        sections: list[str] | None = None
    ) -> dict[str, str]:
        """取得年報/季報的文字內容。找不到回傳空 dict。"""
        ...


@dataclass(frozen=True)
class MarketConfig:
    """一個市場的所有設定，集中管理。

    frozen=True 是因為設定不該在運行時被修改。
    如果需要覆蓋，建立一個新的 MarketConfig 實例。

    用法範例：
        config = MARKET_CONFIGS["TW"]
        if pe > config.valuation_pe_cap:
            rating = "OVERVALUED"
    """
    market_id: str                    # "US" | "TW"
    currency: str                     # "USD" | "TWD"
    timezone: str                     # "America/New_York" | "Asia/Taipei"
    exchange_mic: str                 # "XNYS" | "XTAI"
    lot_size: int = 1                 # 美股=1股, 台股=1000股(一張)

    # ── 估值門檻（取代 financial_scanner.py 的硬編碼）──
    valuation_pe_cap: float = 30.0    # P/E 低於此值視為「合格」
    valuation_tiers: dict = field(default_factory=lambda: {
        "DEEPLY_UNDERVALUED": 10,
        "UNDERVALUED": 18,
        "FAIR": 28,
        "OVERVALUED": 45,
    })
    max_debt_to_equity: float = 1.5
    debt_exempt_sectors: list = field(default_factory=list)

    # ── 資料來源 ──
    filing_types: dict = field(default_factory=lambda: {
        "annual": "10-K",
        "quarterly": "10-Q",
    })
    news_sources: list = field(default_factory=list)
    macro_api_sources: dict = field(default_factory=dict)

    # ── 掃描範圍 ──
    default_scan_universe: list = field(default_factory=list)


from ._constants import DEFAULT_SCAN_UNIVERSE

# ── 預設的市場設定實例 ──
# 把原本散落在 _constants.py 和 financial_scanner.py 的硬編碼集中到這裡
US_MARKET_CONFIG = MarketConfig(
    market_id="US",
    currency="USD",
    timezone="America/New_York",
    exchange_mic="XNYS",
    lot_size=1,
    valuation_pe_cap=30.0,
    valuation_tiers={
        "DEEPLY_UNDERVALUED": 10,
        "UNDERVALUED": 18,
        "FAIR": 28,
        "OVERVALUED": 45,
    },
    max_debt_to_equity=1.5,
    debt_exempt_sectors=[],
    filing_types={"annual": "10-K", "quarterly": "10-Q"},
    news_sources=[
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
    ],
    default_scan_universe=DEFAULT_SCAN_UNIVERSE
)

TW_MARKET_CONFIG = MarketConfig(
    market_id="TW",
    currency="TWD",
    timezone="Asia/Taipei",
    exchange_mic="XTAI",
    lot_size=1000,  # ← 一張 = 1000 股，這是台股跟美股最致命的差異
    valuation_pe_cap=25.0,  # 台灣半導體 P/E 中位數約 15-22，cap 設 25
    valuation_tiers={
        "DEEPLY_UNDERVALUED": 8,
        "UNDERVALUED": 14,
        "FAIR": 22,
        "OVERVALUED": 35,
    },
    max_debt_to_equity=1.5,
    debt_exempt_sectors=["Financials", "Banks", "Insurance"],
    filing_types={"annual": "年報", "quarterly": "季報"},
    news_sources=[
        "https://www.moneydj.com/rss/rss.ashx",
    ],
    macro_api_sources={
        "ndc_signal": "https://index.ndc.gov.tw/n/opendata/...",
    },
)

# 用 dict 做路由查詢
MARKET_CONFIGS: dict[str, MarketConfig] = {
    "US": US_MARKET_CONFIG,
    "TW": TW_MARKET_CONFIG,
}
