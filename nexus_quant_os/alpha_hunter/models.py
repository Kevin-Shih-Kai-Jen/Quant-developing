"""
alpha_hunter/models.py — Alpha Hunter 核心資料模型

所有 dataclass 都是 frozen=False（因為需要漸進式填充）。
所有 Optional 欄位預設為 None。
所有 list 欄位預設為 field(default_factory=list)。
所有 dict 欄位預設為 field(default_factory=dict)。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 列舉型別
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SignalStrength(enum.Enum):
    """信號強度等級。用於 Tier 4 信號生成。"""
    STRONG_BUY  = "STRONG_BUY"    # 3+ 條件符合
    BUY         = "BUY"           # 2 條件符合
    NEUTRAL     = "NEUTRAL"       # <2 條件符合
    AVOID       = "AVOID"         # 負面信號

class ValuationRating(enum.Enum):
    """估值等級。相對於同產業歷史均值。"""
    DEEPLY_UNDERVALUED = "DEEPLY_UNDERVALUED"  # < 0.6x 歷史中位數
    UNDERVALUED        = "UNDERVALUED"          # 0.6x ~ 0.85x
    FAIR               = "FAIR"                 # 0.85x ~ 1.15x
    OVERVALUED         = "OVERVALUED"           # 1.15x ~ 1.5x
    EXTREMELY_OVERVALUED = "EXTREMELY_OVERVALUED"  # > 1.5x

class SupplyChainRelation(enum.Enum):
    """供應鏈關係類型。"""
    SUPPLIER = "SUPPLIER"
    CUSTOMER = "CUSTOMER"
    PARTNER  = "PARTNER"
    COMPETITOR = "COMPETITOR"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 1: 財報資料
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FinancialStatement:
    """一家公司一個季度的財報摘要。

    金額單位由 currency 欄位決定（"USD" 或 "TWD"）。
    所有比率單位：小數（0.15 = 15%）。
    如果 XBRL 中沒有該欄位 → 設為 None（不要設為 0.0）。
    """
    ticker: str                              # e.g. "NVDA" 或 "2330.TW"
    isin_code: Optional[str] = None          # e.g. "TW0002330008" (Edge Case #46)
    listing_date: Optional[datetime] = None  # (Edge Case #46)
    company_name: str = ""                   # e.g. "NVIDIA Corporation"
    filing_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fiscal_year: int = 0
    fiscal_quarter: int = 0
    
    # ── 多市場擴充（Phase 0 新增）──
    cik: Optional[str] = None                # SEC CIK（僅美股有）
    market: str = "US"                       # "US" | "TW"
    currency: str = "USD"                    # "USD" | "TWD"
    stock_id: Optional[str] = None           # 台股代碼（僅台股有，e.g. "2330"）
    fx_rate_to_usd: Optional[float] = None   # 財報截止日的歷史匯率（TWD→USD）

    # 損益表
    revenue: Optional[float] = None                # 營收
    revenue_yoy_growth: Optional[float] = None     # 營收年增率（小數）
    gross_profit: Optional[float] = None           # 毛利
    gross_margin: Optional[float] = None           # 毛利率（小數）
    operating_income: Optional[float] = None       # 營業利益
    operating_margin: Optional[float] = None       # 營業利益率（小數）
    net_income: Optional[float] = None             # 淨利
    eps_diluted: Optional[float] = None            # 稀釋每股盈餘

    # 資產負債表
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_debt: Optional[float] = None
    debt_to_equity: Optional[float] = None         # 負債比（小數）

    # 現金流量表
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # 估值指標（需要搭配市場價格計算）
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None               # 本益比
    ps_ratio: Optional[float] = None               # 股價營收比
    pb_ratio: Optional[float] = None               # 股價淨值比
    ev_ebitda: Optional[float] = None              # 企業價值/EBITDA
    fcf_yield: Optional[float] = None              # 自由現金流殖利率（小數）


@dataclass
class ScanResult:
    """Tier 1 掃描結果：一家公司是否通過基本面篩選。

    每個布林欄位代表一個篩選條件。
    """
    ticker: str
    company_name: str
    latest_statement: FinancialStatement
    isin_code: Optional[str] = None          # (Edge Case #46)
    listing_date: Optional[datetime] = None  # (Edge Case #46)
    scan_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    estimated_eps: Optional[float] = None
    is_estimate: bool = False
    flags: list[str] = field(default_factory=list)
    eps_yoy: float = 0.0                     # 記錄計算出的 EPS YoY

    # 篩選條件結果（每個都是 bool）
    passes_revenue_growth: bool = False      # 營收 YoY > 10%
    passes_margin_expansion: bool = False    # 毛利率 QoQ 改善
    passes_valuation: bool = False           # P/E < 產業中位數 * 1.2
    passes_cash_flow: bool = False           # FCF > 0 且 FCF yield > 3%
    passes_debt_health: bool = False         # D/E < 1.5

    valuation_rating: ValuationRating = ValuationRating.FAIR

    @property
    def pass_count(self) -> int:
        """通過的條件數量（0-5）。"""
        return sum([
            self.passes_revenue_growth,
            self.passes_margin_expansion,
            self.passes_valuation,
            self.passes_cash_flow,
            self.passes_debt_health,
        ])

    @property
    def is_candidate(self) -> bool:
        """通過 3 個以上條件 → 候選標的。"""
        return self.pass_count >= 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 2: 供應鏈
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SupplyChainEdge:
    """供應鏈圖中的一條邊。"""
    source_ticker: str                     # e.g. "TSMC"（供應商）
    target_ticker: str                     # e.g. "NVDA"（客戶）
    relation: SupplyChainRelation
    revenue_pct: Optional[float] = None    # 佔營收百分比（小數，e.g. 0.25 = 25%）
    confidence: float = 0.5                # NLP 提取信心度 [0, 1]
    depth: int = 0                         # 在哪一層被發現的
    source_filing: Optional[str] = None    # 來源文件（e.g. "10-K 2024"）
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SupplyChainGraph:
    """一家公司的完整供應鏈圖譜。"""
    center_ticker: str                     # 中心公司（e.g. "NVDA"）
    edges: list[SupplyChainEdge] = field(default_factory=list)
    build_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def suppliers(self) -> list[SupplyChainEdge]:
        return [e for e in self.edges if e.relation == SupplyChainRelation.SUPPLIER]

    @property
    def customers(self) -> list[SupplyChainEdge]:
        return [e for e in self.edges if e.relation == SupplyChainRelation.CUSTOMER]

    def get_all_tickers(self) -> set[str]:
        """回傳圖中所有 ticker（包含中心公司）。"""
        tickers = {self.center_ticker}
        for e in self.edges:
            tickers.add(e.source_ticker)
            tickers.add(e.target_ticker)
        return tickers


@dataclass
class RecursiveSupplyChainGraph:
    """多層遞迴供應鏈圖譜。向後相容：不影響現有 SupplyChainGraph。"""
    center_ticker: str
    edges: list[SupplyChainEdge] = field(default_factory=list)
    build_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── 遞迴特有欄位 ──
    max_depth: int = 1
    node_depths: dict[str, int] = field(default_factory=dict)
    visited_tickers: set[str] = field(default_factory=set)
    api_calls_used: int = 0
    llm_source: dict[str, str] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        tickers = {self.center_ticker}
        for e in self.edges:
            tickers.add(e.source_ticker)
            tickers.add(e.target_ticker)
        return len(tickers)

    @property
    def layers(self) -> dict[int, list[str]]:
        result: dict[int, list[str]] = {}
        for ticker, depth in self.node_depths.items():
            result.setdefault(depth, []).append(ticker)
        return result

    def get_edges_at_depth(self, depth: int) -> list[SupplyChainEdge]:
        tickers_at_depth = {t for t, d in self.node_depths.items() if d == depth}
        return [
            e for e in self.edges
            if e.source_ticker in tickers_at_depth or e.target_ticker in tickers_at_depth
        ]

    def to_dict(self) -> dict:
        """轉為 JSON-safe 字典（用於 API 回傳）。"""
        return {
            "center_ticker": self.center_ticker,
            "max_depth": self.max_depth,
            "total_nodes": self.total_nodes,
            "total_edges": len(self.edges),
            "api_calls_used": self.api_calls_used,
            "build_timestamp": self.build_timestamp.isoformat(),
            "layers": {str(k): v for k, v in self.layers.items()},
            "nodes": [
                {
                    "ticker": t,
                    "depth": d,
                    "llm_source": self.llm_source.get(t, "unknown"),
                }
                for t, d in sorted(self.node_depths.items(), key=lambda x: x[1])
            ],
            "edges": [
                {
                    "source_ticker": e.source_ticker,
                    "target_ticker": e.target_ticker,
                    "relation": e.relation.value,
                    "revenue_pct": e.revenue_pct,
                    "confidence": e.confidence,
                    "depth": e.depth,
                }
                for e in self.edges
            ],
            "cost_summary": {
                "gemini_calls": sum(1 for v in self.llm_source.values() if v in ("gemini", "gemini_fallback")),
                "ollama_calls": sum(1 for v in self.llm_source.values() if v == "ollama"),
                "cache_hits": sum(1 for v in self.llm_source.values() if v == "cache"),
            }
        }


@dataclass
class EnrichedNode:
    ticker: str
    depth: int
    llm_source: str = "unknown"
    
    # ── 多市場擴充（Phase 0 新增）──
    market: str = "US"                          # "US" | "TW"
    high_freq_catalyst_score: float = 0.0       # 台股月營收動能 Z-Score
    
    # ── 深度掃描結果 ──
    composite_score: Optional[float] = None      # [0, 1]
    signal_strength: str = "NEUTRAL"
    valuation_rating: str = "N/A"
    fundamental_pass: bool = False
    ai_bullish: bool = False
    technical_confirm: bool = False
    
    # ── 圖論計算結果 ──
    degree_centrality: float = 0.0
    network_alpha_score: float = 0.0
    spillover_delta: float = 0.0
    
    # ── 掃描狀態 ──
    scan_status: str = "pending"  # pending → scanning → done / error
    
    # ── 前端互動用 ──
    top_suppliers: list[str] = field(default_factory=list)
    top_customers: list[str] = field(default_factory=list)
    company_name: str = ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 3: AI 分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AIAnalysis:
    """LLM 對一家公司的深度分析結果。"""
    ticker: str
    analysis_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # MD&A 分析
    management_tone: float = 0.0           # [-1, 1]（負面/正面）
    key_risks: list[str] = field(default_factory=list)       # 最多 5 條
    growth_catalysts: list[str] = field(default_factory=list) # 最多 5 條

    # 催化劑偵測
    has_new_product: bool = False
    has_ma_activity: bool = False           # 併購活動
    has_market_expansion: bool = False      # 新市場擴張
    has_cost_restructuring: bool = False    # 成本重組
    has_regulatory_risk: bool = False       # 監管風險

    # 整體評分
    ai_score: float = 0.0                  # [-1, 1]（看空/看多）
    confidence: float = 0.0                # [0, 1]（信心度）
    summary: str = ""                      # 一段話摘要（英文，200字以內）

    @property
    def catalyst_count(self) -> int:
        return sum([
            self.has_new_product,
            self.has_ma_activity,
            self.has_market_expansion,
            self.has_cost_restructuring,
        ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 4: 最終信號
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AlphaSignal:
    """最終的 Alpha 信號 — 一個潛在的投資機會。"""
    ticker: str
    company_name: str
    signal_strength: SignalStrength
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # 各 Tier 的原始輸出（完整保留，供 audit 追蹤）
    scan_result: Optional[ScanResult] = None
    supply_chain: Optional[SupplyChainGraph] = None
    ai_analysis: Optional[AIAnalysis] = None

    # 信號依據（布林條件）
    fundamental_pass: bool = False          # Tier 1 通過
    supply_chain_healthy: bool = False      # Tier 2 無重大風險
    ai_bullish: bool = False                # Tier 3 AI 看多 (score > 0.3)
    technical_confirm: bool = False         # 價格在 200MA 之上

    # 量化指標
    composite_score: float = 0.0            # [0, 1] 綜合評分
    target_return_3m: Optional[float] = None  # 預期 3 個月報酬（小數）

    @property
    def confirmation_count(self) -> int:
        return sum([
            self.fundamental_pass,
            self.supply_chain_healthy,
            self.ai_bullish,
            self.technical_confirm,
        ])

    def to_dict(self) -> dict:
        """轉為 JSON-safe 字典（用於 API 回傳）。"""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "signal_strength": self.signal_strength.value,
            "generated_at": self.generated_at.isoformat(),
            "composite_score": round(self.composite_score, 4),
            "target_return_3m": self.target_return_3m,
            "fundamental_pass": self.fundamental_pass,
            "supply_chain_healthy": self.supply_chain_healthy,
            "ai_bullish": self.ai_bullish,
            "technical_confirm": self.technical_confirm,
            "confirmation_count": self.confirmation_count,
            "ai_summary": self.ai_analysis.summary if self.ai_analysis else "",
            "catalysts": self.ai_analysis.growth_catalysts if self.ai_analysis else [],
            "risks": self.ai_analysis.key_risks if self.ai_analysis else [],
            "valuation": self.scan_result.valuation_rating.value if self.scan_result else "N/A",
            "supply_chain": {
                "center_ticker": self.supply_chain.center_ticker,
                "edges": [
                    {
                        "source_ticker": e.source_ticker,
                        "target_ticker": e.target_ticker,
                        "relation": e.relation.value,
                        "revenue_pct": e.revenue_pct
                    }
                    for e in self.supply_chain.edges
                ]
            } if self.supply_chain else None,
        }
