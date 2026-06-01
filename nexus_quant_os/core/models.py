"""
nexus_quant_os/core/models.py — 核心防呆 Pydantic 模型 (The Abyss Armor)

所有跨模組溝通的資料，強制使用此處定義的 BaseModel，防禦型別錯誤與資料污染。
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# ==========================================================
# 1. AI 訊號事件 (Phase 3 & 5)
# ==========================================================

class AIAnalysisFeature(BaseModel):
    """從文本中無情萃取的特徵 (降級後的 LLM 輸出)"""
    capex_upgrade: bool = Field(default=False, description="資本支出是否上修")
    inventory_cleared: bool = Field(default=False, description="庫存是否去化完成")
    new_orders: bool = Field(default=False, description="是否取得關鍵新訂單")
    management_bullish: bool = Field(default=False, description="管理層語氣是否樂觀")
    entropy: float = Field(default=0.0, description="自洽性抽樣的資訊熵 (越低越確定)")

class AlphaSignalEvent(BaseModel):
    """
    訊號生成後發佈的事件。
    使用 PermID 替代 Ticker，防禦代號重用與時序錯亂。
    """
    perm_id: str
    ticker: str  # 輔助用，主要邏輯應看 perm_id
    date: str
    action: str  # "STRONG_BUY", "STRONG_SELL", "LOW_CATCH", "NEUTRAL"
    ai_score: float = 0.0
    quant_score: float = 0.0
    final_score: float = 0.0
    features: Optional[AIAnalysisFeature] = None

# ==========================================================
# 2. 交易與訂單事件 (Phase 2 & 4 & 5)
# ==========================================================

class OrderEvent(BaseModel):
    """由 Portfolio Optimizer 生成的委託單"""
    order_id: str
    perm_id: str
    ticker: str
    date: str
    amount: int  # 正數為買，負數為賣
    order_type: str = "VWAP" # "VWAP", "MKT", "LOW_CATCH"
    target_weight: float = 0.0 # 佔投組的目標權重 (HRP)

class OrderFlowImbalance(BaseModel):
    """OFI 毒性訂單流偵測"""
    ticker: str
    timestamp: datetime
    buy_volume: int
    sell_volume: int
    imbalance_ratio: float
    is_toxic: bool = False

# ==========================================================
# 3. 雙時態財報庫事件 (Phase 1 & 4)
# ==========================================================

class BitemporalFinancialData(BaseModel):
    """雙軸時間資料 (Bitemporal)"""
    perm_id: str
    effective_date: str  # 財報所屬時間 (如 2018-Q1)
    knowledge_date: str  # 市場首次知曉時間 (如 2018-05-15)
    eps: Optional[float] = None
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    is_restated: bool = False # 是否為重編後數據
