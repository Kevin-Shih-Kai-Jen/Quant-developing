"""
alpha_hunter/data_client_factory.py — 市場路由器

根據 ticker 自動選擇正確的資料客戶端（SECEdgar 或 TWSE）。

設計原則：
    - 路由邏輯不在 API 層，也不在業務層，而是在「資料層的工廠」裡。
    - 業務層只看到 FinancialDataClient 介面，不知道具體實作。

⚠️ 常見 Bug 警告：
    - 不要對每個請求都建立新的 client 實例！Client 內部有 Session、Cache、
      Rate Limiter，重複建立會浪費資源且可能觸發 Rate Limit。
    - 使用模組級單例 (module-level singleton) 模式。
"""

from __future__ import annotations

import logging
from typing import Optional

from .interfaces import FinancialDataClient
from .ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)

# ── 模組級單例（懶載入）──
_us_client: Optional[FinancialDataClient] = None
_tw_client: Optional[FinancialDataClient] = None


class DataClientFactory:
    """市場路由器。根據 ticker 自動回傳正確的資料客戶端。

    用法：
        client = DataClientFactory.get_client("2330.TW")
        stmt = client.get_latest_financials("2330.TW")

        client2 = DataClientFactory.get_client("NVDA")
        stmt2 = client2.get_latest_financials("NVDA")
    """

    @staticmethod
    def get_client(ticker: str) -> FinancialDataClient:
        """根據 ticker 回傳對應的資料客戶端。

        美股 → SECEdgarClient（既有的）
        台股 → TWSEClient（Phase 1 實作，Phase 0 先 raise NotImplementedError）

        ⚠️ 回傳的是模組級單例，不要在呼叫端手動 close 或 del。
        """
        global _us_client, _tw_client
        market = TickerResolver.detect_market(ticker)

        if market == "TW":
            if _tw_client is None:
                from .twse_client import TWSEClient
                _tw_client = TWSEClient()
            return _tw_client

        # 美股
        if _us_client is None:
            from .sec_edgar import SECEdgarClient
            _us_client = SECEdgarClient()
        return _us_client
