"""
alpha_hunter/entity_resolver.py — 實體消歧義模組

解決公司名稱不一致問題（如「台積電」、「台積」、「TSMC」皆指向 2330.TW）。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EntityResolver:
    """實體消歧義，利用快取與簡易規則/LLM 減少重複運算。"""

    _cache: dict[str, str] = {
        "台積電": "2330.TW",
        "台積": "2330.TW",
        "TSMC": "2330.TW",
        "聯電": "2303.TW",
        "UMC": "2303.TW",
        "鴻海": "2317.TW",
        "FOXCONN": "2317.TW",
        "聯發科": "2454.TW",
        "MEDIATEK": "2454.TW",
        # ── Edge Case #36: NDA 代稱映射 ──
        # 台灣科技業新聞常用暗語（NDA 代稱）指涉國際客戶，
        # 例如「北美大客戶」幾乎都是指 Apple。
        # 若不做映射，供應鏈圖譜無法正確連結到對應的客戶節點。
        "北美大客戶": "AAPL",
        "水果牌": "AAPL",
        "美系手機大廠": "AAPL",
        "AI晶片霸主": "NVDA",
        "AI GPU大廠": "NVDA",
        "美系GPU大廠": "NVDA",
        "黃仁勳": "NVDA",
        "美系伺服器大廠": "DELL",
    }

    @classmethod
    def resolve_entity(cls, name: str) -> Optional[str]:
        """將不規範的公司名稱轉為標準 ticker。

        實作：先查字典快取，若無則可呼叫 LLM 進行消歧義，再寫入快取。
        """
        clean_name = name.strip().upper()

        # 1. 查快取
        if clean_name in cls._cache:
            return cls._cache[clean_name]

        # 2. TODO: 呼叫 LLM 進行消歧義
        # e.g., ticker = llm_resolve(clean_name)
        # if ticker:
        #     cls._cache[clean_name] = ticker
        #     return ticker

        return None

    # ── Edge Case #37: 集團作帳與張冠李戴 ──
    # 台灣大型集團（如仁寶、緯創）旗下有多家上市公司，
    # 新聞提到母公司名稱時，可能實際指的是子公司。
    # 例如「緯創拿到 AI 伺服器大單」→ 實際受惠的是子公司「緯穎」(6669.TW)，
    # 而非母公司「緯創」(3231.TW) 本身。
    # 若不做語境消歧義，會買錯股票。
    CONGLOMERATE_RULES: dict[str, dict] = {
        "緯創": {
            "default": "3231.TW",       # 緯創資通（母公司）
            "keywords": ["伺服器", "機架", "AI"],
            "redirect": "6669.TW",      # 緯穎（伺服器子公司）
        },
        "仁寶": {
            "default": "2324.TW",       # 仁寶電腦（母公司）
            "keywords": ["伺服器"],
            "redirect": "6895.TW",      # 仁寶旗下伺服器相關子公司
        },
    }

    @classmethod
    def resolve_conglomerate(cls, name: str, context: str) -> Optional[str]:
        """Edge Case #37: 集團作帳與張冠李戴 — 否定過濾。

        根據新聞語境判斷：當提到集團母公司名稱時，
        檢查上下文關鍵字以決定實際受惠的是母公司還是子公司。

        Args:
            name: 公司名稱（如「緯創」）
            context: 新聞標題或內文片段，用於語境判斷

        Returns:
            解析後的 ticker（如 "6669.TW"），若名稱不在規則表中則回傳 None

        範例：
            >>> EntityResolver.resolve_conglomerate("緯創", "緯創拿下 AI 伺服器大單")
            '6669.TW'
            >>> EntityResolver.resolve_conglomerate("緯創", "緯創筆電出貨量成長")
            '3231.TW'
        """
        clean_name = name.strip()

        if clean_name not in cls.CONGLOMERATE_RULES:
            return None

        rule = cls.CONGLOMERATE_RULES[clean_name]

        # 檢查上下文是否包含任何觸發關鍵字
        for keyword in rule["keywords"]:
            if keyword in context:
                logger.info(
                    "集團消歧義：'%s' + 關鍵字 '%s' → 重導至 %s",
                    clean_name, keyword, rule["redirect"],
                )
                return rule["redirect"]

        # 無觸發關鍵字 → 回傳母公司預設 ticker
        return rule["default"]
