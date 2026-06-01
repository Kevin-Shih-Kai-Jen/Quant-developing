"""
alpha_hunter/ticker_resolver.py — 跨市場 Ticker 解析與標準化

解決的問題：
    1. TSM (美股 ADR) 和 2330.TW (台股原股) 是同一家公司，
       圖譜裡只能有一個節點（母節點 = 原上市地）。
    2. 判斷一個 ticker 是美股還是台股。
    3. 台股 ticker 送去 yfinance 時需要加 .TW 後綴。

設計原則：
    - 這是一個「純函數」工具類別，不做任何 I/O（不打 API、不讀檔案）。
    - 所有映射表都是靜態的 dict，可以在測試中輕鬆 Mock。
    - 所有方法都是 @classmethod 或 @staticmethod，不需要實例化。

⚠️ 常見 Bug 警告：
    - 不要用 ticker.isdigit() 判斷台股！"2330.TW" 不是純數字。
    - 不要假設所有 4 碼數字都是台股！可能是權證代碼（需要 Universe Filter 擋）。
    - KY 股的 ticker 格式是 "xxxx-KY"（如 "5871-KY"），不要漏掉。
"""

from __future__ import annotations

import re
from typing import Optional

from .interfaces import MARKET_CONFIGS, MarketConfig


class TickerResolver:
    """跨市場 Ticker 解析器。"""

    # ── ADR ↔ 原股映射表 ──
    # 用途：圖譜建構時，NVDA 的 10-K 提到 "TSMC"，LLM 推測代碼 "TSM"。
    # 我們需要把 TSM 折疊回 2330.TW，避免圖譜出現重複節點。
    ADR_TO_PRIMARY: dict[str, str] = {
        "TSM": "2330.TW",     # 台積電
        "UMC": "2303.TW",     # 聯電
        "ASX": "3711.TW",     # 日月光投控
        "IMOS": "8150.TW",    # 南茂科技
        "SPIL": "2325.TW",    # 矽品（已下市但保留映射）
        "CHT": "2412.TW",     # 中華電信
    }

    # 反向映射（自動產生）
    PRIMARY_TO_ADR: dict[str, str] = {v: k for k, v in ADR_TO_PRIMARY.items()}

    # ── Edge Case #39: 英文代號的台股公司 ──
    # 部分台股上市公司使用英文 ticker（如 M31、EMEMORY），
    # 這些代號符合美股格式（1-5 個大寫字母），但實際上是台股。
    # 若不特別處理，detect_market() 會誤判為美股，導致：
    #   1. 使用錯誤的資料源（SEC EDGAR 而非 TWSE）
    #   2. 無法取得財報和月營收
    #   3. 供應鏈圖譜節點錯誤
    TW_ENGLISH_NAMES: dict[str, str] = {
        "M31": "6643.TW",         # M31（矽智財 IP）
        "EMEMORY": "3529.TW",     # 力旺（嵌入式記憶體 IP）
        "EGIS": "6462.TW",        # 神盾（指紋辨識 IC）
        "ADATA": "3260.TW",       # 威剛（記憶體模組）
        "ITE": "6213.TW",         # 聯陽（IO 控制 IC）
        "ALI": "3533.TW",         # 嘉澤端子（連接器）
    }

    # ── Ticker 格式判斷用正則 ──
    # 美股：1-5 個大寫字母（如 NVDA, AAPL, BRK.B → 不含 .TW）
    US_PATTERN = re.compile(r'^[A-Z]{1,5}$')
    # 台股普通股：4 碼數字，可選 .TW 或 .TWO 後綴
    TW_PATTERN = re.compile(r'^\d{4}(\.TW|\.TWO)?$')
    # 台股 KY 股：4 碼數字 + -KY 或 .TW
    TW_KY_PATTERN = re.compile(r'^\d{4}-KY(\.TW)?$')
    # 衍生品垃圾代碼（權證、牛熊證）：5-6 碼純數字
    DERIVATIVE_PATTERN = re.compile(r'^\d{5,6}$')

    @classmethod
    def detect_market(cls, ticker: str) -> str:
        """判斷 ticker 屬於哪個市場。

        回傳 "US" 或 "TW"。
        如果無法判斷，預設回傳 "US"（向後相容）。

        範例：
            detect_market("NVDA")      → "US"
            detect_market("2330.TW")   → "TW"
            detect_market("2330")      → "TW"
            detect_market("TSM")       → "US"  （ADR 本身在美股交易）
            detect_market("5871-KY")   → "TW"
        """
        raw = ticker.upper().strip()

        # 明確帶 .TW / .TWO → 台股
        if raw.endswith(('.TW', '.TWO')):
            return "TW"

        # KY 股
        if cls.TW_KY_PATTERN.match(raw):
            return "TW"

        # 純 4 碼數字 → 台股
        if re.match(r'^\d{4}$', raw):
            return "TW"

        # Edge Case #39: 英文代號的台股公司
        # 必須在預設回傳 "US" 之前檢查，否則會誤判
        if raw in cls.TW_ENGLISH_NAMES:
            return "TW"

        # 其他 → 美股（包含 TSM 這種 ADR）
        return "US"

    @classmethod
    def get_market_config(cls, ticker: str) -> MarketConfig:
        """取得 ticker 對應的市場設定。"""
        market = cls.detect_market(ticker)
        return MARKET_CONFIGS[market]

    @classmethod
    def to_canonical(cls, raw_ticker: str) -> str:
        """將所有別名轉為系統唯一的「母實體 ID」。

        用途：圖譜節點 ID 必須用這個函數產生的值。
        規則：ADR 代碼 → 轉為原上市地代碼。其他不變。

        範例：
            to_canonical("TSM")      → "2330.TW"
            to_canonical("NVDA")     → "NVDA"
            to_canonical("2330.TW")  → "2330.TW"
            to_canonical("2330")     → "2330.TW"  （補上 .TW 後綴）
        """
        raw = raw_ticker.upper().strip()

        # 先查 ADR 映射
        if raw in cls.ADR_TO_PRIMARY:
            return cls.ADR_TO_PRIMARY[raw]

        # Edge Case #39: 英文代號的台股公司 → 轉為標準台股 ticker
        if raw in cls.TW_ENGLISH_NAMES:
            return cls.TW_ENGLISH_NAMES[raw]

        # 純 4 碼數字 → 補上 .TW
        if re.match(r'^\d{4}$', raw):
            return f"{raw}.TW"

        return raw

    @classmethod
    def to_yfinance(cls, ticker: str) -> str:
        """將 ticker 轉為 yfinance 可接受的格式。

        yfinance 要求台股必須帶 .TW 後綴。
        美股直接回傳原 ticker。

        範例：
            to_yfinance("NVDA")      → "NVDA"
            to_yfinance("2330.TW")   → "2330.TW"
            to_yfinance("2330")      → "2330.TW"
            to_yfinance("TSM")       → "TSM"  （ADR 在美股交易，用 TSM 就好）
        """
        raw = ticker.upper().strip()
        if re.match(r'^\d{4}$', raw):
            return f"{raw}.TW"
        return raw

    @classmethod
    def is_derivative(cls, ticker: str) -> bool:
        """判斷是否為衍生品（權證、牛熊證）垃圾代碼。

        這些代碼沒有財報，不能放進圖譜。
        """
        return bool(cls.DERIVATIVE_PATTERN.match(ticker.strip()))
