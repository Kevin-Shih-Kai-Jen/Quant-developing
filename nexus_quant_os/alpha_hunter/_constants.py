"""
alpha_hunter/_constants.py — 全域常數

修改任何常數都必須同步更新本文件的註解。
"""

from pathlib import Path

# ── 資料儲存路徑 ──────────────────────────────────────────
ALPHA_HUNTER_DATA_DIR = Path("data/alpha_hunter")
EDGAR_CACHE_DIR       = ALPHA_HUNTER_DATA_DIR / "edgar_cache"
SUPPLY_CHAIN_CACHE    = ALPHA_HUNTER_DATA_DIR / "supply_chain.json"
SIGNALS_HISTORY_DIR   = ALPHA_HUNTER_DATA_DIR / "signals"

# ── SEC EDGAR API ─────────────────────────────────────────
EDGAR_BASE_URL      = "https://data.sec.gov"
EDGAR_COMPANY_URL   = f"{EDGAR_BASE_URL}/submissions/CIK{{cik}}.json"
EDGAR_XBRL_URL      = f"{EDGAR_BASE_URL}/api/xbrl/companyfacts/CIK{{cik}}.json"
EDGAR_USER_AGENT    = "NexusQuantOS research@example.com"  # SEC 要求
EDGAR_RATE_LIMIT    = 10   # SEC 限制：10 requests/second
EDGAR_REQUEST_INTERVAL = 0.12  # 1/10 + buffer

# ── 篩選閾值 ──────────────────────────────────────────────
MIN_REVENUE_GROWTH_YOY = 0.10        # 10% 年增率
MIN_FCF_YIELD          = 0.03        # 3% FCF yield
MAX_DEBT_TO_EQUITY     = 1.5         # D/E < 1.5
VALUATION_PEER_MULTIPLIER = 1.2      # P/E < 產業中位數 × 1.2
AI_BULLISH_THRESHOLD   = 0.3         # AI score > 0.3 → 看多
TECHNICAL_MA_PERIOD    = 200         # 200 日移動平均

# ── LLM 設定 ──────────────────────────────────────────────
LLM_MAX_INPUT_CHARS    = 8000        # MD&A 截斷長度
LLM_TEMPERATURE        = 0.2         # 低溫度 → 穩定輸出
LLM_TIMEOUT            = 60.0        # 秒
LLM_MAX_RETRIES        = 3

# ── 掃描範圍 ──────────────────────────────────────────────
DEFAULT_SCAN_UNIVERSE = [
    # 科技
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSM", "AMD", "CRM",
    "ADBE", "INTC", "QCOM", "TXN", "AMAT",
    # 半導體供應鏈
    "ASML", "LRCX", "KLAC", "MRVL", "MU", "ON", "NXPI",
    # 科技基建
    "ORCL", "NOW", "SNOW", "PLTR", "PANW", "CRWD", "FTNT",
    # 消費
    "COST", "WMT", "HD", "NKE", "SBUX", "MCD",
    # 醫療
    "UNH", "JNJ", "LLY", "PFE", "ABT", "TMO",
    # 金融
    "JPM", "V", "MA", "GS", "BLK",
    # 工業
    "CAT", "DE", "HON", "UPS", "LMT",
]
# 共 50 檔，可自行增減但不要超過 100（SEC rate limit）

# ── Cache TTL ─────────────────────────────────────────────
EDGAR_CACHE_TTL_DAYS   = 1           # EDGAR 資料快取 1 天
SUPPLY_CHAIN_TTL_DAYS  = 90          # 供應鏈圖譜快取 90 天
AI_ANALYSIS_TTL_DAYS   = 7           # AI 分析快取 7 天
