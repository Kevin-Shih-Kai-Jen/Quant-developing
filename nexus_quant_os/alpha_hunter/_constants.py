"""
alpha_hunter/_constants.py — 全域常數

修改任何常數都必須同步更新本文件的註解。
"""

from pathlib import Path

# ── 資料儲存路徑 ──────────────────────────────────────────
# Bug #13 fix: Use project root instead of CWD-relative path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # nexus_quant_os/alpha_hunter → nexus_quant_os → project root
ALPHA_HUNTER_DATA_DIR = _PROJECT_ROOT / "data" / "alpha_hunter"
EDGAR_CACHE_DIR       = ALPHA_HUNTER_DATA_DIR / "edgar_cache"
SUPPLY_CHAIN_CACHE    = ALPHA_HUNTER_DATA_DIR / "supply_chain.json"
SUPPLY_CHAIN_NODES_DIR = ALPHA_HUNTER_DATA_DIR / "supply_chain_nodes"
SIGNALS_HISTORY_DIR   = ALPHA_HUNTER_DATA_DIR / "signals"
AI_CACHE_DIR          = ALPHA_HUNTER_DATA_DIR / "ai_cache"

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
LLM_MAX_RETRIES        = 2

# ── Ollama ────────────────────────────────────────────────
OLLAMA_BASE_URL        = "http://localhost:11434"
OLLAMA_MODEL           = "gemma4:e4b"
OLLAMA_TIMEOUT         = 120.0       # 秒
OLLAMA_MAX_INPUT_CHARS = 4000        # Gemma4:e4b 上限較小

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

# ── 台股交易成本 (Edge Case #31: 未計入交易稅導致回測績效膨脹) ──
# 台股賣出時強制扣除證券交易稅，買賣皆扣券商手續費。
# 回測時必須同時扣除這三項成本，否則會高估策略績效。
TW_SELL_TAX = 0.003          # 證券交易稅 0.3%（賣出時強制扣除）
TW_BROKER_FEE = 0.001425     # 券商手續費 0.1425%（買賣皆扣）
TW_BROKER_DISCOUNT = 0.6     # 電子下單手續費折扣（通常六折）

# ── 台股掃描範圍 (Edge Case #34: 包含歷史下市股防禦倖存者偏誤) ──
# 只掃描目前上市的權值股會導致「倖存者偏誤」(survivorship bias)，
# 讓回測績效看起來比實際好。此清單包含歷史重要個股以緩解此問題。
DEFAULT_TW_SCAN_UNIVERSE = [
    "2330.TW", "2454.TW", "2317.TW", "2308.TW", "2382.TW",  # 台積電, 聯發科, 鴻海, 台達電, 廣達
    "2303.TW", "3711.TW", "2412.TW", "2881.TW", "2882.TW",  # 聯電, 日月光, 中華電, 富邦金, 國泰金
    "2886.TW", "2891.TW", "3008.TW", "2345.TW", "6669.TW",  # 兆豐金, 中信金, 大立光, 智邦, 緯穎
    "2357.TW", "3034.TW", "2379.TW", "2395.TW", "3037.TW",  # 華碩, 聯詠, 瑞昱, 研華, 欣興
]
