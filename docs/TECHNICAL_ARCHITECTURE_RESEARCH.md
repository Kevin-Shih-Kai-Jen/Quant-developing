# Nexus Quant OS — 深度技術架構研究報告

> **來源**: Deep Think 深度分析  
> **日期**: 2026-05-26  
> **範疇**: Paper Trading 架構 / AI Agent 記憶體設計 / 供應鏈 NLP Pipeline  
> **狀態**: 技術規格確定，待實作

---

## 1. Alpaca Paper Trading API — 最佳架構設計

### 核心原則：狀態對帳模式 (State Reconciliation Pattern)

> **Alpaca 是唯一的真實來源 (Single Source of Truth)**。  
> 絕不在本地維護虛擬持倉帳本。執行邏輯必須是「無狀態的自我修復」。

```
每日執行流程:

  ┌─── 1. 決策層 (MoE Router) ───────────────────────┐
  │  輸入: 最新市場數據                                │
  │  輸出: 目標權重 {"SPY": 0.4, "SHY": 0.6, ...}     │
  └──────────────────────────────────────────────────┘
                     ↓
  ┌─── 2. 同步層 (Alpaca API) ───────────────────────┐
  │  GET /v2/account       → 真實總淨值 (Total Equity) │
  │  GET /v2/positions     → 真實持倉 (Positions)       │
  │  DELETE /v2/orders     → 清除昨日殘單               │
  └──────────────────────────────────────────────────┘
                     ↓
  ┌─── 3. 對帳層 (Delta Calculation) ────────────────┐
  │  Delta = (總淨值 × 目標權重) - 當前持倉價值         │
  │  Delta > 0 → 需要買入                              │
  │  Delta < 0 → 需要賣出                              │
  └──────────────────────────────────────────────────┘
                     ↓
  ┌─── 4. 路由層 (Order Execution) ──────────────────┐
  │  ⚠️ 永遠先賣後買 (Sell-first, Buy-second)         │
  │  → 釋放購買力後才能正確計算買入金額                 │
  │  → 每筆訂單帶冪等性 client_order_id                │
  └──────────────────────────────────────────────────┘
```

### Edge Case 防禦矩陣

| 邊界情境 | 潛在災難 | 防禦機制 |
|:--|:--|:--|
| **網路斷線 / API 逾時** | 重複下單買爆倉位 (Double Spend) | **冪等性 (Idempotency)**：每筆訂單帶 `client_order_id` (如 `buy_SPY_20260526`)。搭配 `tenacity` 指數退避重試，Alpaca 自動阻擋重複 ID |
| **部分成交 (Partial Fill)** | 持倉與預期不符，本地記帳大亂 | **放棄追單**：`TimeInForce='day'`。每日首步 `cancel_all_orders()` 清殘單，讀取實際持倉，未補齊的部位自動算進今天的 Delta |
| **休市 / 提早收盤** | 假日報錯，錯過半天市 | **動態排程**：啟動首步呼叫 `alpaca.get_clock()`，`is_open=False` 直接退出，利用 `next_close` 動態調整 |
| **股票分割 / 合併** | 股價突變，系統誤算部位 | **信任券商**：嚴格遵守狀態對帳模式，完全信任 `get_positions()`，券商後台已處理除權息 |

### 實作要點

```python
# 冪等性下單範例
from datetime import date

def place_order(api, symbol, qty, side):
    """帶冪等性的下單函數"""
    order_id = f"{side}_{symbol}_{date.today().isoformat()}"
    return api.submit_order(
        symbol=symbol,
        qty=abs(qty),
        side=side,
        type='market',
        time_in_force='day',
        client_order_id=order_id,  # 冪等性保證
    )

# 每日排程首步：清除殘單 + 讀取真實狀態
def daily_reconcile(api):
    api.cancel_all_orders()          # 清除昨日殘單
    account = api.get_account()       # 真實淨值
    positions = api.list_positions()  # 真實持倉
    clock = api.get_clock()
    if not clock.is_open:
        return None  # 休市，直接退出
    return account, positions
```

---

## 2. 免費 LLM 長期記憶 Agent — 三層混合記憶架構

### 核心原則：讀寫分離的分層記憶 (Tiered Memory)

> 借鑒作業系統與 MemGPT 概念。  
> 絕不把所有對話塞進 Context Window（會失憶 + 幻覺）。

```
┌─────────────────────────────────────────────────────────┐
│                 AI 理財顧問 Agent                        │
│                                                         │
│  ┌─── Layer 1: 短期工作記憶 ─────────────────────────┐ │
│  │  最近 5~10 輪對話 (List)                           │ │
│  │  用途: 維持指代連貫 (「那這檔呢？」)               │ │
│  │  生命週期: 單次 Session                            │ │
│  └────────────────────────────────────────────────────┘ │
│                        ↕                                │
│  ┌─── Layer 2: 情節記憶 (Episodic) ──────────────────┐ │
│  │  輕量 Vector DB (ChromaDB / SQLite-VSS)            │ │
│  │  對話 + 決策理由 → 向量化 → RAG 檢索              │ │
│  │  用途: 「為什麼我上個月賣了 NVDA？」               │ │
│  │  生命週期: 永久保存，按相似度檢索                  │ │
│  └────────────────────────────────────────────────────┘ │
│                        ↕                                │
│  ┌─── Layer 3: 結構化狀態記憶 ⭐ 最核心 ─────────────┐ │
│  │  JSON / SQLite 存儲硬性財務約束                     │ │
│  │  {                                                 │ │
│  │    "risk_tolerance": "low",                        │ │
│  │    "total_assets": 500000,                         │ │
│  │    "constraints": [                                │ │
│  │      {"type": "military", "end": "2026-07-01"}     │ │
│  │    ],                                              │ │
│  │    "positions": {"SPY": 200, "QQQ": 100}           │ │
│  │  }                                                 │ │
│  │  用途: 作為最高權重 System Prompt 注入              │ │
│  │  生命週期: 永久，每次對話更新                      │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Agent 協作 Pipeline（「當兵一個月」完整範例）

```
User: [貼出券商截圖] 「我下個月去當兵」

Step 1 — 感知與寫入 (Gemini 2.0 Flash)
  ├─ Vision: 截圖 OCR → 解析持倉 {"NVDA": 50, "SPY": 200}
  ├─ NLU: 偵測約束條件變更
  └─ Tool Call: update_profile(
       key="constraint",
       value={"type": "military", "start": "2026-07", "duration": "30d"}
     )

Step 2 — 記憶提取 (Memory Router)
  ├─ 讀取 Layer 3 JSON Profile → 發現 military constraint
  ├─ RAG 搜尋 Layer 2 → 找到歷史決策 "上次長假前減碼 NVDA"
  └─ 組裝 Context: Profile + 歷史經驗 + 最新市場數據

Step 3 — 金融推理 (Qwen 2.5 72B)
  ├─ System Prompt 注入: JSON Profile（最高權重）
  ├─ 推理鏈:
  │   當兵 = 無法停損
  │   → 必須壓低波動
  │   → 排除 NVDA (30-day vol > 40%)
  │   → 偏好防禦型 ETF
  └─ 輸出: "建議: 40% SHY + 25% GLD + 20% SPY + 15% TLT"
       附帶理由與風險說明
```

### 模型分工

| 職責 | 模型 | 為什麼 |
|:--|:--|:--|
| **前端感知** (圖片、對話) | Gemini 2.0 Flash | 免費、原生多模態、意圖識別強 |
| **後端推理** (金融分析) | Qwen 2.5 72B (OpenRouter) | 免費、FinBen 基準最高分 |
| **備選推理** | DeepSeek-V3 | 超低成本、長鏈推理極強 |
| **未來升級** | Gemini 3.5 Pro | 付費、全面最強 |

---

## 3. SEC 10-K 供應鏈 NLP Pipeline — 技術架構與精確度

### Pipeline 五階段架構

```
10-K Filing (100,000+ 字)
        ↓
┌─── Stage 1: 定向降噪解析 ────────────────────────┐
│  正則 / sec-parsers 嚴格提取:                      │
│  - Item 1 (Business)           ← 核心客戶揭露      │
│  - Item 1A (Risk Factors)      ← 客戶依賴風險      │
│  - Item 7 (MD&A)               ← 營收結構變化      │
│  效果: 10 萬字 → ~5,000 字                         │
└───────────────────────────────────────────────────┘
        ↓
┌─── Stage 2: 語意分塊與過濾 ──────────────────────┐
│  關鍵字篩選高潛力段落:                              │
│  supplier, customer, rely on, accounted for,        │
│  principal customer, sole source, supply agreement  │
│  效果: 5,000 字 → ~500-1,000 字                    │
└───────────────────────────────────────────────────┘
        ↓
┌─── Stage 3: LLM 結構化提取 ─────────────────────┐
│  Qwen 2.5 72B + JSON Schema 強制輸出:              │
│  [                                                 │
│    {                                               │
│      "source": "NVDA",                             │
│      "target": "TSMC",                             │
│      "relation_type": "SUPPLIER",                  │
│      "confidence": 0.95,                           │
│      "evidence": "We rely on Taiwan Semi..."       │
│    }                                               │
│  ]                                                 │
│  ⚠️ 強制附上原文 evidence → 防幻覺                 │
└───────────────────────────────────────────────────┘
        ↓
┌─── Stage 4: 實體消歧義 ⚠️ 最大技術瓶頸 ─────────┐
│  財報寫法: "Alphabet", "Google", "GCP"              │
│  → 全部映射到標準 Ticker: GOOGL                     │
│                                                    │
│  實作:                                              │
│  - 本地映射字典 (手工 + 自動擴展)                   │
│  - Fuzzy Matching (fuzzywuzzy / rapidfuzz)          │
│  - LLM 二次確認 (低信心度 < 0.7 的結果)             │
└───────────────────────────────────────────────────┘
        ↓
┌─── Stage 5: 圖譜構建 ──────────────────────────┐
│  Neo4j / NetworkX 知識圖譜                        │
│  - 節點: 公司 (附 Ticker, Sector, Market Cap)     │
│  - 邊: 供應/客戶關係 (附 confidence, evidence)    │
│  - 計算: Centrality → 找出瓶頸節點                │
│                                                   │
│  NVDA ──supply──→ TSMC ──supply──→ ASML           │
│    ↑                ↑                              │
│    └── AMD ─────────┘                             │
│                                                   │
│  瓶頸節點 = 被低估的潛在飆股                       │
└───────────────────────────────────────────────────┘
```

### 精確度預期

| 指標 | 預期值 | 原因 |
|:--|:--|:--|
| **準確率 (Precision)** | **85% - 95%** | LLM + evidence 驗證，極少無中生有 |
| **召回率 (Recall)** | **30% - 50%** | ⚠️ **法規硬傷，非技術瓶頸** |

### 召回率低落的根因：SEC 法規限制

> **SEC Regulation S-K Item 101** 規定：只有佔總營收超過 **10%** 的大客戶才需要強制揭露。

這意味著：
- 分散型客戶結構的公司（如 Costco 有數百萬消費者）幾乎不會揭露任何「客戶」
- 大廠的保密協定讓供應商只敢寫「Customer A accounted for 15% of revenues」而**刻意隱藏名字**

### 🔥 突破 Recall 極限的 Alpha 策略

```
數據源擴展: 10-K 文本 + Earnings Call Transcripts (法說會逐字稿)

為什麼法說會有用？
  - 華爾街分析師在 Q&A 環節會咄咄逼人
  - CEO 常會不小心說溜嘴「Customer A」的真實身分
  - 例: "Our largest customer, which is in the smartphone space..."
        → 高信心推斷為 Apple

交叉比對策略:
  10-K: "Customer A accounted for 15% of revenues"
  法說會: "...our partnership with a leading GPU company..."
  → LLM 推論: Customer A = NVIDIA (信心度 0.85)

預期效果:
  Recall 從 30-50% 提升至 55-70%
  這將遠超傳統量化機構的供應鏈數據覆蓋率
```

### 免費法說會數據源

| 來源 | 覆蓋範圍 | 費用 |
|:--|:--|:--|
| **SEC EDGAR 8-K** | 部分公司附逐字稿 | ✅ 免費 |
| **The Motley Fool Transcripts** | S&P 500 主要公司 | ✅ 免費 (需爬蟲) |
| **Seeking Alpha** | 廣泛覆蓋 | ⚠️ 需帳號 (免費 tier) |
| **Financial Modeling Prep** | API 提供逐字稿 | ✅ 免費額度內 |

---

## 實作優先級建議

```
Phase 1 (Week 1-2):
  └─ Alpaca Paper Trading
     - 實作狀態對帳模式
     - 冪等性下單 + Edge Case 防禦
     - Docker cron 排程

Phase 2 (Week 3-5):
  └─ AI 理財顧問 Agent
     - 三層記憶架構
     - Gemini Flash 前端 + Qwen 後端
     - 券商截圖辨識

Phase 3 (Week 6-10):
  └─ Alpha 獵手 Agent (新專案)
     - SEC EDGAR 解析 Pipeline
     - LLM 結構化提取
     - 供應鏈圖譜構建
     - 法說會逐字稿交叉比對
```

---

*本文件為 Deep Think 深度分析產出，作為 Nexus Quant OS 未來功能實作的技術規格書。*
