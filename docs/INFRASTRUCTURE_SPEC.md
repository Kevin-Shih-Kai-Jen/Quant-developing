# Nexus Quant OS — 2026 量化金融基礎設施規格書

> **來源**: Deep Research 深度調查  
> **日期**: 2026-05-26  
> **範疇**: 金融 LLM 基準 / 交易執行引擎 / 零成本供應鏈知識圖譜  
> **狀態**: 技術規格確定，待實作

---

## 模組 1：金融 LLM 推論引擎與基準測試

### 基準測試矩陣

| 基準 | 測試重點 | 風險層級 | 對應防護 |
|:--|:--|:--|:--|
| **FinBen** | 量化數學 (FinQA)、防數字幻覺、股票 Agent 交易 | 🟡 中 | 核心估值邏輯防護 |
| **FLARE** | 時間序列預測、財務試算表 | 🔴 高 | ⚠️ 易誘發多步推理幻覺 |
| **CFinBench** | 9.9 萬題，大中華區合規、稅法 | 🟡 中 | 在地化風險過濾器 |

### 最佳架構：雙層混合裁判 (Dual-Layer MoE)

```
┌─────────────────────────────────────────────────────────┐
│              雙層金融 LLM 推論架構                        │
│                                                         │
│  ┌─── L1: 邊緣本地 / 高頻推論 ──────────────────────┐  │
│  │                                                    │  │
│  │  模型: Fin-R1 (7B)                                 │  │
│  │  底座: Qwen2.5-7B + 6 萬條金融 CoT                │  │
│  │  訓練: SFT + GRPO                                  │  │
│  │                                                    │  │
│  │  ⚠️ 關鍵限制:                                      │  │
│  │    純 RL 會導致金融邏輯崩潰跳躍                     │  │
│  │    必須 SFT 先暖啟動再 GRPO 微調                    │  │
│  │                                                    │  │
│  │  效能: FinQA 76.0 (越級擊敗 70B 模型)              │  │
│  │  延遲: 低延遲，適合即時決策                        │  │
│  │  用途: 日內即時推論、特徵生成、快速篩選             │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                        ↓ 非同步覆核                      │
│  ┌─── L2: 雲端 / 非同步覆核 ────────────────────────┐  │
│  │                                                    │  │
│  │  主力: Qwen2.5-72B                                 │  │
│  │  角色: LLM-as-a-Judge                              │  │
│  │  職責: 最終投資邏輯與合規事實覆核                   │  │
│  │                                                    │  │
│  │  備選: DeepSeek-R1 (671B)                          │  │
│  │  專攻: 跨國長文本、複雜圖表邏輯                    │  │
│  │                                                    │  │
│  │  用途: 覆核 L1 的推論結果、合規檢查、              │  │
│  │        複雜財報分析、跨市場邏輯驗證                 │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Fin-R1 關鍵技術細節

| 項目 | 規格 |
|:--|:--|
| 底座模型 | Qwen2.5-7B |
| 訓練數據 | 6 萬條金融 Chain-of-Thought |
| 訓練方法 | SFT (監督微調) → GRPO (群體相對策略優化) |
| FinQA 分數 | **76.0** (越級擊敗 70B 級模型) |
| 部署方式 | 本地 (Mac M1 可跑 / Docker) |
| 推論延遲 | < 500ms (7B 級別) |

> ⚠️ **訓練陷阱**：純 RL (Reinforcement Learning) 在金融領域會導致「邏輯崩潰跳躍」—— 模型跳過中間推理步驟直接輸出結論，在金融計算中這是致命的。必須先 SFT 暖啟動建立穩定的推理鏈路，再用 GRPO 進行偏好優化。

---

## 模組 2：交易執行引擎與微觀結構防禦

### 目標：消除回測至實盤失真 (Backtest-to-Live Drift)

### API 選型矩陣

| API | 實盤延遲 | Paper 環境品質 | 最佳用途 | 關鍵限制 |
|:--|:--|:--|:--|:--|
| **IBKR (TWS)** | < 100ms | ✅ **無失真** — 網路拓樸與撮合邏輯與實盤 1:1 一致 | 🏆 HFT / 延遲敏感型首選 | 需自費訂閱 L1/L2 數據、API 學習曲線陡峭 |
| **Alpaca** | ~14ms | ⚠️ **模擬延遲悖論** — 內部引擎導致 700ms~數秒延遲；免費版僅 IEX | 中低頻 / Python 生態首選 | ⛔ 嚴禁高頻 |
| **Tradier** | < 100ms | ⚠️ 強制 15 分鐘延遲，無 WebSocket | 選擇權多腿策略首選 | 需外掛 Polygon/Databento |

### Alpaca 防禦性架構指令

> 基於 Nexus Quant OS 的日頻交易特性，Alpaca 是最佳選擇。  
> 但必須嚴格遵守以下防禦規則：

```
⛔ 嚴禁事項:
  - 嚴禁高頻交易 (Paper 環境有 700ms+ 延遲)
  - 嚴禁信任 Paper 環境的成交價格作為滑價基準

✅ 強制規則:
  1. 綑綁 Bracket Orders (OCO/OTO) 鎖住雙邊敞口
     → 每筆買入必須同時設定 Stop Loss + Take Profit
     → 防止 API 斷線後裸倉暴露

  2. 強制微型實盤測試
     → Paper 驗證通過後，用 $100~$500 實盤跑 1 週
     → 確認滑價、成交延遲與 Paper 環境的差異

  3. 冪等性對帳 (與 Deep Think 結論一致)
     → client_order_id 防重複下單
     → 每日首步 cancel_all_orders() + 讀取真實持倉
```

### IBKR 未來升級路線

當系統成熟後，建議從 Alpaca 遷移至 IBKR：

```
遷移檢查清單:
  □ 系統穩定運行 > 3 個月
  □ Paper Trading Sharpe > 1.0
  □ 準備好 L1/L2 數據訂閱費用
  □ 學習 IBKR TWS API (ib_insync 套件)
  □ 建立 IBKR Paper Account 並行測試 2 週
  □ 確認撮合邏輯一致後切換
```

---

## 模組 3：零成本 S&P 500 供應鏈知識圖譜 (KG) ETL

### 免費數據源與技術限制

| 數據源 | API 限制 | 抓取內容 | 用途 |
|:--|:--|:--|:--|
| **GLEIF Level 2 RR-CDF** | REST API, 60 次/分 | Ticker → LEI 映射、垂直母子公司結構 | 實體對齊、企業集團識別 |
| **SEC EDGAR 10-K** | REST API, 10 次/秒 (需合法 User-Agent) | 全文財報 | 供應鏈關係提取 |
| ↳ Exhibit 21 | 同上 | 海外子公司清單 | 避稅結構、微型營運節點 |
| ↳ Item 1 & 1A | 同上 | 業務描述 + 風險因素 | 重大供應商/客戶 (>10% 營收) |

### 4 步自動化 ETL 管線

```
Step 1: 實體對齊 (Entity Mapping)
  ┌──────────────────────────────────────────────┐
  │  Ticker/CIK → LEI 映射                       │
  │  目的: 消除存活者偏差與名稱混淆               │
  │                                              │
  │  "Alphabet Inc." ─┐                          │
  │  "Google LLC"     ├─→ LEI: 5493006... → GOOGL │
  │  "GCP"            ─┘                          │
  │                                              │
  │  工具: GLEIF API + 本地映射字典               │
  └──────────────────────────────────────────────┘
                     ↓
Step 2: 垂直萃取 (Structured Parsing)
  ┌──────────────────────────────────────────────┐
  │  ⛔ 棄用 Regex (太脆弱)                       │
  │  ✅ 使用 edgartools 或 sec-api.io            │
  │                                              │
  │  解析 10-K HTML 表格:                         │
  │  - Exhibit 21 → 海外子公司列表               │
  │  - Item 1     → 業務描述中的供應商/客戶       │
  │  - Item 1A    → 風險因素中的依賴關係          │
  └──────────────────────────────────────────────┘
                     ↓
Step 3: LLM 關係萃取 (Structured Extraction)
  ┌──────────────────────────────────────────────┐
  │  模型: Fin-R1 (7B, 本地推論)                  │
  │  文本切塊 → Prompt 強制 JSON 輸出:            │
  │                                              │
  │  [                                           │
  │    {                                         │
  │      "source": "NVDA",                       │
  │      "target": "TSMC",                       │
  │      "relation_type": "SUPPLIER",            │
  │      "evidence": "We rely on Taiwan Semi...",│
  │      "ratio": "100% of GPU manufacturing"    │
  │    }                                         │
  │  ]                                           │
  │                                              │
  │  ⚠️ ratio 欄位 = Alpha 金礦                  │
  │  → 比 Deep Think 版本多了營收佔比量化          │
  └──────────────────────────────────────────────┘
                     ↓
Step 4: 圖譜部署 (Knowledge Graph)
  ┌──────────────────────────────────────────────┐
  │  Neo4j 知識圖譜                               │
  │                                              │
  │  Cypher 查詢範例:                              │
  │  MATCH (s)-[:SUPPLIES*1..3]->(c:SP500)       │
  │  WHERE s.name = "TSMC"                       │
  │  RETURN c                                    │
  │  → 秒級計算「TSMC 斷鏈的衝擊半徑」            │
  │                                              │
  │  應用場景:                                     │
  │  - 地緣政治風險 (台海) → TSMC 斷鏈影響哪些股  │
  │  - 瓶頸節點識別 → 被低估的關鍵供應商          │
  │  - 傳染效應模擬 → 一家倒閉會拖垮哪些公司      │
  └──────────────────────────────────────────────┘
```

### vs Deep Think 版本的差異

| 項目 | Deep Think | Deep Research | 改進 |
|:--|:--|:--|:--|
| LLM 提取模型 | Qwen 2.5 72B (雲端) | **Fin-R1 7B (本地)** | 零成本 + 低延遲 |
| 實體消歧 | Fuzzy Matching | **GLEIF LEI 映射** | 更準確、標準化 |
| JSON Schema | source/target/relation/evidence | + **ratio (營收佔比)** | 更量化 |
| 解析工具 | sec-parsers | **edgartools / sec-api.io** | 更穩定 |
| 圖譜查詢 | NetworkX | **Neo4j + Cypher** | 秒級多跳查詢 |

---

## 系統行動迴圈 (System Loop)

### 完整自動化閉環

```
┌─────────────────────────────────────────────────────────────┐
│                  ANTIGRAVITY SYSTEM LOOP                     │
│                                                             │
│  [DATA] 數據層                                               │
│  ├─ 排程抓取: GLEIF + EDGAR (每季財報季後觸發)               │
│  ├─ Fin-R1 JSON 萃取 (本地 7B 推論)                         │
│  └─ 更新 Neo4j 供應鏈拓樸                                   │
│                     ↓                                        │
│  [REASONING] 推理層                                          │
│  ├─ Neo4j 向量化 → Worldview (Agentic RAG)                  │
│  ├─ Fin-R1 產出初步權重 (L1 快速推論)                       │
│  └─ Qwen 2.5 72B 裁判覆核 (L2 事實檢查)                    │
│                     ↓                                        │
│  [EXECUTION] 執行層                                          │
│  ├─ 路由至 Alpaca API                                        │
│  ├─ 封裝 Bracket Orders (OCO/OTO) 鎖雙邊敞口               │
│  └─ API 對帳修正 Agent Memory (State Reconciliation)        │
│                     ↓                                        │
│  [MONITOR] 監控層                                            │
│  ├─ 績效追蹤 + Drift 偵測                                   │
│  ├─ 數據品質告警                                             │
│  └─ Telegram 通知                                            │
│                     ↓                                        │
│              ┌──────────────┐                                │
│              │  迴圈重複 ↩  │                                │
│              └──────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### 排程頻率

| 任務 | 頻率 | 觸發方式 |
|:--|:--|:--|
| 市場數據更新 | 每日 | cron 09:00 EST |
| MoE 推論 + 下單 | 每日 | cron 09:25 EST |
| 供應鏈圖譜更新 | 每季 | 財報季後手動 / cron |
| GLEIF 實體映射 | 每月 | cron 月初 |
| 模型重訓練 | 每月 | cron + 手動觸發 |
| 系統健康檢查 | 每日 | cron 08:00 EST |

---

## Deep Think + Deep Research 綜合結論

### 兩份研究的互補性

| 維度 | Deep Think 貢獻 | Deep Research 貢獻 |
|:--|:--|:--|
| **交易架構** | 狀態對帳模式 + Edge Case 防禦 | API 選型矩陣 + Bracket Orders 防禦 |
| **AI Agent** | 三層記憶體設計 + 協作 Pipeline | 雙層 MoE 推論架構 + Fin-R1 規格 |
| **供應鏈 NLP** | Pipeline 精確度分析 + 法說會策略 | GLEIF 實體映射 + Neo4j Cypher + ratio 量化 |

### 最終技術堆疊

```
推論: Fin-R1 (7B, 本地) → Qwen 2.5 72B (雲端覆核)
記憶: JSON Profile + ChromaDB + Working Memory
交易: Alpaca Paper → IBKR (未來)
圖譜: SEC EDGAR + GLEIF → Fin-R1 提取 → Neo4j
監控: pytest + Telegram Bot + Great Expectations
部署: Docker + cron + caffeinate (Mac) / GitHub Actions (雲端)
```

---

*本文件為 Deep Research 深度調查產出，與 [TECHNICAL_ARCHITECTURE_RESEARCH.md](TECHNICAL_ARCHITECTURE_RESEARCH.md) (Deep Think) 互為補充。*
