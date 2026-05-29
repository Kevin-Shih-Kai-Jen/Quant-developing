# Nexus Quant OS — 未來功能路線圖 (Roadmap)

> **最後更新**: 2026-05-26
> **狀態**: 規劃中 — 待 Deep Think / Deep Research 深度研究後啟動

---

## 功能總覽

| # | 功能 | 優先級 | 難度 | 成本 | 預估時程 |
|:--|:--|:--|:--|:--|:--|
| 1 | 模擬交易 API (Paper Trading) | 🟢 已完成 (Moomoo) | ⭐⭐⭐ | 免費 | 1-2 週 |
| 2 | AI 個人理財顧問 Agent | 🟡 高 | ⭐⭐⭐⭐ | 免費~低 | 2-3 週 |
| 3 | Alpha 獵手 Agent (供應鏈追蹤) | 🟠 高 | ⭐⭐⭐⭐⭐ | 中~高 | 4-8 週 |
| 4 | 持續創新功能建議 | 🟢 持續 | — | — | 持續 |
| 5 | 自動化 Bug 偵測 | 🔴 最高 | ⭐⭐ | 免費 | 1 週 |

---

## Phase 5: 模擬交易 API (✅ 已完成)

**目標**: 連接真實的模擬交易 API，讓模型每天自動下單並將報告推播至 Discord。

**技術方案**: FutuOpenD (Moomoo API) + Discord Webhook 通知 + TCP Fail-fast 防護機制。

**成果與現況**:
- 已完成 `FutuBroker` 實作，達成「大腦(模型)」與「手腳(執行)」完全分離的狀態對帳模式 (State Reconciliation)。
- 建立 TCP 端口快速探測，避免 Moomoo 官方 SDK 無限重試導致排程卡死。
- 自動抓取真實 FRED 與 yfinance 數據，推論後一鍵下單並自動發送 Discord 交易報告。

---

## Phase 6: AI 個人理財顧問 Agent

**目標**: 對話式互動，記錄個人財務狀況，根據情境 (如當兵) 推薦最適配置。

**技術方案**: Gemini 2.0 Flash (圖片辨識) + Qwen 2.5 72B (金融推理)

**關鍵需求**:
- 券商截圖辨識持倉
- 情境感知 (時間限制、風險偏好)
- 長期記憶與個人化

---

## Phase 7: Alpha 獵手 Agent

**目標**: 自動掃描財報、追蹤供應鏈、找出被低估的潛力股。

**技術方案**: SEC EDGAR + NLP 供應鏈提取 + DeepSeek-V3 分析

**關鍵需求**:
- 盡量免費
- 供應鏈遞迴追蹤 (NVDA → TSMC → ASM → ...)
- 輸出「3 個月潛力標的」清單

> 💡 **建議開新專案**，與 Nexus Quant OS 分開維護。

---

## 建議執行順序

```
第 1-2 週:  功能 5 (Bug 監控) + 功能 1 (✅ Paper Trading 已完成)
第 3-5 週:  功能 2 (AI 理財顧問)
第 6-10 週: 功能 3 (Alpha 獵手, 新專案)
持續:       功能 4 (持續創新)
```

---

*詳細技術規格請參考 Deep Think / Deep Research 產出的研究報告。*
