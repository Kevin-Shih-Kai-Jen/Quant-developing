# Nexus Quant OS — 功能路線圖 (Roadmap)

> **最後更新**: 2026-05-29
> **當前版本**: v3.2 — 全自動化量化交易作業系統

---

## 系統架構六大層級

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: 數據管線 (Data Pipeline)                                  │
│    yfinance 價格 + FRED 總經 → PiT 對齊 → 特徵工程                  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: 推論引擎 (Inference)                                      │
│    PyTorch MoE Router (4 Expert × Top-2 Routing)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: 風險防火牆 (Risk Firewall)                                │
│    HMM 政體偵測 + OOD 異常偵測 → 4 級風險等級                        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: LLM 情緒引擎 (Sentiment Engine)                          │
│    Gemini Flash + Ollama Gemma4 備援 + 多樣性新聞抽樣                │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5: 投組優化 (Portfolio Optimization)                          │
│    CVXPY MVO + Regime Allocator + Weight Smoother                  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6: 執行與監控 (Execution & Monitoring)                       │
│    FastAPI 中樞 + Moomoo 下單 + Bloomberg 風格 Dashboard             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 版本演化與功能完成度

| Phase | 功能 | 版本 | 狀態 |
|:--|:--|:--|:--|
| 0 | 原型期 — Web Scraping + KDJ 策略 | v0.x | ✅ 已完成 |
| 1 | MoE Router + PiT Alignment 核心架構 | v1.0 | ✅ 已完成 |
| 2 | HMM/OOD 智能風險防火牆 + Regime Allocator | v2.0 | ✅ 已完成 |
| 3 | CVXPY 優化器重構 + VIX 動態視窗 | v2.2 | ✅ 已完成 |
| 4 | 七層前瞻偏誤審計 + Alpha 歸因實驗 | v2.2.1 | ✅ 已完成 |
| 5 | Moomoo 自動交易 + Discord 推播 + Bloomberg Dashboard | v3.0 | ✅ 已完成 |
| 5.1 | 數據管線穩定化 (快取/合成降級/增量下載) | v3.1 | ✅ 已完成 |
| 5.2 | LLM 雙軌備援 + 多樣性新聞抽樣 + Gemma4 本地推論 | v3.2 | ✅ 已完成 |
| 6 | AI 個人理財顧問 Agent (多模態) | — | 🟡 規劃中 |
| 7 | Alpha 獵手 Agent (供應鏈追蹤) | — | 🟠 規劃中 |

---

## ✅ 已完成功能詳解

### Phase 0~1: 核心架構 (v0.x → v1.0)
- MoE (Mixture of Experts) 路由架構，4 專家 × Top-2 路由，63 維特徵輸入 → 7 資產權重輸出
- PiT (Point-in-Time) 對齊引擎，所有總經資料套用真實發布延遲 (CPI 延遲 45 天等)
- Walk-Forward Backtest 引擎 (70% train / 30% OOS)

### Phase 2: 風控升級 (v2.0)
- **HMM 政體偵測器**: 3 狀態隱馬爾可夫模型 (BULL / BEAR / EXTREME_SHOCK)
- **OOD 異常偵測器**: Isolation Forest + AutoEncoder，偵測訓練時從未見過的市場模式
- **Firewall Core**: 四級風險等級 (GREEN → CAUTION → WARNING → EMERGENCY)
- **Regime Allocator**: 政體先驗混合，熊市自動重倉 TLT/GLD 防禦
- **Weight Smoother**: EMA 平滑 + 信號穩定度檢查 + 最短持有期限制

### Phase 3: 優化器重構 (v2.2)
- 從 scipy 遷移至 CVXPY + OSQP，收斂率從 ~85% 提升至 100%
- Ridge 正則化消除反向 ETF (SH/PSQ) 共線性
- VIX 動態視窗：V 型反轉偵測 (VIX>25 且 5MA 下降 → 加倉視窗從 5 天縮至 1 天)

### Phase 4: 嚴格驗證 (v2.2.1)
- 七層前瞻偏誤審計 — 全數通過
- Scaler 指紋鑑定 — 證實模型確實只看 2019 年以前的舊數據
- Alpha 歸因實驗 — 排除 NVDA/AVGO 後，策略最大回撤仍僅為大盤的一半

### Phase 5: 實盤串接與系統化 (v3.0 → v3.2)
- **Moomoo (FutuOpenD) 自動交易**: 狀態對帳模式 + TCP Fail-Fast 防卡死
- **Discord Webhook**: 交易報告自動推播
- **Bloomberg 風格 Dashboard**: Glassmorphism 深色主題 SPA，三大模組 (總覽/AI 顧問/系統監控)
- **數據管線穩定化**: CSV 增量快取 + FRED 合成降級 + 快取新鮮度驗證
- **LLM 雙軌情緒引擎**: Gemini Flash (主) + Ollama Gemma4:e4b (備援)，配額超限時無縫切換
- **多樣性新聞抽樣**: 關鍵字分桶 (MACRO/TECH/MARKETS/BONDS/OTHERS) + Round-Robin 均衡抽樣
- **智能撤單防呆**: 休市時主動撤銷掛單，避免沖銷產生雙邊手續費
- **負現金警告 UI**: T+2 交割導致的暫時性負現金顯示智能提示

---

## 🟡 Phase 6: AI 個人理財顧問 Agent (規劃中)

**目標**: 對話式互動，圖片辨識券商截圖，情境感知推薦配置。

**技術方案**: Gemini 2.0 Flash (多模態圖片辨識) + 本地 Gemma (金融推理)

**關鍵需求**:
- 券商 APP 截圖 → 自動辨識持倉
- 情境感知 (例：「下個月去當兵不能操作」→ 推薦防禦型配置)
- 長期記憶 (SQLite 儲存個人偏好與對話歷史)
- 串接 Nexus Quant OS Pipeline 即時推論

**現有基礎**: 前端聊天室 UI 已建好，後端 `advisor/` 模組骨架已存在。

---

## 🟠 Phase 7: Alpha 獵手 Agent (規劃中)

**目標**: 自動掃描財報、追蹤供應鏈、找出被低估的潛力股。

**技術方案**: SEC EDGAR XBRL + NLP 供應鏈提取 + DeepSeek-V3

**關鍵需求**:
- 盡量免費 (SEC EDGAR + Yahoo Finance + SimFin)
- 供應鏈遞迴追蹤 (NVDA → TSMC → ASM → ...)
- 輸出「3 個月潛力標的」清單

> 💡 **建議獨立開新專案**，與 Nexus Quant OS 核心分開維護。

---

## 🟢 持續創新 (Continuous Innovation)

### 短期 (1-2 週)
| 功能 | 難度 |
|:--|:--|
| Telegram Bot 即時推播 | ⭐ |
| 多模型 Ensemble (3-5 隨機種子投票) | ⭐⭐ |
| Drawdown 緊急制動 (日虧 >3% → 100% SHY) | ⭐ |

### 中期 (2-4 週)
| 功能 | 難度 |
|:--|:--|
| Walk-Forward 自動重訓練 (月度滾動窗口) | ⭐⭐ |
| 動態資產池 (每季自動篩選最佳 ETF) | ⭐⭐⭐ |
| 多時間框架融合 (日線 + 週線) | ⭐⭐⭐ |

### 長期 (1-3 個月)
| 功能 | 難度 |
|:--|:--|
| 多資產類別 (加密貨幣 ETF / 商品期貨) | ⭐⭐⭐ |
| Fama-French 5-Factor Alpha 歸因報告 | ⭐⭐⭐ |
| 壓力測試模擬器 (2008 金融海嘯重演) | ⭐⭐ |

---

## 建議執行順序

```
已完成:  Phase 0-5.2 (核心系統 + 自動交易 + 風控 + LLM 情緒)
下一步:  Phase 6 (AI 理財顧問) → 2-3 週
未來:    Phase 7 (Alpha 獵手, 新專案) → 4-8 週
持續:    短/中/長期創新功能
```

---

## 專案文件索引

| 文件 | 說明 |
|:--|:--|
| [README.md](README.md) | 專案總覽與快速入門 |
| [DEVELOPMENT_JOURNAL.md](DEVELOPMENT_JOURNAL.md) | 完整開發歷程 (Phase 0 → 5.2) |
| [ROADMAP.md](ROADMAP.md) | 本文件 — 功能路線圖 |
| [docs/TECHNICAL_ARCHITECTURE_RESEARCH.md](docs/TECHNICAL_ARCHITECTURE_RESEARCH.md) | AI Agent 記憶體與對帳模式深度研究 |
| [docs/INFRASTRUCTURE_SPEC.md](docs/INFRASTRUCTURE_SPEC.md) | 量化金融基礎設施規格 |
| [audit_report_data_integrity.md](audit_report_data_integrity.md) | 七層前瞻偏誤審計 + Alpha 歸因 |
| [backtest_report_2020_2026_OOS.md](backtest_report_2020_2026_OOS.md) | 2020-2026 OOS 回測報告 |

---

*「好的量化系統是一台報酬放大器 + 風險壓縮器。」*
