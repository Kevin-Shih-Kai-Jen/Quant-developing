# 🧠 Nexus Quant OS

Nexus Quant OS 是一個專為對抗「前瞻偏誤 (Look-Ahead Bias)」與「極端黑天鵝事件」所設計的新世代量化交易引擎。它結合了最前沿的 AI 架構（Mixture of Experts, MoE）、時間序列對齊引擎，以及基於雙重防護的智能風險防火牆。

在最新的 **v3.2 版本** 中，Nexus Quant OS 已經進化為一個完整的作業系統，內建了 **Bloomberg 風格的視覺化儀表板**、**LLM 雙軌情緒引擎 (Gemini + Ollama)**、**多樣性新聞抽樣**，以及 **Moomoo (富途) 自動化交易串接**。

---

## 🌟 為什麼該用 Nexus Quant OS？

1. **徹底消滅前瞻偏誤 (Zero Look-Ahead Bias)**
   系統內建 `PiT Alignment` (Point-in-Time) 引擎，會嚴格模擬真實世界的發布延遲（例如總經數據延遲 45 天），確保模型在任何時間點都只能看到「當下已知」的資訊。
2. **動態市場適應 (Dynamic Market Adaptation)**
   採用 **MoE (混合專家) 路由架構**，同時佈署多個不同專長的 AI 模型。技術面與總經專家會根據市場政體自動切換權重。
3. **終極避險：智能風險防火牆 (Intelligent Risk Firewall)**
   結合 **HMM (隱馬爾可夫模型)** 與 **OOD (異常偵測)**。當觸發紅燈時，防火牆會無視 AI 的買入建議，強制將曝險降至零（100% 現金）。
4. **LLM 雙軌情緒引擎 (Dual-Track Sentiment Engine)**
   結合 Gemini Flash (雲端主力) 與 Ollama Gemma4 (本地備援)，抓取即時財經新聞並透過多樣性抽樣 (Diversity Sampling) 產生情緒調整因子，微調投組權重。
5. **實盤與模擬自動下單 (Automated Execution & Alerts)**
   內建 Futu OpenD 接口，支援 Moomoo 模擬帳戶 (Paper Trading) 與實盤交易，全自動執行投組再平衡，並自動發送 Discord 交易報告。

---

## 🔥 最新更新：v3.2 版本亮點

- **🤖 LLM 雙軌備援**：Gemini Flash 配額超限時，自動無縫切換至本地 Ollama (`gemma4:e4b`) 進行情緒分析，不再因 API 限制而中斷。
- **📰 多樣性新聞抽樣 (Diversity Sampling)**：從 186 條 RSS 新聞中，透過關鍵字分桶 (MACRO/TECH/MARKETS/BONDS/OTHERS) + Round-Robin 均衡抽樣，確保 LLM 看到的 15 條新聞涵蓋所有市場面向。
- **💾 數據管線穩定化**：增量快取 + 合成降級 (Synthetic Fallback) + 快取新鮮度驗證，確保 FRED 或 yfinance 斷線時系統仍能正常運作。
- **🛡️ 智能撤單防呆**：休市時自動撤銷掛單，而非反向沖銷，避免雙邊手續費。
- **💰 負現金智能提示**：T+2 交割導致暫時性負現金時，Dashboard 顯示解釋提示，消除使用者疑慮。

---

## 🏗️ 核心架構 (System Architecture)

Nexus Quant OS 的管線可拆解為六大核心層：

```
┌─────────────────────────────────────────────────────────────────┐
│  L1 數據管線    yfinance + FRED → PiT 對齊 → 特徵工程           │
│  L2 推論引擎    PyTorch MoE Router (4 Expert × Top-2)           │
│  L3 風險防火牆  HMM 政體偵測 + OOD 異常偵測 → 4 級風險等級       │
│  L4 情緒引擎    Gemini Flash + Ollama Gemma4 + 多樣性新聞抽樣    │
│  L5 投組優化    CVXPY MVO + Regime Allocator + Weight Smoother  │
│  L6 執行監控    FastAPI + Moomoo 下單 + Bloomberg Dashboard      │
└─────────────────────────────────────────────────────────────────┘
```

### 目錄結構

```
nexus_quant_os/
├── api/                          # FastAPI 系統中樞 + 前端 SPA
│   ├── server.py                 # REST API (Pipeline/Dashboard/Health/Advisor)
│   └── static/index.html         # Bloomberg 風格 Glassmorphism Dashboard
│
├── nexus_quant_os/               # 核心引擎 (Python Package)
│   ├── data_pipelines/           # L1: 數據管線
│   │   ├── data_loader.py        # yfinance + FRED 下載 + 快取 + 合成降級
│   │   ├── aligner.py            # PiT (Point-in-Time) 對齊引擎
│   │   └── dataset_builder.py    # 滾動特徵工程 + Scaler 正規化
│   │
│   ├── models/                   # L2: 推論引擎
│   │   ├── moe_router.py         # MoE Router (4 Expert × Top-2 Routing)
│   │   ├── experts/              # 專家模組 (TabNet, MLP, FinBERT 等)
│   │   └── checkpoints/          # 已訓練模型權重 (.pt)
│   │
│   ├── risk_firewall/            # L3: 風險防火牆
│   │   ├── firewall_core.py      # 四級風險等級 + Sigmoid 平滑縮放
│   │   ├── hmm_regime_detector.py  # HMM 3 狀態政體偵測
│   │   └── ood_anomaly_detector.py # AutoEncoder + Isolation Forest
│   │
│   ├── llm/                      # L4: LLM 情緒引擎
│   │   ├── news_fetcher.py       # RSS 新聞 + 多樣性抽樣 (Diversity Sampling)
│   │   ├── gemini_client.py      # Gemini Flash (雲端主力)
│   │   ├── gemma_client.py       # Ollama Gemma4:e4b (本地備援)
│   │   ├── deepseek_client.py    # DeepSeek-V3 (備選)
│   │   └── sentiment_aggregator.py # 多來源情緒融合 → 調整因子
│   │
│   ├── portfolio/                # L5: 投組優化
│   │   ├── optimizer.py          # CVXPY + OSQP 均值方差優化
│   │   ├── regime_allocator.py   # 政體先驗混合 (BULL/BEAR/SHOCK 劇本)
│   │   ├── weight_smoother.py    # EMA 平滑 + 信號穩定度 + 持倉天數限制
│   │   └── risk_parity.py        # 風險平價權重計算
│   │
│   ├── execution/                # L6: 交易執行
│   │   ├── futu_broker.py        # Moomoo (FutuOpenD) 券商串接
│   │   ├── broker_base.py        # 券商抽象基底類別
│   │   ├── broker_router.py      # 智能券商路由
│   │   └── trade_logger.py       # 交易日誌記錄
│   │
│   ├── advisor/                  # AI 理財顧問
│   │   ├── advisor_v2.py         # Gemini 對話引擎
│   │   ├── memory.py             # SQLite 長期記憶
│   │   ├── pipeline_bridge.py    # 串接 Pipeline 即時推論
│   │   └── tools.py              # Agent 工具函數
│   │
│   ├── monitoring/               # 系統監控
│   │   └── health_check.py       # 多層健康檢查 (數據/模型/系統)
│   │
│   ├── notifications/            # 通知推播
│   │   └── discord_notifier.py   # Discord Webhook 交易報告
│   │
│   ├── training/                 # 模型訓練
│   │   ├── train_moe.py          # MoE Router 完整訓練流程
│   │   └── rolling_trainer.py    # Walk-Forward 滾動訓練
│   │
│   └── plugins/                  # 插件系統
│       ├── base_plugin.py        # 插件基底類別
│       ├── assets/               # 資產類插件
│       └── macro/                # 總經類插件
│
├── scripts/                      # 自動排程腳本
│   ├── daily_health_check.py     # 每日健康檢查
│   ├── install_schedule.sh       # macOS launchd 排程安裝
│   └── *.plist                   # launchd 排程設定 (盤前/盤後)
│
├── tests/                        # 測試套件 (pytest)
│   ├── test_data_integrity.py    # 前瞻偏誤回歸測試
│   ├── test_model_sanity.py      # 模型權重合法性測試
│   ├── test_optimizer.py         # 優化器收斂性測試
│   ├── test_futu_broker.py       # 券商 API 串接測試
│   └── test_advisor.py           # AI 顧問功能測試
│
├── main.py                       # 完整訓練 + 回測入口
├── backtest.py                   # 獨立回測引擎
├── run_moomoo_trade.py           # Moomoo 自動交易入口
├── daily_run.sh                  # 每日自動排程腳本
├── Nexus Quant OS.command        # macOS 一鍵啟動捷徑
├── Dockerfile                    # Docker 容器化部署
└── docker-compose.yml            # Docker Compose 編排
```

---

## 🚀 如何使用 (Quick Start)

### 1. 環境準備
請確保您的機器已安裝 Python 3.10+，並註冊以下免費 API Keys：
- **FRED API Key** (美國聯準會經濟數據)
- **Gemini API Key** (Google AI Studio)

### 2. 環境變數設定
在專案根目錄下建立 `.env` 檔案：
```env
FRED_API_KEY=您的_FRED_KEY
GEMINI_API_KEY=您的_GEMINI_KEY
```

### 3. 一鍵啟動 (Mac 用戶)
直接在桌面或資料夾中**雙擊執行** `Nexus Quant OS.command`，系統會自動：
1. 啟動虛擬環境與 FastAPI 伺服器
2. 在預設瀏覽器中打開 `http://127.0.0.1:8080`
3. 進入 Nexus Quant OS 視覺化終端！

### 4. 開發者手動啟動
```bash
source .venv/bin/activate
python api/server.py
```

### 5. 本地 LLM 備援 (可選)
安裝 [Ollama](https://ollama.ai) 並拉取模型：
```bash
ollama pull gemma4:e4b
```
系統會在 Gemini 配額超限時自動切換至本地模型。

---

## 📊 績效驗證 (Performance Validation)

在 2020-2026 OOS (Out-of-Sample) 期間，模型在未見過新冠崩盤與激進升息的環境下，達成優異表現：

| 指標 | MoE 策略 | SPY 買進持有 |
|:--|:--|:--|
| 總報酬 | **+374.0%** | +151.2% |
| CAGR | **+27.77%** | +15.56% |
| Sharpe | **+1.536** | +0.812 |
| 最大回撤 | **-8.74%** | -33.72% |

> *註：回測不包含 Moomoo 實盤滑價，實際交易請注意風險。*

---

## 📚 專案文件 (Documentation)

| 文件 | 說明 |
|:--|:--|
| [DEVELOPMENT_JOURNAL.md](DEVELOPMENT_JOURNAL.md) | 開發歷程 — 從爬蟲原型到 v3.2 的技術演化與架構決策 |
| [ROADMAP.md](ROADMAP.md) | 功能路線圖與版本演化追蹤 |
| [docs/TECHNICAL_ARCHITECTURE_RESEARCH.md](docs/TECHNICAL_ARCHITECTURE_RESEARCH.md) | 架構研究 — AI Agent 記憶體、Moomoo 對帳模式 |
| [docs/INFRASTRUCTURE_SPEC.md](docs/INFRASTRUCTURE_SPEC.md) | 量化金融基礎設施規格 |
| [audit_report_data_integrity.md](audit_report_data_integrity.md) | 前瞻偏誤審計 + Alpha 歸因實驗 |
| [backtest_report_2020_2026_OOS.md](backtest_report_2020_2026_OOS.md) | 2020-2026 OOS 回測報告 |

---

## ⚖️ 免責聲明 (Disclaimer)
本專案為量化交易架構的研究與展示用途。系統給出的任何資產配置權重與預測均不構成投資建議。金融市場具備高度不可預測性，請自行評估風險，開發者不對任何實際交易所產生的損失負責。
