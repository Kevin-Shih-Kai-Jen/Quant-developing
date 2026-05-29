# 🧠 Nexus Quant OS

Nexus Quant OS 是一個專為對抗「前瞻偏誤 (Look-Ahead Bias)」與「極端黑天鵝事件」所設計的新世代量化交易引擎。它結合了最前沿的 AI 架構（Mixture of Experts, MoE）、時間序列對齊引擎，以及基於雙重防護的智能風險防火牆。

在最新的 **v3.0 版本** 中，Nexus Quant OS 已經進化為一個完整的作業系統，內建了 **Bloomberg 風格的視覺化儀表板**、**Gemini 2.5 AI 理財顧問**，以及 **Moomoo (富途) 自動化交易串接**。

---

## 🌟 為什麼該用 Nexus Quant OS？

傳統的量化模型與回測系統常面臨幾個致命痛點，而 Nexus Quant OS 提出了系統性的解決方案：

1. **徹底消滅前瞻偏誤 (Zero Look-Ahead Bias)**
   系統內建 `PiT Alignment` (Point-in-Time) 引擎，會嚴格模擬真實世界的發布延遲（例如總經數據延遲 45 天），確保模型在任何時間點都只能看到「當下已知」的資訊。
2. **動態市場適應 (Dynamic Market Adaptation)**
   採用 **MoE (混合專家) 路由架構**，同時佈署多個不同專長的 AI 模型。技術面與總經專家會根據市場政體自動切換權重。
3. **終極避險：智能風險防火牆 (Intelligent Risk Firewall)**
   結合 **HMM (隱馬爾可夫模型)** 與 **OOD (異常偵測)**。當觸發紅燈時，防火牆會無視 AI 的買入建議，強制將曝險降至零（100% 現金）。
4. **【NEW】個人化 AI 理財顧問 (Gemini AI Advisor)**
   內建支援長期記憶的 AI 顧問，不僅能隨時回答金融問題，還支援多模態視覺分析（可直接上傳券商對帳單截圖），並根據系統的量化訊號給出解讀。
5. **【NEW】實盤與模擬自動下單 (Automated Execution & Alerts)**
   內建 Futu OpenD 接口，支援 Moomoo 模擬帳戶 (Paper Trading) 與實盤交易，全自動執行投組再平衡，並自動發送 Discord 交易報告。

---

## 🔥 最新更新：v3.0 版本亮點 (AI-Driven OS Terminal)

- **🖥️ 視覺化終端儀表板 (SPA Dashboard)**：使用純前端技術 (Vanilla JS, CSS Glassmorphism) 打造極具科技感的深色主題介面，整合總覽、AI 顧問與系統監控三大模組。
- **🧠 記憶體化 AI 顧問 (Advisor Memory)**：利用 SQLite 儲存歷史對話，AI 顧問現在能記住你的風險偏好、資金狀況與個人背景，提供量身打造的建議。
- **⚡ 一鍵啟動 (One-Click Startup)**：擺脫繁瑣的終端機指令，現在只需雙擊 `NexusOS.command` 捷徑，即可自動啟動後端伺服器並在瀏覽器打開終端介面。
- **🛡️ 依賴與穩定性優化**：全面升級至 Pydantic V2，修復環境變數解析，並引入自動化清理機制，確保伺服器穩定長駐。

---

## 🏗️ 核心架構 (System Architecture)

Nexus Quant OS 的管線可拆解為四大核心層：

1. **使用者介面與智能體層 (UI & Agent Layer)**
   - 基於 FastAPI 提供 REST API。
   - 前端 Web 終端機提供實時監控與 AI 聊天。
   - `GeminiAdvisor` 提供自然語言互動與視覺辨識。
2. **數據攝取與 PiT 對齊層 (Data & PiT Layer)**
   - 抓取 yfinance 高頻價格數據與 FRED 低頻總經數據。
   - 透過 `enforce_pit_alignment` 實施 Timestamp 隔離。
3. **混合專家決策層 (MoE Router)**
   - **Expert-0 (技術面專精)** / **Expert-1 (總經面專精)** / **Expert-2 (綜合全能)**。
   - Router 根據當日特徵矩陣計算專家利用率 (Utilization)，輸出原始資產配置權重。
4. **風險防火牆與執行層 (Risk & Execution Layer)**
   - HMM / OOD 輸出 `scale_factor` 進行風險阻斷。
   - `FutuBroker` 透過 Moomoo API 執行真實的訂單買賣。

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
直接在桌面或資料夾中**雙擊執行** `NexusOS.command`，系統會自動：
1. 啟動虛擬環境與 FastAPI 伺服器
2. 在預設瀏覽器中打開 `http://127.0.0.1:8080`
3. 進入 Nexus Quant OS 視覺化終端！

### 4. 開發者手動啟動
```bash
source .venv/bin/activate
python -m uvicorn api.server:app --host 127.0.0.1 --port 8080
```

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
| [DEVELOPMENT_JOURNAL.md](DEVELOPMENT_JOURNAL.md) | 開發歷程 — 從爬蟲原型到 v3.0 的技術演化與架構決策 |
| [ROADMAP.md](ROADMAP.md) | 未來功能路線圖 |
| [docs/TECHNICAL_ARCHITECTURE_RESEARCH.md](docs/TECHNICAL_ARCHITECTURE_RESEARCH.md) | 架構研究 — AI Agent 記憶體、Moomoo 對帳模式 |
| [audit_report_data_integrity.md](audit_report_data_integrity.md) | 前瞻偏誤審計 + Alpha 歸因實驗 |

---

## ⚖️ 免責聲明 (Disclaimer)
本專案為量化交易架構的研究與展示用途。系統給出的任何資產配置權重與預測均不構成投資建議。金融市場具備高度不可預測性，請自行評估風險，開發者不對任何實際交易所產生的損失負責。
