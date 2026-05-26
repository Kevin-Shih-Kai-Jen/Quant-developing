# Nexus Quant OS

Nexus Quant OS 是一個專為對抗「前瞻偏誤 (Look-Ahead Bias)」與「極端黑天鵝事件」所設計的新世代量化交易引擎。它結合了最前沿的 AI 架構（Mixture of Experts, MoE）、時間序列對齊引擎，以及基於雙重防護的智能風險防火牆。

這套系統的設計初衷不僅是追求高回報 (Alpha)，更是在市場出現未知恐慌與結構性崩盤時，能夠自動閃避並保護本金。

---

## 🌟 為什麼該用 Nexus Quant OS？

傳統的量化模型與回測系統常面臨幾個致命痛點，而 Nexus Quant OS 提出了系統性的解決方案：

1. **徹底消滅前瞻偏誤 (Zero Look-Ahead Bias)**
   許多模型在回測時表現極佳，實盤卻崩潰，原因在於不慎使用了「未來資訊」（例如：在月初就用到了月底才公布的 CPI 數據）。本系統內建 `PiT Alignment` (Point-in-Time) 引擎，會嚴格模擬真實世界的發布延遲（例如總經數據延遲 45 天），確保模型在任何時間點都只能看到「當下已知」的資訊。
2. **動態市場適應 (Dynamic Market Adaptation)**
   死板的資產配置無法應付多變的市場。本系統採用 **MoE (混合專家) 路由架構**，同時佈署多個不同專長的 AI 模型。當市場處於技術面主導時，系統會動態分配更多權重給技術分析專家；當宏觀經濟發生巨變時，總經專家會接管主導權。
3. **終極避險：智能風險防火牆 (Intelligent Risk Firewall)**
   AI 是盲目的，但防火牆不是。我們的防火牆獨立於 AI 預測之外，並具備「一票否決權」。它結合了：
   - **HMM (隱馬爾可夫模型)**：即時偵測市場政體（牛市/熊市/極端崩盤）。
   - **OOD (Autoencoder + Isolation Forest)**：偵測「歷史未見」的新型態黑天鵝。
   當觸發紅燈時，防火牆會無視 AI 的買入建議，強制將曝險降至零（100% 現金）。

---

## 🔥 最新更新：v2.2 版本亮點

v2.2 版本針對投組優化器與政體配置器進行了深度重構，大幅提升在極端市況與一般牛市下的表現：
- **徹底遷移至 CVXPY (OSQP Solver)**：取代原本容易收斂失敗的 `scipy.optimize`，並加入微小 Ridge 正則化消除反向 ETF 間的共線性，確保 100% 存在凸優化可行解。
- **政體動態資金利用率 (Dynamic Regime Allocation)**：當 HMM 判斷為 `BULL` 時，強迫 100% 資金參與多頭市場不留現金；僅在 `BEAR` 時，才允許將資金投入現金 (SHY) 避險，徹底釋放多頭市場的 Alpha。
- **V-Shape 抄底機制 (VIX Momentum)**：結合 VIX 5MA 動能判斷，當 VIX > 25 且處於下降趨勢時，視為 V 型反轉強烈信號，強制縮小加倉延遲 (Smoother Window) 至 1 天，防止踏空。
- **終極防呆測試**：使用 2000-2019 訓練的模型，在完全未見過新冠與升息的 2020-2026 OOS 期間，創造了 374.0% 報酬 (27.7% CAGR) 的驚人成績。

---

## 🏗️ 核心模型與系統架構

Nexus Quant OS 的管線 (Pipeline) 可拆解為三大核心層：

### 1. 數據攝取與 PiT 對齊層
- 支援高頻價格數據與低頻總經數據。
- 透過 `enforce_pit_alignment` 函數，對所有宏觀數據實施 Timestamp 隔離，確保特徵工程與模型推論的絕對純淨。

### 2. 混合專家決策層 (MoE Router)
模型內部由一個 Router 與多個 Expert 組成：
- **Expert-0 (技術面專精)**：處理短期的價量特徵（如 `daily_return`, `realised_vol`, `volume_zscore`）。
- **Expert-1 (總經面專精 - TabNet)**：專門解讀低頻但影響深遠的宏觀指標（如 CPI、失業率、信用利差），並採用 TabNet 結構以對抗雜訊。
- **Expert-2 (綜合全能)**：負責將技術面與基本面做深度交叉驗證。
Router 會根據當日的輸入特徵 (Feature Matrix)，計算出每位專家的利用率 (Utilization)，並輸出原始的資產配置權重。

### 3. 風險防火牆與執行層
這是在提交訂單前的最後一道防線，根據 HMM 與 OOD 的評估，輸出一個 `scale_factor` (0.0 ~ 1.0)。
- **GREEN (綠燈)**：`scale_factor = 1.0`，完全放行。
- **CAUTION (黃燈)** / **WARNING (橘燈)**：依嚴重程度縮減部位 (例如 0.7 甚至 0.25)。
- **EMERGENCY (紅燈)**：`scale_factor = 0.0`，全面空手。

---

## 🚀 如何使用 (Installation & Quick Start)

專案已完全容器化 (Dockerized)，保證跨平台的執行一致性。

### 1. 環境準備
請確保您的機器已安裝 `Docker` 與 `Docker Compose`。
同時，您需要註冊一個免費的 FRED API Key（用於抓取總經數據）。

### 2. 環境變數設定
在專案根目錄下建立 `.env` 檔案，並填入以下內容：
```env
# FRED (美國聯準會經濟數據) API 憑證
FRED_API_KEY=您的_API_KEY_這裡
```
*(請放心，`.env` 已被加入 `.gitignore` 中，不會上傳至 GitHub)*

### 3. 啟動系統 (訓練與推論)
**步驟一：訓練模型與防火牆**
初次使用或需要重新校準模型時，請執行：
```bash
docker compose run --rm quant_engine python -m nexus_quant_os.training.train_moe
```
這將會訓練 MoE 權重以及 HMM / Autoencoder 防火牆，並自動將 Checkpoint 儲存下來。

**步驟二：單次執行 (Single-shot Execution)**
若要直接跑出當天的最新預測與最終權重：
```bash
docker compose run --rm quant_engine python main.py
```
執行完畢後，終端機會印出非常詳細的 7 步驟執行日誌與最終部位分配。

---

## 🧩 插件生態系統 (Plugin Ecosystem) 擴充指南

Nexus Quant OS 內建高度解耦的插件介面 (`BasePlugin`)，讓您可以輕鬆地將新的非結構化資料或替代數據（如新聞情緒、供應鏈數據）引入系統中。

### 未來插件該怎麼裝？
1. **開發插件**：在 `nexus_quant_os/plugins/` 目錄下建立新的 Python 檔案（例如 `social_sentiment_plugin.py`）。
2. **繼承介面**：該類別必須繼承自 `BasePlugin`，並實作 `fetch_data()` 與 `get_feature_names()` 方法。
3. **註冊插件**：在主程式或 DataLoader 中，將實作好的 Plugin 實體化，系統會自動在特徵工程 (Feature Engineering) 階段將其產生的 DataFrame 與主資料集進行 `PiT Alignment` 與合併。
4. **模型相容**：只要新的特徵欄位被加入，`QuantMoERouter` 會在下一次訓練時自動將輸入維度 (Input Dim) 擴展，無需大幅改寫核心神經網路代碼。

---

## 📊 績效驗證 (Performance Validation)

本系統經過嚴格的七層前瞻偏誤審計，確認**無數據洩漏**。

### OOS 壓力測試 (2020-2026, 模型僅訓練 2019 前數據)

| 指標 | MoE 策略 | SPY 買進持有 |
|:--|:--|:--|
| 總報酬 | **+374.0%** | +151.2% |
| CAGR | **+27.77%** | +15.56% |
| Sharpe | **+1.536** | +0.812 |
| 最大回撤 | **-8.74%** | -33.72% |

### Alpha 歸因對照實驗

- **等權基準** (10 ETF 瞎抱): +168.0% → 模型擇時額外貢獻 +206%
- **排除 NVDA/AVGO** (8 ETF): Max DD **-9.42%** (SPY: -18.76%) / **Sharpe 1.376**

> 詳見 [數據完整性審計報告](audit_report_data_integrity.md)

---

## 📚 專案文件 (Documentation)

| 文件 | 說明 |
|:--|:--|
| [DEVELOPMENT_JOURNAL.md](DEVELOPMENT_JOURNAL.md) | 完整開發歷程 — 從爬蟲原型到 v2.2 的技術演化與架構決策 |
| [ROADMAP.md](ROADMAP.md) | 未來功能路線圖 — Paper Trading、AI 理財顧問、Alpha 獵手 |
| [audit_report_data_integrity.md](audit_report_data_integrity.md) | 七層前瞻偏誤審計 + Scaler 指紋鑑定 + Alpha 歸因實驗 |
| [backtest_report_2020_2026_OOS.md](backtest_report_2020_2026_OOS.md) | 2020-2026 完整 OOS 回測績效報告 |

---

## ⚖️ 免責聲明 (Disclaimer)
本專案為量化交易架構的研究與展示用途。系統給出的任何資產配置權重與預測均不構成投資建議。金融市場具備高度不可預測性，請自行評估風險，開發者不對任何實際交易所產生的損失負責。
