# Nexus Quant OS — 開發歷程 (Development Journal)

> **最後更新**: 2026-05-26
> **作者**: Kevin Shih
> **專案**: Nexus Quant OS — 新世代量化交易引擎

本文件完整記錄了 Nexus Quant OS 從零到一的技術演化歷程，展示架構決策背後的思考脈絡與解決問題的過程。

---

## 📅 版本演化時間軸

```
v0.x (早期原型)        v1.0 (核心架構)        v2.0 (風控升級)         v2.2 (優化器重構)
     │                      │                      │                       │
     ├─ Web Scraping        ├─ MoE Router          ├─ HMM Risk Firewall   ├─ CVXPY Migration       ├─ Moomoo Execution
     ├─ KDJ Indicator       ├─ PiT Alignment       ├─ OOD Anomaly Det.    ├─ VIX Momentum          ├─ Discord Alerts
     ├─ Basic Strategy      ├─ Feature Engineer    ├─ Regime Allocator    ├─ Dynamic Allocation    ├─ Fail-Fast TCP Check
     └─ Mixed Strategy      ├─ Docker Pipeline     ├─ Weight Smoother     ├─ Data Integrity Audit  └─ OS Architecture
                            └─ Backtest Engine     └─ Expert Specializ.   └─ Alpha Attribution
```

---

## Phase 0: 原型期 — 從爬蟲到指標 (Early Prototype)

### 💡 初始動機
一開始只是想建立一個自動化的股票分析系統，從最基礎的功能開始：

### 里程碑
- **Web Scraping v1/v2**: 建立數據爬蟲，從公開數據源抓取股票資訊
- **KDJ Indicator**: 實作第一個技術指標（KDJ 隨機指標）
- **Basic Trading Strategy**: 基於 KDJ 的簡單買賣信號
- **First Mixed Strategy**: 嘗試結合多個指標的混合策略

### 🔑 關鍵學習
> 「單一指標策略在回測中看起來有效，但在實際不同市場環境下會失效。需要一個能自適應不同市場環境的系統。」

這個認知直接催生了 MoE (混合專家) 的架構選擇。

---

## Phase 1: 核心架構 — MoE + PiT 引擎 (v1.0)

### 🏗️ 架構決策

#### 為什麼選擇 MoE 而不是單一模型？

傳統的量化模型有一個致命缺陷：它們假設市場永遠遵循同一套規則。但現實是：
- 牛市中，技術面動能信號有效
- 熊市中，總經面的風險指標更重要
- 震盪市中，情緒面的均值回歸信號才是關鍵

**解決方案**：引入 MoE (Mixture of Experts) 架構。讓多個專家各司其職，由 Router 根據當前市場特徵動態分配權重。

```
                    ┌─── Expert-0: 技術面 MLP (短期價量)
Feature Vector ──→ Router ─┤
   (90D)           (Gate)  ├─── Expert-1: 總經面 TabNet (宏觀指標)
                           │
                           ├─── Expert-2: 全能型 MLP (交叉驗證)
                           │
                           └─── Expert-3: 情緒面 MLP (極端市況)
```

#### 為什麼要 PiT (Point-in-Time) 對齊？

**血淚教訓**：早期回測績效驚人，但分析後發現模型在 1 月 2 日就「看到」了 1 月 31 日才公布的 CPI 數據。這就是前瞻偏誤 (Look-Ahead Bias)。

**解決方案**：建立 `enforce_pit_alignment` 引擎，對所有宏觀數據嚴格實施時間戳隔離：
- CPI：延遲 45 天
- 失業率：延遲 35 天
- 工業生產指數：延遲 17 天

### 里程碑
- **QuantMoERouter**: 4 專家 × Top-2 路由，63 維輸入 → 7 資產權重
- **PiT Alignment Engine**: 確保零前瞻偏誤的時間序列對齊
- **Feature Engineer**: 9 個特徵（4 技術面 + 5 總經面）
- **Docker Pipeline**: 完全容器化，一鍵訓練 + 推論
- **Walk-Forward Backtest**: 時間序列分割（70% train / 30% OOS）

---

## Phase 2: 風控升級 — 智能風險防火牆 (v2.0)

### 🛡️ 問題發現

> 「模型在 2020 年 COVID 崩盤期間依然推薦做多。AI 不知道它不知道什麼。」

MoE Router 只學過訓練數據的分佈。當市場出現「歷史從未見過」的模式時，模型的預測是不可信的。

### 🏗️ 架構決策

#### 雙重防護防火牆

```
         ┌─── HMM 政體偵測器 ───┐
         │  3 狀態隱馬爾可夫     │   ─── 「市場現在是牛市還是熊市？」
         │  BULL / BEAR / SHOCK │
         └──────────────────────┘
                    +
         ┌─── OOD 異常偵測器 ───┐
         │  Isolation Forest    │   ─── 「今天的市場模式在訓練數據中見過嗎？」
         │  + Autoencoder       │
         └──────────────────────┘
                    ↓
              FirewallDecision
           scale_factor: 0.0 ~ 1.0
```

- **GREEN (1.0)**: 一切正常，全額放行
- **CAUTION (0.7)**: 輕微異常，減碼 30%
- **WARNING (0.25)**: 明顯危險，僅保留 25% 部位
- **EMERGENCY (0.0)**: 全面撤退，100% 現金

#### 政體自適應配置器 (Regime Allocator)

根據 HMM 判斷的市場政體，動態調整資產配置策略：
- **BULL**：100% 資金投入多頭，不留現金
- **BEAR**：增加 SHY/GLD 避險部位
- **SHOCK**：強制降低曝險至最低水平

#### 權重平滑器 (Weight Smoother)

避免模型因微小信號變化而頻繁換倉（導致高交易成本）：
- EMA 平滑 (α=0.30)
- 最小調倉閾值 4%
- 最短持有期 5 天

### 里程碑
- **IntelligentRiskFirewall**: HMM (3-state) + Isolation Forest + Autoencoder
- **RegimeAllocator**: 政體感知的動態配置
- **WeightSmoother**: 含 Expert-0 豁免的智能平滑
- **資產池擴展**: 7 → 10 檔 ETF（新增 AVGO, NVDA, PSQ）

---

## Phase 3: 優化器重構 — CVXPY 遷移 (v2.2)

### 🐛 問題發現

> 「`scipy.optimize.minimize` 在反向 ETF (SH, PSQ) 加入後頻繁收斂失敗。」

根因分析：SH ≈ -1×SPY, PSQ ≈ -1×QQQ，共線性導致 Hessian 矩陣接近奇異，scipy 的 SLSQP 無法處理。

### 🏗️ 架構決策

#### 為什麼從 scipy 遷移到 cvxpy？

| 特性 | scipy.optimize | cvxpy + OSQP |
|:--|:--|:--|
| 凸優化保證 | ❌ 可能陷入局部最優 | ✅ 數學保證全局最優 |
| 共線性處理 | ❌ Hessian 奇異崩潰 | ✅ Ridge 正則化消除 |
| 約束處理 | 鬆散（penalty-based） | 嚴格（硬約束） |
| 失敗率 | ~15-20% | ~0% |

#### 技術細節
```python
# Ridge 正則化解決共線性
Σ_reg = Σ + 1e-4 * I  # 微小單位矩陣確保正定性

# CVXPY 凸優化
w = cp.Variable(n_assets)
objective = cp.Minimize(cp.quad_form(w, Σ_reg) - λ * μ @ w)
constraints = [w >= 0, cp.sum(w) == 1, w <= max_weight]
```

#### VIX 動態視窗 (V-Shape 抄底機制)
```
VIX > 25 且 5MA 下降趨勢 → 判定為 V 型反轉
  → 平滑窗口從 5 天縮至 1 天
  → 快速跟上反彈不踏空
```

### 里程碑
- **CVXPY + OSQP**: 100% 收斂率的凸優化器
- **Ridge Regularization**: 消除反向 ETF 共線性
- **VIX Momentum**: V 型反轉偵測 + 動態加倉
- **Dynamic Bull Allocation**: BULL 政體強制 100% 投入

---

## Phase 4: 嚴格驗證 — 數據完整性審計 (v2.2.1)

### ❓ 自我質疑

> 「+374% 的報酬太好了。我不信任它。讓我證明它不是幻覺。」

這是量化研究中最重要的品質：**對自己的成果保持懷疑**。

### 🔍 七層前瞻偏誤審計

我們對系統進行了 7 層逐級穿透式審計：

| # | 審計層 | 結果 | 方法 |
|:--|:--|:--|:--|
| 1 | 特徵洩漏 | ✅ 乾淨 | 確認 `forward_return` 不在 `FEATURE_COLS` |
| 2 | 回測報酬計算 | ✅ 乾淨 | 確認 t 日決定權重，t+1 日結算 |
| 3 | 共變異數矩陣 | ✅ 乾淨 | 確認 `hist_window = y_raw[start:t]` |
| 4 | HMM 防火牆 | ✅ 乾淨 | 確認使用 `spy_feat[:t+1]` |
| 5 | 批次推論 | ✅ 乾淨 | MoE 為前饋架構，無跨時間步依賴 |
| 6 | 權重平滑器 | ✅ 乾淨 | 逐 t 因果迭代 |
| 7 | 政體配置器 | ✅ 乾淨 | 逐 t 因果迭代 |

### 🧪 Scaler 指紋鑑定 — 鐵證

直接解剖 Checkpoint 中 StandardScaler 儲存的統計量：

| 指標 | Scaler 記錄 | 2009-2016 真實值 | 2020-2024 真實值 |
|:--|:--|:--|:--|
| CPI YoY | 1.4915% | 1.4987% ✅ | 4.2069% ❌ |
| 失業率 | 7.4288% | 7.4214% ✅ | 4.9452% ❌ |

**結論**: 模型確實只看了 2019 年以前的舊數據。

### 🧪 Alpha 歸因對照實驗

#### 實驗 A: 等權基準 (10 ETF 瞎抱)
- 等權: +168.0% | MoE 策略: **+374.0%**
- **模型的擇時貢獻了額外 +206% 超額報酬**

#### 實驗 B: 排除 NVDA/AVGO (8 ETF 極限測試)
- SPY: +40.3% / Max DD -18.76%
- Ex-Semi 策略: +34.2% / **Max DD -9.42%** / **Sharpe 1.376**
- **即使沒有飆股，模型的最大回撤僅為大盤的一半**

### 🔑 關鍵發現

> 這套系統最強的不是「預測漲跌」，而是**極致的風控能力**。
> 不論資產池內容為何，最大回撤均控制在 10% 以內。
> 它是一台「報酬放大器 + 風險壓縮器」。

---

## 技術債與已知限制

| 項目 | 說明 | 狀態 |
|:--|:--|:--|
| 資產選擇偏誤 | NVDA/AVGO 為事後選入的飆股 | ⚠️ 已記錄 |
| 單一訓練視窗 | 無 Walk-Forward 自動重訓練 | 📋 未來功能 |
| 無實時交易 | 僅回測，無 Paper/Live Trading 串接 | ✅ 已解決 (v3.1 Moomoo) |
| CPU-only 訓練 | Docker on Mac M1 限制 | ℹ️ 已知 |

---

## Phase 5: 自動化執行與監控 — 串接真實世界 (v3.1)

### 🌍 核心突破
系統正式跨出回測沙盒，連接真實金融世界。

### 🏗️ 架構決策
1. **執行層與決策層分離 (Separation of Concerns)**
   - 決策層 (MoE Router) 僅負責輸出目標權重，對資產餘額一無所知。
   - 執行層 (`FutuBroker`) 透過 Moomoo API 獲取真實帳戶狀態 (Total Equity, Positions)，進行對帳 (Reconciliation) 與差額計算，實現動態再平衡。

2. **防卡死機制 (Fail-Fast TCP Check)**
   - **痛點**：Moomoo 官方 SDK 在 FutuOpenD 未啟動時會陷入無限重試，導致自動排程卡死。
   - **解法**：在啟動 SDK 前，先以 1 秒超時進行底層 TCP Socket 握手探測，若無法連線則乾淨俐落地拋出異常結束程式，節省系統資源。

3. **雙層防護防頻繁交易**
   - 繼承回測中 `WeightSmoother` 的 4% 變動門檻。
   - 在執行層 (`FutuBroker`) 再加上 2% 容錯與 $50 最小訂單金額限制，完美避免微小震盪導致的手續費消耗。

### 里程碑
- **Moomoo Paper Trading**: 全自動模擬交易閉環完成。
- **Discord Webhook**: 排程執行完畢自動發送交易報告至 Discord。
- **Data Robustness**: 解決 FRED 密鑰缺失時的合成數據生成中斷 Bug。

---

## 未來路線圖

詳見 [ROADMAP.md](ROADMAP.md)

| 階段 | 功能 | 預估時程 |
|:--|:--|:--|
| Phase 5 | Moomoo Paper Trading API 串接 | 1-2 週 |
| Phase 6 | AI 個人理財顧問 Agent | 2-3 週 |
| Phase 7 | Alpha 獵手 Agent (供應鏈追蹤) | 4-8 週 |
| 持續 | 自動化 Bug 偵測 + 持續創新 | 永續 |

---

## 文件索引

| 文件 | 說明 |
|:--|:--|
| [README.md](README.md) | 專案總覽與快速入門 |
| [DEVELOPMENT_JOURNAL.md](DEVELOPMENT_JOURNAL.md) | 本文件 — 完整開發歷程 |
| [ROADMAP.md](ROADMAP.md) | 未來功能路線圖 |
| [docs/TECHNICAL_ARCHITECTURE_RESEARCH.md](docs/TECHNICAL_ARCHITECTURE_RESEARCH.md) | Deep Think 深度技術架構研究 |
| [docs/INFRASTRUCTURE_SPEC.md](docs/INFRASTRUCTURE_SPEC.md) | Deep Research 量化金融基礎設施規格 |
| [audit_report_data_integrity.md](audit_report_data_integrity.md) | 數據完整性審計報告 |
| [backtest_report_2020_2026_OOS.md](backtest_report_2020_2026_OOS.md) | 2020-2026 OOS 回測報告 |

---

*「好的量化系統不是一天建成的。每一次失敗都是一次架構升級的機會。」*
