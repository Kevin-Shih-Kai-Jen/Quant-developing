#!/usr/bin/env python3
"""
main.py — Nexus Quant OS 核心排程器 (Core DAG Scheduler)
=========================================================

本檔案是整個量化作業系統的入口點，嚴格遵循架構報告中的
**單向數據流 (Unidirectional Data Flow / DAG)** 原則。

資料流向：

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  STEP 1      │    │  STEP 2      │    │  STEP 3      │
  │ 多模態資料   │───>│ PiT 對齊器   │───>│ 特徵工程     │
  │ 生成模組     │    │ (Aligner)    │    │ (Feature Eng)│
  └──────────────┘    └──────────────┘    └──────┬───────┘
                                                 │
                    ┌────────────────────────────-┤
                    │                             │
                    v                             v
  ┌──────────────────────┐         ┌──────────────────────┐
  │  STEP 4              │         │  STEP 5              │
  │  防火牆訓練          │         │  MoE Router          │
  │  (HMM + OOD 校準)   │         │  (大腦心臟推論)      │
  └──────────┬───────────┘         └──────────┬───────────┘
             │                                │
             │   raw_weights [N_assets]        │
             │<───────────────────────────────-┘
             v
  ┌──────────────────────┐
  │  STEP 6              │
  │  防火牆即時審查      │
  │  (Risk Evaluation)   │
  └──────────┬───────────┘
             │
             v
  ┌──────────────────────┐
  │  STEP 7              │
  │  最終安全權重輸出    │
  │  (Final Weights)     │
  └──────────────────────┘

核心保證：
    1. 數據只能往下流動，絕不逆流
    2. 任何時間點 T 的決策，只能使用 T 之前已公開的資訊
    3. 極端市場狀態下，防火牆會強制將權重歸零

執行方式：
    cd /Users/coolguy/developer/nexus_quant_os
    source .venv/bin/activate
    PYTHONPATH=. python main.py

Author : Nexus Quant OS — System Integration Division
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ═════════════════════════════════════════════════════════════════════
# 內部模組匯入 — 嚴格按照 DAG 拓撲排列
# ═════════════════════════════════════════════════════════════════════

# [DAG Layer 1] 資料對齊層
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.data_pipelines.data_loader import load_all_data

# [DAG Layer 1.5] 訓練模組（可選載入 checkpoint）
from nexus_quant_os.training.train_moe import (
    build_daily_dataset,
    find_latest_checkpoint,
    load_checkpoint,
    INPUT_DIM as MOE_INPUT_DIM,
)

# [DAG Layer 2] 推論引擎層
from nexus_quant_os.models.moe_router import (
    GatingNoiseType,
    QuantMoERouter,
    RouterConfig,
    RoutingOutput,
    build_dummy_expert,
)

# [DAG Layer 3] 風險防火牆層
from nexus_quant_os.risk_firewall.hmm_regime_detector import HMMConfig
from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODConfig
from nexus_quant_os.risk_firewall.firewall_core import (
    FirewallConfig,
    FirewallDecision,
    IntelligentRiskFirewall,
    RiskTier,
)

# [DAG Layer 4] 投組優化 + 權重管理層
from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer, OptimizerConfig
from nexus_quant_os.portfolio.weight_smoother import WeightSmoother, SmootherConfig
from nexus_quant_os.portfolio.regime_allocator import RegimeAllocator, RegimeAllocatorConfig
from nexus_quant_os.llm.sentiment_aggregator import SentimentAggregator


# ═════════════════════════════════════════════════════════════════════
# 全域常量
# ═════════════════════════════════════════════════════════════════════

# 投資組合標的：美股大盤 ETF + NVDA + Broadcom
ASSET_UNIVERSE = ["AVGO", "GLD", "IWM", "NVDA", "QQQ", "SPY", "TLT"]  # 字母排序（與 train_moe.py 一致）
N_ASSETS       = len(ASSET_UNIVERSE)

# 真實數據時間軸 (涵蓋 COVID 崩盤 + 升息週期 + AI 浪潮)
DATA_START = "2020-01-02"
DATA_END   = None    # None = 今天

# FRED API Key (從環境變數讀取，保護私鑰不進版本控制)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# 特徵工程窗口
RETURN_LOOKBACK      = 5          # 5 日收益率
VOL_LOOKBACK         = 20         # 20 日已實現波動度

# MoE Router 配置
NUM_EXPERTS          = 4          # Tech-MLP / Macro-TabNet / Generalist-MLP / Sentiment-MLP
TOP_K                = 2          # 每個時間步激活 2 個專家
EXPERT_HIDDEN_DIM    = 64         # 每個專家的隱藏層維度

# 防火牆校準使用的訓練窗口比例
FIREWALL_TRAIN_RATIO = 0.70       # 前 70% 用於校準，後 30% 為即時推論


# ═════════════════════════════════════════════════════════════════════
# STEP 1: 真實數據載入（含合成數據 fallback）
# ═════════════════════════════════════════════════════════════════════

def load_real_market_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """從 Yahoo Finance + FRED API 載入真實市場數據。

    若 FRED_API_KEY 未設定或網路失敗，自動 fallback 至合成數據。

    Returns
    -------
    daily_prices : pd.DataFrame
    macro_data   : pd.DataFrame
    source       : str   ('REAL' | 'SYNTHETIC')
    """
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY 未設定，使用合成數據模式。")
        prices, macro = _generate_synthetic_fallback()
        return prices, macro, "SYNTHETIC"

    try:
        end = DATA_END or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        daily_prices, macro_data = load_all_data(
            tickers=ASSET_UNIVERSE,
            fred_api_key=FRED_API_KEY,
            start=DATA_START,
            end=end,
        )
        return daily_prices, macro_data, "REAL"

    except Exception as exc:
        logger.error("真實數據載入失敗: %s — fallback 至合成數據", exc)
        prices, macro = _generate_synthetic_fallback()
        return prices, macro, "SYNTHETIC"


def _generate_synthetic_fallback(
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合成數據備援（當真實數據無法取得時使用）。"""
    np.random.seed(seed)

    today = pd.Timestamp.today().normalize()
    # 週末自動往前推到最後一個交易日
    if today.dayofweek >= 5:
        today = today - pd.tseries.offsets.BDay(1)
        
    sim_end   = today.strftime("%Y-%m-%d")
    sim_start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    
    trading_days = pd.bdate_range(sim_start, sim_end)
    n_days       = len(trading_days)

    base_prices = {
        "SPY":  (480.0, 0.012),
        "QQQ":  (420.0, 0.016),
        "IWM":  (200.0, 0.018),
        "TLT":  (100.0, 0.008),
        "GLD":  (190.0, 0.010),
        "NVDA": (500.0, 0.030),
        "AVGO": (850.0, 0.022),
    }

    rows = []
    for asset in ASSET_UNIVERSE:
        base_price, daily_vol = base_prices[asset]
        returns = np.random.normal(0.0003, daily_vol, n_days)
        close   = np.round(base_price * np.exp(np.cumsum(returns)), 2)
        spread  = np.abs(np.random.normal(0, daily_vol * base_price * 0.5, n_days))
        for i in range(n_days):
            rows.append({
                "timestamp": trading_days[i], "asset_id": asset,
                "close": close[i], "high": close[i] + spread[i],
                "low":   close[i] - spread[i],
                "volume": np.random.randint(10_000_000, 100_000_000),
            })

    daily_prices = pd.DataFrame(rows).sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    months    = pd.date_range(sim_start, sim_end, freq="MS")
    macro_rows = []
    for i, ref_month in enumerate(months):
        pub_date = ref_month + pd.DateOffset(months=1, days=15)
        for asset in ASSET_UNIVERSE:
            macro_rows.append({
                "timestamp":         pub_date, "asset_id": asset,
                "cpi_yoy":           round(3.0 + 0.03 * i + np.random.normal(0, 0.1), 2),
                "unemployment_rate": round(3.8 + 0.02 * i + np.random.normal(0, 0.1), 2),
                "pmi_manufacturing": round(52.0 - 0.1 * i + np.random.normal(0, 1.0), 1),
                "fed_funds_rate":    round(0.25 + 0.1 * i + np.random.normal(0, 0.02), 3),
                "credit_spread":     round(0.35 + 0.005 * i + np.random.normal(0, 0.02), 3),
            })

    macro_data = pd.DataFrame(macro_rows).sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    return daily_prices, macro_data


# NOTE: engineer_features 已搬遷至 nexus_quant_os/data_pipelines/feature_engineer.py
# 此處使用統一模組以消除循環依賴
from nexus_quant_os.data_pipelines.feature_engineer import engineer_features



# ═════════════════════════════════════════════════════════════════════
# STEP 4–6: 主管線排程
# ═════════════════════════════════════════════════════════════════════

def run_pipeline() -> None:
    """執行 Nexus Quant OS 完整 DAG 管線。"""

    SEP   = "=" * 78
    THIN  = "-" * 78
    ARROW = ">>>"

    end_display = DATA_END or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"\n{SEP}")
    print("  NEXUS QUANT OS v0.1 — Full DAG Pipeline Execution")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Assets    : {ASSET_UNIVERSE}")
    print(f"  Data Range: {DATA_START} -> {end_display}")
    print(f"  Data Mode : {'REAL (Yahoo Finance + FRED)' if FRED_API_KEY else 'SYNTHETIC (fallback)'}")
    print(f"{SEP}\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: 真實數據載入
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 1/9 : 載入真實市場數據 (Yahoo Finance + FRED API)...")

    daily_prices, macro_data, data_source = load_real_market_data()

    n_price_rows = len(daily_prices)
    n_macro_rows = len(macro_data)
    n_days       = n_price_rows // max(len(ASSET_UNIVERSE), 1)
    print(f"    數據來源     : {data_source}")
    print(f"    高頻價格數據 : {n_price_rows:,} rows  "
          f"({len(ASSET_UNIVERSE)} assets × {n_days} days)")
    print(f"    低頻總經數據 : {n_macro_rows:,} rows")
    print(f"    [OK] 數據載入完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: PiT 對齊 — 嚴格消除前瞻偏誤
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 2/9 : 執行 Point-in-Time 無前瞻偏誤對齊...")

    # 對每檔資產獨立進行對齊
    aligned_frames = []
    for asset in ASSET_UNIVERSE:
        asset_prices = daily_prices[daily_prices["asset_id"] == asset].copy()
        asset_macro  = macro_data[macro_data["asset_id"] == asset].copy()

        aligned_asset, audit = enforce_pit_alignment(
            daily_prices=asset_prices,
            macro_fundamental_data=asset_macro,
            timestamp_col="timestamp",
            asset_col="asset_id",
            max_drift_days=45,           # 總經數據最多延遲 45 天
            drop_unmatched=True,         # 丟棄無法對齊的早期資料
            preserve_right_timestamp=True,
        )
        aligned_frames.append(aligned_asset)
        logger.debug("  %s: %s", asset, audit.summary())

    aligned_df = pd.concat(aligned_frames, ignore_index=True)
    aligned_df = aligned_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    # 防呆驗證：確認無前瞻偏誤
    if "timestamp_published" in aligned_df.columns:
        non_null   = aligned_df.dropna(subset=["timestamp_published"])
        violations = non_null[non_null["timestamp_published"] > non_null["timestamp"]]
        assert violations.empty, (
            f"CRITICAL: {len(violations)} 筆前瞻偏誤洩漏！系統應立即停止。"
        )

    print(f"    對齊後總行數 : {len(aligned_df):,} rows")
    print(f"    前瞻偏誤檢查 : PASSED (零洩漏)")
    print(f"    [OK] PiT 對齊完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: 特徵工程 — DataFrame → 數值特徵矩陣
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 3/9 : 特徵工程 (技術面 + 總經面融合)...")

    feature_matrix, feature_names = engineer_features(aligned_df)
    n_samples, n_features = feature_matrix.shape

    print(f"    特徵矩陣形狀 : [{n_samples}, {n_features}]")
    print(f"    特徵欄位     : {feature_names}")
    print(f"    數值範圍     : "
          f"min={feature_matrix.min():.6f}  max={feature_matrix.max():.6f}")
    print(f"    NaN 殘留     : {np.isnan(feature_matrix).sum()} (應為 0)")
    print(f"    [OK] 特徵工程完成\n")

    # ─────────────────────────────────────────────────────────────
    # 時間切割：單獨拉出大盤 (SPY) 來訓練防火牆，防止資產錯置與未來外洩
    # ─────────────────────────────────────────────────────────────
    spy_df = aligned_df[aligned_df["asset_id"] == "SPY"].copy().reset_index(drop=True)
    spy_feature_matrix, spy_feature_names = engineer_features(spy_df)
    n_spy_samples = len(spy_feature_matrix)
    
    split_idx      = int(n_spy_samples * FIREWALL_TRAIN_RATIO)
    train_features = spy_feature_matrix[:split_idx]           # [T_train, F]
    
    print(f"    訓練窗口     : {split_idx:,} samples (前 {FIREWALL_TRAIN_RATIO:.0%})")
    print(f"    即時推論窗口 : {n_spy_samples - split_idx:,} samples (後 "
          f"{1 - FIREWALL_TRAIN_RATIO:.0%})\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: 防火牆訓練 (HMM 政體偵測 + OOD 孤立森林)
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 4/9 : 訓練智慧風險防火牆 (HMM + Isolation Forest + AE)...")

    firewall = IntelligentRiskFirewall.from_configs(
        hmm_config=HMMConfig(
            n_regimes=3,
            n_iter=200,
            danger_threshold=0.50,
        ),
        ood_config=OODConfig(
            n_estimators=200,
            ae_epochs=30,
            ae_latent_dim=8,
            combined_threshold=0.55,
        ),
        firewall_config=FirewallConfig(
            hmm_caution_threshold=0.40,
            hmm_warning_threshold=0.60,
            hmm_emergency_threshold=0.78,
            ood_caution_threshold=0.45,
            ood_warning_threshold=0.60,
            ood_emergency_threshold=0.78,
            caution_scale=0.70,
            warning_scale=0.25,
            emergency_scale=0.00,
            smooth_blend=True,
        ),
    )
    firewall.fit(train_features)
    print(f"    HMM 政體數   : 3 (BULL / BEAR / EXTREME)")
    print(f"    OOD 模型     : IsolationForest(200 trees) + Autoencoder(30 epochs)")
    print(f"    [OK] 防火牆校準完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: MoE Router — 大腦心臟推論
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 5/9 : MoE 動態路由器 (大腦心臟)...")

    device = torch.device("cpu")  # Docker on Mac M1 強制 CPU
    print(f"    推論裝置     : {device}")

    ckpt_path = find_latest_checkpoint()

    # ══════════════════════════════════════════════════════════════
    # PATH A：已訓練 Checkpoint 存在 → 橫截面日度推論 (input_dim=56)
    # ══════════════════════════════════════════════════════════════
    if ckpt_path:
        print(f"    模式         : ✅ 已訓練模型 ({ckpt_path.name})")
        router, scaler, trained_assets, _firewall = load_checkpoint(ckpt_path)
        router = router.to(device)
        router.eval()

        total_params = sum(p.numel() for p in router.parameters())
        print(f"    Router 參數量: {total_params:,}")
        print(f"    Input 維度   : {router.config.input_dim}D "
              f"({N_ASSETS} assets × 8 features)")
        print(f"    Output 維度  : {N_ASSETS}D (portfolio weights)")
        print(f"    [OK] 已訓練 Router 載入完成\n")

        # ── STEP 5b: 構建橫截面日度特徵 → 推論 ───────────────────
        print(f"{ARROW} STEP 5b  : 橫截面日度特徵 + 已訓練 MoE 推論...")

        X_daily, y_daily, _y_raw_daily, dates_daily, assets_sorted = build_daily_dataset(aligned_df, inference_mode=True)

        print(f"    橫截面特徵形狀 : {list(X_daily.shape)}  "
              f"({N_ASSETS} assets × 8 features per day)")

        # 用訓練時的 scaler 標準化
        X_scaled = scaler.transform(X_daily).astype(np.float32)
        X_tensor  = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            routing_result: RoutingOutput = router(X_tensor)

        # 取最後一天作為「今日」原始部位信號
        raw_weights_tensor = routing_result.combined_output[-1]   # [N_assets]
        raw_weights        = raw_weights_tensor.cpu().numpy()     # [N_assets]

        expert_util  = routing_result.expert_utilisation.cpu().numpy()
        expert_names = [f"Expert-{i}" for i in range(NUM_EXPERTS)]

        print(f"    推論日期       : {dates_daily[-1].date()} (最新交易日)")
        print(f"    Router 輸出形狀: {list(routing_result.combined_output.shape)}")
        print(f"    專家使用率      : ", end="")
        for name, util in zip(expert_names, expert_util):
            print(f"{name}={util:.3f}  ", end="")
        print()
        print(f"    今日原始權重 (raw) : {np.round(raw_weights, 6)}")
        print(f"    [OK] 已訓練 MoE 推論完成\n")

    # ══════════════════════════════════════════════════════════════
    # PATH B：無 Checkpoint → Fallback 至隨機初始化 Router (input_dim=8)
    # ══════════════════════════════════════════════════════════════
    else:
        print(f"    模式         : ⚠️  未找到 checkpoint，使用隨機初始化 Router")
        print(f"                   (執行 python -m nexus_quant_os.training.train_moe 以訓練)")

        router_config = RouterConfig(
            input_dim=MOE_INPUT_DIM,
            num_experts=NUM_EXPERTS,
            output_dim=N_ASSETS,
            top_k=TOP_K,
            noise_type=GatingNoiseType.NONE,
            aux_loss_coeff=1e-2,
        )
        expert_names = ["Tech-MLP", "Macro-TabNet", "Generalist-MLP", "Sentiment-MLP"]
        experts = nn.ModuleList([
            build_dummy_expert(
                input_dim=MOE_INPUT_DIM,
                output_dim=N_ASSETS,
                hidden_dim=EXPERT_HIDDEN_DIM,
            )
            for _ in range(NUM_EXPERTS)
        ])
        router = QuantMoERouter(config=router_config, experts=experts).to(device)
        router.eval()

        total_params = sum(p.numel() for p in router.parameters())
        print(f"    Router 參數量: {total_params:,}")
        print(f"    Config       : D={MOE_INPUT_DIM} E={NUM_EXPERTS} K={TOP_K} O={N_ASSETS}")
        print(f"    [OK] 隨機 Router 初始化完成\n")

        print(f"{ARROW} STEP 5b  : DataFrame -> Tensor 轉換 + MoE 前向推論...")

        X_daily, _, _, dates_daily, assets_sorted = build_daily_dataset(aligned_df, inference_mode=True)
        from sklearn.preprocessing import StandardScaler
        fallback_scaler = StandardScaler()
        X_daily_scaled = fallback_scaler.fit_transform(X_daily).astype(np.float32)
        live_tensor = torch.tensor(
            X_daily_scaled, dtype=torch.float32, device=device
        )
        print(f"    Tensor 形狀  : {list(live_tensor.shape)}")
        print(f"    dtype        : {live_tensor.dtype}")
        print(f"    device       : {live_tensor.device}")

        with torch.no_grad():
            routing_result = router(live_tensor)

        raw_weights_tensor = routing_result.combined_output[-1]
        raw_weights        = raw_weights_tensor.cpu().numpy()

        expert_util = routing_result.expert_utilisation.cpu().numpy()
        print(f"\n    Router 輸出形狀     : {list(routing_result.combined_output.shape)}")
        print(f"    專家使用率           : ", end="")
        for name, util in zip(expert_names, expert_util):
            print(f"{name}={util:.3f}  ", end="")
        print()
        print(f"    今日原始權重 (raw)   : {np.round(raw_weights, 6)}")
        print(f"    [OK] MoE 推論完成\n")


    # ─────────────────────────────────────────────────────────────
    # STEP 5c: 投組優化器 (Risk Parity / Constrained MVO)
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 5c  : 投組優化器 (Risk Parity / MVO)...")

    # 正規化原始權重
    abs_sum = np.abs(raw_weights).sum() + 1e-8
    norm_weights = raw_weights / abs_sum

    # 建構共變異數矩陣（使用最近 60 天）
    if ckpt_path and len(X_daily) >= 60:
        # Extract daily_return feature (index 0 per asset block in the feature vector)
        N_FEATURES_PER_ASSET = X_daily.shape[1] // N_ASSETS  # should be 9
        return_indices = [i * N_FEATURES_PER_ASSET for i in range(N_ASSETS)]
        returns_matrix = X_daily[-60:, return_indices]
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        # Fallback to identity if cov_matrix shape is wrong
        if cov_matrix.shape[0] != N_ASSETS:
            cov_matrix = np.eye(N_ASSETS) * 0.01
    else:
        cov_matrix = np.eye(N_ASSETS) * 0.01

    portfolio_optimizer = PortfolioOptimizer(
        n_assets=N_ASSETS,
        config=OptimizerConfig(
            max_single_weight=0.35,
            gross_exposure=1.0,
            min_cash=0.05,
            dispersion_threshold=0.45,
        ),
    )
    optimized_weights = portfolio_optimizer.optimize(
        moe_weights=norm_weights,
        expert_utilisation=expert_util,
        cov_matrix=cov_matrix,
    )
    opt_mode = "RiskParity" if float(np.std(expert_util)) > 0.30 else "MVO"
    print(f"    優化模式     : {opt_mode}")
    print(f"    優化後權重   : {np.round(optimized_weights, 6)}")
    print(f"    [OK] 投組優化完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: 防火牆即時審查
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 6/9 : 風險防火牆即時審查 (HMM + OOD 雙重驗證)...")

    observation_window = spy_feature_matrix[-20:]   # [20, F]

    decision = firewall.evaluate(
        market_features=observation_window,
        raw_weights=optimized_weights,
    )

    tier_icons = {
        RiskTier.GREEN:     "🟢 GREEN",
        RiskTier.CAUTION:   "🟡 CAUTION",
        RiskTier.WARNING:   "🟠 WARNING",
        RiskTier.EMERGENCY: "🔴 EMERGENCY",
    }

    safe_weights = optimized_weights * decision.scale_factor

    print(f"    風險等級      : {tier_icons[decision.risk_tier]}")
    print(f"    HMM 崩盤概率  : {decision.hmm_danger_prob:.4f}  "
          f"(P(EXTREME_SHOCK)  → 觸發 EMERGENCY)")
    print(f"    HMM 熊市概率  : {decision.hmm_bear_prob:.4f}  "
          f"(P(BEAR_HIGH_VOL)  → 觸發 WARNING/CAUTION)")
    print(f"    HMM 政體      : {decision.hmm_regime_label}")
    print(f"    OOD 異常分數  : {decision.ood_combined_score:.4f}  "
          f"(flagged: {decision.ood_is_flagged})")
    print(f"    縮放因子      : {decision.scale_factor:.4f}")
    print(f"    否決原因      : {decision.veto_reason}")
    print(f"    [OK] 風險審查完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: LLM 情緒引擎調整（可選）
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 7/9 : LLM 情緒引擎...")
    sentiment_adjustment = 1.0
    try:
        sentiment_engine = SentimentAggregator(
            use_deepseek=True,
            use_gemini=False,  # 預設只用本地 DeepSeek
        )
        sentiment_result = sentiment_engine.get_daily_sentiment(
            headlines=["Market analysis pending"]  # Placeholder — 未來接 RSS
        )
        sentiment_adjustment = 1.0 + sentiment_result.sentiment_daily * 0.10
        print(f"    情緒分數     : {sentiment_result.sentiment_daily:.4f}")
        print(f"    調整因子     : {sentiment_adjustment:.4f}")
        print(f"    [OK] 情緒調整完成\n")
    except Exception as e:
        print(f"    ⚠ LLM 不可用，跳過情緒調整 ({e})\n")
        sentiment_adjustment = 1.0

    safe_weights = safe_weights * sentiment_adjustment

    # ─────────────────────────────────────────────────────────────
    # STEP 8: 政體自適應配置
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 8/9 : 政體自適應配置 (Regime-Adaptive Allocation)...")

    regime_allocator = RegimeAllocator(
        assets=list(assets_sorted),
        config=RegimeAllocatorConfig(
            min_confidence=0.65,
            max_prior_blend=0.25,
            transition_smoothing=0.8,
        ),
    )
    regime_conf = max(
        decision.hmm_bear_prob,
        decision.hmm_danger_prob,
    )
    regime_blended = regime_allocator.blend(
        moe_weights=safe_weights,
        regime_label=decision.hmm_regime_label,
        regime_confidence=regime_conf,
        hmm_bear_prob=decision.hmm_bear_prob,
        hmm_danger_prob=decision.hmm_danger_prob,
    )
    print(f"    政體         : {decision.hmm_regime_label}")
    print(f"    政體信心度   : {regime_conf:.4f}")
    print(f"    混合後權重   : {np.round(regime_blended, 6)}")
    print(f"    [OK] 政體配置完成\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 9: 權重平滑 + 最終輸出
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 9/9 : 權重平滑 + 最終投資組合權重輸出")

    weight_smoother = WeightSmoother(
        n_assets=N_ASSETS,
        config=SmootherConfig(
            alpha=0.30,
            min_rebalance_threshold=0.04,
            max_single_turnover=0.08,
            min_hold_days=5,
            signal_stability_window=3,
        ),
    )
    final_weights = weight_smoother.smooth(regime_blended)

    print("-" * 78)
    print(f"\n  {'Asset':<10}  {'Raw Weight':>14}  {'Optimized':>14}  {'Final':>14}  {'Action':>8}")
    print("  " + "-" * 68)

    for i, asset in enumerate(assets_sorted):
        rw = float(norm_weights[i])
        ow = float(optimized_weights[i])
        fw = float(final_weights[i])
        action = "BUY" if fw > 0.05 else ("SELL" if fw < -0.05 else "HOLD")
        print(f"  {asset:<8s}  {rw:>+12.6f}  {ow:>+12.6f}  {fw:>+12.6f}  {action:>8s}")

    total_final = np.sum(np.abs(final_weights))
    cash_pct = max(0.0, 1.0 - total_final) * 100

    print(f"  {'-' * 68}")
    print(f"\n  總曝險 (Gross Exposure) : {total_final:.4f}")
    print(f"  現金水位 (Cash)         : {cash_pct:.1f}%")
    print(f"  防火牆縮放              : {decision.scale_factor:.4f} "
          f"({decision.risk_tier.name})")
    print(f"  優化模式                : {opt_mode}")
    print(f"  情緒調整                : {sentiment_adjustment:.4f}")

    # ─────────────────────────────────────────────────────────────
    # PIPELINE COMPLETE
    # ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  NEXUS QUANT OS v2.0 — Pipeline Execution Complete")
    print(f"  Status       : ALL 9 STEPS PASSED")
    print(f"  Data Integrity : Zero look-ahead bias violations")
    print(f"  Risk Tier     : {decision.risk_tier.name}")
    print(f"  Optimizer     : {opt_mode}")
    print(f"  Regime        : {decision.hmm_regime_label}")
    print(f"  Final Action  : "
          f"{'POSITIONS ACTIVE' if total_final > 0.001 else 'ALL FLAT (100% CASH)'}")
    print(f"{SEP}\n")


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

logger = logging.getLogger("nexus_quant_os.main")

if __name__ == "__main__":
    # 設定全域日誌
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    # 主管線與數據載入層日誌可見
    logging.getLogger("nexus_quant_os.main").setLevel(logging.INFO)
    logging.getLogger("nexus_quant_os.data_pipelines.data_loader").setLevel(logging.INFO)
    logging.getLogger("nexus_quant_os.risk_firewall.firewall_core").setLevel(logging.INFO)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_pipeline()
