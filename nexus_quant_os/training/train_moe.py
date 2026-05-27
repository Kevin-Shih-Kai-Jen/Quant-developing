"""
nexus_quant_os/training/train_moe.py — MoE Router 真實數據訓練腳本
====================================================================

訓練配置：
    架構    : 橫截面日度 (Cross-Sectional Daily)
              每天將所有 N_assets 資產的特徵橫向拼接，作為 Router 的輸入
    輸入維度 : N_assets × N_features = 7 × 9 = 63
    輸出維度 : N_assets = 7 (每檔資產的投資組合權重)
    Expert  : 差異化架構
              Expert-0（技術面）: 3 層 MLP，輸入 4 tech features × 7 assets = 28D
              Expert-1（總經面）: TabNet Sequential Attention，輸入 5 macro features × 7 assets = 35D
              Expert-2（泛化型）: 3 層 MLP，輸入全部 9 features × 7 assets = 63D
    損失    : 0.40 × MSE + 0.30 × (−Sharpe) + 0.15 × AuxLoss + 0.15 × Turnover
    Epochs  : 200（含 Early Stopping，patience=30）
    優化器  : Adam + CosineAnnealingLR
    正則化  : weight_decay=1e-4, gradient clipping=1.0

PiT 保證：
    訓練目標為 t+1 的前瞻報酬 (forward_return)
    特徵 X[t] 只含 t 時刻已公開的資訊
    時序分割（非隨機打亂）以避免模型選擇偏誤

Author : Nexus Quant OS — Model Training Division
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# 確保專案根目錄在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from nexus_quant_os.models.moe_router import (
    GatingNoiseType,
    QuantMoERouter,
    RouterConfig,
    RoutingOutput,
)
from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.risk_firewall.firewall_core import IntelligentRiskFirewall, HMMConfig, OODConfig, FirewallConfig
import pickle

logger = logging.getLogger("nexus_quant_os.training.train_moe")

# ═════════════════════════════════════════════════════════════════════
# 全域訓練常量
# ═════════════════════════════════════════════════════════════════════

ASSET_UNIVERSE  = [
    "AVGO", "GLD", "IWM", "NVDA",
    "QQQ", "SPY", "TLT",
]
N_ASSETS        = len(ASSET_UNIVERSE)     # 7
N_FEATURES      = 9                       # 每資產特徵數（v2: +yield_curve_slope）
INPUT_DIM       = N_ASSETS * N_FEATURES   # 63 = 橫截面輸入維度
OUTPUT_DIM      = N_ASSETS                # 7 = 投資組合權重維度

NUM_EXPERTS     = 4
TOP_K           = 2
HIDDEN_DIM      = 128   # MLP 第一隱藏層維度
EPOCHS          = 200
BATCH_SIZE      = 64
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 40
TRAIN_RATIO     = 0.70
VOL_LOOKBACK    = 20
AUX_LOSS_COEFF  = 0.10  # 0.15 — 平衡負載 vs 學習信號（強制 Router 分配路由給 Expert-0）

CHECKPOINT_DIR  = _PROJECT_ROOT / "nexus_quant_os" / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")
DATA_START      = "2020-01-01"

FEATURE_COLS = [
    "daily_return", "realised_vol", "high_low_spread", "volume_zscore",
    "cpi_yoy", "unemployment_rate", "pmi_manufacturing", "credit_spread",
    "yield_curve_slope",
]

MACRO_COLS = [
    "cpi_yoy", "unemployment_rate", "pmi_manufacturing",
    "credit_spread", "yield_curve_slope",
]
TECH_COLS  = ["daily_return", "realised_vol", "high_low_spread", "volume_zscore"]

# ═════════════════════════════════════════════════════════════════════
# 差異化 Expert 特徵索引（在 63D 橫截面向量中的位置）
# ═════════════════════════════════════════════════════════════════════
# 每個 asset 的特徵按 FEATURE_COLS 排列，共 9 個特徵
# asset_i 的第 j 個特徵在 63D 向量中的索引 = i * N_FEATURES + j
#   TECH  features: indices 0,1,2,3  (per asset)
#   MACRO features: indices 4,5,6,7,8 (per asset)

TECH_FEATURE_INDICES = []
for i in range(N_ASSETS):
    for j in range(4):  # daily_return, realised_vol, high_low_spread, volume_zscore
        TECH_FEATURE_INDICES.append(i * N_FEATURES + j)
# → 28 indices

MACRO_FEATURE_INDICES = []
for i in range(N_ASSETS):
    for j in range(4, N_FEATURES):  # cpi_yoy, unemployment_rate, pmi, credit_spread, yield_curve
        MACRO_FEATURE_INDICES.append(i * N_FEATURES + j)
# → 35 indices


# ═════════════════════════════════════════════════════════════════════
# Expert 架構
# ═════════════════════════════════════════════════════════════════════

from nexus_quant_os.models.experts.tabular_tabnet import TabNetExpert


def build_expert_mlp(
    input_dim:  int,
    output_dim: int,
    hidden_dim: int   = 128,
    dropout:    float = 0.1,
) -> nn.Module:
    """3 層 MLP Expert with LayerNorm + ReLU + Dropout。"""
    H = hidden_dim
    return nn.Sequential(
        nn.Linear(input_dim, H),
        nn.LayerNorm(H),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(H, H // 2),
        nn.LayerNorm(H // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(H // 2, output_dim),
    )


class SpecializedExpert(nn.Module):
    """差異化專家包裝器 — 從全域 63D 輸入中提取特定特徵子集。

    Router 的 gating network 仍然看全部 63D（全局路由決策），
    但每個 Expert 的內部網路只處理自己負責的特徵子集。

    Parameters
    ----------
    feature_indices : list[int]  在 63D 向量中的索引位置
    expert_network  : nn.Module  實際的推論網路（MLP 或 TabNet）
    """

    def __init__(
        self,
        feature_indices: list[int],
        expert_network:  nn.Module,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "feature_indices",
            torch.tensor(feature_indices, dtype=torch.long),
        )
        self.network = expert_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 63] → 提取子集 → Expert 推論 → [B, 7]"""
        x_subset = x[:, self.feature_indices]  # [B, subset_dim]
        return self.network(x_subset)


def build_specialized_experts(
    output_dim: int = OUTPUT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dropout:    float = 0.1,
) -> nn.ModuleList:
    """建構 4 個差異化 Expert：

    Expert-0（技術面 MLP）：4 tech features × 7 assets = 28D 輸入
    Expert-1（總經面 TabNet）：5 macro features × 7 assets = 35D 輸入
    Expert-2（泛化型 MLP）：全部 9 features × 7 assets = 63D 輸入
    Expert-3（情緒面 MLP）：全部 9 features × 7 assets = 63D 輸入（窄 MLP，hidden=64）
    """
    # Expert-0: 技術面 — 價格、波動、成交量
    expert_0 = SpecializedExpert(
        feature_indices=TECH_FEATURE_INDICES,   # 28D
        expert_network=build_expert_mlp(
            input_dim=len(TECH_FEATURE_INDICES),  # 28
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ),
    )

    # Expert-1: 總經面 — CPI、失業率、PMI、信用利差、殖利率曲線（TabNet）
    expert_1 = SpecializedExpert(
        feature_indices=MACRO_FEATURE_INDICES,  # 35D
        expert_network=TabNetExpert(
            input_dim=len(MACRO_FEATURE_INDICES),  # 35
            output_dim=output_dim,
            n_steps=3,
            hidden_dim=hidden_dim,
            gamma=1.5,
        ),
    )

    # Expert-2: 泛化型 — 看全部特徵
    all_indices = list(range(INPUT_DIM))  # 63D
    expert_2 = SpecializedExpert(
        feature_indices=all_indices,
        expert_network=build_expert_mlp(
            input_dim=INPUT_DIM,  # 63
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ),
    )

    # Expert-3: 情緒面 — 使用全部特徵，但更窄的 MLP (hidden=64)
    # 在 live 推論時，LLM 情緒信號作為外部調整信號加成到最終權重上
    # 訓練時完全靠量化特徵學習，讓 Gating Network 自行分配路由
    expert_3 = SpecializedExpert(
        feature_indices=all_indices,  # 63D（同 Expert-2）
        expert_network=build_expert_mlp(
            input_dim=INPUT_DIM,  # 63
            output_dim=output_dim,
            hidden_dim=64,        # 窄 MLP，避免與 Expert-2 冗餘
            dropout=dropout * 1.5,  # 稍高 dropout 以增加差異化
        ),
    )

    return nn.ModuleList([expert_0, expert_1, expert_2, expert_3])


# ═════════════════════════════════════════════════════════════════════
# 損失函數：0.40×MSE + 0.30×(−Sharpe) + 0.15×Aux + 0.15×Turnover
# ═════════════════════════════════════════════════════════════════════

class CombinedPortfolioLoss(nn.Module):
    """可微分組合損失函數 v2.0。

    Primary   : MSE(predicted_weights, normalized_forward_returns)
    Secondary : −Sharpe(portfolio_returns)   [最大化 Sharpe]
    Auxiliary : MoE 負載平衡損失             [防止 Expert Collapse]
    Turnover  : 換手率懲罰                   [降低交易成本]
    """

    def __init__(
        self,
        mse_weight:      float = 0.40,
        sharpe_weight:   float = 0.30,
        aux_weight:      float = 0.15,
        turnover_weight: float = 0.15,
        turnover_coeff:  float = 1.0,
        eps:             float = 1e-8,
    ) -> None:
        super().__init__()
        self.mse_weight      = mse_weight
        self.sharpe_weight   = sharpe_weight
        self.aux_weight      = aux_weight
        self.turnover_weight = turnover_weight
        self.turnover_coeff  = turnover_coeff
        self.eps             = eps

    def forward(
        self,
        predicted_weights: torch.Tensor,  # [B, N_assets]
        y_norm:         torch.Tensor,  # [B, N_assets] (橫截面標準化，用於 MSE 擬合)
        y_raw:          torch.Tensor,  # [B, N_assets] (未標準化的真實前瞻報酬，用於 Sharpe)
        aux_loss:       torch.Tensor,  # scalar
        prev_weights:   torch.Tensor | None = None,  # [B, N_assets] 前一天的預測權重
    ) -> tuple[torch.Tensor, dict[str, float]]:

        y_pred = predicted_weights

        # ── MSE loss ──────────────────────────────────────────────
        mse = F.mse_loss(y_pred, y_norm)

        # ── Sharpe ratio loss ─────────────────────────────────────
        # 投資組合日報酬 = Σ w[t,a] × r[t,a] (使用真實報酬計算 Sharpe)
        portfolio_ret = (y_pred * y_raw).sum(dim=1)  # [B]
        mean_ret  = portfolio_ret.mean()
        std_ret   = portfolio_ret.std() + self.eps
        sharpe    = mean_ret / std_ret
        sharpe_loss = -sharpe  # 最小化負 Sharpe = 最大化 Sharpe

        # ── Turnover penalty ─────────────────────────────────────
        # 計算相鄰時間步之間的權重變動（懲罰頻繁換倉）
        if prev_weights is not None:
            turnover = torch.abs(y_pred - prev_weights).sum(dim=1).mean()
        else:
            # Batch 內相鄰樣本的差異作為近似
            if y_pred.shape[0] > 1:
                turnover = torch.abs(y_pred[1:] - y_pred[:-1]).sum(dim=1).mean()
            else:
                turnover = torch.tensor(0.0, device=y_pred.device)
        turnover_loss = turnover * self.turnover_coeff

        # ── Total ─────────────────────────────────────────────────
        total = (
            self.mse_weight      * mse           +
            self.sharpe_weight   * sharpe_loss   +
            self.aux_weight      * aux_loss      +
            self.turnover_weight * turnover_loss
        )

        return total, {
            "mse":      mse.item(),
            "sharpe":   sharpe.item(),
            "aux":      aux_loss.item() if hasattr(aux_loss, 'item') else float(aux_loss),
            "turnover": turnover_loss.item(),
            "total":    total.item(),
        }


# ═════════════════════════════════════════════════════════════════════
# 數據集構建：橫截面日度格式
# ═════════════════════════════════════════════════════════════════════

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算每個 (asset, date) 的技術面特徵 + 前瞻報酬。"""
    df = df.copy()

    # 技術面特徵
    df["daily_return"]    = df.groupby("asset_id")["close"].pct_change()
    df["realised_vol"]    = (
        df.groupby("asset_id")["daily_return"]
        .transform(lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).std())
    )
    df["high_low_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    vol_mean = df.groupby("asset_id")["volume"].transform(
        lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).mean()
    )
    vol_std = df.groupby("asset_id")["volume"].transform(
        lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).std()
    )
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)

    # 總經特徵：ffill → fillna(0)（殘留的 NaN 以 0 填補）
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = (
                df.groupby("asset_id")[col]
                .transform(lambda x: x.ffill())
            )
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # 前瞻報酬（訓練目標）：t+1 的報酬，在 t 時預測
    df["forward_return"] = df.groupby("asset_id")["close"].pct_change().shift(-1)

    return df


def build_daily_dataset(
    aligned_df: pd.DataFrame,
    inference_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """構建橫截面日度訓練數據集。

    關鍵設計：
        - 每天將所有 N_assets 資產的 9 個特徵橫向拼接
          → X[t] shape = [N_assets × N_features] = [63]
        - 目標為下一個交易日的橫截面標準化報酬
          → y[t] = r[t+1] / Σ|r[t+1]|   shape = [N_assets] = [7]
        - 只保留所有資產均有完整數據的日期

    Parameters
    ----------
    aligned_df    : pd.DataFrame   enforce_pit_alignment 輸出
    inference_mode : bool           推論模式：不要求 forward_return，
                                    保留最後一天（最新交易日）的數據

    Returns
    -------
    X      : np.ndarray  [T_dates, 63]
    y      : np.ndarray  [T_dates, 7]   (橫截面標準化前瞻報酬)
    y_raw  : np.ndarray  [T_dates, 7]   (未標準化的真實前瞻報酬)
    dates  : pd.DatetimeIndex
    assets : list[str]                  (字母排序，與欄位順序一致)
    """
    df = _add_technical_features(aligned_df)

    # 推論模式：只要求技術指標欄位非 NaN，不要求 forward_return
    # 這樣最後一天（尚無明天報酬）不會被丟棄
    if inference_mode:
        df = df.dropna(subset=TECH_COLS).reset_index(drop=True)
    else:
        df = df.dropna(subset=TECH_COLS + ["forward_return"]).reset_index(drop=True)

    # 確保使用字母排序（與訓練時一致）
    assets = sorted(df["asset_id"].unique().tolist())
    logger.info("Assets in dataset: %s", assets)

    # 找出所有資產都有數據的共同日期
    dates_sets = [
        set(df.loc[df["asset_id"] == a, "timestamp"].dt.normalize())
        for a in assets
    ]
    common_dates = sorted(set.intersection(*dates_sets))
    logger.info("Common dates: %d", len(common_dates))

    X_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    valid_dates: list         = []

    for date in common_dates:
        day_df = df[df["timestamp"].dt.normalize() == date].drop_duplicates(subset="asset_id").set_index("asset_id")

        if not all(a in day_df.index for a in assets):
            continue

        feat_row: list[float] = []
        ret_row:  list[float] = []
        valid = True

        for asset in assets:
            row   = day_df.loc[asset]
            feats = []
            for c in FEATURE_COLS:
                v = float(row.get(c, np.nan))
                if np.isnan(v) or np.isinf(v):
                    valid = False
                    break
                feats.append(v)

            if not valid:
                break

            fwd = float(row.get("forward_return", np.nan))
            # 推論模式：forward_return 為 NaN 是正常的（最後一天無明日報酬）
            if inference_mode and (np.isnan(fwd) or np.isinf(fwd)):
                fwd = 0.0  # 填 0 作為佔位符，推論時不會用到 y
            elif np.isnan(fwd) or np.isinf(fwd):
                valid = False
                break

            feat_row.extend(feats)
            ret_row.append(fwd)

        if valid:
            X_rows.append(feat_row)
            y_rows.append(ret_row)
            valid_dates.append(date)

    X = np.array(X_rows, dtype=np.float32)   # [T_dates, 63]
    y = np.array(y_rows, dtype=np.float32)   # [T_dates, 7]

    # 安全護欄：確保 y 至少是 2D（防止單日期邊界情況）
    if y.ndim == 1:
        y = y.reshape(-1, len(assets))

    # 橫截面標準化：每天的報酬除以絕對值之和
    # 結果：sum(|y[t]|) = 1，可直接解讀為市場中性投資組合權重
    abs_sum = np.abs(y).sum(axis=1, keepdims=True) + 1e-8
    y_norm  = y / abs_sum

    logger.info("Dataset: X=%s  y=%s  dates=%d", X.shape, y_norm.shape, len(valid_dates))
    return X, y_norm, y, pd.DatetimeIndex(valid_dates), assets


# ═════════════════════════════════════════════════════════════════════
# 驗證指標計算
# ═════════════════════════════════════════════════════════════════════

def compute_val_metrics(
    router:  QuantMoERouter,
    X_val:   np.ndarray,       # [T_val, 63]
    y_val:   np.ndarray,       # [T_val, 7]
    y_raw_val: np.ndarray,     # [T_val, 7]
    scaler:  StandardScaler,
    device:  torch.device,
) -> dict:
    """計算驗證集 Sharpe、MSE 與方向命中率。"""
    router.eval()
    X_scaled = scaler.transform(X_val).astype(np.float32)

    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        out = router(X_t)
        w   = out.combined_output.cpu().numpy()  # [T_val, 7]

    # Sharpe（基於真實投資組合日報酬）
    port_ret = (w * y_raw_val).sum(axis=1)           # [T_val]
    sharpe   = float(port_ret.mean() / (port_ret.std() + 1e-8))

    # MSE (對齊標準化標籤)
    mse = float(np.mean((w - y_val) ** 2))

    # 方向命中率（基於真實報酬）
    hit_rate = float(np.mean(np.sign(w) == np.sign(y_raw_val)))

    return {"sharpe": sharpe, "mse": mse, "hit_rate": hit_rate}


# ═════════════════════════════════════════════════════════════════════
# 主訓練迴圈
# ═════════════════════════════════════════════════════════════════════

def train_moe_router(
    X_train: np.ndarray,   # [T_train, 63]
    y_train: np.ndarray,   # [T_train, 7]
    y_raw_train: np.ndarray,
    X_val:   np.ndarray,   # [T_val, 63]
    y_val:   np.ndarray,   # [T_val, 7]
    y_raw_val: np.ndarray,
    epochs:     int = 200,
    batch_size: int = 64,
) -> tuple[QuantMoERouter, StandardScaler]:
    """用橫截面日度數據訓練 QuantMoERouter。

    Returns
    -------
    router : 訓練完畢的 QuantMoERouter（已還原最佳權重）
    scaler : 已 fit 的 StandardScaler（推論時需使用）
    """
    device = torch.device("cpu")  # Docker on Mac M1，強制 CPU

    # ── 特徵標準化 ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)

    # ── 建立 Router ───────────────────────────────────────────────
    router_config = RouterConfig(
        input_dim=INPUT_DIM,           # 63
        num_experts=NUM_EXPERTS,       # 3
        output_dim=OUTPUT_DIM,         # 7
        top_k=TOP_K,                   # 2
        noise_type=GatingNoiseType.GAUSSIAN,  # 訓練階段加入高斯噪音
        aux_loss_coeff=AUX_LOSS_COEFF, # 0.10（修復 Expert Collapse）
    )
    experts = build_specialized_experts(
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
    )
    router = QuantMoERouter(config=router_config, experts=experts).to(device)

    total_params = sum(p.numel() for p in router.parameters())

    # ── 優化器 + 排程器 ────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        router.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    # CosineAnnealing：LR 從 LR 衰減到 LR×0.05
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR * 0.05
    )
    criterion = CombinedPortfolioLoss(
        mse_weight=0.40, sharpe_weight=0.30, aux_weight=0.15,
        turnover_weight=0.15, turnover_coeff=1.0,
    )

    # ── Early Stopping 狀態 ────────────────────────────────────────
    best_val_sharpe  = -np.inf
    best_state_dict  = None
    patience_counter = 0

    # ── Tensor 化 ─────────────────────────────────────────────────
    X_t = torch.tensor(X_train_sc, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train,    dtype=torch.float32, device=device)
    y_raw_t = torch.tensor(y_raw_train, dtype=torch.float32, device=device)
    T   = len(X_t)

    # ── 表頭輸出 ─────────────────────────────────────────────────
    SEP = "═" * 76
    print(f"\n{SEP}")
    print(f"  MoE Router Training  |  {epochs} Epochs  |  CPU")
    print(f"  Expert-0  : MLP (Tech, {len(TECH_FEATURE_INDICES)}D)")
    print(f"  Expert-1  : TabNet (Macro, {len(MACRO_FEATURE_INDICES)}D)")
    print(f"  Expert-2  : MLP (All, {INPUT_DIM}D)")
    print(f"  Expert-3  : MLP (Sentiment, {INPUT_DIM}D, narrow)")
    print(f"  Params    : {total_params:,}")
    print(f"  Input     : {INPUT_DIM}D  ({N_ASSETS} assets × {N_FEATURES} features)")
    print(f"  Loss      : 0.40×MSE + 0.30×(−Sharpe) + 0.15×Aux + 0.15×Turnover  |  aux_coeff={AUX_LOSS_COEFF}")
    print(f"  Train     : {T} days  |  Val : {len(X_val)} days")
    print(f"  Optimizer : Adam(lr={LR})  CosineAnnealing  wd={WEIGHT_DECAY}")
    print(f"  EarlyStop : patience={PATIENCE}")
    print(f"{SEP}")
    print(f"\n  {'Ep':>4}  {'TotalLoss':>10}  {'MSE':>8}  "
          f"{'TrSharpe':>9}  {'ValSharpe':>10}  {'HitRate':>8}  {'LR':>9}")
    print(f"  {'-'*76}")

    # ══════════════════════════════════════════════════════════════
    # 訓練迴圈
    # ══════════════════════════════════════════════════════════════
    for epoch in range(1, epochs + 1):
        router.train()

        # ── v2.0 時序感知訓練：不再打亂，保留時序以計算真實 Turnover ──
        # 為了 Turnover Penalty 的準確性，使用時間順序而非隨機排列
        X_seq = X_t
        y_seq = y_t
        y_raw_seq = y_raw_t

        batch_losses:  list[float] = []
        batch_mse:     list[float] = []
        batch_sharpe:  list[float] = []
        batch_turnover: list[float] = []

        # 追蹤前一個 batch 的最後一組預測權重
        prev_batch_last_weights: torch.Tensor | None = None

        for start in range(0, T, batch_size):
            X_b = X_seq[start : start + batch_size]  # [B, 63]
            y_b = y_seq[start : start + batch_size]  # [B, 7]
            y_raw_b = y_raw_seq[start : start + batch_size]

            optimizer.zero_grad()
            out: RoutingOutput = router(X_b)

            # 構建 prev_weights：用前一步的預測作為「昨天的持倉」
            curr_w = out.combined_output  # [B, 7]
            if prev_batch_last_weights is not None:
                # 將前一 batch 的最後權重與當前 batch 的前 B-1 個拼接
                prev_w = torch.cat([
                    prev_batch_last_weights.unsqueeze(0),
                    curr_w[:-1].detach(),
                ], dim=0)  # [B, 7]
            else:
                prev_w = torch.cat([
                    torch.zeros(1, curr_w.shape[1], device=device),
                    curr_w[:-1].detach(),
                ], dim=0)  # [B, 7]

            total_loss, metrics = criterion(
                curr_w,               # [B, 7]
                y_b,                  # [B, 7]
                y_raw_b,
                out.aux_loss,
                prev_weights=prev_w,  # v2.0: 換手率懲罰
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()

            # 更新前一 batch 的尾端權重（detach 避免計算圖膨脹）
            prev_batch_last_weights = curr_w[-1].detach()

            batch_losses.append(metrics["total"])
            batch_mse.append(metrics["mse"])
            batch_sharpe.append(metrics["sharpe"])
            batch_turnover.append(metrics["turnover"])

        scheduler.step()

        avg_loss   = float(np.mean(batch_losses))
        avg_mse    = float(np.mean(batch_mse))
        avg_sharpe = float(np.mean(batch_sharpe))
        lr_now     = float(scheduler.get_last_lr()[0])

        # Validation
        val_m      = compute_val_metrics(router, X_val, y_val, y_raw_val, scaler, device)
        val_sharpe = val_m["sharpe"]
        hit_rate   = val_m["hit_rate"]

        # Early stopping
        improved = val_sharpe > best_val_sharpe
        if improved:
            best_val_sharpe  = val_sharpe
            best_state_dict  = {k: v.clone() for k, v in router.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # 輸出（每 10 epoch、首 epoch、最後 epoch）
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs or improved:
            marker = " ← best" if improved else ""
            print(
                f"  {epoch:>4}  {avg_loss:>10.4f}  {avg_mse:>8.5f}  "
                f"{avg_sharpe:>9.4f}  {val_sharpe:>10.4f}  {hit_rate:>7.1%}  "
                f"{lr_now:>9.2e}{marker}"
            )

        if patience_counter >= PATIENCE:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    print(f"\n  Best Validation Sharpe : {best_val_sharpe:+.4f}")

    # 還原最佳 checkpoint
    if best_state_dict is not None:
        router.load_state_dict(best_state_dict)

    return router, scaler


# ═════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═════════════════════════════════════════════════════════════════════

def save_checkpoint(
    router:     QuantMoERouter,
    scaler:     StandardScaler,
    assets:     list[str],
    val_sharpe: float,
    firewall:   Optional[IntelligentRiskFirewall] = None,
    path:       Optional[Path] = None,
) -> Path:
    """儲存模型權重、Router 配置、StandardScaler 及 Firewall。"""
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path  = CHECKPOINT_DIR / f"moe_router_{stamp}.pt"

    firewall_bytes = pickle.dumps(firewall) if firewall is not None else None

    torch.save({
        "model_state_dict": router.state_dict(),
        "router_config": {
            "input_dim":      INPUT_DIM,
            "num_experts":    NUM_EXPERTS,
            "output_dim":     OUTPUT_DIM,
            "top_k":          TOP_K,
            "aux_loss_coeff": AUX_LOSS_COEFF,
            "expert_type":    "specialized_v2",  # 標記差異化架構
        },
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "firewall_bytes": firewall_bytes,
        "assets":       assets,
        "val_sharpe":   val_sharpe,
        "hidden_dim":   HIDDEN_DIM,
        "feature_cols": FEATURE_COLS,
        "timestamp":    datetime.now().isoformat(),
    }, path)

    logger.info("Checkpoint saved → %s", path)
    return path


def load_checkpoint(
    path: Path,
) -> tuple[QuantMoERouter, StandardScaler, list[str], Optional[IntelligentRiskFirewall]]:
    """從 checkpoint 重建 Router + StandardScaler + Firewall。"""
    ckpt      = torch.load(path, map_location="cpu", weights_only=False)
    cfg       = ckpt["router_config"]
    hidden    = ckpt.get("hidden_dim", HIDDEN_DIM)

    # 判斷是新版差異化架構還是舊版同質 MLP
    expert_type = cfg.get("expert_type", "homogeneous")
    if expert_type == "specialized_v2":
        experts = build_specialized_experts(
            output_dim=cfg["output_dim"],
            hidden_dim=hidden,
            dropout=0.0,   # 推論階段：無 Dropout
        )
    else:
        # 向後相容舊版 checkpoint
        experts = nn.ModuleList([
            build_expert_mlp(cfg["input_dim"], cfg["output_dim"], hidden, dropout=0.0)
            for _ in range(cfg["num_experts"])
        ])

    config = RouterConfig(
        input_dim=cfg["input_dim"],
        num_experts=cfg["num_experts"],
        output_dim=cfg["output_dim"],
        top_k=cfg["top_k"],
        noise_type=GatingNoiseType.NONE,   # 推論階段：無噪音
        aux_loss_coeff=cfg["aux_loss_coeff"],
    )
    router = QuantMoERouter(config=config, experts=experts)
    router.load_state_dict(ckpt["model_state_dict"])
    router.eval()

    scaler        = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"],  dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1000  # Dummy value for strict sklearn API compliance

    assets = ckpt.get("assets", ASSET_UNIVERSE)
    logger.info(
        "Loaded checkpoint: %s | val_sharpe=%.4f",
        path.name, ckpt.get("val_sharpe", float("nan")),
    )

    firewall = None
    if ckpt.get("firewall_bytes") is not None:
        firewall = pickle.loads(ckpt["firewall_bytes"])

    return router, scaler, assets, firewall


def build_raw_daily_returns(
    aligned_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """回測用：取得每天各資產的實際前瞻報酬（未標準化）。

    Returns
    -------
    y_raw  : np.ndarray  [T_dates, N_assets]  每日實際 % 報酬
    dates  : pd.DatetimeIndex
    assets : list[str]  (字母排序)
    """
    df = _add_technical_features(aligned_df)
    df = df.dropna(subset=TECH_COLS + ["forward_return"]).reset_index(drop=True)
    assets = sorted(df["asset_id"].unique().tolist())

    dates_sets = [
        set(df.loc[df["asset_id"] == a, "timestamp"].dt.normalize())
        for a in assets
    ]
    common_dates = sorted(set.intersection(*dates_sets))

    y_rows: list[list[float]] = []
    valid_dates: list         = []

    for date in common_dates:
        day_df = df[df["timestamp"].dt.normalize() == date].set_index("asset_id")
        if not all(a in day_df.index for a in assets):
            continue
        ret_row = []
        valid   = True
        for asset in assets:
            fwd = float(day_df.loc[asset].get("forward_return", np.nan))
            if np.isnan(fwd) or np.isinf(fwd):
                valid = False
                break
            ret_row.append(fwd)
        if valid:
            y_rows.append(ret_row)
            valid_dates.append(date)

    y_raw = np.array(y_rows, dtype=np.float32)  # [T_dates, N_assets]
    return y_raw, pd.DatetimeIndex(valid_dates), assets


def find_latest_checkpoint() -> Optional[Path]:
    """回傳 CHECKPOINT_DIR 中最新的 checkpoint，若無則 None。"""
    ckpts = sorted(CHECKPOINT_DIR.glob("moe_router_*.pt"))
    return ckpts[-1] if ckpts else None


# ═════════════════════════════════════════════════════════════════════
# 獨立執行入口
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if not FRED_API_KEY:
        print("❌ FRED_API_KEY 未設定，請確認 .env 檔案或環境變數。")
        sys.exit(1)

    # ── STEP 1: 載入真實數據 ──────────────────────────────────────
    print("\n>>> [1/5] 載入真實市場數據...")
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    daily_prices, macro_data = load_all_data(
        tickers=ASSET_UNIVERSE,
        fred_api_key=FRED_API_KEY,
        start=DATA_START,
        end=end,
    )
    print(f"    價格: {len(daily_prices):,} rows  |  總經: {len(macro_data):,} rows")

    # ── STEP 2: PiT 對齊 ─────────────────────────────────────────
    print(">>> [2/5] PiT 無前瞻偏誤對齊...")
    frames = []
    for asset in ASSET_UNIVERSE:
        ap = daily_prices[daily_prices["asset_id"] == asset].copy()
        am = macro_data[macro_data["asset_id"] == asset].copy()
        aligned, _ = enforce_pit_alignment(
            daily_prices=ap,
            macro_fundamental_data=am,
            timestamp_col="timestamp",
            asset_col="asset_id",
            max_drift_days=45,
            drop_unmatched=True,
            preserve_right_timestamp=True,
        )
        frames.append(aligned)
    aligned_df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )
    print(f"    對齊後: {len(aligned_df):,} rows")

    # ── STEP 3: 構建橫截面日度數據集 ─────────────────────────────
    print(">>> [3/5] 構建橫截面日度數據集...")
    X, y_norm, y_raw, dates, assets = build_daily_dataset(aligned_df)
    print(f"    X={X.shape}  y_norm={y_norm.shape}  y_raw={y_raw.shape}  dates: {dates[0].date()} → {dates[-1].date()}")

    # 時序分割（嚴格按時間順序，不隨機打亂）
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, y_train, y_raw_train = X[:split_idx], y_norm[:split_idx], y_raw[:split_idx]
    X_val,   y_val,   y_raw_val   = X[split_idx:], y_norm[split_idx:], y_raw[split_idx:]
    print(f"    Train: {len(X_train)} days ({dates[0].date()} ~ {dates[split_idx-1].date()})")
    print(f"    Val  : {len(X_val)}   days ({dates[split_idx].date()} ~ {dates[-1].date()})")

    # ── STEP 4: 訓練 ─────────────────────────────────────────────
    print("\n>>> [4/5] 訓練 MoE Router...")
    router, scaler = train_moe_router(
        X_train, y_train, y_raw_train,
        X_val, y_val, y_raw_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
    )

    print("\n>>> [5/5] 最終驗證指標...")
    val_m = compute_val_metrics(router, X_val, y_val, y_raw_val, scaler, torch.device("cpu"))
    print(f"    Sharpe   : {val_m['sharpe']:+.4f}")
    print(f"    MSE      : {val_m['mse']:.6f}")
    print(f"    Hit Rate : {val_m['hit_rate']:.1%}")

    print("\n>>> [6/5] 訓練 IntelligentRiskFirewall...")
    from nexus_quant_os.data_pipelines.feature_engineer import engineer_features
    # 取出 SPY 並過濾日期，使其只對訓練期的數據擬合
    spy_df = aligned_df[aligned_df["asset_id"] == "SPY"].copy().reset_index(drop=True)
    spy_feat_matrix, _ = engineer_features(spy_df)
    train_spy_matrix = spy_feat_matrix[:split_idx]

    firewall = IntelligentRiskFirewall.from_configs(
        hmm_config=HMMConfig(n_regimes=3, n_iter=200, danger_threshold=0.50),
        ood_config=OODConfig(n_estimators=200, ae_epochs=30, ae_latent_dim=8, combined_threshold=0.55),
        firewall_config=FirewallConfig(
            hmm_caution_threshold=0.40, hmm_warning_threshold=0.60, hmm_emergency_threshold=0.78,
            ood_caution_threshold=0.45, ood_warning_threshold=0.60, ood_emergency_threshold=0.78,
            caution_scale=0.70, warning_scale=0.25, emergency_scale=0.00, smooth_blend=True
        )
    )
    firewall.fit(train_spy_matrix)

    ckpt_path = save_checkpoint(router, scaler, assets, val_m["sharpe"], firewall=firewall)
    print(f"\n✅ Checkpoint 已儲存：{ckpt_path.name}")
    print("   下一步：docker compose run --rm quant_engine python main.py")
