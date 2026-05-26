"""
data_pipelines/feature_engineer.py — 統一特徵工程模組
======================================================

將原本分散在 main.py 與 train_moe.py 中的特徵工程邏輯
集中到此模組，避免循環依賴與維度不一致。

特徵清單 (9 個)：
    技術面 (4)：daily_return, realised_vol, high_low_spread, volume_zscore
    總經面 (5)：cpi_yoy, unemployment_rate, pmi_manufacturing,
               credit_spread, yield_curve_slope

Author : Nexus Quant OS — Data Engineering Division
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus_quant_os.data_pipelines.feature_engineer")

# 滾動窗口大小
VOL_LOOKBACK = 20

# 統一特徵欄位定義（與 train_moe.py 一致）
FEATURE_COLS = [
    "daily_return", "realised_vol", "high_low_spread", "volume_zscore",
    "cpi_yoy", "unemployment_rate", "pmi_manufacturing",
    "credit_spread", "yield_curve_slope",
]

MACRO_COLS = [
    "cpi_yoy", "unemployment_rate", "pmi_manufacturing",
    "credit_spread", "yield_curve_slope",
]

TECH_COLS = ["daily_return", "realised_vol", "high_low_spread", "volume_zscore"]


def engineer_features(aligned_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """將對齊後的 DataFrame 轉換為數值特徵矩陣。

    計算的特徵（共 9 個）：
        - daily_return      : 日報酬率 (close-to-close)
        - realised_vol      : 20 日滾動波動度
        - high_low_spread   : 日內振幅 (high - low) / close
        - volume_zscore     : 成交量 z-score (20 日滾動)
        - cpi_yoy           : CPI 年增率 (來自 PiT 對齊)
        - unemployment_rate : 失業率
        - pmi_manufacturing : PMI 製造業指數
        - credit_spread     : 信用利差
        - yield_curve_slope : 殖利率曲線斜率（v2.0 新增）

    Parameters
    ----------
    aligned_df : pd.DataFrame
        經過 enforce_pit_alignment 處理後的 DataFrame。

    Returns
    -------
    feature_matrix : np.ndarray  shape [T, F]  (F=9)
    feature_names  : list[str]
    """
    df = aligned_df.copy()

    # ── 高頻技術特徵 ──────────────────────────────────────────────
    df["daily_return"] = df.groupby("asset_id")["close"].pct_change()
    df["realised_vol"] = (
        df.groupby("asset_id")["daily_return"]
        .transform(lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).std())
    )
    df["high_low_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # 成交量 z-score（20 日滾動標準化）
    vol_mean = df.groupby("asset_id")["volume"].transform(
        lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).mean()
    )
    vol_std = df.groupby("asset_id")["volume"].transform(
        lambda x: x.rolling(VOL_LOOKBACK, min_periods=5).std()
    )
    df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)

    # ── 低頻總經特徵（已由 Aligner 嚴格對齊） ────────────────────
    # 策略：在每個資產組內 forward-fill（已公開的值持續有效）
    #       殘留的 NaN 以 0 填補
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = (
                df.groupby("asset_id")[col]
                .transform(lambda x: x.ffill())
            )
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    # ── 合併為特徵矩陣 ────────────────────────────────────────────
    # 清除 NaN 行（前幾天的 MA 或無總經資料的早期日子）
    df = df.dropna(subset=TECH_COLS).reset_index(drop=True)

    feature_matrix = df[FEATURE_COLS].values.astype(np.float64)  # [T, 9]

    logger.info(
        "特徵工程完成 | shape=%s  features=%d",
        feature_matrix.shape, len(FEATURE_COLS),
    )

    return feature_matrix, FEATURE_COLS
