import os
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, "/Users/coolguy/developer/nexus_quant_os")

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.data_pipelines.feature_engineer import engineer_features, FEATURE_COLS

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("❌ FRED_API_KEY 未設定，無法下載總經數據")
    sys.exit(1)

# 使用 SPY 作為代表資產進行共線性測試
TICKER = "SPY"

print(f"下載 {TICKER} 數據並建立特徵矩陣...")
daily_prices, macro_data = load_all_data(
    tickers=[TICKER],
    fred_api_key=FRED_API_KEY,
    start="2010-01-01",
    end="2020-12-31"
)

aligned, _ = enforce_pit_alignment(
    daily_prices=daily_prices,
    macro_fundamental_data=macro_data,
    timestamp_col="timestamp",
    asset_col="asset_id",
    max_drift_days=45,
    drop_unmatched=True
)

features, cols = engineer_features(aligned)
df_features = pd.DataFrame(features, columns=cols)

# 計算 Spearman 相關矩陣
corr = df_features.corr(method="spearman")
print("\n=== Spearman Rank Correlation Matrix ===")
print(corr.round(2))

# 找出高度相關的特徵對 (|corr| > 0.7)
print("\n=== High Correlation Pairs (|corr| > 0.7) ===")
high_corr = []
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(corr.iloc[i, j]) > 0.7:
            print(f"{cols[i]} <--> {cols[j]}: {corr.iloc[i, j]:.2f}")
            high_corr.append((cols[i], cols[j]))
if not high_corr:
    print("無高度相關的特徵對。")

# 計算 VIF
print("\n=== Variance Inflation Factor (VIF) ===")
vif_data = pd.DataFrame()
vif_data["Feature"] = df_features.columns
# VIF 需處理常數項，因此對特徵進行標準化
df_norm = (df_features - df_features.mean()) / df_features.std()
vif_data["VIF"] = [variance_inflation_factor(df_norm.values, i) for i in range(df_norm.shape[1])]
vif_data = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data.to_string(index=False))

if vif_data["VIF"].max() > 10:
    print("\n⚠️ 警告：發現嚴重共線性 (VIF > 10)！這可能是 Expert-0 被忽略的原因。")
else:
    print("\n✅ 特徵之間無嚴重共線性 (VIF < 10)。Router 冷落 Expert-0 應該是優化目標或網路結構設定導致，引入 Load Balancing Loss 將會有效！")
