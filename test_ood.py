import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/coolguy/developer/nexus_quant_os")
from main import load_real_market_data
from nexus_quant_os.data_pipelines.feature_engineer import engineer_features
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODConfig, OODAnomalyDetector

daily_prices, macro_data, ds = load_real_market_data()
spy_prices = daily_prices[daily_prices["asset_id"] == "SPY"].copy()
spy_macro = macro_data[macro_data["asset_id"] == "SPY"].copy()

aligned_spy, _ = enforce_pit_alignment(
    daily_prices=spy_prices,
    macro_fundamental_data=spy_macro,
    timestamp_col="timestamp",
    asset_col="asset_id",
    max_drift_days=45,
    drop_unmatched=True,
    preserve_right_timestamp=True,
)

spy_feat, feat_names = engineer_features(aligned_spy)

split_idx = int(len(spy_feat) * 0.8)
train_f = spy_feat[:split_idx]
latest_f = spy_feat[-1:]

ood = OODAnomalyDetector(OODConfig(n_estimators=200, ae_epochs=30, ae_latent_dim=8))
ood.fit(train_f)
score = ood.detect(latest_f)

print(f"Latest Date: {aligned_spy['timestamp'].iloc[-1]}")
print(f"OOD Score: {score.combined_anomaly_score:.4f} (Threshold: 0.8)")
print(f"AE_Z: {score.ae_zscore:.4f}")
print(f"IF_Z: {score.isolation_forest_score:.4f}")

train_mean = np.mean(train_f, axis=0)
train_std = np.std(train_f, axis=0)
print("\n--- Feature Deviation (Z-score vs Train) ---")
for name, val, m, s in zip(feat_names, latest_f[0], train_mean, train_std):
    z = (val - m) / (s + 1e-8)
    if abs(z) > 1.5:
        print(f"🚨 {name}: Z = {z:+.2f} (Val: {val:.4f}, Mean: {m:.4f}, Std: {s:.4f})")
    else:
        print(f"   {name}: Z = {z:+.2f} (Val: {val:.4f}, Mean: {m:.4f}, Std: {s:.4f})")
