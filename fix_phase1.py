import re

# Fix feature_engineer.py
f_eng = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/data_pipelines/feature_engineer.py"
with open(f_eng, "r") as f:
    content = f.read()
content = content.replace("pmi_manufacturing", "industrial_production")
# Add clip for volume_zscore
content = content.replace('df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)', 
                          'df["volume_zscore"] = ((df["volume"] - vol_mean) / (vol_std + 1e-8)).clip(-10, 10)')
with open(f_eng, "w") as f:
    f.write(content)

# Fix data_loader.py (synthetics & RNG)
d_loader = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/data_pipelines/data_loader.py"
with open(d_loader, "r") as f:
    content = f.read()
content = content.replace('np.random.seed(42)', 'rng = np.random.default_rng(42)')
content = content.replace('macro_daily["industrial_production"] = 100.0',
                          'macro_daily["industrial_production"] = 100.0\n        macro_daily["yield_curve_slope"] = 0.5')
with open(d_loader, "w") as f:
    f.write(content)

print("Phase 1 Fixed")
