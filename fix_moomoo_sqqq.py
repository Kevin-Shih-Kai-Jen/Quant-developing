import re

file_path = "/Users/coolguy/developer/nexus_quant_os/run_moomoo_trade.py"
with open(file_path, "r") as f:
    content = f.read()

old_sqqq = """    total_exposure = sum(target_weights.values())"""

new_sqqq = """    # ── 緊急避險：SQQQ 動態配置 ──
    if decision.hmm_danger_prob > 0.78 and decision.scale_factor <= 0.5:
        target_weights["SQQQ"] = 0.10
        print("    🚨 EMERGENCY: Allocating 10% to SQQQ for crash protection!")

    total_exposure = sum(target_weights.values())"""

if old_sqqq in content:
    content = content.replace(old_sqqq, new_sqqq, 1) # Replace only the first occurrence
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully added SQQQ logic to run_moomoo_trade.py")
else:
    print("Could not find old_sqqq block.")

