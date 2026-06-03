import re

file_path = "/Users/coolguy/developer/nexus_quant_os/run_moomoo_trade.py"
with open(file_path, "r") as f:
    content = f.read()

old_target = """        # ── Target allocation ─────────────────────────────────────────
        print("  ▸ Target Allocation:")"""

new_target = """        # ── HODL Exemption (NVDA, AVGO) ───────────────────────────────
        exempt_assets = ["NVDA", "AVGO"]
        for p in positions:
            if p.symbol in exempt_assets:
                current_weight = p.market_value / account.equity if account.equity > 0 else 0.0
                model_target = target_weights.get(p.symbol, 0.0)
                # Ensure we never sell these exempted assets below their current weight
                if model_target < current_weight:
                    target_weights[p.symbol] = current_weight
                    print(f"    🛡️ HODL Exemption: {p.symbol} target forced to {current_weight:.1%} (Model wanted {model_target:.1%})")

        # Re-calculate total exposure after HODL overrides
        total_exposure = sum(target_weights.values())
        metadata["cash_pct"] = max(0.0, 1.0 - total_exposure) * 100

        # ── Target allocation ─────────────────────────────────────────
        print("  ▸ Target Allocation:")"""

if old_target in content:
    content = content.replace(old_target, new_target, 1)
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully added HODL logic to run_moomoo_trade.py")
else:
    print("Could not find old_target block.")
