import sys
import re

file_path = "/Users/coolguy/developer/nexus_quant_os/run_sqqq_backtest.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add yfinance import and fetch SQQQ
yfinance_patch = """
    # ── 0. Fetch SQQQ Data ────────────────────────────────────────────
    print(">>> [0/5] Fetching SQQQ data for hedging...")
    import yfinance as yf
    sqqq_df = yf.download("SQQQ", start=bt_start, end=bt_end, progress=False)
    if not sqqq_df.empty:
        sqqq_ret = sqqq_df['Adj Close'].pct_change().dropna()
        # yfinance returns multi-index or single level depending on version, handle it
        if isinstance(sqqq_ret, pd.DataFrame):
            sqqq_ret = sqqq_ret.iloc[:, 0]
        sqqq_ret.index = sqqq_ret.index.tz_localize(None).normalize()
    else:
        sqqq_ret = pd.Series()
        
    print(">>> [1/5] 載入真實市場數據...")
"""
content = content.replace('    print(">>> [1/5] 載入真實市場數據...")', yfinance_patch)


# 2. Add Firewall state simulation and SQQQ weight array
firewall_patch = """
    print(f"    ✓ 權重平滑已套用（含 VIX 動態視窗與 Expert-0 豁免）")

    # ── v2.0 Phase 2.5: 套用 Firewall 動態風控與 SQQQ 避險 ──
    w_final = np.zeros_like(w_smooth)
    sqqq_weights = np.zeros(min_len)
    
    sqqq_val_ret = np.zeros(min_len)
    for i, date in enumerate(dates_val[:min_len]):
        if date in sqqq_ret.index:
            sqqq_val_ret[i] = float(sqqq_ret.loc[date])
            
    if firewall is not None:
        firewall.shock_state["US"] = {"days_in_shock": 0, "days_safe": 0, "last_scale_factor": 1.0}
        for t in range(min_len):
            start_idx = max(0, t - 20)
            obs = spy_val_feat[start_idx : t + 1]
            raw_w = w_smooth[t].copy()
            
            if len(obs) >= 5:
                decision = firewall.evaluate(obs, raw_w, market="US")
                w_final[t] = decision.adjusted_weights
                scale = decision.scale_factor
                
                if decision.hmm_danger_prob > 0.78 and scale <= 0.5:
                    sqqq_weights[t] = 0.10
            else:
                w_final[t] = raw_w
        print(f"    ✓ Firewall 風控已套用 (SQQQ 避險啟動次數: {np.sum(sqqq_weights > 0)})")
    else:
        w_final = w_smooth.copy()
"""
content = content.replace('    print(f"    ✓ 權重平滑已套用（含 VIX 動態視窗與 Expert-0 豁免）")', firewall_patch)

# 3. Replace w_smooth with w_final in limit reject mask
content = content.replace('if w_smooth[t, i] != w_smooth[t-1, i]:', 'if w_final[t, i] != w_final[t-1, i]:')
content = content.replace('w_smooth[t, i] = w_smooth[t-1, i]', 'w_final[t, i] = w_final[t-1, i]')

# 4. Replace turnover calculation
content = content.replace('w_prev = np.vstack([np.zeros((1, w_smooth.shape[1])), w_smooth[:-1]])', 'w_prev = np.vstack([np.zeros((1, w_final.shape[1])), w_final[:-1]])')
content = content.replace('w_diff = w_smooth - w_prev', 'w_diff = w_final - w_prev')

# 5. Add SQQQ transaction cost
sqqq_tc_patch = """
    # 加入 SQQQ 交易成本
    sqqq_prev = np.concatenate([[0.0], sqqq_weights[:-1]])
    sqqq_diff = sqqq_weights - sqqq_prev
    daily_tc += np.abs(sqqq_diff) * TRANSACTION_COST_BPS * 1e-4
"""
content = content.replace('daily_tc += cost', 'daily_tc += cost\n' + sqqq_tc_patch)

# 6. Replace return calculation
content = content.replace('port_ret    = (w_smooth * y_raw_val[:min_len]).sum(axis=1)', 'port_ret    = (w_final * y_raw_val[:min_len]).sum(axis=1) + (sqqq_weights * sqqq_val_ret)')

with open(file_path, "w") as f:
    f.write(content)
print("Patch applied.")
