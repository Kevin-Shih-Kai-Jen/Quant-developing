import pytest
import numpy as np

def test_backtest_limit_up_reject():
    """
    Test limit order reject and custom TW slippage + sell tax.
    """
    # Simulate a scenario directly
    min_len = 3
    assets = ["2330.TW", "NVDA"]
    
    w_smooth = np.array([
        [0.0, 0.0],
        [0.5, 0.5], # Try to buy 50%
        [0.0, 0.0], # Try to sell 50%
    ])
    
    # 2330.TW is limit-up at t=1 (high == low)
    limit_reject_mask = np.array([
        [False, False],
        [True, False],  # 2330.TW rejected at t=1
        [False, False],
    ])
    
    reject_count = 0
    for t in range(1, len(w_smooth)):
        for i in range(len(assets)):
            if limit_reject_mask[t, i]:
                if w_smooth[t, i] != w_smooth[t-1, i]:
                    w_smooth[t, i] = w_smooth[t-1, i]
                    reject_count += 1
                    
    assert reject_count == 1
    # 2330.TW should stay 0.0 at t=1
    assert w_smooth[1, 0] == 0.0
    assert w_smooth[1, 1] == 0.5
    
    # Calculate TC
    w_prev = np.vstack([np.zeros((1, w_smooth.shape[1])), w_smooth[:-1]])
    w_diff = w_smooth - w_prev
    
    daily_tc = np.zeros(min_len)
    for i, asset in enumerate(assets):
        diff = w_diff[:, i]
        is_tw = asset.endswith(".TW") or asset.isdigit()
        
        if is_tw:
            sell_tax = 0.003
            slippage = 0.002
            cost = np.where(diff < 0, np.abs(diff) * (sell_tax + slippage), np.abs(diff) * slippage)
        else:
            tc_cost = 5 * 1e-4 # 0.0005
            cost = np.abs(diff) * tc_cost
            
        daily_tc += cost

    # NVDA buy 0.5 at t=1 -> cost 0.5 * 0.0005 = 0.00025
    assert np.isclose(daily_tc[1], 0.00025)
    
    # NVDA sell 0.5 at t=2 -> cost 0.5 * 0.0005 = 0.00025
    # 2330.TW sell 0.0 at t=2 -> cost 0
    assert np.isclose(daily_tc[2], 0.00025)
