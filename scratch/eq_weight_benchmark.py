import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nexus_quant_os.data_pipelines.data_loader import load_all_data
from nexus_quant_os.training.train_moe import ASSET_UNIVERSE
from backtest import compute_metrics, TRANSACTION_COST_BPS

start = "2020-01-01"
end = "2026-05-26"
fred_api_key = os.environ.get("FRED_API_KEY", "")

daily_prices, _ = load_all_data(
    tickers=ASSET_UNIVERSE,
    fred_api_key=fred_api_key,
    start=start,
    end=end
)

# Calculate daily returns for each asset
pivot_close = daily_prices.pivot(index="timestamp", columns="asset_id", values="close")
pivot_ret = pivot_close.pct_change().dropna()

# Equal weight daily rebalanced
weights = np.ones(len(ASSET_UNIVERSE)) / len(ASSET_UNIVERSE)
port_ret = (pivot_ret * weights).sum(axis=1).values

# Subtract transaction costs roughly (assuming 0 turnover since it's just drift, but let's be conservative)
# Actually, daily rebalancing to equal weight requires some turnover.
# Let's assume a simplified turnover
w_actual = np.tile(weights, (len(pivot_ret), 1))
w_prev = np.vstack([np.zeros((1, len(ASSET_UNIVERSE))), w_actual[:-1]])
w_diff = w_actual - w_prev
daily_tc = np.abs(w_diff).sum(axis=1) * (TRANSACTION_COST_BPS * 1e-4)
port_ret_tc = port_ret - daily_tc

metrics = compute_metrics(port_ret_tc, "Equal Weight")
spy_ret = pivot_ret["SPY"].values
spy_metrics = compute_metrics(spy_ret, "SPY")

print("=== Equal Weight (10 Assets) Benchmark 2020-2026 ===")
print(f"Total Return: {metrics['total_return']:+.1%} (SPY: {spy_metrics['total_return']:+.1%})")
print(f"CAGR: {metrics['cagr']:+.2%} (SPY: {spy_metrics['cagr']:+.2%})")
print(f"Sharpe: {metrics['sharpe']:+.3f} (SPY: {spy_metrics['sharpe']:+.3f})")
print(f"Max DD: {metrics['max_drawdown']:.2%} (SPY: {spy_metrics['max_drawdown']:.2%})")
