import re

with open("nexus_quant_os/portfolio/cvxpy_optimizer.py", "r") as f:
    code = f.read()

# Fix 1: Line 181 in optimize()
code = code.replace(
    "weights = self._constrained_mvo(expected_returns, cov_matrix, moe_weights, hmm_bear_prob)",
    "weights = self._constrained_mvo(expected_returns, cov_matrix, moe_weights, hmm_bear_prob, current_weights, exec_config)"
)

# Fix 2: Restore optimize_batch (optimize_series) to call self.optimize
bad_block = """            # --- Plan B: Constrained MVO ---
            try:
                logger.info("Optimizer mode: Constrained MVO")
                w_mvo = self._constrained_mvo(
                    expected_returns=moe_weight_series[t],
                    cov_matrix=cov,
                    moe_weights=moe_weight_series[t],
                    hmm_bear_prob=hmm_bear_probs[t] if hmm_bear_probs is not None else 0.0,
                    current_weights=current_weights,
                    exec_config=exec_config,
                )
                optimized[t] = self._apply_constraints(w_mvo)
            except Exception:
                optimized[t] = moe_weight_series[t]"""

good_block = """            bp = hmm_bear_probs[t] if hmm_bear_probs is not None else 0.0
            optimized[t] = self.optimize(
                moe_weights=moe_weight_series[t],
                expert_utilisation=expert_util_2d[t],
                cov_matrix=cov,
                returns_history=hist_window,
                hmm_bear_prob=float(bp),
                # Note: optimize_series usually does not have current_weights or exec_config context for historical steps.
            )"""

code = code.replace(bad_block, good_block)

with open("nexus_quant_os/portfolio/cvxpy_optimizer.py", "w") as f:
    f.write(code)

