import re

file_path = "/Users/coolguy/developer/nexus_quant_os/run_moomoo_trade.py"
with open(file_path, "r") as f:
    content = f.read()

# I will replace the entire STEP 7 logic with the correct flow.
old_step_7 = """    # ── STEP 7: Portfolio Optimization ────────────────────────────
    print("  ▸ STEP 7: Portfolio optimization...")

    # Apply firewall scaling
    scaled_weights = raw_weights * decision.scale_factor
    # Long-only 模式：裁剪負權重（模擬模式下不允許做空）
    neg_count = (scaled_weights < 0).sum()
    if neg_count > 0:
        logger.warning("裁剪 %d 個負權重 (long-only 模式)", neg_count)
    scaled_weights = np.maximum(scaled_weights, 0)
    weight_sum = scaled_weights.sum()
    if weight_sum > 1e-8:
        norm_weights = scaled_weights / weight_sum
    else:
        norm_weights = np.zeros_like(scaled_weights)

    # Try CVXPY optimization
    try:
        # Build recent return matrix for optimizer
        returns_frames = []
        for asset in assets_sorted:
            asset_df = aligned_df[aligned_df["asset_id"] == asset].copy()
            asset_df = asset_df.sort_values("timestamp")
            asset_returns = asset_df["close"].pct_change().dropna().values
            returns_frames.append(asset_returns[-252:])  # Last year

        min_len = min(len(r) for r in returns_frames)
        return_matrix = np.column_stack([r[-min_len:] for r in returns_frames])

        optimizer = PortfolioOptimizer(
            n_assets=len(assets_sorted),
            config=OptimizerConfig(max_single_weight=0.15),
        )
        cov_matrix = np.cov(return_matrix, rowvar=False)
        optimized_weights = optimizer.optimize(
            moe_weights=norm_weights,
            expert_utilisation=np.zeros(1),
            cov_matrix=cov_matrix,
            returns_history=return_matrix,
        )
        opt_mode = "CVXPY-MVO"
    except Exception as exc:
        logger.warning("Optimizer failed: %s — using normalized weights", exc)
        optimized_weights = norm_weights
        opt_mode = "FALLBACK-NORMALIZED"

    # Apply concentration cap
    MAX_WEIGHT = 0.15
    optimized_weights = np.minimum(optimized_weights, MAX_WEIGHT)
    weight_sum = optimized_weights.sum()
    if weight_sum > 1e-8:
        optimized_weights = optimized_weights / weight_sum

    # Weight smoothing
    weight_smoother = WeightSmoother(
        n_assets=len(assets_sorted),
        config=SmootherConfig(alpha=0.30, min_rebalance_threshold=0.04),
    )
    final_weights = weight_smoother.smooth(optimized_weights)"""

new_step_7 = """    # ── STEP 7: Portfolio Optimization ────────────────────────────
    print("  ▸ STEP 7: Portfolio optimization...")

    # 1. Base weights from MoE (normalized)
    base_weights = np.maximum(raw_weights, 0)
    b_sum = base_weights.sum()
    if b_sum > 1e-8:
        base_weights /= b_sum
    else:
        base_weights = np.ones_like(raw_weights) / len(raw_weights)

    # 2. Try CVXPY optimization (on 100% deployed assumption)
    try:
        returns_frames = []
        for asset in assets_sorted:
            asset_df = aligned_df[aligned_df["asset_id"] == asset].copy()
            asset_df = asset_df.sort_values("timestamp")
            asset_returns = asset_df["close"].pct_change().dropna().values
            returns_frames.append(asset_returns[-252:])

        min_len = min(len(r) for r in returns_frames)
        return_matrix = np.column_stack([r[-min_len:] for r in returns_frames])

        optimizer = PortfolioOptimizer(
            n_assets=len(assets_sorted),
            config=OptimizerConfig(max_single_weight=0.15),
        )
        cov_matrix = np.cov(return_matrix, rowvar=False)
        optimized_weights = optimizer.optimize(
            moe_weights=base_weights,
            expert_utilisation=np.zeros(1),
            cov_matrix=cov_matrix,
            returns_history=return_matrix,
        )
        opt_mode = "CVXPY-MVO"
    except Exception as exc:
        logger.warning("Optimizer failed: %s — using base weights", exc)
        optimized_weights = base_weights
        opt_mode = "FALLBACK-NORMALIZED"

    # 3. Apply concentration cap & Weight Smoothing
    MAX_WEIGHT = 0.15
    optimized_weights = np.minimum(optimized_weights, MAX_WEIGHT)
    if optimized_weights.sum() > 1e-8:
        optimized_weights /= optimized_weights.sum()

    weight_smoother = WeightSmoother(
        n_assets=len(assets_sorted),
        config=SmootherConfig(alpha=0.30, min_rebalance_threshold=0.04),
    )
    smooth_weights = weight_smoother.smooth(optimized_weights)

    # 4. FINAL STEP: Apply Firewall Scale & SQQQ Hedging
    final_weights = smooth_weights * decision.scale_factor"""

if old_step_7 in content:
    content = content.replace(old_step_7, new_step_7)
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully patched STEP 7 in run_moomoo_trade.py")
else:
    print("Could not find old_step_7 block.")

