#!/usr/bin/env python3
"""
run_moomoo_trade.py — End-to-End: Quant Model → Moomoo SIMULATE Execution
===========================================================================

Runs the full Nexus Quant OS pipeline and automatically executes the
resulting portfolio weights on the Moomoo SIMULATE (paper trading) account.

╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  SIMULATE ONLY — This script CANNOT place real-money trades ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    cd /Users/coolguy/developer/nexus_quant_os
    source .venv/bin/activate
    PYTHONPATH=. python run_moomoo_trade.py
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from nexus_quant_os.notifications.discord_notifier import (
    DiscordNotifier,
    TradeRecord,
    TradeReport,
)
from nexus_quant_os.monitoring.health_check import (
    run_pre_trade_checks,
    check_weight_sanity,
    format_health_report,
    Severity,
)

# ═══════════════════════════════════════════════════════════════════
# 0. LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logging.getLogger("nexus_quant_os.main").setLevel(logging.INFO)
logging.getLogger("nexus_quant_os.data_pipelines.data_loader").setLevel(logging.INFO)
logging.getLogger("nexus_quant_os.risk_firewall.firewall_core").setLevel(logging.INFO)
logging.getLogger("nexus_quant_os.execution.futu_broker").setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("nexus_quant_os.run_moomoo_trade")
SEP = "═" * 78

# ═══════════════════════════════════════════════════════════════════
# 1. REUSE MAIN PIPELINE COMPONENTS
# ═══════════════════════════════════════════════════════════════════

# Import everything from main.py — this gives us access to:
# ASSET_UNIVERSE, N_ASSETS, FRED_API_KEY, FIREWALL_TRAIN_RATIO,
# load_real_market_data(), engineer_features(), etc.
from main import (
    ASSET_UNIVERSE,
    N_ASSETS,
    FIREWALL_TRAIN_RATIO,
    load_real_market_data,
)
from nexus_quant_os.data_pipelines.aligner import enforce_pit_alignment
from nexus_quant_os.data_pipelines.feature_engineer import engineer_features
from nexus_quant_os.training.train_moe import (
    build_daily_dataset,
    find_latest_checkpoint,
    load_checkpoint,
)
from nexus_quant_os.models.moe_router import RoutingOutput
from nexus_quant_os.risk_firewall.hmm_regime_detector import HMMConfig
from nexus_quant_os.risk_firewall.ood_anomaly_detector import OODConfig
from nexus_quant_os.risk_firewall.firewall_core import (
    FirewallConfig,
    IntelligentRiskFirewall,
)
from nexus_quant_os.portfolio.optimizer import PortfolioOptimizer, OptimizerConfig
from nexus_quant_os.portfolio.weight_smoother import WeightSmoother, SmootherConfig
from nexus_quant_os.portfolio.regime_allocator import RegimeAllocator, RegimeAllocatorConfig
from nexus_quant_os.execution.futu_broker import FutuBroker


def run_pipeline_and_get_weights() -> tuple[dict[str, float], dict, TradeReport]:
    """Run the full quant pipeline and return target weights + report."""
    report = TradeReport(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))

    print(f"\n{SEP}")
    print("  🧠 NEXUS QUANT OS — Pipeline → Moomoo SIMULATE Execution")
    print(f"{SEP}\n")

    # ── STEP 1: Data Loading ──────────────────────────────────────
    print("  ▸ STEP 1: Loading real market data...")
    daily_prices, macro_data, data_source = load_real_market_data()
    report.data_source = data_source
    report.n_price_rows = len(daily_prices)
    report.n_macro_rows = len(macro_data)
    print(f"    Source: {data_source} | "
          f"Prices: {len(daily_prices):,} rows | "
          f"Macro: {len(macro_data):,} rows\n")

    # ── STEP 2: PiT Alignment ────────────────────────────────────
    print("  ▸ STEP 2: Point-in-Time alignment...")
    aligned_frames = []
    for asset in ASSET_UNIVERSE:
        asset_prices = daily_prices[daily_prices["asset_id"] == asset].copy()
        asset_macro = macro_data[macro_data["asset_id"] == asset].copy()
        aligned_asset, _ = enforce_pit_alignment(
            daily_prices=asset_prices,
            macro_fundamental_data=asset_macro,
            timestamp_col="timestamp",
            asset_col="asset_id",
            max_drift_days=45,
            drop_unmatched=True,
            preserve_right_timestamp=True,
        )
        aligned_frames.append(aligned_asset)

    aligned_df = pd.concat(aligned_frames, ignore_index=True)
    aligned_df = aligned_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    report.n_aligned_rows = len(aligned_df)
    print(f"    Aligned: {len(aligned_df):,} rows | {N_ASSETS} assets\n")

    # ── STEP 3: Feature Engineering ───────────────────────────────
    print("  ▸ STEP 3: Feature engineering...")
    feature_matrix, feature_names = engineer_features(aligned_df)
    report.n_features = feature_matrix.shape[1]
    print(f"    Features: {feature_matrix.shape}\n")

    # ── HEALTH GATE 1: Data Quality (Layer 1) ────────────────────
    print("  ▸ HEALTH CHECK: Data & feature quality...")
    health = run_pre_trade_checks(
        prices_df=daily_prices,
        macro_df=macro_data,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
    )
    for c in health.checks:
        icon = "✅" if c.severity == Severity.OK else (
            "⚠️" if c.severity == Severity.WARNING else "🚨"
        )
        print(f"    {icon} {c.name}: {c.message}")

    if not health.is_healthy:
        msg = format_health_report(health)
        print(f"\n    🚨 CRITICAL health check failed — aborting pipeline")
        report.error = f"Health check failed: {health.n_critical} critical issue(s)"
        # Notify via Discord (will be called from __main__)
        raise RuntimeError(
            f"Pre-trade health gate FAILED: {health.n_critical} critical, "
            f"{health.n_warnings} warnings.\n" + msg
        )
    print(f"    {health.summary()}\n")

    # ── STEP 4: Risk Firewall ─────────────────────────────────────
    print("  ▸ STEP 4: Training risk firewall...")
    spy_df = aligned_df[aligned_df["asset_id"] == "SPY"].copy().reset_index(drop=True)
    spy_feature_matrix, _ = engineer_features(spy_df)
    split_idx = int(len(spy_feature_matrix) * FIREWALL_TRAIN_RATIO)
    train_features = spy_feature_matrix[:split_idx]

    firewall = IntelligentRiskFirewall.from_configs(
        hmm_config=HMMConfig(n_regimes=3, n_iter=200, danger_threshold=0.50),
        ood_config=OODConfig(
            n_estimators=200, ae_epochs=30, ae_latent_dim=8,
            combined_threshold=0.55,
        ),
        firewall_config=FirewallConfig(
            hmm_caution_threshold=0.40,
        ),
    )
    firewall.fit(train_features)
    print(f"    Firewall trained on {split_idx} samples\n")

    # ── STEP 5: MoE Router Inference ──────────────────────────────
    print("  ▸ STEP 5: Running MoE Router inference...")
    import torch.nn as nn
    from nexus_quant_os.models.moe_router import (
        GatingNoiseType, QuantMoERouter, RouterConfig, build_dummy_expert,
    )
    from nexus_quant_os.training.train_moe import INPUT_DIM as MOE_INPUT_DIM

    device = torch.device("cpu")
    ckpt_path = find_latest_checkpoint()

    # Build daily cross-sectional features
    X_daily, _, _, dates_daily, assets_sorted = build_daily_dataset(
        aligned_df, inference_mode=True,
    )
    actual_input_dim = X_daily.shape[1]

    router_loaded = False
    if ckpt_path:
        try:
            router, scaler, trained_assets, _ = load_checkpoint(ckpt_path)
            router = router.to(device).eval()
            X_scaled = scaler.transform(X_daily).astype(np.float32)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            router_loaded = True
            report.router_mode = "CHECKPOINT"
            report.checkpoint_name = ckpt_path.name
            print(f"    ✅ Checkpoint loaded: {ckpt_path.name}")
        except Exception as e:
            print(f"    ⚠️ Checkpoint mismatch: {type(e).__name__}")
            print(f"    → Falling back to random router")

    if not router_loaded:
        # Path B: Random router (same as main.py fallback)
        n_experts = 4
        router_config = RouterConfig(
            input_dim=actual_input_dim,
            num_experts=n_experts,
            output_dim=len(assets_sorted),
            top_k=2,
            noise_type=GatingNoiseType.NONE,
            aux_loss_coeff=1e-2,
        )
        experts = nn.ModuleList([
            build_dummy_expert(
                input_dim=actual_input_dim,
                output_dim=len(assets_sorted),
                hidden_dim=64,
            )
            for _ in range(n_experts)
        ])
        router = QuantMoERouter(config=router_config, experts=experts).to(device)
        router.eval()
        X_tensor = torch.tensor(X_daily.astype(np.float32), dtype=torch.float32, device=device)
        report.router_mode = "RANDOM_FALLBACK"
        print(f"    Random router: dim={actual_input_dim}, experts={n_experts}")

    with torch.no_grad():
        routing_result: RoutingOutput = router(X_tensor)

    raw_weights = routing_result.combined_output[-1].cpu().numpy()
    print(f"    Assets: {list(assets_sorted)}")
    print(f"    Raw weights: {np.round(raw_weights, 4)}\n")

    # ── HEALTH GATE 2: Weight Sanity (Layer 2) ───────────────────
    weight_check = check_weight_sanity(
        raw_weights, asset_names=list(assets_sorted), max_single_weight=0.40,
    )
    icon = "✅" if weight_check.severity == Severity.OK else (
        "⚠️" if weight_check.severity == Severity.WARNING else "🚨"
    )
    print(f"    {icon} {weight_check.name}: {weight_check.message}")

    if weight_check.severity == Severity.CRITICAL:
        report.error = f"Weight sanity CRITICAL: {weight_check.message}"
        raise RuntimeError(f"Model weight sanity FAILED: {weight_check.message}")
    print()

    # ── STEP 6: Firewall Evaluation ───────────────────────────────
    print("  ▸ STEP 6: Risk firewall evaluation...")
    latest_spy_features = spy_feature_matrix[-1:]
    decision = firewall.evaluate(latest_spy_features, raw_weights)
    report.risk_tier = decision.risk_tier.name
    report.scale_factor = decision.scale_factor
    report.regime = decision.hmm_regime_label
    report.hmm_danger = decision.hmm_danger_prob
    report.ood_score = decision.ood_combined_score
    print(f"    Risk Tier: {decision.risk_tier.name} | "
          f"Scale: {decision.scale_factor:.2f} | "
          f"Regime: {decision.hmm_regime_label}\n")

    # ── STEP 7: Portfolio Optimization ────────────────────────────
    print("  ▸ STEP 7: Portfolio optimization...")

    # Apply firewall scaling
    scaled_weights = raw_weights * decision.scale_factor
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
            config=OptimizerConfig(max_weight=0.15),
        )
        optimized_weights = optimizer.optimize(
            expected_returns=norm_weights,
            return_matrix=return_matrix,
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
        n_assets=N_ASSETS,
        config=SmootherConfig(alpha=0.30, min_rebalance_threshold=0.04),
    )
    final_weights = weight_smoother.smooth(optimized_weights)

    print(f"    Optimizer: {opt_mode}")
    print(f"\n    {'Asset':<8} {'Raw':>8} {'Final':>8}")
    print(f"    {'─'*8} {'─'*8} {'─'*8}")

    # Build target weights dict
    target_weights: dict[str, float] = {}
    for i, asset in enumerate(assets_sorted):
        rw = float(raw_weights[i])
        fw = float(final_weights[i])
        if fw > 0.01:
            target_weights[asset] = fw
        print(f"    {asset:<8} {rw:>+7.4f} {fw:>+7.4f}")

    total_exposure = sum(target_weights.values())
    cash_pct = max(0.0, 1.0 - total_exposure) * 100

    print(f"\n    Exposure: {total_exposure:.2%} | Cash: {cash_pct:.1f}%")

    metadata = {
        "risk_tier": decision.risk_tier.name,
        "scale_factor": decision.scale_factor,
        "regime": decision.hmm_regime_label,
        "opt_mode": opt_mode,
        "n_positions": len(target_weights),
        "total_exposure": total_exposure,
        "cash_pct": cash_pct,
    }

    report.optimizer_mode = opt_mode
    report.target_weights = dict(target_weights)
    report.total_exposure = total_exposure
    report.cash_pct = cash_pct

    return target_weights, metadata, report


def execute_on_moomoo(
    target_weights: dict[str, float],
    metadata: dict,
    report: TradeReport,
) -> TradeReport:
    """Execute target weights on Moomoo SIMULATE account.

    Returns the enriched TradeReport with execution details.
    """

    print(f"\n{SEP}")
    print("  📊 MOOMOO SIMULATE EXECUTION")
    print(f"{SEP}\n")

    # ── Connect ───────────────────────────────────────────────────
    print("  ▸ Connecting to FutuOpenD...")
    try:
        broker = FutuBroker(host="127.0.0.1", port=11111)
    except (ConnectionError, ImportError) as e:
        print(f"    ❌ Cannot connect: {e}")
        report.error = str(e)
        return report

    # ── Clean up stale orders ─────────────────────────────────────
    # Cancel any SUBMITTED orders that were placed outside market hours
    # and never filled.  These freeze cash and cause negative balances.
    print("  ▸ Cleaning up stale pending orders...")
    n_cancelled = broker.cancel_all_pending()
    if n_cancelled > 0:
        print(f"    🧹 Cancelled {n_cancelled} stale order(s)")
        import time
        time.sleep(2)  # wait for account to update
    else:
        print("    ✅ No stale orders")

    # ── Pre-trade snapshot ────────────────────────────────────────
    account = broker.get_account()
    positions = broker.get_positions()
    report.pre_equity = account.equity
    report.pre_cash = account.cash

    print(f"    ✅ Connected (SIMULATE)")
    print(f"    Equity    : ${account.equity:,.2f}")
    print(f"    Cash      : ${account.cash:,.2f}")
    print(f"    Positions : {len(positions)}\n")

    # ── Target allocation ─────────────────────────────────────────
    print("  ▸ Target Allocation:")
    print(f"    {'Symbol':<8} {'Weight':>8} {'Target $':>12}")
    print(f"    {'─'*8} {'─'*8} {'─'*12}")
    for sym, w in sorted(target_weights.items(), key=lambda x: -x[1]):
        print(f"    {sym:<8} {w:>7.1%} ${w * account.equity:>11,.2f}")

    cash_pct = metadata.get("cash_pct", 0)
    print(f"    {'CASH':<8} {cash_pct/100:>7.1%} "
          f"${account.equity * cash_pct/100:>11,.2f}")
    print(f"\n    Risk: {metadata.get('risk_tier')} | "
          f"Regime: {metadata.get('regime')} | "
          f"Optimizer: {metadata.get('opt_mode')}\n")

    # ── Reconcile ─────────────────────────────────────────────────
    print("  ▸ Reconciling portfolio...")
    intents = broker.reconcile(target_weights)

    if not intents:
        print("    ✅ Portfolio aligned — no trades needed\n")
        # Still capture post-trade state
        report.post_equity = account.equity
        report.post_cash = account.cash
        return report

    print(f"    {len(intents)} trade intents:")
    for intent in intents:
        emoji = "🟢" if intent.side == "BUY" else "🔴"
        print(f"    {emoji} {intent.side:>4} {intent.symbol:<8} "
              f"x{intent.qty:>5.0f}  {intent.reason}")

    # ── Execute ───────────────────────────────────────────────────
    print(f"\n  ▸ Executing {len(intents)} orders on SIMULATE...")
    results = broker.execute(intents)

    filled = [r for r in results if r.status == "FILLED"]
    rejected = [r for r in results if r.status == "REJECTED"]

    # Populate report
    for r in results:
        report.trades.append(TradeRecord(
            symbol=r.symbol,
            side=r.side,
            qty=r.qty,
            price=r.filled_price,
            status=r.status,
            order_id=r.order_id,
        ))
    report.n_filled = len(filled)
    report.n_rejected = len(rejected)

    print(f"\n  📋 Results:")
    print(f"    {'Symbol':<8} {'Side':>4} {'Qty':>6} {'Price':>10} {'Status':>10}")
    print(f"    {'─'*8} {'─'*4} {'─'*6} {'─'*10} {'─'*10}")
    for r in results:
        emoji = "✅" if r.status == "FILLED" else "❌"
        print(f"    {r.symbol:<8} {r.side:>4} {r.qty:>6.0f} "
              f"${r.filled_price:>9.2f} {emoji} {r.status}")

    print(f"\n    {len(filled)} filled, {len(rejected)} rejected")

    # ── Post-trade snapshot ───────────────────────────────────────
    post_account = broker.get_account()
    post_positions = broker.get_positions()
    report.post_equity = post_account.equity
    report.post_cash = post_account.cash

    print(f"\n  📊 Post-Trade:")
    print(f"    Equity: ${post_account.equity:,.2f} | "
          f"Cash: ${post_account.cash:,.2f} | "
          f"Positions: {len(post_positions)}")

    if post_positions:
        print(f"\n    {'Symbol':<8} {'Qty':>6} {'Cost':>8} {'MktVal':>12} {'P&L':>10}")
        print(f"    {'─'*8} {'─'*6} {'─'*8} {'─'*12} {'─'*10}")
        for p in post_positions:
            print(f"    {p.symbol:<8} {p.qty:>6.0f} ${p.avg_cost:>7.2f} "
                  f"${p.market_value:>11,.2f} ${p.unrealized_pl:>9.2f}")

    print(f"\n{SEP}")
    print(f"  ✅ COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}\n")

    return report


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*78}")
    print("  🚀 NEXUS QUANT OS v2.4 — Full Pipeline → Moomoo SIMULATE")
    print(f"  ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*78}")

    notifier = DiscordNotifier()   # reads DISCORD_WEBHOOK_URL from .env
    report = TradeReport(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))

    try:
        target_weights, metadata, report = run_pipeline_and_get_weights()

        if not target_weights:
            print("\n  ⚠️ No target weights — exiting")
            report.error = "No target weights produced"
            notifier.send_alert(
                title="Pipeline Warning",
                message="Pipeline produced no target weights.",
                severity="warning",
            )
            sys.exit(1)

        report = execute_on_moomoo(target_weights, metadata, report)

        # Send daily report to Discord
        if notifier.is_configured:
            notifier.send_daily_report(report)
            print("  📨 Discord notification sent")

    except Exception as exc:
        report.error = str(exc)
        report.traceback_str = traceback.format_exc()
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        notifier.send_error(report)
        print(f"\n  ❌ FAILED — {exc}")
        sys.exit(1)
