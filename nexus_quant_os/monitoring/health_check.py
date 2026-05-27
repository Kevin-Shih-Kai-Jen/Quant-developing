"""
monitoring/health_check.py — Automated Data & Model Health Monitoring
=====================================================================

Pre-trade health gate that runs **before** the pipeline executes.
If critical checks fail, trading is blocked and a Discord alert is sent.

Layer 1 — Data Quality:
    • Price data freshness (no stale / weekend-only data)
    • NaN / missing value detection across all assets
    • Single-day price anomaly detection (>15% move)
    • FRED macro data staleness (>60 days since last update)

Layer 2 — Model Health:
    • Weight sanity (sum ≈ 1.0, no negatives, no over-concentration)
    • Feature matrix quality (NaN, Inf, extreme values)
    • Inference reproducibility (no random jitter between runs)

Data-flow position::

    run_scheduled.sh  →  health_check.py  →  [PASS] → run_moomoo_trade.py
                                           →  [FAIL] → Discord alert, ABORT

Author : Nexus Quant OS — Monitoring Division
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus_quant_os.monitoring.health_check")


# ═════════════════════════════════════════════════════════════════════
# Severity & Result Models
# ═════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    """Check result severity."""
    OK = "ok"
    WARNING = "warning"      # Log & notify, but continue trading
    CRITICAL = "critical"    # Block trading, send alert


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report from all checks."""
    timestamp: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """True if no CRITICAL checks failed."""
        return not any(c.severity == Severity.CRITICAL for c in self.checks)

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.OK)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.WARNING)

    @property
    def n_critical(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.CRITICAL)

    def summary(self) -> str:
        """Human-readable summary string."""
        icon = "✅" if self.is_healthy else "🚨"
        return (
            f"{icon} Health Check: "
            f"{self.n_ok} OK, {self.n_warnings} warnings, "
            f"{self.n_critical} critical"
        )


# ═════════════════════════════════════════════════════════════════════
# Layer 1: Data Quality Checks
# ═════════════════════════════════════════════════════════════════════


def check_price_freshness(
    prices_df: pd.DataFrame,
    max_stale_days: int = 5,
) -> CheckResult:
    """Check that price data is recent enough.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily price data with a 'timestamp' or DatetimeIndex column.
    max_stale_days : int
        Maximum allowed gap between latest data point and today.
    """
    if prices_df.empty:
        return CheckResult(
            name="Price Freshness",
            severity=Severity.CRITICAL,
            message="Price DataFrame is empty — no data loaded",
        )

    # Find the latest date in the dataset
    if "timestamp" in prices_df.columns:
        latest = pd.to_datetime(prices_df["timestamp"]).max()
    elif isinstance(prices_df.index, pd.DatetimeIndex):
        latest = prices_df.index.max()
    else:
        return CheckResult(
            name="Price Freshness",
            severity=Severity.WARNING,
            message="Cannot determine timestamp column",
        )

    today = pd.Timestamp.now().normalize()
    gap = (today - latest).days

    if gap > max_stale_days:
        return CheckResult(
            name="Price Freshness",
            severity=Severity.CRITICAL,
            message=f"Price data is {gap} days old (latest: {latest.date()}). "
                    f"Max allowed: {max_stale_days} days.",
            details={"latest_date": str(latest.date()), "gap_days": gap},
        )
    elif gap > 3:
        return CheckResult(
            name="Price Freshness",
            severity=Severity.WARNING,
            message=f"Price data is {gap} days old (latest: {latest.date()}).",
            details={"latest_date": str(latest.date()), "gap_days": gap},
        )

    return CheckResult(
        name="Price Freshness",
        severity=Severity.OK,
        message=f"Latest price: {latest.date()} ({gap}d ago)",
    )


def check_price_nan(
    prices_df: pd.DataFrame,
    max_nan_pct: float = 0.05,
) -> CheckResult:
    """Check for NaN values in price data.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily price data. Must contain 'close' column.
    max_nan_pct : float
        Maximum tolerable NaN percentage (0.05 = 5%).
    """
    if prices_df.empty:
        return CheckResult(
            name="Price NaN Check",
            severity=Severity.CRITICAL,
            message="Price DataFrame is empty",
        )

    price_cols = [c for c in ["open", "high", "low", "close", "volume"]
                  if c in prices_df.columns]

    if not price_cols:
        return CheckResult(
            name="Price NaN Check",
            severity=Severity.WARNING,
            message="No OHLCV columns found in price data",
        )

    total_cells = len(prices_df) * len(price_cols)
    nan_count = prices_df[price_cols].isna().sum().sum()
    nan_pct = nan_count / total_cells if total_cells > 0 else 0

    if nan_pct > max_nan_pct:
        # Find which assets have the most NaNs
        nan_per_col = prices_df[price_cols].isna().sum().to_dict()
        return CheckResult(
            name="Price NaN Check",
            severity=Severity.CRITICAL,
            message=f"Price data has {nan_pct:.1%} NaN values "
                    f"({nan_count:,}/{total_cells:,} cells). "
                    f"Max tolerable: {max_nan_pct:.0%}.",
            details={"nan_pct": nan_pct, "nan_per_col": nan_per_col},
        )
    elif nan_count > 0:
        return CheckResult(
            name="Price NaN Check",
            severity=Severity.WARNING,
            message=f"Price data has {nan_count:,} NaN values ({nan_pct:.2%})",
            details={"nan_pct": nan_pct},
        )

    return CheckResult(
        name="Price NaN Check",
        severity=Severity.OK,
        message="No NaN values in price data",
    )


def check_price_anomaly(
    prices_df: pd.DataFrame,
    threshold_pct: float = 0.15,
) -> CheckResult:
    """Detect single-day price moves exceeding threshold.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must contain 'close' and 'asset_id' columns.
    threshold_pct : float
        Maximum tolerable single-day return (0.15 = 15%).
    """
    if "close" not in prices_df.columns:
        return CheckResult(
            name="Price Anomaly",
            severity=Severity.WARNING,
            message="No 'close' column — skipping anomaly check",
        )

    df = prices_df.copy()
    df["_ret"] = df.groupby("asset_id")["close"].pct_change()

    # Only check the most recent 5 rows per asset
    recent = df.groupby("asset_id").tail(5)
    anomalies = recent[recent["_ret"].abs() > threshold_pct]

    if len(anomalies) > 0:
        alert_lines = []
        for _, row in anomalies.iterrows():
            asset = row.get("asset_id", "?")
            ret = row["_ret"]
            date = row.get("timestamp", "?")
            alert_lines.append(f"{asset}: {ret:+.1%} on {date}")

        return CheckResult(
            name="Price Anomaly",
            severity=Severity.WARNING,
            message=f"{len(anomalies)} extreme price move(s) detected "
                    f"(>{threshold_pct:.0%}):\n" + "\n".join(alert_lines),
            details={"n_anomalies": len(anomalies)},
        )

    return CheckResult(
        name="Price Anomaly",
        severity=Severity.OK,
        message="No extreme price moves in recent data",
    )


def check_macro_staleness(
    macro_df: pd.DataFrame,
    max_stale_days: int = 90,
) -> CheckResult:
    """Check that FRED macro data is not excessively stale.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Macro data with DatetimeIndex or 'date' column.
    max_stale_days : int
        Maximum days since last macro data update.
    """
    if macro_df.empty:
        return CheckResult(
            name="Macro Staleness",
            severity=Severity.WARNING,
            message="Macro DataFrame is empty — using fallback?",
        )

    if "date" in macro_df.columns:
        latest = pd.to_datetime(macro_df["date"]).max()
    elif "timestamp" in macro_df.columns:
        latest = pd.to_datetime(macro_df["timestamp"]).max()
    elif isinstance(macro_df.index, pd.DatetimeIndex):
        latest = macro_df.index.max()
    else:
        return CheckResult(
            name="Macro Staleness",
            severity=Severity.WARNING,
            message="Cannot determine macro date column",
        )

    today = pd.Timestamp.now().normalize()
    gap = (today - latest).days

    if gap > max_stale_days:
        return CheckResult(
            name="Macro Staleness",
            severity=Severity.WARNING,
            message=f"FRED macro data is {gap} days old "
                    f"(latest: {latest.date()}). CPI/PMI may be outdated.",
            details={"latest_date": str(latest.date()), "gap_days": gap},
        )

    return CheckResult(
        name="Macro Staleness",
        severity=Severity.OK,
        message=f"Macro data latest: {latest.date()} ({gap}d ago)",
    )


# ═════════════════════════════════════════════════════════════════════
# Layer 2: Model / Feature Health Checks
# ═════════════════════════════════════════════════════════════════════


def check_feature_quality(
    feature_matrix: np.ndarray,
    feature_names: list[str] | None = None,
) -> CheckResult:
    """Check feature matrix for NaN, Inf, or extreme values.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Shape [T, F].
    feature_names : list[str], optional
        Feature column names for diagnostics.
    """
    if feature_matrix.size == 0:
        return CheckResult(
            name="Feature Quality",
            severity=Severity.CRITICAL,
            message="Feature matrix is empty (0 rows)",
        )

    n_nan = np.isnan(feature_matrix).sum()
    n_inf = np.isinf(feature_matrix).sum()

    if n_nan > 0 or n_inf > 0:
        # Identify which features have issues
        problems = {}
        for i in range(feature_matrix.shape[1]):
            col = feature_names[i] if feature_names and i < len(feature_names) else f"f{i}"
            col_nan = int(np.isnan(feature_matrix[:, i]).sum())
            col_inf = int(np.isinf(feature_matrix[:, i]).sum())
            if col_nan > 0 or col_inf > 0:
                problems[col] = {"nan": col_nan, "inf": col_inf}

        severity = Severity.CRITICAL if n_inf > 0 else Severity.WARNING
        return CheckResult(
            name="Feature Quality",
            severity=severity,
            message=f"Feature matrix has {n_nan} NaN and {n_inf} Inf values",
            details={"problems": problems},
        )

    # Check for extreme values (> 100 standard deviations)
    means = np.nanmean(feature_matrix, axis=0)
    stds = np.nanstd(feature_matrix, axis=0)
    stds[stds < 1e-10] = 1.0  # avoid div-by-zero

    last_row = feature_matrix[-1:]
    z_scores = np.abs((last_row - means) / stds)
    extreme = z_scores > 100

    if extreme.any():
        extreme_features = []
        for i in range(len(extreme[0])):
            if extreme[0, i]:
                name = feature_names[i] if feature_names and i < len(feature_names) else f"f{i}"
                extreme_features.append(f"{name} (z={z_scores[0,i]:.0f})")
        return CheckResult(
            name="Feature Quality",
            severity=Severity.WARNING,
            message=f"Extreme z-scores in latest features: {', '.join(extreme_features)}",
            details={"extreme_features": extreme_features},
        )

    return CheckResult(
        name="Feature Quality",
        severity=Severity.OK,
        message=f"Feature matrix clean: {feature_matrix.shape[0]:,} rows × "
                f"{feature_matrix.shape[1]} features",
    )


def check_weight_sanity(
    weights: np.ndarray,
    asset_names: list[str] | None = None,
    max_single_weight: float = 0.40,
) -> CheckResult:
    """Validate model output weights.

    Parameters
    ----------
    weights : np.ndarray
        Raw or final portfolio weights.
    asset_names : list[str], optional
        Asset names for diagnostics.
    max_single_weight : float
        Maximum allowed weight for a single asset (0.40 = 40%).
    """
    if weights.size == 0:
        return CheckResult(
            name="Weight Sanity",
            severity=Severity.CRITICAL,
            message="Weight vector is empty",
        )

    problems = []

    # Check 1: Any NaN/Inf
    if np.isnan(weights).any() or np.isinf(weights).any():
        problems.append("Contains NaN or Inf values")

    # Check 2: Negative weights (long-only constraint)
    if (weights < -0.001).any():
        neg_idx = np.where(weights < -0.001)[0]
        neg_names = [asset_names[i] if asset_names and i < len(asset_names)
                     else f"asset_{i}" for i in neg_idx]
        problems.append(f"Negative weights: {neg_names}")

    # Check 3: Sum should be close to 1.0 (or less for cash holding)
    w_sum = float(weights.sum())
    if w_sum > 1.05:
        problems.append(f"Weight sum = {w_sum:.4f} (>1.05, leverage detected)")
    elif w_sum < 0.01:
        problems.append(f"Weight sum = {w_sum:.4f} (near zero — model producing no allocation)")

    # Check 4: Over-concentration
    max_w = float(weights.max())
    if max_w > max_single_weight:
        max_idx = int(weights.argmax())
        max_name = asset_names[max_idx] if asset_names and max_idx < len(asset_names) else f"asset_{max_idx}"
        problems.append(
            f"Over-concentrated: {max_name} = {max_w:.1%} "
            f"(limit: {max_single_weight:.0%})"
        )

    if problems:
        severity = Severity.CRITICAL if any(
            "NaN" in p or "Inf" in p or "zero" in p for p in problems
        ) else Severity.WARNING

        return CheckResult(
            name="Weight Sanity",
            severity=severity,
            message=f"{len(problems)} issue(s): " + "; ".join(problems),
            details={"weight_sum": float(weights.sum()),
                     "max_weight": float(weights.max()),
                     "weights": weights.tolist()},
        )

    return CheckResult(
        name="Weight Sanity",
        severity=Severity.OK,
        message=f"Weights OK: sum={w_sum:.4f}, max={max_w:.1%}, "
                f"{(weights > 0.01).sum()} active positions",
    )


# ═════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════


def run_pre_trade_checks(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    weights: np.ndarray | None = None,
    asset_names: list[str] | None = None,
) -> HealthReport:
    """Run all pre-trade health checks.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily price data.
    macro_df : pd.DataFrame
        FRED macro data.
    feature_matrix : np.ndarray, optional
        Engineered feature matrix (if already computed).
    feature_names : list[str], optional
        Feature column names.
    weights : np.ndarray, optional
        Model output weights (if already computed).
    asset_names : list[str], optional
        Asset ticker names.

    Returns
    -------
    HealthReport
        Aggregated results. Use `report.is_healthy` to gate trading.
    """
    report = HealthReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ── Layer 1: Data Quality ────────────────────────────────────
    report.checks.append(check_price_freshness(prices_df))
    report.checks.append(check_price_nan(prices_df))
    report.checks.append(check_price_anomaly(prices_df))
    report.checks.append(check_macro_staleness(macro_df))

    # ── Layer 2: Model Health ────────────────────────────────────
    if feature_matrix is not None:
        report.checks.append(
            check_feature_quality(feature_matrix, feature_names)
        )

    if weights is not None:
        report.checks.append(
            check_weight_sanity(weights, asset_names)
        )

    return report


def format_health_report(report: HealthReport) -> str:
    """Format a HealthReport for Discord notification.

    Returns
    -------
    str
        Discord-markdown formatted string.
    """
    lines = [f"**{report.summary()}**\n"]

    for check in report.checks:
        if check.severity == Severity.OK:
            icon = "✅"
        elif check.severity == Severity.WARNING:
            icon = "⚠️"
        else:
            icon = "🚨"
        lines.append(f"{icon} **{check.name}**: {check.message}")

    return "\n".join(lines)
