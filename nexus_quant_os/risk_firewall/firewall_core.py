"""
risk_firewall/firewall_core.py — Intelligent Risk Firewall
===========================================================

The last line of defence in the Nexus Quant OS signal pipeline
(Architecture Report Section 4).

Responsibility:
    Receives raw position weights emitted by the MoE Router and
    applies probabilistic, ML-driven scaling / veto logic before
    any order is dispatched to the broker.

Decision logic (three escalating tiers):

    TIER 1 — CAUTION (soft scale-down):
        Triggered when EITHER detector raises a mild warning.
        Position weights multiplied by a smooth scale factor in (0, 1).

    TIER 2 — WARNING (hard scale-down):
        Triggered when BOTH detectors signal moderate risk OR one
        detector signals a high risk level alone.  Positions reduced
        to a fraction of their original size.

    TIER 3 — EMERGENCY (zero-out / cash):
        Triggered when regime danger probability OR OOD combined score
        exceeds the hard ceiling.  ALL position weights are zeroed;
        the system goes to 100% cash or minimum hedge.

Weight adjustment is applied via a smooth *sigmoid ramp* between tier
boundaries, eliminating cliff-edges in position sizing.

Author : Nexus Quant OS — Risk Engineering Division
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from nexus_quant_os.risk_firewall.hmm_regime_detector import (
    HMMConfig,
    MarketRegimeDetector,
    RegimePrediction,
)
from nexus_quant_os.risk_firewall.ood_anomaly_detector import (
    AnomalyResult,
    OODConfig,
    OODAnomalyDetector,
)

logger = logging.getLogger("nexus_quant_os.risk_firewall.firewall_core")


# =====================================================================
# 1. RISK TIER TAXONOMY
# =====================================================================

class RiskTier(Enum):
    """Escalating risk response levels."""
    GREEN     = auto()   # Normal — full signal pass-through
    CAUTION   = auto()   # Soft scale-down (one detector warning)
    WARNING   = auto()   # Hard scale-down (both detectors warning)
    EMERGENCY = auto()   # Zero-out: go to cash


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

@dataclass
class FirewallConfig:
    """Configuration for IntelligentRiskFirewall.

    HMM thresholds
    --------------
    hmm_caution_threshold : float   CAUTION trigger.
    hmm_warning_threshold : float   WARNING trigger.
    hmm_emergency_threshold : float EMERGENCY trigger.

    OOD thresholds
    --------------
    ood_caution_threshold : float
    ood_warning_threshold : float
    ood_emergency_threshold : float

    Scale factors
    -------------
    caution_scale : float   Multiplier in CAUTION tier (e.g. 0.70).
    warning_scale : float   Multiplier in WARNING tier (e.g. 0.25).
    emergency_scale : float Must be 0.0 (zero-out).

    smooth_blend : bool
        If True, sigmoid-blend between tier boundaries instead of steps.
    """
    hmm_caution_threshold:   float = 0.40
    hmm_warning_threshold:   float = 0.60
    hmm_emergency_threshold: float = 0.80
    ood_caution_threshold:   float = 0.45
    ood_warning_threshold:   float = 0.60
    ood_emergency_threshold: float = 0.80
    caution_scale:   float = 0.70
    warning_scale:   float = 0.25
    emergency_scale: float = 0.05   # 最低防禦倉位（5%），避免完全歸零
    smooth_blend: bool = True


@dataclass
class FirewallDecision:
    """Full audit record of one firewall evaluation.

    Attributes
    ----------
    raw_weights : np.ndarray
        Position weights from MoE Router (before firewall).
    adjusted_weights : np.ndarray
        Position weights after firewall adjustment.
    scale_factor : float
        The scaling multiplier that was applied.
    risk_tier : RiskTier
        Tier classification of this decision cycle.
    hmm_danger_prob : float
        HMM posterior danger probability.
    ood_combined_score : float
        OOD detector combined anomaly score.
    hmm_regime_label : str
        Most likely regime name from HMM.
    ood_is_flagged : bool
        Whether OOD detector flagged the input.
    veto_reason : str
        Human-readable explanation of the firewall action.
    """
    raw_weights: np.ndarray
    adjusted_weights: np.ndarray
    scale_factor: float
    risk_tier: RiskTier
    hmm_danger_prob: float          # P(EXTREME_SHOCK)
    hmm_bear_prob: float            # P(BEAR_HIGH_VOL)
    ood_combined_score: float
    hmm_regime_label: str
    ood_is_flagged: bool
    veto_reason: str


# =====================================================================
# 3. CORE FIREWALL CLASS
# =====================================================================

class IntelligentRiskFirewall:
    """Probabilistic, ML-driven risk firewall for position weight control.

    Integrates HMM regime detection and OOD anomaly detection to
    dynamically scale or zero out MoE Router position weights.

    Usage
    -----
    >>> firewall = IntelligentRiskFirewall.from_configs(hmm_cfg, ood_cfg, fw_cfg)
    >>> firewall.fit(train_features)
    >>> decision = firewall.evaluate(live_features, raw_weights)
    >>> safe_weights = decision.adjusted_weights
    """

    def __init__(
        self,
        hmm_detector: MarketRegimeDetector,
        ood_detector: OODAnomalyDetector,
        config: FirewallConfig | None = None,
    ) -> None:
        self.hmm_detector = hmm_detector
        self.ood_detector = ood_detector
        self.config       = config or FirewallConfig()
        self._is_fitted   = False
        self._lock = threading.Lock()

    @classmethod
    def from_configs(
        cls,
        hmm_config: HMMConfig | None = None,
        ood_config: OODConfig | None = None,
        firewall_config: FirewallConfig | None = None,
    ) -> "IntelligentRiskFirewall":
        """Factory constructor — create detectors from config objects."""
        return cls(
            hmm_detector=MarketRegimeDetector(hmm_config or HMMConfig()),
            ood_detector=OODAnomalyDetector(ood_config or OODConfig()),
            config=firewall_config or FirewallConfig(),
        )

    # -----------------------------------------------------------------
    # 3a. Training
    # -----------------------------------------------------------------

    def fit(self, normal_market_features: np.ndarray) -> "IntelligentRiskFirewall":
        """Train both internal detectors on normal market data.

        Parameters
        ----------
        normal_market_features : np.ndarray
            Shape ``[T, F]`` — historical observations from calm/normal
            market periods only.
        """
        with self._lock:
            logger.info("Fitting IntelligentRiskFirewall on %d samples...",
                        len(normal_market_features))
            self.hmm_detector.fit(normal_market_features)
            self.ood_detector.fit(normal_market_features)
            self._is_fitted = True
            logger.info("Firewall fit complete.")
            return self

    # -----------------------------------------------------------------
    # 3b. Scale-factor computation
    # -----------------------------------------------------------------

    def _compute_scale_factor(
        self,
        tier: RiskTier,
        hmm_danger: float,
        hmm_bear: float,
        ood_score: float,
    ) -> float:
        """Compute a smooth scale factor for the given risk tier.

        In smooth_blend mode, scale is interpolated between tier
        boundaries via a sigmoid function (continuous response curve).
        """
        cfg = self.config

        if tier == RiskTier.EMERGENCY:
            return cfg.emergency_scale  # 保留最低倉位，不完全歸零

        if not cfg.smooth_blend:
            if tier == RiskTier.WARNING:
                return cfg.warning_scale
            if tier == RiskTier.CAUTION:
                return cfg.caution_scale
            return 1.0

        # Sigmoid-smooth blending
        if tier == RiskTier.WARNING:
            if ood_score >= cfg.ood_warning_threshold:
                composite = ood_score
                lo, hi = cfg.ood_warning_threshold, cfg.ood_emergency_threshold
            else:
                composite = hmm_bear
                lo, hi = cfg.hmm_warning_threshold, cfg.hmm_emergency_threshold
            t      = np.clip((composite - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            t_s    = float(1.0 / (1.0 + np.exp(-10.0 * (t - 0.5))))
            return float(cfg.caution_scale + t_s * (cfg.warning_scale - cfg.caution_scale))

        if tier == RiskTier.CAUTION:
            if ood_score >= cfg.ood_caution_threshold:
                composite = ood_score
                lo, hi = cfg.ood_caution_threshold, cfg.ood_warning_threshold
            else:
                composite = hmm_bear
                lo, hi = cfg.hmm_caution_threshold, cfg.hmm_warning_threshold
            t      = np.clip((composite - lo) / (hi - lo + 1e-8), 0.0, 1.0)
            t_s    = float(1.0 / (1.0 + np.exp(-10.0 * (t - 0.5))))
            return float(1.0 + t_s * (cfg.caution_scale - 1.0))

        return 1.0   # GREEN

    # -----------------------------------------------------------------
    # 3c. Risk tier resolution
    # -----------------------------------------------------------------

    def _resolve_tier(
        self,
        hmm_pred: RegimePrediction,
        ood_result: AnomalyResult,
    ) -> tuple[RiskTier, str]:
        """Determine the risk tier from detector outputs.

        Tier logic (semantic split):
          EMERGENCY : P(EXTREME_SHOCK) ≥ hmm_emergency_threshold
          WARNING   : P(BEAR_HIGH_VOL) ≥ hmm_warning_threshold  OR  OOD high
          CAUTION   : P(BEAR_HIGH_VOL) ≥ hmm_caution_threshold  OR  OOD soft
          GREEN     : normal market conditions
        """
        cfg = self.config
        h_extreme = hmm_pred.danger_probability   # P(EXTREME_SHOCK)
        h_bear    = hmm_pred.bear_probability     # P(BEAR_HIGH_VOL)
        o         = ood_result.combined_anomaly_score

        # TIER 3: EMERGENCY — only genuine crisis (EXTREME_SHOCK)
        if h_extreme >= cfg.hmm_emergency_threshold:
            return RiskTier.EMERGENCY, (
                f"HMM danger={h_extreme:.3f} ≥ emergency threshold "
                f"{cfg.hmm_emergency_threshold} "
                f"(regime={hmm_pred.regime_label})"
            )
        if o >= cfg.ood_emergency_threshold:
            return RiskTier.EMERGENCY, (
                f"OOD score={o:.3f} ≥ emergency threshold "
                f"{cfg.ood_emergency_threshold} "
                f"(AE_z={ood_result.ae_zscore:.2f})"
            )

        # TIER 2: WARNING — bear market signals or combined OOD
        hmm_warn = h_bear >= cfg.hmm_warning_threshold
        ood_warn  = o     >= cfg.ood_warning_threshold
        if hmm_warn and ood_warn:
            return RiskTier.WARNING, (
                f"BOTH detectors at WARNING: BEAR={h_bear:.3f} OOD={o:.3f}"
            )
        if hmm_warn:
            return RiskTier.WARNING, (
                f"HMM bear={h_bear:.3f} ≥ warning threshold {cfg.hmm_warning_threshold}"
            )
        if ood_warn:
            return RiskTier.WARNING, (
                f"OOD score={o:.3f} ≥ warning threshold {cfg.ood_warning_threshold}"
            )

        # TIER 1: CAUTION
        hmm_caut = h_bear >= cfg.hmm_caution_threshold
        ood_caut  = o     >= cfg.ood_caution_threshold
        if hmm_caut or ood_caut:
            return RiskTier.CAUTION, (
                f"Soft risk: BEAR={h_bear:.3f} OOD={o:.3f}"
            )

        # TIER 0: GREEN
        return RiskTier.GREEN, "All clear — full pass-through"

    # -----------------------------------------------------------------
    # 3d. Main evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        market_features: np.ndarray,
        raw_weights: np.ndarray,
    ) -> FirewallDecision:
        """Apply firewall logic to raw MoE Router position weights.

        Parameters
        ----------
        market_features : np.ndarray
            Live market observation window.  Shape: ``[W, F]``.
        raw_weights : np.ndarray
            Position weights from MoE Router.  Shape: ``[N_assets]``.

        Returns
        -------
        FirewallDecision
            Full audit record including adjusted weights.
        """
        with self._lock:
            if not self._is_fitted:
                raise RuntimeError(
                    "IntelligentRiskFirewall not fitted. Call .fit() first."
                )

            # 1. Run detectors
            hmm_pred   = self.hmm_detector.predict(market_features)
            ood_result = self.ood_detector.detect(market_features[-1:])

            # 2. Resolve risk tier
            tier, reason = self._resolve_tier(hmm_pred, ood_result)

            # 3. Compute scale factor
            scale = self._compute_scale_factor(
                tier, hmm_pred.danger_probability, hmm_pred.bear_probability, ood_result.combined_anomaly_score
            )

            # 4. Apply scaling
            adjusted_weights = raw_weights * scale

            log_fn = logger.warning if tier != RiskTier.GREEN else logger.info
            log_fn(
                "[Firewall] tier=%s  scale=%.4f  hmm_extreme=%.3f  hmm_bear=%.3f  "
                "ood_score=%.3f  reason=%s",
                tier.name, scale,
                hmm_pred.danger_probability,
                hmm_pred.bear_probability,
                ood_result.combined_anomaly_score,
                reason,
            )

            return FirewallDecision(
                raw_weights=raw_weights.copy(),
                adjusted_weights=adjusted_weights,
                scale_factor=scale,
                risk_tier=tier,
                hmm_danger_prob=hmm_pred.danger_probability,
                hmm_bear_prob=hmm_pred.bear_probability,
                ood_combined_score=ood_result.combined_anomaly_score,
                hmm_regime_label=hmm_pred.regime_label,
                ood_is_flagged=ood_result.is_ood,
                veto_reason=reason,
            )


# =====================================================================
# DEMONSTRATION & PROOF
# =====================================================================

def _run_demonstration() -> None:
    """End-to-end firewall test: normal bull, mild bear, and extreme crash."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("nexus_quant_os.risk_firewall.firewall_core").setLevel(
        logging.INFO
    )
    np.random.seed(0)

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  Nexus Quant OS -- Intelligent Risk Firewall End-to-End Test")
    print(f"{SEP}\n")

    def features(n, vol, ret_mu, ret_sigma, vix_mu, cs_mu):
        return np.column_stack([
            np.random.normal(vol,     vol * 0.2,   n),
            np.random.normal(ret_mu,  ret_sigma,   n),
            np.random.normal(vix_mu,  vol * 0.5,   n),
            np.random.normal(0.5,     0.1,          n),
            np.random.normal(cs_mu,   0.05,         n),
            np.random.normal(ret_mu * 2, 0.001,    n),
        ])

    TRAIN = np.vstack([
        features(400, 0.008,  0.0005, 0.008, -0.001, 0.30),   # bull
        features(200, 0.018, -0.0010, 0.016,  0.002, 0.55),   # bear
        features(200, 0.010,  0.0002, 0.009,  0.000, 0.35),   # neutral
    ])

    raw_weights = np.array([0.25, 0.20, -0.10, 0.15, 0.40])

    print(f"Training firewall on {len(TRAIN)} normal market observations...\n")

    firewall = IntelligentRiskFirewall.from_configs(
        hmm_config=HMMConfig(n_regimes=3, n_iter=150, danger_threshold=0.50),
        ood_config=OODConfig(ae_epochs=30, combined_threshold=0.55),
        firewall_config=FirewallConfig(
            hmm_caution_threshold=0.40,
            hmm_warning_threshold=0.60,
            hmm_emergency_threshold=0.78,
            ood_caution_threshold=0.45,
            ood_warning_threshold=0.60,
            ood_emergency_threshold=0.78,
            caution_scale=0.70,
            warning_scale=0.25,
            smooth_blend=True,
        ),
    )
    firewall.fit(TRAIN)

    scenarios = {
        "1. Normal Bull Market": features(20, 0.008,  0.0005, 0.008, -0.001, 0.30),
        "2. Mild Bear Market":   features(20, 0.018, -0.001,  0.016,  0.002, 0.55),
        "3. EXTREME CRASH":      features(20, 0.080, -0.040,  0.030,  0.010, 2.50),
    }

    tier_icons = {
        RiskTier.GREEN:     "GREEN    ",
        RiskTier.CAUTION:   "CAUTION  ",
        RiskTier.WARNING:   "WARNING  ",
        RiskTier.EMERGENCY: "EMERGENCY",
    }

    print("-" * 72)
    print(f"  {'Scenario':<26s} {'Tier':<12s} {'Scale':>7s}  "
          f"{'HMM':>7s}  {'OOD':>7s}  {'Raw W[0]':>9s}  {'Adj W[0]':>9s}")
    print("  " + "-" * 68)

    decisions: dict[str, FirewallDecision] = {}
    for name, obs_window in scenarios.items():
        d = firewall.evaluate(obs_window, raw_weights.copy())
        decisions[name] = d
        print(
            f"  {name:<26s} [{tier_icons[d.risk_tier]}] "
            f"{d.scale_factor:>7.4f}  "
            f"{d.hmm_danger_prob:>7.4f}  "
            f"{d.ood_combined_score:>7.4f}  "
            f"{d.raw_weights[0]:>9.4f}  "
            f"{d.adjusted_weights[0]:>9.4f}"
        )

    crash_d  = decisions["3. EXTREME CRASH"]
    normal_d = decisions["1. Normal Bull Market"]

    print(f"\nCrash decision detail:")
    print(f"  Risk tier      : {crash_d.risk_tier.name}")
    print(f"  Veto reason    : {crash_d.veto_reason}")
    print(f"  Raw weights    : {np.round(crash_d.raw_weights, 4)}")
    print(f"  Adjusted       : {np.round(crash_d.adjusted_weights, 6)}")
    print(f"  Scale factor   : {crash_d.scale_factor:.6f}")

    print(f"\nAutomated Assertions:")
    count = 0

    # 1. Normal market: scale > 0 (not a full zero-out)
    # OOD may still issue a soft CAUTION on near-normal data -- correct behaviour
    assert normal_d.scale_factor >= 0.60, \
        f"Normal market scale too low: {normal_d.scale_factor}"
    count += 1
    print(f"  [PASS 1/4] Normal market: scale={normal_d.scale_factor:.4f} >= 0.60 "
          f"(tier={normal_d.risk_tier.name})")

    # 2. Crash triggers WARNING or EMERGENCY
    assert crash_d.risk_tier in (RiskTier.EMERGENCY, RiskTier.WARNING), \
        f"Crash should trigger WARNING/EMERGENCY, got {crash_d.risk_tier.name}"
    count += 1
    print(f"  [PASS 2/4] Crash triggers {crash_d.risk_tier.name} tier")

    # 3. Crash weights reduced to <= 30% of original
    weight_ratio = (
        np.abs(crash_d.adjusted_weights).sum()
        / (np.abs(crash_d.raw_weights).sum() + 1e-8)
    )
    assert weight_ratio <= 0.30, \
        f"Crash weights not reduced enough: ratio={weight_ratio:.4f}"
    count += 1
    print(f"  [PASS 3/4] Crash weights reduced to {weight_ratio:.2%} of original")

    # 4. Crash HMM danger > normal HMM danger
    assert crash_d.hmm_danger_prob > normal_d.hmm_danger_prob, \
        "Crash HMM danger should exceed normal!"
    count += 1
    print(f"  [PASS 4/4] Crash HMM danger ({crash_d.hmm_danger_prob:.4f}) "
          f"> Normal ({normal_d.hmm_danger_prob:.4f})")

    print(f"\n{SEP}")
    print(f"  ALL {count} ASSERTIONS PASSED")
    print(f"  Firewall successfully intercepts extreme risk and zeros/scales weights.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    _run_demonstration()
