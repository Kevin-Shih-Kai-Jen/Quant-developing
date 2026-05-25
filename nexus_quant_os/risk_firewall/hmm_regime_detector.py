"""
risk_firewall/hmm_regime_detector.py -- Gaussian HMM Market Regime Detector
============================================================================

Implements the probabilistic market-regime identification layer described
in Architecture Report Section 4.1.

Core concept:
    Financial markets are governed by unobservable latent *regimes*
    (e.g. "low-volatility bull", "high-volatility bear", "extreme
    inflation shock").  A Gaussian Hidden Markov Model (HMM) is trained
    on macro-economic & volatility observations to infer the posterior
    probability of each regime at every point in time.

    The firewall consumes these probabilities to dynamically scale
    position weights -- a high probability of being in a *dangerous*
    regime triggers automatic de-risking.

Mathematical formulation:
    HMM five-tuple  (H, O, T, Psi, Pi) -- see Architecture Report 4.1.
      H  : set of N_STATES hidden market states
      O  : observable feature vector at time t
      T  : N x N transition probability matrix  (learned via EM)
      Psi: Gaussian emission distributions       (learned via EM)
      Pi : initial state probability vector

    Inference uses the Forward Algorithm to compute:
        P(regime = DANGEROUS | observations 1..t)

Author : Nexus Quant OS -- Risk Engineering Division
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger("nexus_quant_os.risk_firewall.hmm_regime_detector")


# =====================================================================
# 1. REGIME TAXONOMY
# =====================================================================

class MarketRegime(IntEnum):
    """Named market regimes.  Indices matched to HMM states after
    post-hoc labelling based on emission mean volatility."""
    BULL_LOW_VOL  = 0   # Low volatility, positive drift -- "risk-on"
    BEAR_HIGH_VOL = 1   # High volatility, negative drift -- "risk-off"
    EXTREME_SHOCK = 2   # Extreme volatility, fat tails -- "crisis"


# Regimes that trigger EMERGENCY in the firewall (true crisis only)
DANGEROUS_REGIMES: frozenset[int] = frozenset({
    MarketRegime.EXTREME_SHOCK,   # Extreme vol / fat-tail crisis
})

# Regimes that trigger WARNING/CAUTION (normal bear market)
BEAR_REGIMES: frozenset[int] = frozenset({
    MarketRegime.BEAR_HIGH_VOL,   # High vol, negative drift — risk-off
})


# =====================================================================
# 2. CONFIGURATION
# =====================================================================

@dataclass
class HMMConfig:
    """Configuration for MarketRegimeDetector.

    Attributes
    ----------
    n_regimes : int
        Number of hidden market states (N).  Typically 2-4.
    n_iter : int
        Maximum EM iterations for Baum-Welch training.
    covariance_type : str
        Covariance structure: "full", "diag", "tied", "spherical".
    tol : float
        EM convergence tolerance.
    random_state : int
        Seed for reproducible initialisation.
    danger_threshold : float
        Minimum cumulative posterior probability of any *dangerous*
        regime that triggers a risk escalation signal (0-1).
    """
    n_regimes: int = 3
    n_iter: int = 500
    covariance_type: str = "full"
    tol: float = 1e-4
    random_state: int = 42
    danger_threshold: float = 0.55


@dataclass
class RegimePrediction:
    """Output of a single predict call.

    Attributes
    ----------
    most_likely_regime : int
        Viterbi-decoded most probable regime index.
    regime_probabilities : np.ndarray
        Posterior probability vector over all regimes.  Shape: [N].
    danger_probability : float
        Summed probability of being in any dangerous regime in [0, 1].
    is_dangerous : bool
        True if danger_probability >= config.danger_threshold.
    regime_label : str
        Human-readable name of the most likely regime.
    """
    most_likely_regime: int
    regime_probabilities: np.ndarray
    danger_probability: float    # P(EXTREME_SHOCK) — triggers EMERGENCY
    bear_probability: float       # P(BEAR_HIGH_VOL) — triggers WARNING/CAUTION
    is_dangerous: bool
    regime_label: str


# =====================================================================
# 3. CORE DETECTOR CLASS
# =====================================================================

class MarketRegimeDetector:
    """Gaussian HMM wrapper for latent market regime detection.

    Usage
    -----
    >>> detector = MarketRegimeDetector(HMMConfig(n_regimes=3))
    >>> detector.fit(training_observations)   # shape: [T, F]
    >>> pred = detector.predict(live_window)  # shape: [W, F]
    >>> print(pred.danger_probability)
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.config = config or HMMConfig()
        self._hmm: GaussianHMM | None = None
        self._is_fitted: bool = False
        # Maps HMM state index -> MarketRegime after post-hoc labelling
        self._state_to_regime: dict[int, int] = {}
        self._regime_names: dict[int, str] = {}

    # -----------------------------------------------------------------
    # 3a. Training
    # -----------------------------------------------------------------

    def fit(
        self,
        observations: np.ndarray,
        lengths: Sequence[int] | None = None,
    ) -> "MarketRegimeDetector":
        """Train the Gaussian HMM on historical market observations.

        Parameters
        ----------
        observations : np.ndarray
            2-D feature matrix of shape ``[T, F]``.
            Recommended features: daily returns, realised volatility,
            VIX change, yield-curve slope, credit spread.
        lengths : sequence of int, optional
            Lengths of individual sub-sequences when *observations*
            concatenates multiple independent time series.
            IMPORTANT: for a 3-state HMM, training data MUST include
            examples of all 3 regimes (use lengths to concatenate
            separate sequences without temporal leakage).

        Returns
        -------
        self
        """
        if observations.ndim != 2:
            raise ValueError(
                f"observations must be 2-D [T, F], got shape {observations.shape}"
            )

        n_samples, n_features = observations.shape
        logger.info(
            "Fitting GaussianHMM | regimes=%d  samples=%d  features=%d  "
            "cov=%s  iter=%d",
            self.config.n_regimes, n_samples, n_features,
            self.config.covariance_type, self.config.n_iter,
        )

        self._hmm = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._hmm.fit(observations, lengths=lengths)

        self._is_fitted = True
        self._label_regimes()

        logger.info(
            "HMM training complete | converged=%s | log_prob=%.4f",
            self._hmm.monitor_.converged,
            self._hmm.monitor_.history[-1] if self._hmm.monitor_.history else float("nan"),
        )
        self._log_transition_matrix()
        return self

    def _label_regimes(self) -> None:
        """Post-hoc label states by mean of realised_vol (feature index 1).

        Volatility is the correct regime discriminator:
            Lowest  vol -> BULL_LOW_VOL
            Highest vol -> EXTREME_SHOCK
            Middle  vol -> BEAR_HIGH_VOL

        NOTE: We use feature index 1 (realised_vol), NOT index 0 (daily_return).
        daily_return has both positive and negative means, so sorting by it
        maps the highest-return state to EXTREME_SHOCK — semantically wrong.
        """
        assert self._hmm is not None
        # Robustly pick the volatility feature (index 1 if available, else 0)
        vol_idx       = min(1, self._hmm.means_.shape[1] - 1)
        means         = self._hmm.means_[:, vol_idx]   # [N] — realised vol proxy
        sorted_states = np.argsort(means)               # ascending: low vol → high vol

        n = self.config.n_regimes
        if n == 2:
            self._state_to_regime = {
                int(sorted_states[0]): MarketRegime.BULL_LOW_VOL,
                int(sorted_states[1]): MarketRegime.BEAR_HIGH_VOL,
            }
        elif n == 3:
            self._state_to_regime = {
                int(sorted_states[0]): MarketRegime.BULL_LOW_VOL,
                int(sorted_states[1]): MarketRegime.BEAR_HIGH_VOL,
                int(sorted_states[2]): MarketRegime.EXTREME_SHOCK,
            }
        else:
            self._state_to_regime = {}
            for i, state in enumerate(sorted_states):
                if i == 0:
                    self._state_to_regime[int(state)] = MarketRegime.BULL_LOW_VOL
                elif i == n - 1:
                    self._state_to_regime[int(state)] = MarketRegime.EXTREME_SHOCK
                else:
                    self._state_to_regime[int(state)] = MarketRegime.BEAR_HIGH_VOL

        self._regime_names = {
            v: MarketRegime(v).name for v in self._state_to_regime.values()
        }
        logger.info("State labelling: %s", self._state_to_regime)

    def _log_transition_matrix(self) -> None:
        assert self._hmm is not None
        T     = self._hmm.transmat_
        lines = ["Transition matrix (row=from, col=to):"]
        header = "       " + "  ".join(f"S{j:>2d}" for j in range(T.shape[1]))
        lines.append(header)
        for i, row in enumerate(T):
            regime  = self._state_to_regime.get(i, i)
            row_str = "  ".join(f"{p:.3f}" for p in row)
            lines.append(f"  S{i:>2d} [{MarketRegime(regime).name[:8]:>8s}]  {row_str}")
        logger.info("\n".join(lines))

    # -----------------------------------------------------------------
    # 3b. Inference
    # -----------------------------------------------------------------

    def predict(self, observation_window: np.ndarray) -> RegimePrediction:
        """Compute posterior regime probabilities for a recent window.

        Parameters
        ----------
        observation_window : np.ndarray
            Shape ``[W, F]`` (W >= 1).  Uses the LAST timestep's
            posterior as the "current" regime probability.
        """
        self._check_fitted()
        if observation_window.ndim == 1:
            observation_window = observation_window.reshape(1, -1)

        # posteriors shape: [W, N]
        posteriors        = self._hmm.predict_proba(observation_window)
        current_posterior = posteriors[-1]   # [N]

        # Map HMM state indices to named regime indices
        regime_probs = np.zeros(len(MarketRegime))
        for state_idx, prob in enumerate(current_posterior):
            regime_idx = self._state_to_regime.get(state_idx, state_idx)
            if regime_idx < len(regime_probs):
                regime_probs[regime_idx] += prob

        # Viterbi most likely path
        viterbi_states    = self._hmm.predict(observation_window)
        most_likely_state = int(viterbi_states[-1])
        most_likely_regime = self._state_to_regime.get(most_likely_state, most_likely_state)

        danger_probability = float(
            sum(regime_probs[r] for r in DANGEROUS_REGIMES if r < len(regime_probs))
        )
        bear_probability = float(
            sum(regime_probs[r] for r in BEAR_REGIMES if r < len(regime_probs))
        )
        is_dangerous = danger_probability >= self.config.danger_threshold

        return RegimePrediction(
            most_likely_regime=most_likely_regime,
            regime_probabilities=regime_probs,
            danger_probability=danger_probability,
            bear_probability=bear_probability,
            is_dangerous=is_dangerous,
            regime_label=MarketRegime(most_likely_regime).name,
        )

    def predict_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Return danger probabilities for every timestep.

        Returns
        -------
        np.ndarray  shape: [T]
        """
        self._check_fitted()
        posteriors   = self._hmm.predict_proba(observations)   # [T, N]
        danger_probs = np.zeros(len(posteriors))
        for t in range(len(posteriors)):
            for state_idx, p in enumerate(posteriors[t]):
                regime_idx = self._state_to_regime.get(state_idx, state_idx)
                if regime_idx in DANGEROUS_REGIMES:
                    danger_probs[t] += p
        return danger_probs

    # -----------------------------------------------------------------
    # 3c. Utilities
    # -----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._hmm is None:
            raise RuntimeError("MarketRegimeDetector not fitted. Call .fit() first.")

    @property
    def transition_matrix(self) -> np.ndarray:
        """Learned transition matrix, shape [N, N]."""
        self._check_fitted()
        return self._hmm.transmat_  # type: ignore[return-value]

    @property
    def emission_means(self) -> np.ndarray:
        """Emission means per state, shape [N, F]."""
        self._check_fitted()
        return self._hmm.means_  # type: ignore[return-value]


# =====================================================================
# DEMONSTRATION & PROOF
# =====================================================================

def _run_demonstration() -> None:
    """Verify HMM detects extreme market regimes.

    KEY DESIGN PRINCIPLE:
        A 3-state HMM MUST see all 3 regimes during training.
        Training on only bull+bear leaves the 3rd state degenerate.
        We use the `lengths` argument to concatenate separate labeled
        sub-sequences without introducing temporal look-ahead.
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    np.random.seed(0)

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  Nexus Quant OS -- HMM Regime Detector Demonstration")
    print(f"{SEP}\n")

    # Features: [realised_vol, daily_return, vix_change]
    def make_regime(n, vol, ret_mean, ret_std):
        vol_obs = np.abs(np.random.normal(vol, vol * 0.15, n))
        ret_obs = np.random.normal(ret_mean, ret_std, n)
        vix_chg = np.random.normal(-ret_mean * 3, vol * 0.8, n)
        return np.column_stack([vol_obs, ret_obs, vix_chg])

    bull   = make_regime(300, vol=0.008, ret_mean= 0.0006, ret_std=0.008)
    bear   = make_regime(200, vol=0.022, ret_mean=-0.0015, ret_std=0.018)
    crisis = make_regime(100, vol=0.065, ret_mean=-0.0050, ret_std=0.055)

    # Training: all 3 regimes, separate sequences via lengths
    train_data    = np.vstack([bull[:200], bear, crisis[:60], bull[200:300]])
    train_lengths = [200, 200, 60, 100]

    # Test: calm window followed by extreme crash (vol x10, returns x8)
    test_calm   = make_regime(30, vol=0.008, ret_mean= 0.0005, ret_std=0.008)
    test_crisis = make_regime(20, vol=0.090, ret_mean=-0.0080, ret_std=0.075)
    test_data   = np.vstack([test_calm, test_crisis])

    print(f"Training: {len(train_data)} obs")
    print(f"  bull=300, bear=200, crisis=60")
    print(f"Test    : 30 calm + 20 EXTREME CRISIS days")
    print(f"  Crisis vol={0.090:.3f} vs bull vol={0.008:.3f} (x{0.090/0.008:.0f})\n")

    config   = HMMConfig(n_regimes=3, n_iter=300, danger_threshold=0.45)
    detector = MarketRegimeDetector(config)
    detector.fit(train_data, lengths=train_lengths)

    print("\nEmission means per state (vol | return | vix_chg):")
    for i, means in enumerate(detector.emission_means):
        regime = detector._state_to_regime.get(i, i)
        print(f"  State {i} [{MarketRegime(regime).name:>16s}]:  "
              f"vol={means[0]:.4f}  ret={means[1]:.5f}  vix={means[2]:.5f}")

    danger_seq = detector.predict_sequence(test_data)

    print(f"\nDanger probability over test window:")
    print(f"  {'Day':>4s}  {'Period':>8s}  {'Regime':>16s}  {'Danger':>8s}  Alert")
    print("  " + "-" * 56)
    for i, obs in enumerate(test_data):
        pred   = detector.predict(obs.reshape(1, -1))
        period = "CALM  " if i < 30 else "CRISIS"
        alert  = "[FIRE]" if pred.is_dangerous else "ok"
        print(
            f"  {i:>4d}  {period}  {pred.regime_label:>16s}  "
            f"{pred.danger_probability:>8.4f}  {alert}"
        )

    print(f"\nAutomated Assertions:")

    crisis_danger = danger_seq[30:]
    calm_danger   = danger_seq[:30]

    # 1. Crisis window danger >= calm window danger (absolute check on max)
    assert crisis_danger.mean() >= calm_danger.mean(), (
        f"Crisis danger ({crisis_danger.mean():.4f}) < calm ({calm_danger.mean():.4f})"
    )
    assert crisis_danger.max() >= 0.90, (
        f"Peak crisis danger {crisis_danger.max():.4f} should be >= 0.90"
    )
    print(f"  [PASS 1/3] Crisis danger max={crisis_danger.max():.4f}  "
          f"mean={crisis_danger.mean():.4f}  vs  Calm mean={calm_danger.mean():.4f}")

    # 2. At least some crisis days trigger the firewall
    crisis_preds = [
        detector.predict(test_data[30 + i].reshape(1, -1)) for i in range(20)
    ]
    n_triggered = sum(p.is_dangerous for p in crisis_preds)
    assert n_triggered >= 3, (
        f"Only {n_triggered}/20 crisis days triggered!"
    )
    print(f"  [PASS 2/3] Firewall triggered on {n_triggered}/20 crisis days")

    # 3. Transition matrix rows sum to 1
    row_sums = detector.transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    print(f"  [PASS 3/3] Transition matrix rows sum to 1.0")

    print(f"\n{SEP}")
    print(f"  ALL 3 ASSERTIONS PASSED -- HMM Regime Detector operational.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    _run_demonstration()
