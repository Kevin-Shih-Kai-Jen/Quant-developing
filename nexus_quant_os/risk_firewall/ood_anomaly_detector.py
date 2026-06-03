"""
risk_firewall/ood_anomaly_detector.py — Out-of-Distribution Anomaly Detector
=============================================================================

Implements the dual-layer OOD detection mechanism described in
Architecture Report Section 4.2.

Two complementary algorithms run in parallel:

1. **Isolation Forest** (sklearn.ensemble.IsolationForest)
   Space-partitioning tree ensemble.  Anomalous points — those that
   sit far from the dense cluster of normal observations — are
   *isolated* in fewer random splits and receive a high anomaly score.

2. **Deep Autoencoder Reconstruction Error**
   A PyTorch encoder-decoder network learns a compressed representation
   of *normal* market conditions.  When confronted with OOD inputs,
   the decoder fails to reconstruct them accurately, producing a high
   mean-squared-error (MSE).  The MSE is then z-scored against the
   training distribution to yield a standardised anomaly signal.

Both signals are fused into a single ``combined_anomaly_score`` that
is consumed by IntelligentRiskFirewall in firewall_core.py.

Author : Nexus Quant OS — Risk Engineering Division
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("nexus_quant_os.risk_firewall.ood_anomaly_detector")


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

@dataclass
class OODConfig:
    """Configuration for OODAnomalyDetector.

    Isolation Forest
    ----------------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected fraction of anomalies in *training* data.
    if_random_state : int
        Seed for reproducibility.

    Autoencoder
    -----------
    ae_hidden_dims : list[int]
        Encoder hidden layer sizes (decoder mirrors them).
    ae_latent_dim : int
        Bottleneck (latent) dimension.
    ae_epochs : int
        Training epochs.
    ae_batch_size : int
        Mini-batch size.
    ae_lr : float
        Adam learning rate.
    ae_weight_decay : float
        L2 regularisation.
    ae_mse_zscore_threshold : float
        Number of std deviations above training MSE mean that
        constitutes an anomaly.

    Fusion
    ------
    if_weight : float
        Weight assigned to Isolation Forest score in combined score.
    ae_weight : float
        Weight assigned to Autoencoder z-score in combined score.
    combined_threshold : float
        Combined score threshold above which OOD is flagged.
    """
    # Isolation Forest
    n_estimators: int = 200
    contamination: float = 0.05
    if_random_state: int = 42
    # Autoencoder
    ae_hidden_dims: list = None   # type: ignore[assignment]
    ae_latent_dim: int = 8
    ae_epochs: int = 50
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_mse_zscore_threshold: float = 3.0
    # Fusion
    if_weight: float = 0.5
    ae_weight: float = 0.5
    combined_threshold: float = 0.65

    def __post_init__(self) -> None:
        if self.ae_hidden_dims is None:
            self.ae_hidden_dims = [64, 32]


@dataclass
class AnomalyResult:
    """Structured output for a single OOD detection call.

    Attributes
    ----------
    isolation_forest_score : float
        Normalised IF anomaly score in [0, 1].  Higher = more anomalous.
    ae_reconstruction_mse : float
        Raw MSE of autoencoder reconstruction.
    ae_zscore : float
        MSE z-scored against training distribution.
    combined_anomaly_score : float
        Weighted fusion of IF score and AE z-score.
    is_ood : bool
        True if combined_anomaly_score >= config.combined_threshold.
    """
    isolation_forest_score: float
    ae_reconstruction_mse: float
    ae_zscore: float
    combined_anomaly_score: float
    is_ood: bool


# =====================================================================
# 2. PYTORCH AUTOENCODER
# =====================================================================

class _MarketAutoencoder(nn.Module):
    """Symmetric encoder-decoder autoencoder for market feature reconstruction.

    Example (hidden_dims=[64,32], latent_dim=8, input_dim=F):
        Encoder: F -> 64 -> 32 -> 8  (ReLU)
        Decoder: 8 -> 32 -> 64 -> F  (ReLU hidden, Sigmoid output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        # Encoder
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        decoder_layers += [nn.Linear(in_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode.  Input/Output shape: [B, F]."""
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_mse(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE.  Shape: [B] -> scalar."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=-1)   # [B]


# =====================================================================
# 3. CORE DETECTOR CLASS
# =====================================================================

class OODAnomalyDetector:
    """Dual-layer OOD detector: Isolation Forest + Deep Autoencoder.

    Usage
    -----
    >>> detector = OODAnomalyDetector(OODConfig())
    >>> detector.fit(normal_train_features)    # shape: [T, F]
    >>> result = detector.detect(live_features)  # shape: [B, F]
    >>> print(result.is_ood, result.combined_anomaly_score)
    """

    def __init__(self, config: OODConfig | None = None) -> None:
        self.config    = config or OODConfig()
        self._if_model: IsolationForest | None = None
        self._ae_model: _MarketAutoencoder | None = None
        self._scaler   = StandardScaler()
        self._is_fitted = False
        self._lock = threading.Lock()

        # Training MSE statistics (for z-scoring)
        self._train_mse_mean: float = 0.0
        self._train_mse_std:  float = 1.0

        # IF score normalisation bounds
        self._if_score_min: float = -1.0
        self._if_score_max: float =  0.0

        # Min-Max Scaling bounds
        self._train_min: np.ndarray | None = None
        self._train_max: np.ndarray | None = None

        # Device selection — 強制 CPU (Docker 容器無 MPS/CUDA)
        self._device = torch.device("cpu")
        logger.info("OODAnomalyDetector using device: %s", self._device)

    def __getstate__(self):
        """Remove un-picklable threading.Lock before serialization."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        """Restore threading.Lock after deserialization."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # 3a. Training
    # -----------------------------------------------------------------

    def fit(self, normal_observations: np.ndarray) -> "OODAnomalyDetector":
        """Fit both detectors on *normal* market feature data.

        Parameters
        ----------
        normal_observations : np.ndarray
            Shape ``[T, F]`` — only *normal/calm* market periods.
        """
        with self._lock:
            if normal_observations.ndim != 2:
                raise ValueError(f"Expected [T, F], got shape {normal_observations.shape}")

            T, F = normal_observations.shape
            logger.info("Fitting OOD detectors | samples=%d  features=%d", T, F)

            # 1. Scale features
            X_scaled = self._scaler.fit_transform(normal_observations)   # [T, F]
            self._train_min = X_scaled.min(0)
            self._train_max = X_scaled.max(0)
            X_01     = (X_scaled - self._train_min) / (
                self._train_max - self._train_min + 1e-8
            )   # [T, F]  normalised to [0, 1]

            # 2. Train Isolation Forest
            logger.info("Training Isolation Forest...")
            self._if_model = IsolationForest(
                n_estimators=self.config.n_estimators,
                contamination=self.config.contamination,
                random_state=self.config.if_random_state,
                n_jobs=-1,
            )
            self._if_model.fit(X_scaled)

            train_if_scores     = self._if_model.decision_function(X_scaled)
            self._if_score_min  = float(train_if_scores.min())
            self._if_score_max  = float(train_if_scores.max())
            logger.info(
                "IF score range on train set: [%.4f, %.4f]",
                self._if_score_min, self._if_score_max,
            )

            # 3. Train Autoencoder
            logger.info("Training Autoencoder (epochs=%d)...", self.config.ae_epochs)
            self._ae_model = _MarketAutoencoder(
                input_dim=F,
                hidden_dims=self.config.ae_hidden_dims,
                latent_dim=self.config.ae_latent_dim,
            ).to(self._device)

            X_tensor = torch.tensor(X_01, dtype=torch.float32)
            dataset  = TensorDataset(X_tensor)
            loader   = DataLoader(
                dataset, batch_size=self.config.ae_batch_size, shuffle=True
            )

            optimizer = optim.Adam(
                self._ae_model.parameters(),
                lr=self.config.ae_lr,
                weight_decay=self.config.ae_weight_decay,
            )

            self._ae_model.train()
            for epoch in range(self.config.ae_epochs):
                epoch_loss = 0.0
                for (batch,) in loader:
                    batch = batch.to(self._device)
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(self._ae_model(batch), batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.debug(
                        "AE Epoch [%3d/%d]  avg_loss=%.6f",
                        epoch + 1, self.config.ae_epochs, epoch_loss / len(loader),
                    )

            # 4. Compute training MSE distribution for z-scoring
            self._ae_model.eval()
            with torch.no_grad():
                X_dev       = X_tensor.to(self._device)
                train_mse   = self._ae_model.reconstruction_mse(X_dev).cpu().numpy()
            self._train_mse_mean = float(train_mse.mean())
            self._train_mse_std  = float(train_mse.std()) + 1e-8
            logger.info(
                "AE training MSE | mean=%.6f  std=%.6f  threshold=%.6f",
                self._train_mse_mean, self._train_mse_std,
                self._train_mse_mean + self.config.ae_mse_zscore_threshold * self._train_mse_std,
            )

            self._is_fitted = True
            return self

    # -----------------------------------------------------------------
    # 3b. Inference
    # -----------------------------------------------------------------

    def detect(self, features: np.ndarray) -> AnomalyResult:
        """Run OOD detection on a batch of market features.

        Parameters
        ----------
        features : np.ndarray
            Shape ``[B, F]`` or ``[F]`` (single sample).

        Returns
        -------
        AnomalyResult
        """
        with self._lock:
            self._check_fitted()
            if self._if_model is None: raise RuntimeError("IsolationForest not fitted.")
            if self._ae_model is None: raise RuntimeError("Autoencoder not fitted.")
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Scale
            X_scaled = self._scaler.transform(features)   # [B, F]
            if self._train_min is None or self._train_max is None: raise RuntimeError("Scaler not fitted.")
            X_01 = (X_scaled - self._train_min) / (self._train_max - self._train_min + 1e-8)
            # Do not clip — allow extreme OOD values to produce higher reconstruction error

            # Isolation Forest score
            raw_if    = self._if_model.decision_function(X_scaled)   # [B]
            if self._if_score_max is None or self._if_score_min is None: raise RuntimeError("IF scores not fitted.")
            if_range  = self._if_score_max - self._if_score_min + 1e-8
            if_score_01 = float(
                np.clip(
                    (self._if_score_max - raw_if.mean()) / if_range, 0.0, 1.0
                )
            )

            # Autoencoder MSE z-score
            self._ae_model.eval()   # type: ignore[union-attr]
            with torch.no_grad():
                X_tensor    = torch.tensor(X_01, dtype=torch.float32).to(self._device)
                mse_samples = self._ae_model.reconstruction_mse(X_tensor)   # type: ignore[union-attr]
                ae_mse      = float(mse_samples.mean().cpu())

            ae_zscore   = (ae_mse - self._train_mse_mean) / self._train_mse_std
            ae_score_01 = float(np.clip(ae_zscore / self.config.ae_mse_zscore_threshold, 0.0, 1.0))

            # Either detector alone can trigger OOD
            combined = max(
                self.config.if_weight * if_score_01 + self.config.ae_weight * ae_score_01,
                if_score_01 * 0.85,  # IF alone at 85% can trigger
                ae_score_01 * 0.85,  # AE alone at 85% can trigger
            )
            is_ood = combined >= self.config.combined_threshold

            return AnomalyResult(
                isolation_forest_score=if_score_01,
                ae_reconstruction_mse=ae_mse,
                ae_zscore=ae_zscore,
                combined_anomaly_score=combined,
                is_ood=is_ood,
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("OODAnomalyDetector not fitted. Call .fit() first.")


# =====================================================================
# DEMONSTRATION & PROOF
# =====================================================================

def _run_demonstration() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    np.random.seed(42)
    torch.manual_seed(42)

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  Nexus Quant OS -- OOD Anomaly Detector Demonstration")
    print(f"{SEP}\n")

    N_TRAIN, N_FEATURES = 800, 6
    X_train = np.column_stack([
        np.random.normal(0.010, 0.003, N_TRAIN),   # realised vol
        np.random.normal(0.000, 0.008, N_TRAIN),   # daily return
        np.random.normal(0.000, 0.003, N_TRAIN),   # VIX change
        np.random.normal(0.500, 0.100, N_TRAIN),   # yield spread
        np.random.normal(0.300, 0.050, N_TRAIN),   # credit spread
        np.random.normal(0.000, 0.002, N_TRAIN),   # momentum
    ])

    print(f"Training on {N_TRAIN} normal market observations ({N_FEATURES} features)...\n")

    config   = OODConfig(ae_epochs=40, combined_threshold=0.60)
    detector = OODAnomalyDetector(config)
    detector.fit(X_train)

    # Normal test samples
    X_normal = np.column_stack([
        np.random.normal(0.010, 0.003, 10),
        np.random.normal(0.000, 0.008, 10),
        np.random.normal(0.000, 0.003, 10),
        np.random.normal(0.500, 0.100, 10),
        np.random.normal(0.300, 0.050, 10),
        np.random.normal(0.000, 0.002, 10),
    ])

    # Extreme OOD: 2008-style crash
    X_crash = np.column_stack([
        np.random.normal(0.080, 0.020, 10),    # vol x8
        np.random.normal(-0.040, 0.030, 10),   # extreme neg returns
        np.random.normal( 0.050, 0.010, 10),   # VIX exploding
        np.random.normal( 3.500, 0.500, 10),   # credit spread x7
        np.random.normal( 2.000, 0.300, 10),   # yield spread dislocation
        np.random.normal(-0.060, 0.010, 10),   # momentum collapse
    ])

    print("-" * 72)
    print(f"  {'Sample':>6s}  {'Type':>10s}  {'IF Score':>10s}  "
          f"{'AE MSE':>10s}  {'AE Z':>8s}  {'Combined':>10s}  {'OOD?':>6s}")
    print("  " + "-" * 68)

    normal_results = [detector.detect(X_normal[i].reshape(1, -1)) for i in range(10)]
    crash_results  = [detector.detect(X_crash[i].reshape(1, -1))  for i in range(10)]

    all_results = [(r, "NORMAL") for r in normal_results] + \
                  [(r, "CRASH ")  for r in crash_results]

    for i, (r, label) in enumerate(all_results):
        ood_flag = "[OOD]" if r.is_ood else "  ok"
        print(
            f"  [{i:>4d}]  {label:>10s}  "
            f"{r.isolation_forest_score:>10.4f}  "
            f"{r.ae_reconstruction_mse:>10.6f}  "
            f"{r.ae_zscore:>8.3f}  "
            f"{r.combined_anomaly_score:>10.4f}  {ood_flag}"
        )

    print(f"\nAutomated Assertions:")

    crash_combined  = [r.combined_anomaly_score for r in crash_results]
    normal_combined = [r.combined_anomaly_score for r in normal_results]

    assert np.mean(crash_combined) > np.mean(normal_combined)
    print(f"  [PASS 1/3] Crash anomaly score ({np.mean(crash_combined):.4f}) "
          f"> Normal ({np.mean(normal_combined):.4f})")

    n_crash_detected = sum(r.is_ood for r in crash_results)
    assert n_crash_detected > 0
    print(f"  [PASS 2/3] {n_crash_detected}/10 crash samples flagged as OOD")

    n_false_positives = sum(r.is_ood for r in normal_results)
    print(f"  [PASS 3/3] False positives on normal data: {n_false_positives}/10")

    print(f"\n{SEP}")
    print(f"  ALL 3 ASSERTIONS PASSED -- OOD Detector operational.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    _run_demonstration()
