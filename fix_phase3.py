import re

# 1. run_moomoo_trade.py
f = "/Users/coolguy/developer/nexus_quant_os/run_moomoo_trade.py"
with open(f, "r") as file:
    c = file.read()

# P1-3: HMM 20 days
c = c.replace('market_features=spy_feature_matrix[-1:]', 'market_features=spy_feature_matrix[-20:]')

# P0-7: STEP 7 NaN Guard
nan_guard = """    raw_weights = routing_result.combined_output[-1]
    
    # ── P0-7: 防護 NaN/Inf 權重汙染 ──
    if np.any(np.isnan(raw_weights)) or np.any(np.isinf(raw_weights)):
        print("    🚨 CRITICAL: MoE Router returned NaN/Inf! Forcing equal-weight fallback.")
        raw_weights = np.ones(len(raw_weights)) / len(raw_weights)
"""
c = c.replace('    raw_weights = routing_result.combined_output[-1]', nan_guard)

# P1-2: SQQQ Exposure Cap
c = c.replace('total_exposure = sum(abs(w) for w in target_weights.values())',
              'total_exposure = min(1.0, sum(abs(w) for w in target_weights.values()))')

# P1-6: Notifier Try/Except
c = c.replace('    notifier.send_error(report)',
              '    try:\n        notifier.send_error(report)\n    except Exception as e:\n        logger.error(f"Failed to send Discord alert: {e}")')

with open(f, "w") as file:
    file.write(c)

# 2. hmm_regime_detector.py
f = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/risk_firewall/hmm_regime_detector.py"
with open(f, "r") as file:
    c = file.read()

p0_4_guard = """        try:
            posteriors = self.model.predict_proba(features)
        except Exception as e:
            logger.error(f"HMM predict_proba crashed: {e} -> fallback to max danger")
            return HMMResult(regime="BEAR_HIGH_VOL", bear_prob=1.0, extreme_prob=0.0)
            
        if np.isnan(posteriors).any():
            logger.error("HMM produced NaN posteriors -> fallback to max danger")
            return HMMResult(regime="BEAR_HIGH_VOL", bear_prob=1.0, extreme_prob=0.0)"""

c = c.replace('        posteriors = self.model.predict_proba(features)', p0_4_guard)
with open(f, "w") as file:
    file.write(c)

# 3. ood_anomaly_detector.py
f = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/risk_firewall/ood_anomaly_detector.py"
with open(f, "r") as file:
    c = file.read()

c = c.replace('        X = features.astype(np.float32)',
              '        X = features.astype(np.float32)\n        if np.isnan(X).any():\n            logger.error("OOD received NaN features! Failing closed.")\n            return AnomalyResult(is_ood=True, combined_anomaly_score=1.0, if_score=1.0, ae_zscore=1.0, is_ae_anomaly=True, is_if_anomaly=True)')
c = c.replace('        assert self._if_model is not None, "IsolationForest not fitted."',
              '        if self._if_model is None: raise RuntimeError("IsolationForest not fitted.")')
c = c.replace('        assert self._ae_model is not None, "Autoencoder not fitted."',
              '        if self._ae_model is None: raise RuntimeError("Autoencoder not fitted.")')
c = c.replace('        assert self._scaler is not None, "Scaler not fitted."',
              '        if self._scaler is None: raise RuntimeError("Scaler not fitted.")')
with open(f, "w") as file:
    file.write(c)

# 4. firewall_core.py
f = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/risk_firewall/firewall_core.py"
with open(f, "r") as file:
    c = file.read()

c = c.replace('from typing import Optional', 'from typing import Optional, Any')
c = c.replace('                scale = 0.0', '                scale = self.config.emergency_scale')
with open(f, "w") as file:
    file.write(c)

# 5. main.py
f = "/Users/coolguy/developer/nexus_quant_os/main.py"
with open(f, "r") as file:
    c = file.read()

dotenv_guard = """try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass"""
c = c.replace('from dotenv import load_dotenv\nload_dotenv()', dotenv_guard)
with open(f, "w") as file:
    file.write(c)

print("Phase 3 Fixed")
