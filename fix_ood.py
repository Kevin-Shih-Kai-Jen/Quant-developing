import re
f = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/risk_firewall/ood_anomaly_detector.py"
with open(f, "r") as file:
    c = file.read()
c = c.replace('            assert self._if_model is not None', '            if self._if_model is None: raise RuntimeError("IsolationForest not fitted.")')
c = c.replace('            assert self._ae_model is not None', '            if self._ae_model is None: raise RuntimeError("Autoencoder not fitted.")')
c = c.replace('            assert self._train_min is not None and self._train_max is not None', '            if self._train_min is None or self._train_max is None: raise RuntimeError("Scaler not fitted.")')
c = c.replace('            assert self._if_score_max is not None and self._if_score_min is not None', '            if self._if_score_max is None or self._if_score_min is None: raise RuntimeError("IF scores not fitted.")')
with open(f, "w") as file:
    file.write(c)
