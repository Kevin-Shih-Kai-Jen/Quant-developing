import numpy as np
import pytest
from nexus_quant_os.monitoring.health_check import check_weight_sanity, Severity

def test_normal_allocation():
    weights = np.array([0.3, 0.3, 0.4])
    result = check_weight_sanity(weights)
    assert result.severity == Severity.OK

def test_100_percent_cash_firewall():
    # When firewall triggers, weights are all zero
    weights = np.array([0.0, 0.0, 0.0])
    result = check_weight_sanity(weights)
    # 0% exposure should NOT be critical
    assert result.severity in [Severity.OK, Severity.WARNING]

def test_negative_weights():
    # If optimizer outputs negative weights (it shouldn't)
    weights = np.array([-0.1, 0.5, 0.6])
    result = check_weight_sanity(weights)
    assert result.severity == Severity.CRITICAL

def test_nan_weights():
    weights = np.array([np.nan, 0.5, 0.5])
    result = check_weight_sanity(weights)
    assert result.severity == Severity.CRITICAL

def test_over_concentration():
    weights = np.array([0.9, 0.1, 0.0])
    result = check_weight_sanity(weights, max_single_weight=0.40)
    assert result.severity == Severity.WARNING
