import pytest
import numpy as np
import logging

from nexus_quant_os.portfolio.cvxpy_optimizer import PortfolioOptimizer, OptimizerConfig

@pytest.fixture
def base_optimizer():
    # 建立一個有 5 檔資產（包含 SHY 現金與 PSQ 反向）的優化器
    names = ["AAPL", "MSFT", "GOOG", "SHY", "PSQ"]
    return PortfolioOptimizer(n_assets=5, asset_names=names, config=OptimizerConfig(max_single_weight=0.30))

# ── 🌪️ 第一象限：神經網路毒藥注入 ──
def test_q1_nan_poisoning(base_optimizer):
    """【大腦中風】測試：MoE 吐出 NaN 或 Inf"""
    moe_weights = np.array([0.2, np.nan, 0.4, 0.1, 0.1]) 
    expert_util = np.array([0.5, np.nan, 0.5])
    cov_matrix = np.eye(5) * 0.01

    weights = base_optimizer.optimize(moe_weights, expert_util, cov_matrix)
    
    # 斷言：系統必須活下來！而且輸出的數字必須完全乾淨
    assert np.isfinite(weights).all(), "🚨 系統崩潰：權重中殘留 NaN 毒數據！"
    assert np.isclose(weights.sum(), 1.0) or np.isclose(weights.sum(), 0.0), "🚨 總權重未滿 100%！"

# ── 🌪️ 第二象限：數學極限崩潰 ──
def test_q2_singular_matrix(base_optimizer):
    """【千股跌停】測試：完美共線性，奇異矩陣攻擊"""
    moe_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expert_util = np.array([0.9, 0.9, 0.9])
    
    # 建立 Rank=1 的極端共變異數矩陣 (完全正相關)
    cov_matrix = np.ones((5, 5)) * 0.02 
    
    weights = base_optimizer.optimize(moe_weights, expert_util, cov_matrix)
    assert np.isfinite(weights).all(), "🚨 系統被奇異矩陣擊穿！"

def test_q2_zero_variance(base_optimizer):
    """【市場停牌】測試：無波動，共變異數矩陣全為 0"""
    moe_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expert_util = np.array([1.0, 0.0, 0.0]) # 強制進入 Risk Parity
    cov_matrix = np.zeros((5, 5))
    
    weights = base_optimizer.optimize(moe_weights, expert_util, cov_matrix)
    assert np.isfinite(weights).all(), "🚨 發生除以零或其他數學崩潰！"
    
def test_q2_fat_finger(base_optimizer):
    """【胖手指】測試：模型幻覺，預期報酬爆表"""
    # 模擬大腦發瘋，預測 AAPL 會有 1,000,000 的報酬率
    moe_weights = np.array([1000000.0, -0.01, 0.0, 0.0, 0.0]) 
    expert_util = np.array([0.9, 0.9, 0.9])
    cov_matrix = np.eye(5) * 0.01
    
    weights = base_optimizer.optimize(moe_weights, expert_util, cov_matrix)
    
    # 根據你的公式 dyn_max，加上各種微調，絕對不可超過 0.51
    assert weights[0] <= 0.501, f"🚨 胖手指防線被擊穿，單檔權重達 {weights[0]*100}%！"
