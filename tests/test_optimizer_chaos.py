import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from nexus_quant_os.portfolio.cvxpy_optimizer import PortfolioOptimizer, OptimizerConfig

# 使用 Hypothesis 自動生成包含 NaN, Inf, 極大值, 極小值的陣列
@settings(max_examples=100, deadline=None)  # 跑 100 種隨機變異, 取消 deadline
@given(
    moe_preds=st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=5, max_size=5),
    expert_utils=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3)
)
def test_optimizer_survives_poison_data(moe_preds, expert_utils):
    """
    【第二象限測試】毒數據注入：
    無論 MoE 吐出多麼荒謬的預期報酬（包含 NaN/Inf），
    Optimizer 絕對不允許拋出 Exception 導致主程式崩潰！
    """
    # 建立 5 檔資產的優化器
    opt = PortfolioOptimizer(n_assets=5, config=OptimizerConfig(max_single_weight=0.30))
    cov_matrix = np.eye(5) * 0.01  # 正常的共變異數矩陣
    
    try:
        # 💣 執行轟炸：餵入有毒的 moe_preds
        weights = opt.optimize(
            moe_weights=np.array(moe_preds),
            expert_utilisation=np.array(expert_utils),
            cov_matrix=cov_matrix,
        )
        
        # 🛡️ 防禦斷言 1：系統存活，且輸出的權重絕對不可以有 NaN 或 Inf
        assert np.isfinite(weights).all(), "🚨 致命錯誤：輸出的交易權重包含 NaN 或 Inf！防線被擊穿。"
        
        # 🛡️ 防禦斷言 2：權重加總必須接近 1.0 (資金完全分配，不允許算出負數總和)
        assert np.isclose(weights.sum(), 1.0) or np.isclose(weights.sum(), 0.0), f"🚨 資金分配總和為 {weights.sum()}，而非 1.0 或 0.0！"
        
        # 🛡️ 防禦斷言 3：胖手指硬約束，單一資產不可超過設定的 30% (加上一點浮點數誤差)
        if np.isclose(weights.sum(), 1.0): # 如果全現金，權重為0
            assert np.max(weights) <= 0.301, f"🚨 胖手指防護失效！最大權重達 {np.max(weights)*100}%"

    except Exception as e:
        # 如果拋出錯誤，代表你的 try-except 沒寫好，測試失敗！
        pytest.fail(f"🚨 系統崩潰！Optimizer 無法處理極端輸入。錯誤訊息: {e}")

def test_optimizer_singular_matrix_attack():
    """
    【第二象限測試】完美共線性攻擊 (Singular Matrix Attack)：
    餵給 CVXPY 10 檔股票，設定它們的走勢一模一樣（相關係數 1.0）。
    驗證 CVXPY 不會報錯，且能優雅降級。
    """
    opt = PortfolioOptimizer(n_assets=10, config=OptimizerConfig(max_single_weight=0.30))
    # 建立完美共線的共變異數矩陣 (全部元素都是 0.01)
    singular_cov = np.ones((10, 10)) * 0.01
    
    # 預期報酬
    moe_preds = np.array([0.05] * 10)
    expert_utils = np.array([0.5, 0.5, 0.5])
    
    try:
        weights = opt.optimize(
            moe_weights=moe_preds,
            expert_utilisation=expert_utils,
            cov_matrix=singular_cov,
        )
        assert np.isfinite(weights).all()
        assert np.isclose(weights.sum(), 1.0) or np.isclose(weights.sum(), 0.0)
    except Exception as e:
        pytest.fail(f"🚨 完美共線矩陣導致系統崩潰: {e}")

@given(
    fat_finger=st.floats(min_value=1e5, max_value=1e10)
)
def test_optimizer_fat_finger_breach(fat_finger):
    """
    【第二象限測試】胖手指突破測試：
    給予某檔股票超高預期報酬，驗證 max_single_weight = 0.30 (30% 上限) 牢不可破。
    """
    opt = PortfolioOptimizer(n_assets=5, config=OptimizerConfig(max_single_weight=0.30))
    cov_matrix = np.eye(5) * 0.01
    
    moe_preds = np.array([0.01, 0.02, 0.03, fat_finger, 0.01])
    expert_utils = np.array([0.5, 0.5, 0.5])
    
    weights = opt.optimize(
        moe_weights=moe_preds,
        expert_utilisation=expert_utils,
        cov_matrix=cov_matrix,
    )
    
    # 就算預期報酬有 100 億，權重仍不可超過 0.30 (加上浮點誤差)
    assert np.max(weights) <= 0.301, f"🚨 胖手指防護失效！最大權重達 {np.max(weights)*100}%"
