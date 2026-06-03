import pytest
import numpy as np
import yaml
import json
import os
import sys
from unittest.mock import patch, MagicMock

# 載入我們升級後的模組
from nexus_quant_os.portfolio.cvxpy_optimizer import PortfolioOptimizer, OptimizerConfig
from main import run_pipeline
from scripts.copilot_terminal import CopilotTerminal

# ════════════════════════════════════════════════════════════
# 🌪️ 測試一：CVXPY 數學約束測試 (Shift-Left Constraints)
# ════════════════════════════════════════════════════════════
def test_optimizer_human_constraints():
    """【防線測試】驗證 CVXPY 能否完美將人類意志轉化為數學邊界，且不會爆倉"""
    opt = PortfolioOptimizer(n_assets=4, asset_names=["AAPL", "AVGO", "NVDA", "CEG"], config=OptimizerConfig())
    
    # 模擬大腦預測 (大家看起來都不錯)
    moe_weights = np.array([0.25, 0.25, 0.25, 0.25])
    expert_util = np.array([0.9, 0.9, 0.9])
    cov_matrix = np.eye(4) * 0.01

    # 模擬人類意志 (YAML 設定)
    exec_config = {
        "blacklist": ["AVGO"],              # 絕對不買博通
        "hodl_symbols": ["NVDA"],           # 死抱輝達
        "forced_positions": [
            {"symbol": "CEG", "min_weight": 0.10}  # 強制買核能至少 10%
        ]
    }
    # 模擬目前真實庫存：我們已經持有 15% 的輝達
    current_weights = {"NVDA": 0.15}

    # 🚀 執行數學優化
    weights = opt.optimize(
        moe_weights, expert_util, cov_matrix, 
        current_weights=current_weights, exec_config=exec_config
    )

    # 🛡️ 裝甲斷言 1：黑名單絕對約束 (AVGO 必須是 0)
    assert weights[1] <= 1e-4, f"🚨 致命錯誤：黑名單 AVGO 居然被分配了 {weights[1]} 的權重！"
    
    # 🛡️ 裝甲斷言 2：HODL 信仰約束 (NVDA 必須 >= 15%，絕對不可賣出)
    assert weights[2] >= 0.149, f"🚨 致命錯誤：NVDA 被無故賣出，權重降至 {weights[2]}！"
    
    # 🛡️ 裝甲斷言 3：強制買入約束 (CEG 必須 >= 10%)
    assert weights[3] >= 0.099, f"🚨 致命錯誤：CEG 強制買入失敗，權重僅 {weights[3]}！"
    
    # 🛡️ 裝甲斷言 4：資金守恆定律 (所有權重加總必須是 100%)
    assert np.isclose(weights.sum(), 1.0), "🚨 致命錯誤：總權重加總不等於 1.0，引發爆倉危機！"

# ════════════════════════════════════════════════════════════
# 🌪️ 測試二：券商動態路由測試 (Execution Engine Routing)
# ════════════════════════════════════════════════════════════
@patch("main.FutuBroker")
@patch("main.ShadowBroker")
def test_execution_dry_run_routing(MockShadow, MockFutu):
    """【演習防線】確保 --dry-run 模式絕對不會觸發真實券商 API"""
    import sys
    import builtins
    
    original_open = builtins.open
    def custom_open(file, *args, **kwargs):
        if "execution_config.yaml" in str(file):
            from unittest.mock import mock_open
            return mock_open(read_data="execution:\n  dry_run: false\n")()
        return original_open(file, *args, **kwargs)


    test_args = ["main.py", "--dry-run"]
    with patch.object(sys, 'argv', test_args):
        # Configure the Mock broker to return valid account structures to prevent TypeError
        MockShadow.return_value.get_account.return_value.equity = 100000.0
        MockShadow.return_value.get_positions.return_value = []
        
        with patch("builtins.open", side_effect=custom_open):

            from main import run_pipeline
            run_pipeline()
            
        MockShadow.assert_called_once()
        MockFutu.assert_not_called()

# ════════════════════════════════════════════════════════════
# 🌪️ 測試三：AI 副駕 JSON 攔截測試 (Copilot Interceptor)
# ════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_copilot_json_interceptor(tmp_path):
    """【AI 防爆】測試：副駕是否能精準攔截 JSON 並改寫 YAML，且不崩潰"""
    # 建立臨時 YAML 檔
    fake_yaml = tmp_path / "execution_config.yaml"
    fake_yaml.write_text("blacklist: ['OLD_STOCK']\n")
    
    terminal = CopilotTerminal()
    terminal.config_path = str(fake_yaml) # 替換為臨時路徑
    
    # 模擬 LLM 吐出帶有雜訊的 JSON
    fake_llm_response = """
    好的經理，我已經準備為您更新系統。
    ```json_tool_call
    {
      "tool": "update_universe_and_rules",
      "add_blacklist": ["AVGO"],
      "custom_python_code": "print('hello')"
    }
    ```
    現在準備啟動管線！
    """
    
    # 攔截 LLM (我們 Mock 掉 ask 方法)
    terminal.llm.ask = MagicMock(return_value=fake_llm_response)
    
    # 🚀 處理輸入
    await terminal.process_user_input("把 AVGO 加入黑名單")
    
    # 🛡️ 裝甲斷言 1：驗證 YAML 是否正確被修改 (AVGO 是否被加進去)
    with open(fake_yaml, "r") as f:
        new_config = yaml.safe_load(f)
    
    assert "AVGO" in new_config["blacklist"], "🚨 AI JSON 攔截失敗：AVGO 未寫入黑名單！"
    assert "OLD_STOCK" in new_config["blacklist"], "🚨 致命錯誤：舊的黑名單被洗掉了！"

# (輔助 Mock 函數)
def mock_open_yaml_config():
    from unittest.mock import mock_open
    return mock_open(read_data="execution:\n  dry_run: false\n")
