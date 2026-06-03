import re

with open("main.py", "r") as f:
    code = f.read()

# 1. Add argparse and yaml and new imports
imports_pattern = r"import sys\nimport warnings"
imports_repl = """import sys
import warnings
import argparse
import yaml
from nexus_quant_os.live_trading.shadow_broker import ShadowBroker
from nexus_quant_os.execution.futu_broker import FutuBroker"""
code = code.replace(imports_pattern, imports_repl)

# 2. Add argument parsing and yaml loading at the top of run_pipeline
run_pipeline_pattern = r"def run_pipeline\(\) -> None:\n    \"\"\"執行 Nexus Quant OS 完整 DAG 管線。\"\"\"\n\n    SEP   = \"=\" \* 78"
run_pipeline_repl = """def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Quant OS — Core DAG Scheduler")
    parser.add_argument("--dry-run", action="store_true", help="演習模式：使用影子券商，不發送真實委託")
    return parser.parse_args()

def run_pipeline() -> None:
    \"\"\"執行 Nexus Quant OS 完整 DAG 管線。\"\"\"

    args = parse_args()

    # 1. 讀取人類高權限指令層
    try:
        with open("execution_config.yaml", "r") as f:
            exec_config = yaml.safe_load(f)
    except FileNotFoundError:
        exec_config = {}

    SEP   = "=" * 78"""
code = re.sub(run_pipeline_pattern, run_pipeline_repl, code)

# 3. Add current_weights fetching for dry_run
# We need to do this BEFORE step 5c. So we will just insert it before STEP 5c.
step_5c_pattern = r"    # ─────────────────────────────────────────────────────────────\n    # STEP 5c: 投組優化器 \(Risk Parity / Constrained MVO\)"
step_5c_repl = """    # ── 準備 Execution Config & 券商持倉 (供 Optimizer 使用) ──
    exec_params = exec_config.get("execution", {})
    dry_run_mode = args.dry_run or exec_params.get("dry_run", False)

    if dry_run_mode:
        broker = ShadowBroker()
    else:
        try:
            broker = FutuBroker(
                host=exec_params.get("broker_host", "127.0.0.1"), 
                port=exec_params.get("broker_port", 11111)
            )
        except ConnectionError as e:
            logger.critical("🚨 無法連線至券商 API: %s", e)
            print("🚨 無法連線至券商 API，將強制退回 Dry-run 模式")
            broker = ShadowBroker()
            dry_run_mode = True

    current_weights = {}
    account = broker.get_account()
    if account.equity > 0:
        for pos in broker.get_positions():
            current_weights[pos.symbol] = pos.market_value / account.equity

    # ─────────────────────────────────────────────────────────────
    # STEP 5c: 投組優化器 (Risk Parity / Constrained MVO)"""
code = re.sub(step_5c_pattern, step_5c_repl, code)

# 4. Modify portfolio_optimizer.optimize to take current_weights and exec_config
opt_call_pattern = r"    optimized_weights = portfolio_optimizer\.optimize\(\n        moe_weights=norm_weights,\n        expert_utilisation=expert_util,\n        cov_matrix=cov_matrix,\n    \)"
opt_call_repl = """    optimized_weights = portfolio_optimizer.optimize(
        moe_weights=norm_weights,
        expert_utilisation=expert_util,
        cov_matrix=cov_matrix,
        current_weights=current_weights,
        exec_config=exec_config,
    )"""
code = re.sub(opt_call_pattern, opt_call_repl, code)

# 5. Add STEP 10 at the end of run_pipeline
step_10_code = """
    # ─────────────────────────────────────────────────────────────
    # STEP 10: Trade Execution (動態路由)
    # ─────────────────────────────────────────────────────────────
    print(f"{ARROW} STEP 10/10 : 券商執行路由 (Broker Execution)")
    if dry_run_mode:
        print("    模式         : 🟢 [DRY-RUN] 演習模式啟動：掛載 ShadowBroker (模擬滑價與虛擬成交)")
    else:
        print("    模式         : 🔴 [LIVE TRADE] 實盤模式啟動：已連線 FutuOpenD")

    target_weights_dict = {asset: float(w) for asset, w in zip(assets_sorted, final_weights)}
    
    intents = broker.reconcile(target_weights=target_weights_dict)
    
    if not intents:
        print("    ✅ 投資組合已是最佳狀態，無須調倉。")
    else:
        results = broker.execute(intents)
        print("    📊 執行結果統整:")
        for r in results:
            print(f"      [{r.status}] {r.side} {r.symbol} x{r.qty} (Price: {r.filled_price:.2f}, ID: {r.order_id})")

    # ─────────────────────────────────────────────────────────────
    # PIPELINE COMPLETE"""

code = code.replace("    # ─────────────────────────────────────────────────────────────\n    # PIPELINE COMPLETE", step_10_code)

with open("main.py", "w") as f:
    f.write(code)

