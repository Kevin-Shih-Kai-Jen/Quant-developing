import sys
import re
import json
import yaml
import asyncio
import subprocess
import os

class DeepSeekClient:
    """Mock LLM Client for demonstration purposes"""
    def ask(self, user_prompt: str) -> str:
        if "博通" in user_prompt or "AVGO" in user_prompt:
            return '''好的，我已經收到您的指令。我會將博通(AVGO)加入永久黑名單，並設定波動率防護條件。現在我將觸發 DAG 管線進行回測。

```json_tool_call
{
    "tool": "update_universe_and_rules",
    "add_blacklist": ["AVGO"],
    "custom_python_code": "def user_dynamic_filter(df):\\n    if 'volatility' in df.columns:\\n        return df['volatility'] < 0.05\\n    return True"
}
```

```json_tool_call
{"tool": "trigger_dag_pipeline"}
```'''
        elif "回測已完成" in user_prompt:
            return "報告經理，已成功將博通加入黑名單並執行回測。本次 Dry-run 回撤成功守在安全範圍內，各項指標均符合預期。"
        return "我了解了。"

class CopilotTerminal:
    def __init__(self):
        # Substitute with actual DeepSeekClient in production
        self.llm = DeepSeekClient()
        self.config_path = "execution_config.yaml"

    async def _run_dag_pipeline_async(self):
        """背景非同步執行 DAG 管線，不卡死主執行緒"""
        print("\n🚀 [系統後端] 正在啟動 DAG 管線 (main.py --dry-run)...")
        # 使用 subprocess 在背景跑，模擬真實系統架構
        process = await asyncio.create_subprocess_exec(
            sys.executable, "main.py", "--dry-run",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": "."}
        )
        stdout, stderr = await process.communicate()
        print("\n✅ [系統後端] 回測/演習執行完畢！正將報告餵回給 Copilot...\n")
        
        # Merge stdout and stderr
        full_log = (stdout.decode() + "\n" + stderr.decode()).strip()
        return full_log

    async def process_user_input(self, user_msg: str):
        # 1. 呼叫 LLM
        llm_response = self.llm.ask(user_prompt=user_msg)
        
        # 顯示 AI 的文字回覆給用戶看
        clean_text = re.sub(r'```json_tool_call.*?```', '', llm_response, flags=re.DOTALL).strip()
        print(f"\n🤖 [AI 副駕]:\n{clean_text}\n")

        # 2. 攔截系統工具指令
        tool_calls = re.findall(r'```json_tool_call\s*(.*?)\s*```', llm_response, re.DOTALL)
        
        for call_str in tool_calls:
            try:
                cmd = json.loads(call_str)
                
                # ── 工具 A：更新系統配置 ──
                if cmd["tool"] == "update_universe_and_rules":
                    print(f"⚙️ [系統執行] 正在寫入 {self.config_path} ...")
                    try:
                        with open(self.config_path, "r") as f:
                            config = yaml.safe_load(f)
                    except FileNotFoundError:
                        config = {"blacklist": []}
                        
                    if "blacklist" not in config or not isinstance(config["blacklist"], list):
                        config["blacklist"] = []
                        
                    config["blacklist"].extend(cmd.get("add_blacklist", []))
                    config["blacklist"] = list(set(config["blacklist"])) # 去重複
                    
                    with open(self.config_path, "w") as f:
                        yaml.dump(config, f, allow_unicode=True)
                        
                    # 寫入沙盒程式碼
                    os.makedirs("nexus_quant_os/plugins", exist_ok=True)
                    with open("nexus_quant_os/plugins/user_rules_sandbox.py", "w") as f:
                        f.write(cmd.get("custom_python_code", ""))
                    print("✅ [系統執行] 黑名單與自訂策略更新成功。")

                # ── 工具 B：觸發 DAG 管線重訓與回測 ──
                elif cmd["tool"] == "trigger_dag_pipeline":
                    # 非同步啟動管線
                    report_log = await self._run_dag_pipeline_async()
                    
                    # 將結果塞回給 LLM 做 Phase 4 的「覆盤分析」
                    follow_up = f"[系統隱藏訊息] 回測已完成。日誌如下:\n{report_log[-1000:]}\n請向用戶報告績效。"
                    final_analysis = self.llm.ask(user_prompt=follow_up)
                    print(f"\n🤖 [AI 覆盤報告]:\n{final_analysis}\n")
                    
            except Exception as e:
                print(f"🚨 指令解析失敗: {e}")

if __name__ == "__main__":
    terminal = CopilotTerminal()
    # 模擬一次完整的人機協作對話
    asyncio.run(terminal.process_user_input(
        "同意你的提案！請幫我把博通(AVGO)加入永久黑名單，設定波動率防護條件，然後去跑回測吧！"
    ))
