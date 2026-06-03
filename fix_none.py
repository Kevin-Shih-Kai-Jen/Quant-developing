with open("nexus_quant_os/portfolio/cvxpy_optimizer.py", "r") as f:
    code = f.read()

code = code.replace(
    'forced_positions = exec_config.get("forced_positions", [])',
    'forced_positions = exec_config.get("forced_positions") or []'
)
code = code.replace(
    'blacklist = set(exec_config.get("blacklist", []))',
    'blacklist = set(exec_config.get("blacklist") or [])'
)
code = code.replace(
    'hodl_symbols = set(exec_config.get("hodl_symbols", []))',
    'hodl_symbols = set(exec_config.get("hodl_symbols") or [])'
)

with open("nexus_quant_os/portfolio/cvxpy_optimizer.py", "w") as f:
    f.write(code)

