with open("main.py", "r") as f:
    code = f.read()

imports = """from __future__ import annotations

import argparse
import yaml
from nexus_quant_os.live_trading.shadow_broker import ShadowBroker
from nexus_quant_os.execution.futu_broker import FutuBroker"""

code = code.replace("from __future__ import annotations", imports)

with open("main.py", "w") as f:
    f.write(code)

