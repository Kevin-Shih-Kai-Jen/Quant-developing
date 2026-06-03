import re

with open("nexus_quant_os/execution/futu_broker.py", "r") as f:
    lines = f.readlines()

out = []
for i, line in enumerate(lines):
    if line.startswith("            logger.error") and "order_list_query failed" in line:
        out.append("                " + line.strip() + "\n")
    elif line.startswith("            logger.error") and "place_order failed" in line:
        out.append("                " + line.strip() + "\n")
    elif line.startswith("            )") and ("modify_order" in lines[i-1] or "trd_env" in lines[i-1]):
        out.append("                " + line.strip() + "\n")
    else:
        out.append(line)

with open("nexus_quant_os/execution/futu_broker.py", "w") as f:
    f.writelines(out)
