f = "/Users/coolguy/developer/nexus_quant_os/nexus_quant_os/data_pipelines/data_loader.py"
with open(f, "r") as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    if "macro_daily[\"industrial_production\"] = 100.0" in line:
        print(f"Line {i}: {repr(line)}")
    if "macro_daily[\"yield_curve_slope\"] = 0.5" in line:
        print(f"Line {i}: {repr(line)}")
