import re

with open("tests/test_chaos_nuclear_armor.py", "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    if "fw.is_fitted = True" in line or "fw.is_fitted_tw = True" in line:
        continue
    out.append(line)
    if "fw = IntelligentRiskFirewall(mock_hmm, mock_ood)" in line:
        out.append("        fw._is_fitted = {'US': True, 'TW': True}\n")

with open("tests/test_chaos_nuclear_armor.py", "w") as f:
    f.writelines(out)
