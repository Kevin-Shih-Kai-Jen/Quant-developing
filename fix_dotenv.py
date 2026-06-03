import re

file_path = "/Users/coolguy/developer/nexus_quant_os/main.py"
with open(file_path, "r") as f:
    content = f.read()

old_env = """FRED_API_KEY = os.environ.get("FRED_API_KEY", "")"""

new_env = """from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")"""

if old_env in content:
    content = content.replace(old_env, new_env, 1)
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully added load_dotenv to main.py")
else:
    print("Could not find old_env block.")
