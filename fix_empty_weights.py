import re

file_path = "/Users/coolguy/developer/nexus_quant_os/run_moomoo_trade.py"
with open(file_path, "r") as f:
    content = f.read()

old_exit = """        if not target_weights:
            print("\\n  ⚠️ No target weights — exiting")
            report.error = "No target weights produced"
            notifier.send_alert(
                title="Pipeline Warning",
                message="Pipeline produced no target weights.",
                severity="warning",
            )
            sys.exit(1)"""

new_exit = """        # If target_weights is empty, it means the firewall ordered 100% cash.
        # We MUST proceed to execute_on_moomoo to actually sell existing positions!
        if not target_weights:
            print("\\n  ⚠️ Target weights empty (100% Cash mode triggered)")
"""

if old_exit in content:
    content = content.replace(old_exit, new_exit, 1)
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully removed the faulty abort logic.")
else:
    print("Could not find the abort logic.")
