import sys
sys.path.insert(0, "/Users/coolguy/developer/nexus_quant_os")
from run_moomoo_trade import run_pipeline_and_get_weights

weights, metadata, report = run_pipeline_and_get_weights()
print("\n--- TEST SCRIPT RESULTS ---")
print("Target Weights:", weights)
print("Metadata:", metadata)
