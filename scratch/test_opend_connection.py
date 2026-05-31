import sys
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

try:
    from nexus_quant_os.execution.futu_broker import FutuBroker
    print("SUCCESS: Imported FutuBroker module successfully.")
except Exception as e:
    print(f"FAILED: Could not import FutuBroker module. Error: {e}")
    sys.exit(1)

try:
    # Attempt to initialize FutuBroker which does a verification connection
    print("Attempting to connect to FutuOpenD at 127.0.0.1:11111...")
    broker = FutuBroker(host="127.0.0.1", port=11111)
    print("\n🎉 SUCCESS: Connected to FutuOpenD successfully!")
    sys.exit(0)
except ConnectionError as ce:
    print(f"\n❌ FAILED: ConnectionError. Details:\n{ce}")
    sys.exit(2)
except Exception as e:
    print(f"\n❌ FAILED: Unexpected error. Details:\n{e}")
    sys.exit(3)
