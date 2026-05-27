#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_scheduled.sh — Nexus Quant OS Automated Daily Execution
# ═══════════════════════════════════════════════════════════════════
#
# Called by macOS launchd on schedule.  Handles:
#   1. Environment setup (.env loading)
#   2. Virtual environment activation
#   3. OpenD connectivity check
#   4. Pipeline execution with Discord notifications
#   5. Error alerting on failure
#
# Usage (manual test):
#   bash /Users/coolguy/developer/nexus_quant_os/scripts/run_scheduled.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="/Users/coolguy/developer/nexus_quant_os"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/.venv"
LOG_FILE="${LOG_DIR}/scheduled_$(date '+%Y%m%d_%H%M%S').log"
ENV_FILE="${PROJECT_DIR}/.env"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# ── Logging ────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1

echo "═══════════════════════════════════════════════════════════"
echo "  🕐 Nexus Quant OS — Scheduled Run"
echo "  📅 $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Load environment variables ─────────────────────────────────────
if [ -f "$ENV_FILE" ]; then
    echo "  ▸ Loading .env..."
    set -a
    source "$ENV_FILE"
    set +a
    echo "    ✅ Environment loaded"
else
    echo "    ❌ .env not found at $ENV_FILE"
    exit 1
fi

# ── Activate virtual environment ───────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "  ▸ Activating venv..."
    source "${VENV_DIR}/bin/activate"
    echo "    ✅ Python: $(python3 --version)"
else
    echo "    ❌ venv not found at $VENV_DIR"
    exit 1
fi

# ── Check OpenD connectivity ──────────────────────────────────────
echo "  ▸ Checking FutuOpenD on 127.0.0.1:11111..."
if nc -z 127.0.0.1 11111 2>/dev/null; then
    echo "    ✅ OpenD is running"
else
    echo "    ⚠️  OpenD not detected — pipeline will still run"
    echo "       (Moomoo execution step will be skipped)"
fi

# ── Execute pipeline ─────────────────────────────────────────────
echo ""
echo "  ▸ Running pipeline..."
cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR" python3 run_moomoo_trade.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "  ✅ Scheduled run completed successfully"
else
    echo ""
    echo "  ❌ Pipeline failed with exit code $EXIT_CODE"
    # Discord error alert is handled inside run_moomoo_trade.py
fi

# ── Cleanup old logs (keep last 30 days) ──────────────────────────
find "$LOG_DIR" -name "scheduled_*.log" -mtime +30 -delete 2>/dev/null || true

echo ""
echo "  📄 Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"

exit $EXIT_CODE
