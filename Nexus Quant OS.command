#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Nexus Quant OS — Double-Click Launcher
# ═══════════════════════════════════════════════════════════════
#
# Double-click this file to start the Nexus Quant OS dashboard.
# It will:
#   1. Load environment variables from .env
#   2. Start the FastAPI server on port 8000
#   3. Open the dashboard in your default browser
#
# To stop: close this terminal window or press Ctrl+C
# ═══════════════════════════════════════════════════════════════

# Navigate to project directory (where this script lives)
cd "$(dirname "$0")"

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   🧠 Nexus Quant OS v3.0                 ║"
echo "  ║   Autonomous Quantitative Trading System  ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# Check Python venv exists
if [ ! -d ".venv" ]; then
    echo "  ❌ Python venv not found at .venv/"
    echo "     Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    read -p "  Press Enter to exit..."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "  ✅ Environment loaded (.env)"
else
    echo "  ⚠️  No .env file found — some features may not work"
fi

# Check if port 8000 is already in use
if lsof -i :8000 -sTCP:LISTEN > /dev/null 2>&1; then
    echo "  ℹ️  Server already running on port 8000"
    echo "  🌐 Opening dashboard..."
    open http://localhost:8000
    exit 0
fi

# Start server in background
echo "  🚀 Starting server on http://localhost:8000 ..."
PYTHONPATH="$(pwd)" .venv/bin/python -m uvicorn api.server:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level warning &

SERVER_PID=$!

# Wait for server to be ready
echo "  ⏳ Waiting for server..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo "  ✅ Server ready!"
        break
    fi
    sleep 1
done

# Open browser
echo "  🌐 Opening dashboard in browser..."
open http://localhost:8000

echo ""
echo "  ─────────────────────────────────────────────"
echo "  Dashboard: http://localhost:8000"
echo "  Press Ctrl+C to stop the server"
echo "  ─────────────────────────────────────────────"
echo ""

# Keep script running (shows server logs)
wait $SERVER_PID
