#!/bin/bash
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH

echo "=== Nexus Quant OS Daily Execution ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Step 1: Clean up stale processes ──────────────────────────────────
echo "1. Checking for lingering processes on port 8080..."
PIDS=$(lsof -t -i:8080)
if [ ! -z "$PIDS" ]; then
  echo "Killing processes on port 8080: $PIDS"
  kill -9 $PIDS
else
  echo "Port 8080 is clear."
fi

# ── Step 2: Start Docker containers ───────────────────────────────────
echo ""
echo "2. Starting Docker containers..."
cd /Users/coolguy/developer/nexus_quant_os
docker compose up -d api

# ── Step 3: Wait for API readiness ────────────────────────────────────
echo ""
echo "3. Waiting for API to become ready..."
for i in {1..15}; do
  if curl -s http://localhost:8080/ > /dev/null; then
    echo "API is up!"
    break
  fi
  echo "Waiting... ($i/15)"
  sleep 2
done

# ── Step 4: Run inference pipeline ────────────────────────────────────
echo ""
echo "4. Triggering daily pipeline execution..."
PIPELINE_RESULT=$(curl -s -X POST http://localhost:8080/api/run_pipeline)
PIPELINE_STATUS=$(echo "$PIPELINE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null)

if [ "$PIPELINE_STATUS" != "success" ]; then
  echo "❌ Pipeline failed! Result:"
  echo "$PIPELINE_RESULT"
  echo "=== Daily Run FAILED ==="
  exit 1
fi
echo "✅ Pipeline succeeded!"

# ── Step 5: Execute simulated trades ──────────────────────────────────
echo ""
echo "5. Executing simulated trades..."
TRADE_RESULT=$(curl -s -X POST http://localhost:8080/api/execute_trades \
  -H "Content-Type: application/json" \
  -d "$PIPELINE_RESULT")
echo "$TRADE_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   Status: {data.get('status', 'unknown')}\")
if 'trades' in data:
    for t in data['trades']:
        print(f\"   {t.get('side','?')} {t.get('symbol','?')}: {t.get('qty',0)} shares @ \${t.get('filled_price',0):.2f}\")
if 'portfolio_value' in data:
    print(f\"   💰 Portfolio Value: \${data['portfolio_value']:,.2f}\")
" 2>/dev/null

# ── Step 6: Record daily portfolio snapshot ───────────────────────────
echo ""
echo "6. Recording portfolio snapshot..."
curl -s http://localhost:8080/api/portfolio | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   💰 Equity: \${data.get('equity', 0):,.2f}\")
print(f\"   💵 Cash: \${data.get('cash', 0):,.2f}\")
if 'positions' in data:
    for p in data['positions']:
        pnl = p.get('unrealized_pl', 0)
        pnl_str = f'+\${pnl:,.2f}' if pnl >= 0 else f'-\${abs(pnl):,.2f}'
        print(f\"   {p['symbol']}: {p['qty']} shares ({pnl_str})\")
" 2>/dev/null

echo ""
echo "=== Daily Run Complete! ==="
