#!/bin/bash
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH

echo "=== Nexus Quant OS Daily Execution ==="
echo "1. Checking for lingering processes on port 8080..."
# Find and kill any process listening on 8080
PIDS=$(lsof -t -i:8080)
if [ ! -z "$PIDS" ]; then
  echo "Killing processes on port 8080: $PIDS"
  kill -9 $PIDS
else
  echo "Port 8080 is clear."
fi

echo "2. Starting Docker containers..."
cd /Users/coolguy/developer/nexus_quant_os
docker compose up -d api

echo "3. Waiting for API to become ready..."
for i in {1..15}; do
  if curl -s http://localhost:8080/ > /dev/null; then
    echo "API is up!"
    break
  fi
  echo "Waiting..."
  sleep 2
done

echo "4. Triggering daily pipeline execution..."
curl -X POST http://localhost:8080/api/run_pipeline

echo ""
echo "=== Daily Run Complete! ==="
