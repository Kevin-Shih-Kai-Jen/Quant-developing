#!/bin/bash
echo "Installing Web UI dependencies..."
pip install -r requirements.txt

echo "Starting Nexus Quant OS API Server..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
