#!/bin/bash
set -euo pipefail

# Kill existing API processes (best-effort)
echo "Stopping existing API processes..."
pkill -f "uvicorn api.server:app" || true
fuser -k 8000/tcp || true
sleep 0.5

# Start the API server – bind to all interfaces
API_CMD="/home/mat/Downloads/pathfind_features/.venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload"
echo "Starting API: $API_CMD"
$API_CMD
