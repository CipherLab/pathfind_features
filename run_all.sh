#!/bin/bash
set -euo pipefail

# Kill existing API and CLI processes (best-effort)
pkill -f "uvicorn api.server:app" || true
pkill -f "tsx index.jsx" || true
pkill -f "babel-node index.js" || true
fuser -k 8000/tcp || true
sleep 0.5

# Start the API server (background)
API_CMD="/home/mat/Downloads/pathfind_features/.venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload"
echo "Starting API: $API_CMD"
$API_CMD &
API_PID=$!

# Ensure API is killed when this script exits
cleanup() {
	echo "Stopping API (PID $API_PID)..."
	kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Start the CLI (foreground) using tsx via npm script
echo "Starting CLI (PATHFIND_API=http://127.0.0.1:8000)..."
cd /home/mat/Downloads/pathfind_features/cli
PATHFIND_API=http://127.0.0.1:8000 npm start