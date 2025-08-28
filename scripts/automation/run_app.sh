#!/bin/bash
set -euo pipefail

# Kill existing Web app processes (best-effort)
echo "Stopping existing Web App processes..."
pkill -f "vite" || true
fuser -k 5173/tcp || true
sleep 0.5

# Determine a usable LAN IP (prefer 192.168.x.x, then 10.x.x.x, then 172.16-31)
LAN_IP=${LAN_IP:-}
if [ -z "${LAN_IP}" ]; then
	# Try to detect via hostname -I
	for ip in $(hostname -I 2>/dev/null || true); do
		if [[ $ip =~ ^192\.168\.[0-9]+\.[0-9]+$ ]]; then LAN_IP=$ip; break; fi
	done
fi
if [ -z "${LAN_IP}" ]; then
	for ip in $(hostname -I 2>/dev/null || true); do
		if [[ $ip =~ ^10\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then LAN_IP=$ip; break; fi
	done
fi
if [ -z "${LAN_IP}" ]; then
	for ip in $(hostname -I 2>/dev/null || true); do
		if [[ $ip =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]+\.[0-9]+$ ]]; then LAN_IP=$ip; break; fi
	done
fi
LAN_IP=${LAN_IP:-127.0.0.1}

# Start the Web App
echo "Building and starting Web App..."
cd /home/mat/Downloads/pathfind_features/web
npm run build
export VITE_API_URL="http://${LAN_IP}:8000"
echo "VITE_API_URL=${VITE_API_URL}"
npm run dev -- --host 0.0.0.0 --port 5173
