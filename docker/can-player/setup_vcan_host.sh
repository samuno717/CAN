#!/usr/bin/env bash
# Helper script to initialize vcan0 on Raspberry Pi / Linux host before running containers
set -euo pipefail

IFACE="${1:-vcan0}"

echo "[INFO] Loading 'vcan' kernel module..."
sudo modprobe vcan

if ! ip link show "${IFACE}" >/dev/null 2>&1; then
    echo "[INFO] Creating virtual CAN interface '${IFACE}'..."
    sudo ip link add dev "${IFACE}" type vcan
fi

sudo ip link set up "${IFACE}"
echo "[SUCCESS] SocketCAN interface '${IFACE}' is UP and ready:"
ip -details link show "${IFACE}"
