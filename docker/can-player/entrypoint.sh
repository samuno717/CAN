#!/usr/bin/env bash
set -euo pipefail

# Configuration via environment variables (or CLI overrides)
SCENARIO="${SCENARIO:-multi}"
CAN_IFACE="${CAN_IFACE:-vcan0}"
LOG_IFACE="${LOG_IFACE:-can0}"
LOOPS="${LOOPS:-1}"                         # Default '1' for full pass of the test dataset ('i' for infinite loop)
GAP_MS="${GAP_MS:-1}"                       # Timestamp gap mode (-g) or default timing
VERBOSE="${VERBOSE:-0}"                     # Set to '1' to print every sent frame (-v)
SKIP_VCAN_SETUP="${SKIP_VCAN_SETUP:-0}"
CUSTOM_LOG_PATH="${CUSTOM_LOG_PATH:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"               # Default 0 = play the full test split log without artificial truncation
START_DELAY_SEC="${START_DELAY_SEC:-3}"
SPEED="${SPEED:-1}"                         # Replay speed factor: timestamps are divided by SPEED before playback

usage() {
    cat <<EOF
Usage: entrypoint.sh [OPTIONS]

Options / Environment Variables:
  --scenario, -s <name>     Scenario to stream: raw | dos | replay | payload | frequency | fuzzing | suspension | multi
                            (env: SCENARIO, default: multi)
  --file, -f <path>         Explicit path to a candump .log file (overrides --scenario)
                            (env: CUSTOM_LOG_PATH)
  --interface, -i <iface>   Target SocketCAN interface on host (env: CAN_IFACE, default: vcan0)
  --log-iface <iface>       Source interface recorded inside the log file (env: LOG_IFACE, default: can0)
  --loops, -l <count>       Number of loops to play, or 'i' for infinite loop (env: LOOPS, default: 1)
  --max-frames <count>      Optional frame limit (env: MAX_FRAMES, default: 0 = full test split)
  --verbose, -v             Print transmitted CAN frames to stdout (env: VERBOSE=1)
  --help, -h                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--scenario)
            SCENARIO="$2"
            shift 2
            ;;
        -f|--file)
            CUSTOM_LOG_PATH="$2"
            shift 2
            ;;
        -i|--interface)
            CAN_IFACE="$2"
            shift 2
            ;;
        --log-iface)
            LOG_IFACE="$2"
            shift 2
            ;;
        -l|--loops)
            LOOPS="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# 1. Resolve candump log file path
if [[ -n "${CUSTOM_LOG_PATH}" ]]; then
    LOG_FILE="${CUSTOM_LOG_PATH}"
else
    case "${SCENARIO,,}" in
        raw|clean)
            if [[ -f "/workspace/data/processed/attacks/raw_attack.log" ]]; then
                LOG_FILE="/workspace/data/processed/attacks/raw_attack.log"
            else
                # Fallback: slice the last 15% (test split = 886,593 frames) of the raw candump file
                RAW_SRC=$(find /workspace/data/raw -maxdepth 1 -name "candump-*.log" | sort | tail -n 1)
                LOG_FILE="/tmp/test_split_clean.log"
                echo "[INFO] Extracting 15% test split (last 886,593 frames) from ${RAW_SRC}..."
                tail -n 886593 "${RAW_SRC}" > "${LOG_FILE}"
            fi
            ;;
        dos|replay|payload|frequency|fuzzing|suspension|multi)
            LOG_FILE="/workspace/data/processed/attacks/${SCENARIO,,}_attack.log"
            ;;
        *)
            echo "[ERROR] Unknown scenario '${SCENARIO}'." >&2
            echo "Allowed scenarios: raw, dos, replay, payload, frequency, fuzzing, suspension, multi" >&2
            exit 1
            ;;
    esac
fi

if [[ -z "${LOG_FILE}" || ! -f "${LOG_FILE}" ]]; then
    echo "[ERROR] Log file not found: '${LOG_FILE:-<empty>}'" >&2
    echo "Make sure data volume is mounted to /workspace/data and attack logs are generated." >&2
    exit 1
fi

# 2. Setup virtual CAN interface (if using vcan* and not skipped)
if [[ "${SKIP_VCAN_SETUP}" != "1" && "${CAN_IFACE}" == vcan* ]]; then
    if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
        echo "[INFO] Interface '${CAN_IFACE}' does not exist. Creating virtual CAN interface..."
        modprobe vcan 2>/dev/null || echo "[WARN] Could not run 'modprobe vcan' (module may already be built-in or loaded on host)."
        ip link add dev "${CAN_IFACE}" type vcan
    fi
    ip link set up "${CAN_IFACE}"
    echo "[INFO] Interface '${CAN_IFACE}' is UP."
fi

PLAY_FILE="${LOG_FILE}"
if [[ "${MAX_FRAMES}" =~ ^[0-9]+$ ]] && [[ "${MAX_FRAMES}" -gt 0 ]]; then
    PLAY_FILE="/tmp/active_${SCENARIO,,}.log"
    head -n "${MAX_FRAMES}" "${LOG_FILE}" > "${PLAY_FILE}"
    echo "[INFO] Sliced first ${MAX_FRAMES} frames into ${PLAY_FILE}"
fi

if [[ "${SPEED}" != "1" && "${SPEED}" != "1.0" ]]; then
    SCALED_FILE="/tmp/speed_${SCENARIO,,}.log"
    # Keep relative timing only (canplayer uses timestamp differences), divided by SPEED
    awk -v s="${SPEED}" '{
        ts = substr($1, 2, length($1) - 2) + 0.0;
        if (NR == 1) t0 = ts;
        printf "(%.6f) %s %s\n", 1000.0 + (ts - t0) / s, $2, $3
    }' "${PLAY_FILE}" > "${SCALED_FILE}"
    PLAY_FILE="${SCALED_FILE}"
    echo "[INFO] Replay speed x${SPEED}: time-scaled log written to ${PLAY_FILE}"
fi

FRAME_COUNT=$(wc -l < "${PLAY_FILE}" | tr -d ' ')
FIRST_TS=$(head -n 1 "${PLAY_FILE}" | awk '{gsub(/[()]/, "", $1); print $1}')
LAST_TS=$(tail -n 1 "${PLAY_FILE}" | awk '{gsub(/[()]/, "", $1); print $1}')

echo "============================================================"
echo " CAN Bus Stream Player (can-utils / SocketCAN)"
echo "------------------------------------------------------------"
echo " Scenario      : ${SCENARIO}"
echo " Log File      : ${PLAY_FILE} (source: ${LOG_FILE})"
echo " Total Frames  : ${FRAME_COUNT}"
echo " Time Span     : ${FIRST_TS} -> ${LAST_TS}"
echo " Mapping       : ${CAN_IFACE}=${LOG_IFACE} (Host <- Log)"
echo " Loops         : ${LOOPS} ('1' = single full test pass, 'i' = infinite)"
echo " Speed         : x${SPEED}"
echo "============================================================"

if [[ "${START_DELAY_SEC}" -gt 0 ]]; then
    echo "[INFO] Waiting ${START_DELAY_SEC}s for can-ids listener to initialize..."
    sleep "${START_DELAY_SEC}"
fi

# do not pass '-x' (this disables local SocketCAN loopback to other processes on the host)
CANPLAYER_ARGS=("-I" "${PLAY_FILE}" "-l" "${LOOPS}")
if [[ "${VERBOSE}" == "1" ]]; then
    CANPLAYER_ARGS+=("-v")
fi

# Map interface: <target_host_iface>=<source_log_iface>
CANPLAYER_ARGS+=("${CAN_IFACE}=${LOG_IFACE}")

echo "[INFO] Starting stream: canplayer ${CANPLAYER_ARGS[*]}"
exec canplayer "${CANPLAYER_ARGS[@]}"
