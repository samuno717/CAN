#!/usr/bin/env bash
# =====================================================================================
#  Automated online IDS experiments on Raspberry Pi 5 (vcan0 + Docker Compose)
#
#  For every combination of  STRIDES x SPEEDS x REPS x MODELS x SCENARIOS  it:
#    1. starts can-ids with recording enabled (RECORD_RUN=1, unique RUN_ID)
#    2. waits until the model is loaded (Prometheus endpoint up)
#    3. replays the scenario log once with can-player (canplayer -> vcan0)
#    4. waits until the IDS stops by itself (IDLE_EXIT_SEC after the last frame)
#    5. exports Prometheus series of the run and cools the CPU down
#  Finally it evaluates all new runs (scripts/online/evaluate_online.py).
#
#  Results: reports/online/<RUN_ID>/{config,summary,evaluation}.json, windows.csv.gz, system.csv ...
#
#  Usage (from the repository root on the Pi):
#    bash scripts/online/run_experiments.sh
#    MODELS="ae_lstm" SCENARIOS="multi" REPS=1 bash scripts/online/run_experiments.sh
#    RUN_TAG=stress SCENARIOS=raw SPEEDS="1 2 4 8" REPS=1 bash scripts/online/run_experiments.sh
#    RUN_TAG=bench MODE=file IDS_PACING=0 SCENARIOS=raw bash scripts/online/run_experiments.sh
# =====================================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
# Every "docker compose run" below gets stdin from /dev/null: with the terminal attached as stdin
# it can keep running after the container has exited and the script never continues.

MODELS="${MODELS:-ae_lstm iforest}"
SCENARIOS="${SCENARIOS:-raw dos replay payload frequency fuzzing suspension multi}"
REPS="${REPS:-1}"
STRIDES="${STRIDES:-1}"
SPEEDS="${SPEEDS:-1}"
MODE="${MODE:-socketcan}"            # socketcan: can-player -> vcan0 -> can-ids | file: can-ids reads the log itself
IDS_PACING="${IDS_PACING:-1}"        # MODE=file only: 0 = as fast as possible (throughput benchmark)
RUN_TAG="${RUN_TAG:-main}"
CAN_IFACE="${CAN_IFACE:-vcan0}"
WITH_MONITORING="${WITH_MONITORING:-1}"   # 1 = keep Prometheus + Grafana running and export series per run
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"
IDLE_EXIT_SEC="${IDLE_EXIT_SEC:-5}"
MAX_RUN_SEC="${MAX_RUN_SEC:-5400}"
METRICS_PORT="${METRICS_PORT:-8000}"
BUILD="${BUILD:-1}"
EVALUATE="${EVALUATE:-1}"

export CAN_IFACE RUN_TAG IDLE_EXIT_SEC METRICS_PORT
export RECORD_RUN=1 LOOPS=1

LOG_DIR="reports/online/_logs"
mkdir -p "${LOG_DIR}"   # created by the host user so host-side logs can be written next to root-owned run dirs

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ "${BUILD}" == "1" ]]; then
    log "Building images (model weights, scalers and thresholds are baked into can-ids)..."
    docker compose build can-ids can-player
fi

if [[ "${MODE}" == "socketcan" && "${CAN_IFACE}" == vcan* ]]; then
    if ! ip link show "${CAN_IFACE}" >/dev/null 2>&1; then
        bash docker/can-player/setup_vcan_host.sh "${CAN_IFACE}"
    fi
fi

if [[ "${WITH_MONITORING}" == "1" ]]; then
    docker compose up -d prometheus grafana
fi

wait_for_ids() {
    local deadline=$((SECONDS + 300))
    until python3 -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:${METRICS_PORT}/metrics', timeout=2)" 2>/dev/null; do
        if (( SECONDS > deadline )); then
            log "ERROR: can-ids did not start within 300 s"; docker logs --tail 50 can-bus-ids || true; return 1
        fi
        sleep 1
    done
}

run_one() {
    local model="$1" scenario="$2" stride="$3" speed="$4" rep="$5"
    export MODEL_TYPE="${model}" SCENARIO="${scenario}" IDS_STRIDE="${stride}" SPEED="${speed}"
    export RUN_ID="$(date +%Y%m%d-%H%M%S)_${RUN_TAG}_${model}_${scenario}_s${stride}_x${speed}_r${rep}"
    log ">>> ${RUN_ID}"

    if [[ "${MODE}" == "file" ]]; then
        IDS_SOURCE=file IDS_PACING="${IDS_PACING}" timeout "${MAX_RUN_SEC}" \
            docker compose run -T --rm --no-deps can-ids < /dev/null > "${LOG_DIR}/${RUN_ID}.ids.log" 2>&1 || log "WARN: IDS exited with error"
    else
        IDS_SOURCE=socketcan docker compose up -d --no-deps --force-recreate can-ids
        wait_for_ids
        timeout "${MAX_RUN_SEC}" docker compose run -T --rm --no-deps can-player < /dev/null > "${LOG_DIR}/${RUN_ID}.player.log" 2>&1 \
            || log "WARN: can-player exited with error (see ${LOG_DIR}/${RUN_ID}.player.log)"
        # The IDS stops by itself IDLE_EXIT_SEC after the last frame (and writes summary.json)
        if ! timeout $((IDLE_EXIT_SEC + 180)) docker wait can-bus-ids >/dev/null; then
            log "WARN: IDS did not stop by itself - sending SIGTERM"
            docker stop -t 30 can-bus-ids >/dev/null || true
        fi
        docker logs can-bus-ids > "${LOG_DIR}/${RUN_ID}.ids.log" 2>&1 || true
    fi

    if [[ "${WITH_MONITORING}" == "1" ]]; then
        sleep 3   # let Prometheus finish the last scrape window
        docker compose run -T --rm --no-deps --entrypoint python can-ids \
            scripts/online/export_prometheus.py --run-dir "reports/online/${RUN_ID}" < /dev/null || log "WARN: Prometheus export failed"
    fi

    log "<<< ${RUN_ID} done, cooling down ${COOLDOWN_SEC}s"
    sleep "${COOLDOWN_SEC}"
}

TOTAL=0
for _ in ${STRIDES}; do for _ in ${SPEEDS}; do for _ in $(seq 1 "${REPS}"); do for _ in ${MODELS}; do for _ in ${SCENARIOS}; do
    TOTAL=$((TOTAL + 1)); done; done; done; done; done
log "Planned runs: ${TOTAL} (models: ${MODELS} | scenarios: ${SCENARIOS} | reps: ${REPS} | strides: ${STRIDES} | speeds: ${SPEEDS} | mode: ${MODE})"

# Repetitions are the outer loop so that slow drifts (temperature, background load) are spread over all configurations
for stride in ${STRIDES}; do
    for speed in ${SPEEDS}; do
        for rep in $(seq 1 "${REPS}"); do
            for model in ${MODELS}; do
                for scenario in ${SCENARIOS}; do
                    run_one "${model}" "${scenario}" "${stride}" "${speed}" "${rep}"
                done
            done
        done
    done
done

if [[ "${EVALUATE}" == "1" ]]; then
    log "Evaluating new runs..."
    docker compose run -T --rm --no-deps --entrypoint python can-ids -m scripts.online.evaluate_online --all < /dev/null \
        | tee "${LOG_DIR}/evaluate_$(date +%Y%m%d-%H%M%S).txt"
fi
log "All experiments finished. Copy reports/online/ to the PC and run: uv run python -m scripts.online.aggregate_online"
