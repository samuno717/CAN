smoke:
	python -m scripts.dvc_stages.train training.epochs=1

# --- Docker / Raspberry Pi 5 SocketCAN Stream & Real-Time IDS Targets ---
vcan-up:
	bash docker/can-player/setup_vcan_host.sh vcan0

can-stream:
	SCENARIO=$${SCENARIO:-multi} LOOPS=$${LOOPS:-1} docker compose up --build can-player

can-stream-loop:
	SCENARIO=$${SCENARIO:-multi} LOOPS=i docker compose up --build can-player

can-monitor:
	docker compose --profile monitor run --rm can-monitor

# Start full Real-Time IDS stack (can-player + can-ids + Prometheus + Grafana)
ids-up:
	MODEL_TYPE=$${MODEL_TYPE:-ae_lstm} SCENARIO=$${SCENARIO:-multi} LOOPS=$${LOOPS:-i} IDS_STRIDE=$${IDS_STRIDE:-1} docker compose up --build

# Shortcuts for comparing AE-LSTM vs Isolation Forest on Raspberry Pi 5
ids-ae:
	MODEL_TYPE=ae_lstm SCENARIO=$${SCENARIO:-multi} LOOPS=$${LOOPS:-i} IDS_STRIDE=$${IDS_STRIDE:-1} docker compose up --build

ids-iforest:
	MODEL_TYPE=iforest SCENARIO=$${SCENARIO:-multi} LOOPS=$${LOOPS:-i} IDS_STRIDE=$${IDS_STRIDE:-1} docker compose up --build

ids-down:
	docker compose down

# --- Online experiments on Raspberry Pi 5 (see scripts/online/README.md) ---
# 1. Detection quality online: every scenario, both models, stride 1, real-time replay on vcan0
exp-main:
	RUN_TAG=main STRIDES=1 REPS=1 bash scripts/online/run_experiments.sh

# 2. Performance with repetitions (clean traffic + multi-attack), stride 1 and 5
exp-perf:
	RUN_TAG=perf SCENARIOS="raw multi" STRIDES="1 5" REPS=3 bash scripts/online/run_experiments.sh

# 3. Stress test: accelerated replay until the IDS stops keeping up
exp-stress:
	RUN_TAG=stress SCENARIOS=raw SPEEDS="2 4 8 16" STRIDES="1 5" REPS=1 bash scripts/online/run_experiments.sh

# 4. Pure processing throughput (no SocketCAN, no pacing) + online/offline consistency source
exp-bench:
	RUN_TAG=bench MODE=file IDS_PACING=0 SCENARIOS="raw multi" STRIDES=1 REPS=1 WITH_MONITORING=0 bash scripts/online/run_experiments.sh

exp-eval:
	docker compose run --rm --no-deps --entrypoint python can-ids -m scripts.online.evaluate_online --all

# On the PC after copying reports/online from the Pi
exp-aggregate:
	uv run python -m scripts.online.aggregate_online
