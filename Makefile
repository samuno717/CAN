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

