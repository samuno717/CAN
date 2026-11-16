# CAN Bus Anomaly Detection & Real-Time IDS (AE-LSTM vs Isolation Forest)

Unsupervised Intrusion Detection System (IDS) for Controller Area Network (CAN) traffic, designed and benchmarked for **Raspberry Pi 5 (8 GB)** on Linux (`SocketCAN`).

The main goal of this project is to detect anomalies and attacks on the CAN bus without relying on labeled attack data during training. The models are trained strictly on normal driving traffic (`candump` logs) and evaluated against synthetic attack scenarios injected into the test split, as well as in a **live streaming Docker + Prometheus + Grafana IDS stack**.

I built this pipeline to compare a sequence-aware deep learning approach (**LSTM Autoencoder**) against a classical baseline (**Isolation Forest**) under identical windowing and thresholding conditions.

---

## How It Works

### 1. Data & Feature Extraction
Raw SocketCAN logs (`data/raw/candump-*.log`) are parsed into structured CSVs and split chronologically into **70% train / 15% validation / 15% test** (no shuffling, to avoid data leakage across time windows).

Each CAN frame is converted into a **26-dimensional feature vector**:
- `delta_t` – time elapsed since the previous frame of the same ECU ID on the bus (MinMax scaled)
- `byte1` .. `byte8` – 8 payload bytes converted from hex to `0–255` (MinMax scaled)
- 17 one-hot encoded columns for the 17 distinct ECU arbitration IDs present in the traffic

### 2. Models
Frames are grouped into sliding windows of length `64` (stride `5` during training, stride `1` during evaluation).

* **LSTM Autoencoder (`AELSTM`)**: 
  Encodes a `[64, 26]` window into a 16-dimensional latent bottleneck (`hidden_size=64 -> latent_dim=16`) and reconstructs the sequence. The anomaly score for a window is its Mean Squared Error (MSE).
* **Isolation Forest (`CANIsolationForest`)**: 
  Since tree models don't take 3D tensors directly, each `[64, 26]` window is summarized into `[mean, std, min, max]` per feature (`104` features total) and passed to an ensemble of 200 trees.

### 3. Thresholding & Persistence Filter
The anomaly threshold is calibrated on the clean validation set as the **99.9th percentile** of the validation scores (`0.04144` MSE for AE-LSTM, `0.14315` for Isolation Forest).

Because a 99.9% threshold still produces ~0.1% false alarms on benign traffic (which adds up quickly on a high-speed bus), I added a **persistence filter (`k = 10`)**: an alarm is only triggered when `k` consecutive sliding windows exceed the threshold.

---

## Simulated Attack Scenarios

Attacks are generated only on the **15% test split** (`src/utils/attacks/`) so the scalers and thresholds never see them beforehand:

1. **DoS (`dos`)** – floods the bus with high-priority `0x002` frames every `0.25 ms`, shifting existing traffic timestamps.
2. **Replay (`replay`)** – captures a block of legitimate traffic and re-injects it later on the bus.
3. **Payload / Masquerade (`payload`)** – overwrites payload bytes of an existing ECU (`0x130`) in-place (random bytes, inversion, constant value, or slow drift) without changing frame timing.
4. **Frequency (`frequency`)** – increases transmission frequency of target ECU `0x130` (5x normal rate).
5. **Fuzzing (`fuzzing`)** – injects frames with random 11-bit CAN IDs and random 8-byte payloads at short intervals.
6. **Suspension (`suspension`)** – silences a specific ECU (`0x130`) for a period of time (simulating a Bus-Off state or UDS `CommunicationControl` kill).
7. **Multi-attack (`multi`)** – sequential trace containing all 6 attacks separated by normal traffic segments.

---

## Results (Test Split, Persistence `k=10`)

Metrics below come from `reports/metrics/model_comparison.json`:

| Scenario | AE-LSTM Precision | AE-LSTM Recall | AE-LSTM F1 | IForest Precision | IForest Recall | IForest F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 0.908 | 0.988 | 0.946 | 1.000 | 0.968 | **0.984** |
| **Replay** | 0.999 | 0.999 | **0.999** | 0.000 | 0.000 | 0.000 |
| **Payload** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Frequency** | 0.987 | 0.990 | **0.988** | 1.000 | 0.442 | 0.613 |
| **Fuzzing** | 0.988 | 0.988 | **0.988** | 1.000 | 0.957 | 0.978 |
| **Suspension** | 0.983 | 0.992 | **0.987** | 1.000 | 0.549 | 0.709 |
| **Multi (Combined)** | 0.988 | 0.840 | **0.908** | 1.000 | 0.518 | 0.682 |

- **Why both models miss the in-place Payload attack:** In the `payload` scenario, only a single ECU's bytes (`0x130`) are randomized in-place while bus timing (`delta_t`) and ID order remain 100% intact. Because 17 one-hot ID columns and `delta_t` dominate the 26-feature MSE average, the error spike on those few frames stays below the global 99.9% threshold. Detecting pure masquerade attacks would require per-ECU models or weighting payload reconstruction loss higher than ID loss.

![Model Comparison](plots/model_comparison_ae_lstm_vs_iforest.svg)

---

## Real-Time CAN IDS (Docker + Prometheus + Grafana on Raspberry Pi 5)

The repository includes a complete real-time Intrusion Detection System (`src/ids/`) and a 4-container Docker Compose stack optimized for Linux / Raspberry Pi 5 (`vcan0` / `can0`):

1. **`can-player`** – replays any chosen scenario (`raw`, `dos`, `replay`, `payload`, `frequency`, `fuzzing`, `suspension`, `multi`) onto a SocketCAN interface (`vcan0` or `can0`) using `canplayer`.
2. **`can-ids`** – sniffs raw SocketCAN frames, performs O(1) stateful feature scaling (`StreamingFeatureExtractor`), evaluates 64-frame sliding windows in micro-batches using either **AE-LSTM** or **Isolation Forest** (`IDSDetector`), applies the `k=10` persistence filter, logs confirmed incidents to `reports/live_alerts.jsonl`, and exposes Prometheus metrics on `:8000/metrics`.
3. **`prometheus`** – scrapes `can-ids` telemetry every `1s` on `:9090`.
4. **`grafana`** – serves a pre-provisioned live dashboard (`CAN Bus Real-Time IDS`) on `http://localhost:3000` (anonymous access enabled, default login `admin` / `admin`).

### Launching the Live IDS Stack on Linux / Raspberry Pi 5

```bash
# 1. Run with AE-LSTM (default) and the combined 'multi' attack scenario
MODEL_TYPE=ae_lstm SCENARIO=multi docker compose up --build

# 2. Run with Isolation Forest for side-by-side comparison
MODEL_TYPE=iforest SCENARIO=multi docker compose up --build

# 3. Test a specific attack scenario (e.g. dos, replay, payload, frequency, fuzzing, suspension, raw)
MODEL_TYPE=ae_lstm SCENARIO=dos docker compose up

# 4. Adjust sliding window stride (e.g. IDS_STRIDE=5) or target physical CAN HAT (CAN_IFACE=can0)
MODEL_TYPE=ae_lstm SCENARIO=fuzzing IDS_STRIDE=5 CAN_IFACE=vcan0 docker compose up
```

Or using `Makefile` shortcuts:
```bash
make ids-ae SCENARIO=multi
make ids-iforest SCENARIO=multi
```

Open **`http://localhost:3000`** (or `http://<raspberry-pi-ip>:3000`) to view the live Grafana dashboard:
- **Security Status Banner:** turns from green (`SECURE (NORMAL TRAFFIC)`) to red (`INTRUSION DETECTED!`) when $k=10$ consecutive windows exceed the calibrated threshold.
- **Real-Time Anomaly Score vs 99.9% Threshold:** live curve of window MSE / IForest score, micro-batch peak, and 50-window rolling mean.
- **Diagnostic Feature Breakdown:** real-time separation of Timing Error (`delta_t`), Payload Error (`byte1..byte8`), and Arbitration ID Error (`can_id_*`).
- **Raspberry Pi 5 Benchmark Panels:** live per-window inference latency (`ms`), bus frame rate (`frames/s`), window throughput (`windows/s`), and CPU / RAM usage for comparing `AE-LSTM` vs `Isolation Forest`.

---

## Project Structure

```text
├── config.py                  # Paths and attack label definitions
├── params.yaml                # Hydra / DVC hyperparameters & attack configs
├── dvc.yaml                   # Full DVC pipeline definition
├── docker-compose.yml         # Full Real-Time IDS stack (can-player, can-ids, Prometheus, Grafana)
├── docker/
│   ├── can-player/            # SocketCAN traffic generator container
│   ├── can-ids/               # Real-Time IDS Python container (ARM64 / x86_64)
│   ├── prometheus/            # Prometheus 1s scrape configuration
│   └── grafana/               # Auto-provisioned datasource & CAN IDS dashboard JSON
├── data/
│   ├── raw/                   # Raw SocketCAN (.log) dumps
│   └── processed/             # Parsed CSVs, scaled .npy splits, generated attacks
├── notebooks/                 # Exploratory notebooks (ECU frequency, step charts, MSE profiling)
├── plots/                     # Generated SVG/PNG charts from evaluation stages
├── reports/metrics/           # JSON metrics, calibrated thresholds, model comparison
├── scripts/
│   ├── dvc_stages/            # Entrypoints for each DVC pipeline stage
│   └── plot_makers/           # Standalone scripts for extra plots (entropy, seeds, etc.)
└── src/
    ├── ids/                   # Real-time streaming feature extractor, detector & Prometheus exporter
    ├── models/                # AELSTM, CANIsolationForest, saved weights & scalers
    └── utils/                 # Log parser, feature scaling, attack generators, trainer/evaluator
```

---

## Offline DVC Pipeline Setup & Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and **DVC** + **Hydra** for running the offline training/evaluation pipeline.

### 1. Install dependencies
```powershell
# Install uv if you don't have it yet
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create the virtual environment and install packages
uv sync
```

### 2. Run the full pipeline with DVC
```powershell
uv run dvc repro
```

### 3. Run individual stages or tweak parameters
```powershell
# Train AE-LSTM for a quick 1-epoch smoke test
uv run python -m scripts.dvc_stages.train training.epochs=1

# Train and evaluate Isolation Forest
uv run python -m scripts.dvc_stages.train_iforest
uv run python -m scripts.dvc_stages.evaluate_iforest

# Generate and evaluate a specific attack scenario (e.g. fuzzing)
uv run python -m scripts.dvc_stages.generate_attack attacks.active_scenario=fuzzing
uv run python -m scripts.dvc_stages.preprocess_attack attacks.active_scenario=fuzzing
uv run python -m scripts.dvc_stages.evaluate_attack attacks.active_scenario=fuzzing

# Generate the side-by-side comparison plot and JSON report
uv run python -m scripts.dvc_stages.compare_models
```
