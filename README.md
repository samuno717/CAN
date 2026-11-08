 # CAN Bus Anomaly Detection (AE-LSTM vs Isolation Forest)

Unsupervised Intrusion Detection System (IDS) for Controller Area Network (CAN) traffic. 

The main goal of this project is to detect anomalies and attacks on the CAN bus without relying on labeled attack data during training. The models are trained strictly on normal driving traffic (`candump` logs) and evaluated against several synthetic attack scenarios injected into the test split.

I built this pipeline to compare a sequence-aware deep learning approach (**LSTM Autoencoder**) against a classical baseline (**Isolation Forest**) under identical windowing and thresholding conditions.

---

## How It Works

### 1. Data & Feature Extraction
Raw SocketCAN logs (`data/raw/candump-*.log`) are parsed into structured CSVs and split chronologically into **70% train / 15% validation / 15% test** (no shuffling, to avoid data leakage across time windows).

Each CAN frame is converted into a **26-dimensional feature vector**:
- `delta_t` – time elapsed since the previous frame on the bus (MinMax scaled)
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

## Project Structure

```text
├── config.py                  # Paths and attack label definitions
├── params.yaml                # Hydra / DVC hyperparameters & attack configs
├── dvc.yaml                   # Full DVC pipeline definition
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
    ├── models/                # AELSTM, CANIsolationForest, saved weights & scalers
    └── utils/                 # Log parser, feature scaling, attack generators, trainer/evaluator
```

---

## Setup & Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and **DVC** + **Hydra** for running the pipeline. Note that `pyproject.toml` is configured for Windows with CUDA 12.8 (`pytorch-cu128`).

### 1. Install dependencies
```powershell
# Install uv if you don't have it yet
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create the virtual environment and install packages
uv sync
```

### 2. Run the full pipeline with DVC
All stages (parsing, feature scaling, evaluation, attack generation, and model comparison) are tracked in `dvc.yaml`. To reproduce everything:
```powershell
uv run dvc repro
```
*(Note: the `train` stage for AE-LSTM is commented out in `dvc.yaml` by default so `dvc repro` uses the pre-trained weights in `src/models/weights/` instead of retraining for 400 epochs every time. Uncomment it if you want to train from scratch.)*

### 3. Run individual stages or tweak parameters
Every script in `scripts/dvc_stages/` uses Hydra and reads from `params.yaml`. You can run them directly or override parameters from the CLI:

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
