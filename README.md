# CAN Bus Anomaly Detection Baseline

## Dependencies
Make sure to have `uv` installed:

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

macOS / Linux:
```sh
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

## Installation

### 1. Clone the repository
```sh
git clone https://github.com/samuno717/CAN
```

### 2. Sync using uv

```sh
uv sync
```

## Usage

### CAN data processing

To run the main process and generate `processed_data.csv`, execute the `main.py` script.

```sh
python -m src.main <filename>
```

### Generating a data report
Make sure to run main.py first to parse the data. This will output `report.html` file.

```sh
python -m src.report_generator
```

### Generating Model Baselines

To view the experiments with activation functions and model architectures, you can run the Jupyter notebook.

This notebok will process the `processed_data.csv`, train a model for each valid CAN ID, and print the progress to the console. 
The final output will be a DataFrame and a plot showing the learned MSE thresholds.

