# CAN Bus Anomaly Detection Baseline

## Installation

### 1. Create and activate venv

```sh
python -m venv venv
source venv/Scripts/activate
```

### 2. Install libraries

```sh
pip install -r requirements.txt
```

## Usage

### 1. CAN data processing

To run the main process and generate `processed_data.csv`, execute the `main.py` script.

```sh
python -m src.main <filename>
```


### 2. Generating Model Baselines

To view the experiments with activation functions and model architectures, you can run the Jupyter notebook.

This notebok will process the `processed_data.csv`, train a model for each valid CAN ID, and print the progress to the console. 
The final output will be a DataFrame and a plot showing the learned MSE thresholds.

