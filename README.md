# CAN Bus Anomaly Detection

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

To run the main process and generate `processed_data.csv`, execute the `data_parser.py` script.

```sh
python -m src.data_parser <filename>
```

### Generating a data report
Make sure to run ``data_parser.py`` first to parse the data. 
\
\
Run this command and it will output `report.html` file in the `/data` folder

```sh
python -m src.report_generator
```

### Generating Charts

To generate charts simply run the corresponding notebooks in `/notebooks` directory. The charts will appear in the `/charts` directory.

Each notebok will process the `processed_data.csv`, so make sure to run `data_parser.py` first. 

For now there are 2 available chart generators. In each you are capable of changing the parameters to adjust the chart.



