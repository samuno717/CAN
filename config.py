from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TENSORS_DIR = PROCESSED_DATA_DIR / "tensors"
TENSORS_PATH = TENSORS_DIR / "tensors.npy"
TENSORS_VAL_PATH = TENSORS_DIR / "tensors_val.npy"

SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = SRC_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
SCALERS_DIR = MODELS_DIR / "scalers"
BYTES_SCALER_PATH = SCALERS_DIR / "bytes_scaler.pkl"
TIME_SCALER_PATH = SCALERS_DIR / "time_scaler.pkl"

ATTACKS_DIR = PROJECT_ROOT / "attacks"

REPORTS_DIR = PROJECT_ROOT / "reports"

PLOTS_DIR = PROJECT_ROOT / 'plots'

NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
