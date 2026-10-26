from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
FEATURES_TRAIN_PATH = FEATURES_DIR / "train_scaled.npy"
FEATURES_VAL_PATH = FEATURES_DIR / "val_scaled.npy"
FEATURES_TEST_PATH = FEATURES_DIR / "test_scaled.npy"

PROCESSED_ATTACKS_DIR = PROCESSED_DATA_DIR / "attacks"

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
SCALERS_DIR = MODELS_DIR / "scalers"
BYTES_SCALER_PATH = SCALERS_DIR / "bytes_scaler.pkl"
TIME_SCALER_PATH = SCALERS_DIR / "time_scaler.pkl"
FEATURE_COLUMNS_PATH = SCALERS_DIR / "feature_columns.json"

ATTACKS_DIR = PROJECT_ROOT / "attacks"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Attack label IDs (0 = normal frame, 1 = DoS, 2 = Replay)
DOS_LABEL = 1
REPLAY_LABEL = 2

ATTACK_NAMES = {
    DOS_LABEL: "DoS",
    REPLAY_LABEL: "Replay",
}

ATTACK_COLORS = {
    DOS_LABEL: "#C0392B",
    REPLAY_LABEL: "#7C3AED",
}
