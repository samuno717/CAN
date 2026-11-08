import json

import hydra
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from config import (
    BYTES_SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURES_DIR,
    FEATURES_TEST_PATH,
    FEATURES_TRAIN_PATH,
    FEATURES_VAL_PATH,
    PROCESSED_DATA_DIR,
    SCALERS_DIR,
    TIME_SCALER_PATH,
)
from src.utils.feature_prep.preprocess import preprocess_and_split
from src.utils.feature_prep.scale_features import scale_features


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    logger.info("Loading processed CSV file...")
    df = pd.DataFrame(pd.read_csv(PROCESSED_DATA_DIR / cfg.preprocessing.processed_file))
    logger.success("File loaded!")

    # Preprocessing and split
    df_train, df_val, df_test = preprocess_and_split(df, cfg.preprocessing.train_ratio, cfg.preprocessing.val_ratio)

    # Scaling
    df_train, df_val, df_test, bytes_scaler, time_scaler = scale_features(df_train, df_val, df_test)

    # Saving scalers and canonical feature column order
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bytes_scaler, BYTES_SCALER_PATH)
    joblib.dump(time_scaler, TIME_SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(df_train.columns), f, indent=2)
    logger.info(f"Scalers and feature schema saved to: {SCALERS_DIR}")

    # Saving 2D matrix
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_TRAIN_PATH, df_train.values.astype(np.float32))
    np.save(FEATURES_VAL_PATH, df_val.values.astype(np.float32))
    np.save(FEATURES_TEST_PATH, df_test.values.astype(np.float32))

    logger.success(f"Training dataset saved to: {FEATURES_TRAIN_PATH}")
    logger.success(f"Validation dataset saved to: {FEATURES_VAL_PATH}")
    logger.success(f"Test dataset saved to: {FEATURES_TEST_PATH}")


if __name__ == "__main__":
    main()
