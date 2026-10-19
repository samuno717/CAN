import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from loguru import logger

from config import (PROCESSED_DATA_DIR, FEATURES_DIR, SCALERS_DIR, TIME_SCALER_PATH, BYTES_SCALER_PATH,
                    FEATURES_TRAIN_PATH, FEATURES_VAL_PATH, FEATURES_TEST_PATH)
from src.utils.feature_prep.preprocess import preprocess_and_split
from src.utils.feature_prep.scale_features import scale_features


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    logger.info("Loading processed CSV file...")
    df = pd.DataFrame(pd.read_csv(PROCESSED_DATA_DIR / 'processed_data.csv'))
    logger.success(f"File loaded!")

    # Preprocessing and split
    df_train, df_val, df_test = preprocess_and_split(df, cfg.data_split.train_ratio, cfg.data_split.val_ratio)

    # Scaling
    df_train, df_val, df_test, bytes_scaler, time_scaler = scale_features(df_train, df_val, df_test)

    # Saving scalers
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bytes_scaler, BYTES_SCALER_PATH)
    joblib.dump(time_scaler, TIME_SCALER_PATH)
    logger.info(f"Scalers saved to: {SCALERS_DIR}")

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

