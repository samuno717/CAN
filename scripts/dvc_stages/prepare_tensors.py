import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from loguru import logger

from config import PROCESSED_DATA_DIR, TENSORS_DIR, SCALERS_DIR, TIME_SCALER_PATH, BYTES_SCALER_PATH, TENSORS_PATH, \
    TENSORS_VAL_PATH
from src.utils.tensor_prep.preprocess import preprocess_and_split
from src.utils.tensor_prep.scale_features import scale_features


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    logger.info("Loading processed CSV file...")
    df = pd.DataFrame(pd.read_csv(PROCESSED_DATA_DIR / 'processed_data.csv'))
    logger.success(f"File loaded!")

    # Preprocessing and split
    split_ratio = cfg.tensor_data.train_split
    df_train, df_val = preprocess_and_split(df, split_ratio)

    # Scaling
    df_train, df_val, bytes_scaler, time_scaler = scale_features(df_train, df_val)

    # Saving scalers
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bytes_scaler, BYTES_SCALER_PATH)
    joblib.dump(time_scaler, TIME_SCALER_PATH)
    logger.info(f"Scalers saved to: {SCALERS_DIR}")

    # Saving 2D matrix
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TENSORS_PATH, df_train.values.astype(np.float32))
    np.save(TENSORS_VAL_PATH, df_val.values.astype(np.float32))

    logger.success(f"Training dataset saved to: {TENSORS_PATH}")
    logger.success(f"Test dataset saved to: {TENSORS_VAL_PATH}")


if __name__ == "__main__":
    main()

