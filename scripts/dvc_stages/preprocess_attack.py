import json

import hydra
import joblib
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from config import (
    BYTES_SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    PROCESSED_ATTACKS_DIR,
    TIME_SCALER_PATH,
)
from src.utils.feature_prep.scale_features import transform_attack_features
from src.utils.raw_data_parser import parse_can_data


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    scenario = str(cfg.attacks.active_scenario)
    attack_log_path = PROCESSED_ATTACKS_DIR / f"{scenario}_attack.log"
    attack_features_path = PROCESSED_ATTACKS_DIR / f"{scenario}_scaled.npy"

    parsed_df = parse_can_data(attack_log_path)

    bytes_scaler = joblib.load(BYTES_SCALER_PATH)
    time_scaler = joblib.load(TIME_SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    scaled_df = transform_attack_features(
        df=parsed_df,
        bytes_scaler=bytes_scaler,
        time_scaler=time_scaler,
        feature_columns=feature_columns,
    )

    np.save(attack_features_path, scaled_df.to_numpy(dtype=np.float32))
    logger.success(f"[{scenario}] Scaled features saved to: {attack_features_path}")


if __name__ == "__main__":
    main()
