import json

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from config import FEATURES_VAL_PATH, METRICS_DIR, PLOTS_DIR, WEIGHTS_DIR
from src.models.isolation_forest import (
    CANIsolationForest,
    extract_iforest_window_features,
    plot_iforest_validation_scores,
)


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    seq_len = int(cfg.model.sequence_length)
    seq_step = int(cfg.model.sequence_step)
    feature_mode = str(cfg.iforest.feature_mode)

    model_path = WEIGHTS_DIR / str(cfg.iforest.path)
    plot_path = PLOTS_DIR / f"iforest_validation_score_n{cfg.iforest.n_estimators}.svg"
    threshold_path = METRICS_DIR / "iforest_threshold.json"

    iforest = CANIsolationForest.load(model_path)

    val_windows = extract_iforest_window_features(
        file_path=FEATURES_VAL_PATH,
        seq_len=seq_len,
        stride=seq_step,
        feature_mode=feature_mode,
    )

    val_scores = iforest.score_windows(val_windows)
    plot_iforest_validation_scores(scores=val_scores, save_path=plot_path, cfg=cfg)

    threshold = float(np.percentile(val_scores, cfg.detection.percentile))
    logger.success(f"[IsolationForest] Calibrated threshold (p={cfg.detection.percentile}%): {threshold:.5f}")

    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "IsolationForest",
                "threshold": threshold,
                "percentile": float(cfg.detection.percentile),
                "n_estimators": int(cfg.iforest.n_estimators),
                "feature_mode": feature_mode,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
