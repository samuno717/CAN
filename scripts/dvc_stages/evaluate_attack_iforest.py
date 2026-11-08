import json

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from config import METRICS_DIR, PLOTS_DIR, PROCESSED_ATTACKS_DIR, WEIGHTS_DIR
from src.models.isolation_forest import (
    CANIsolationForest,
    extract_iforest_window_features,
)
from src.utils.evaluate_attack import (
    align_window_labels,
    compute_detection_metrics,
    plot_attack_evaluation,
)


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    scenario = str(cfg.attacks.active_scenario)
    logger.info(f"[IsolationForest] Evaluating scenario '{scenario}'...")

    seq_len = int(cfg.model.sequence_length)
    eval_stride = int(cfg.attacks.eval_stride)
    persistence_k = int(cfg.detection.persistence_k)
    feature_mode = str(cfg.iforest.feature_mode)
    n_estimators = int(cfg.iforest.n_estimators)

    model_path = WEIGHTS_DIR / str(cfg.iforest.path)
    threshold_path = METRICS_DIR / "iforest_threshold.json"
    features_path = PROCESSED_ATTACKS_DIR / f"{scenario}_scaled.npy"
    labels_path = PROCESSED_ATTACKS_DIR / f"{scenario}_labels.npy"

    metrics_save_path = METRICS_DIR / f"iforest_{scenario}_metrics.json"
    plot_save_path = PLOTS_DIR / f"iforest_{scenario}_attack_reaction_n{n_estimators}.svg"

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    attack_windows = extract_iforest_window_features(
        file_path=features_path,
        seq_len=seq_len,
        stride=eval_stride,
        feature_mode=feature_mode,
    )

    y_true_cls = align_window_labels(
        labels=np.load(labels_path),
        sequence_length=seq_len,
        stride=eval_stride,
    )

    iforest = CANIsolationForest.load(model_path)
    scores = iforest.score_windows(attack_windows)

    metrics = compute_detection_metrics(
        errors=scores,
        y_true_cls=y_true_cls,
        threshold=threshold,
        persistence_k=persistence_k,
        scenario_name=f"iforest_{scenario}",
        metrics_save_path=metrics_save_path,
    )

    plot_attack_evaluation(
        errors=scores,
        y_true_cls=y_true_cls,
        threshold=threshold,
        save_path=plot_save_path,
        persistence_k=persistence_k,
        metrics=metrics,
        scenario_title=f"Isolation Forest \u2014 {scenario.upper()}",
    )


if __name__ == "__main__":
    main()
