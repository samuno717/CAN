import json

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from config import METRICS_DIR, PLOTS_DIR, PROCESSED_ATTACKS_DIR, WEIGHTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.evaluate_attack import (
    align_window_labels,
    compute_detection_metrics,
    plot_attack_evaluation,
)
from src.utils.evaluator import evaluate_model, load_sliding_windows
from src.utils.FastTensorDataLoader import FastTensorDataLoader


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    scenario = str(cfg.attacks.active_scenario)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating scenario '{scenario}' on {device}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seq_len = int(cfg.model.sequence_length)
    hidden_size = int(cfg.model.hidden_size)
    eval_stride = int(cfg.attacks.eval_stride)
    persistence_k = int(cfg.detection.persistence_k)

    model_path = WEIGHTS_DIR / cfg.model.path
    threshold_path = METRICS_DIR / "threshold.json"
    features_path = PROCESSED_ATTACKS_DIR / f"{scenario}_scaled.npy"
    labels_path = PROCESSED_ATTACKS_DIR / f"{scenario}_labels.npy"

    metrics_save_path = METRICS_DIR / f"{scenario}_metrics.json"
    plot_save_path = PLOTS_DIR / f"{scenario}_attack_reaction_h{hidden_size}_e{cfg.training.epochs}.svg"

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    attack_windows = load_sliding_windows(
        file_path=features_path,
        seq_len=seq_len,
        device=torch.device("cpu"),
        stride=eval_stride,
    )
    attack_loader = FastTensorDataLoader(
        dataset_tensor=attack_windows,
        batch_size=min(int(cfg.training.batch_size), 2048),
        shuffle=False,
    )

    y_true_cls = align_window_labels(
        labels=np.load(labels_path),
        sequence_length=seq_len,
        stride=eval_stride,
    )

    model = AELSTM(
        num_features=int(cfg.model.num_features),
        hidden_size=hidden_size,
        latent_dim=int(cfg.model.latent_dim),
        sequence_length=seq_len,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    errors = evaluate_model(model=model, dataloader=attack_loader, device=device)

    metrics = compute_detection_metrics(
        errors=errors,
        y_true_cls=y_true_cls,
        threshold=threshold,
        persistence_k=persistence_k,
        scenario_name=scenario,
        metrics_save_path=metrics_save_path,
    )

    plot_attack_evaluation(
        errors=errors,
        y_true_cls=y_true_cls,
        threshold=threshold,
        save_path=plot_save_path,
        persistence_k=persistence_k,
        metrics=metrics,
        scenario_title=scenario.upper(),
    )


if __name__ == "__main__":
    main()
