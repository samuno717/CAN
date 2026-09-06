import json
import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from config import PLOTS_DIR, WEIGHTS_DIR, ATTACKS_DIR, REPORTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.EcuDataset import EcuDataset
from src.utils.evaluate_attack import (
    align_window_labels,
    compute_detection_metrics,
    evaluate_attack_stream,
    plot_attack_evaluation,
)


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running attack evaluation on device: {device}")

    hidden_size = cfg.model.hidden_size
    sequence_length = cfg.model.sequence_length
    num_features = cfg.model.num_features
    batch_size = cfg.training.batch_size

    model_path = WEIGHTS_DIR / f"ae_lstm_can_model_h{hidden_size}.pth"
    attack_data_path = ATTACKS_DIR / "dos_tensors.npy"
    labels_path = ATTACKS_DIR / "dos_labels.npy"
    threshold_path = REPORTS_DIR / "metrics" / "threshold.json"

    plot_output_path = PLOTS_DIR / f"dos_attack_reaction_h{hidden_size}.png"
    metrics_output_path = REPORTS_DIR / "metrics" / f"dos_metrics_h{hidden_size}.json"

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found at {threshold_path}. "
            f"Run baseline evaluation stage first to calibrate detection threshold."
        )

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
        threshold = float(threshold_data["threshold"])

    logger.info(f"Loaded dynamic threshold from baseline stage: {threshold:.5f}")

    logger.info(f"Loading trained weights from: {model_path}")
    model = AELSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    logger.info(f"Loading attacked dataset: {attack_data_path}")
    dataset = EcuDataset(data_path=attack_data_path, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = evaluate_attack_stream(model=model, dataloader=dataloader, device=device)

    raw_labels = np.load(labels_path)
    window_labels = align_window_labels(raw_labels, sequence_length=sequence_length)

    compute_detection_metrics(
        errors=errors,
        y_true=window_labels,
        threshold=threshold,
        metrics_save_path=metrics_output_path,
    )

    plot_attack_evaluation(
        errors=errors,
        y_true=window_labels,
        threshold=threshold,
        save_path=plot_output_path,
        attack_name="Denial of Service (DoS)",
        display_limit=12000,
    )


if __name__ == "__main__":
    main()