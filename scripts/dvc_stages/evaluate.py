import json
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from config import FEATURES_VAL_PATH, PLOTS_DIR, REPORTS_DIR, WEIGHTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.evaluator import (
    evaluate_model,
    load_sliding_windows,
    plot_reconstruction_errors,
)
from src.utils.FastTensorDataLoader import FastTensorDataLoader


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation runs on: {device}")

    hidden_size = cfg.model.hidden_size
    seq_len = cfg.model.sequence_length
    num_features = cfg.model.num_features

    model_path = WEIGHTS_DIR / cfg.model.path
    plot_path = (PLOTS_DIR / f"reconstruction_error_h{hidden_size}_e{cfg.training.epochs}.svg")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}. Run training first or check the file name.")

    val_windows = load_sliding_windows(file_path=FEATURES_VAL_PATH,
                                       seq_len=seq_len,
                                       device=device,
                                       stride=cfg.model.sequence_step)

    val_loader = FastTensorDataLoader(dataset_tensor=val_windows,
                                      batch_size=cfg.training.batch_size,
                                      shuffle=False)

    logger.info(f"Loading model weights from {model_path.name} [hidden={hidden_size}, seq_len={seq_len}]...")

    model = AELSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        latent_dim=cfg.model.latent_dim,
        sequence_length=seq_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) # loading saved weights to GPU

    val_errors = evaluate_model(model=model, dataloader=val_loader, device=device)
    plot_reconstruction_errors(errors=val_errors, save_path=plot_path, cfg=cfg)

    threshold = float(np.percentile(val_errors, cfg.detection.percentile))
    logger.success(f"Calibrated threshold (p={cfg.detection.percentile}%): {threshold:.5f}")

    logger.warning("Saving baseline threshold...")
    threshold_path = REPORTS_DIR / "metrics" / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "percentile": cfg.detection.percentile,
            },
            f, indent=4,
        )

if __name__ == "__main__":
    main()