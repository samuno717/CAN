import json

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from loguru import logger

from config import PLOTS_DIR, TENSORS_VAL_PATH, WEIGHTS_DIR, REPORTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.EcuDataset import EcuDataset
from src.utils.evaluator import evaluate_model, plot_reconstruction_errors

@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation runs on: {device}")

    hidden_size = cfg.model.hidden_size
    seq_len = cfg.model.sequence_length
    num_features = cfg.model.num_features

    model_path = WEIGHTS_DIR / f"ae_lstm_can_model_h{hidden_size}.pth"
    plot_path = PLOTS_DIR / f"reconstruction_error_h{hidden_size}.png"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}. Run training first.")

    logger.info(f"Loading validation dataset from {TENSORS_VAL_PATH} (window size={seq_len})...")
    dataset = EcuDataset(data_path=TENSORS_VAL_PATH, sequence_length=seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    logger.info(f"Loading architecture [features={num_features}, hidden={hidden_size}, seq_len={seq_len}]...")
    model = AELSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        sequence_length=seq_len,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    errors = evaluate_model(model=model, dataloader=dataloader, device=device)

    plot_reconstruction_errors(errors=errors, save_path=plot_path)

    threshold = float(np.percentile(errors, cfg.detection.percentile))
    threshold_path = REPORTS_DIR / "metrics" / "threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "percentile": cfg.detection.percentile}, f, indent=4)

    logger.success(f"Calibrated threshold ({threshold:.5f}) saved to: {threshold_path}")

if __name__ == "__main__":
    main()