from datetime import datetime
import random
import hydra
import numpy as np
import torch
import torch.nn as nn
import json
from loguru import logger
from omegaconf import DictConfig

from config import FEATURES_TRAIN_PATH, FEATURES_VAL_PATH, WEIGHTS_DIR, METRICS_DIR, PLOTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.FastTensorDataLoader import FastTensorDataLoader
from src.utils.trainer import train_model, plot_training_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for more than 1 GPU

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    seed = cfg.training.seed
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training runs on: {device}")

    model_path = (WEIGHTS_DIR / f"ae_lstm_can_model_h{cfg.model.hidden_size}_e{cfg.training.epochs}.pth")

    seq_len = cfg.model.sequence_length
    seq_step = cfg.model.sequence_step

    logger.info(f"Loading training dataset from {FEATURES_TRAIN_PATH}...")
    train_numpy = np.load(FEATURES_TRAIN_PATH)
    train_tensor = torch.from_numpy(train_numpy).float().to(device)
    train_windows = train_tensor.unfold(0, seq_len, seq_step).permute(0, 2, 1)

    train_loader = FastTensorDataLoader(
        dataset_tensor=train_windows,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    logger.info(f"Train windows: {len(train_windows):,}")

    logger.info(f"Loading validation dataset from {FEATURES_VAL_PATH}...")
    val_numpy = np.load(FEATURES_VAL_PATH)
    val_tensor = torch.from_numpy(val_numpy).float().to(device)
    val_windows = val_tensor.unfold(0, seq_len, seq_step).permute(0, 2, 1)

    val_loader = FastTensorDataLoader(
        dataset_tensor=val_windows,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )
    logger.info(f"Validation windows: {len(val_windows):,}")

    logger.info("Initializing AE-LSTM model...")
    model = AELSTM(
        num_features=cfg.model.num_features,
        hidden_size=cfg.model.hidden_size,
        latent_dim=cfg.model.latent_dim,
        sequence_length=seq_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.training.epochs,
        device=device,
        save_path=model_path,
    )

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    history_path = (METRICS_DIR / f"history_h{cfg.model.hidden_size}_seed{seed}.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "history": history}, f, indent=2)

    plot_path = (PLOTS_DIR / "training_loss_history" / f"train_loss_h{cfg.model.hidden_size}_seed{seed}.svg")

    train_params = {
        "hidden_size": cfg.model.hidden_size,
        "latent_dim": cfg.model.latent_dim,
        "sequence_length": cfg.model.sequence_length,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "seed": seed,
        "optimizer": "Adam"
    }
    plot_training_loss(history=history, save_path=plot_path, params=train_params)

if __name__ == "__main__":
    main()