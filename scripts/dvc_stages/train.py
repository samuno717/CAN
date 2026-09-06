import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from loguru import logger
from torch.utils.data import DataLoader

from config import TENSORS_PATH, WEIGHTS_DIR
from src.models.ae_lstm import AELSTM
from src.utils.trainer import train_model
from src.utils.EcuDataset import EcuDataset


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    logger.info("Checking GPU availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training done on: {device}")

    model_path = WEIGHTS_DIR / f'ae_lstm_can_model_h{cfg.model.hidden_size}.pth'

    logger.info(f"Loading dataset from {TENSORS_PATH}...")
    dataset = EcuDataset(data_path=TENSORS_PATH, sequence_length=cfg.model.sequence_length)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    logger.info(f"Successfully created {len(dataset)} time windows.")

    logger.info("Creating model and optimizer...")
    model = AELSTM(num_features=cfg.model.num_features,
                   hidden_size=cfg.model.hidden_size,
                   sequence_length=cfg.model.sequence_length,
                   ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    model = train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.training.epochs,
        device=device
    )

    torch.save(model.state_dict(), model_path)
    logger.success(f"Weights saved to: {model_path}")


if __name__ == "__main__":
    main()