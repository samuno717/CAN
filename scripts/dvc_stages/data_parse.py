import hydra
from loguru import logger
from omegaconf import DictConfig

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.raw_data_parser import parse_can_data


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig):
    raw_datafile = RAW_DATA_DIR / cfg.parsing.raw_file
    output_path = PROCESSED_DATA_DIR / cfg.parsing.output_file

    parsed_data = parse_can_data(raw_datafile)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_data.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()
