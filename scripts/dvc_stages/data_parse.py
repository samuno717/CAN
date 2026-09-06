from loguru import logger

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.raw_data_parser import parse_can_data


def main():
    raw_datafile = RAW_DATA_DIR / 'candump-2026-04-28_162402.log'
    output_path = PROCESSED_DATA_DIR / 'processed_data.csv'

    parsed_data = parse_can_data(raw_datafile)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_data.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()
