from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from config import PROJECT_ROOT


def parse_can_data(raw_data_file: Path) -> pd.DataFrame:
    datafile_size = raw_data_file.stat().st_size

    records = []
    previous_time_for_id = {}
    first_log_timestamp = None
    final_log_timestamp = None
    ids = set()

    with open(raw_data_file, 'r', encoding='utf-8') as logfile, \
            tqdm(total=datafile_size, desc="Processing frames", unit="B", unit_scale=True, unit_divisor=1024) as pbar:

        for l in logfile:
            pbar.update(len(l.encode('utf-8')))

            raw_line = l.strip().split()
            if not raw_line:
                continue

            raw_line.pop(1)

            current_timestamp = float(raw_line[0].strip("()"))

            if first_log_timestamp is None:
                first_log_timestamp = current_timestamp
            final_log_timestamp = current_timestamp

            raw_can_id = raw_line[1].split("#")[0]
            can_id = int(raw_can_id, 16)
            ids.add(can_id)

            raw_payload = raw_line[1].split("#")[1]
            split_payload = [raw_payload[i:i+2] for i in range(0, len(raw_payload), 2)]
            payload = [int(i, 16) for i in split_payload]
            payload += [0] * (8 - len(payload))

            if can_id in previous_time_for_id:
                delta_t = current_timestamp - previous_time_for_id[can_id]
            else:
                delta_t = 0.0

            previous_time_for_id[can_id] = current_timestamp
            delta_t = round(delta_t, 6)

            records.append([current_timestamp, delta_t, can_id] + payload)

    df = pd.DataFrame(records, columns=np.array(["timestamp", "delta_t", "can_id", "byte1", "byte2", "byte3", "byte4", "byte5", "byte6", "byte7", "byte8"]))
    logger.success("Data parsing done!")

    total_time = final_log_timestamp - first_log_timestamp

    logger.info(
        f"\n  Number of ECUs: {len(ids)}"
        f"\n  Total drive time:  {total_time:.2f} seconds"
        f"\n  Number of frames: {len(df)}"
    )

    return df
