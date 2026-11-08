from pathlib import Path

import numpy as np
import pandas as pd

from config import DOS_LABEL


def load_test_raw_df(raw_data: Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> pd.DataFrame:
    """Loads clean CAN frames from the test split."""
    with open(raw_data, "r", encoding="utf-8") as logfile:
        lines = [line.strip() for line in logfile if line.strip()]

    val_end = int(len(lines) * (train_ratio + val_ratio))
    test_rows = [line.split() for line in lines[val_end:]]
    df = pd.DataFrame(test_rows)
    df["label"] = np.int8(0)
    return df


def inject_dos_df(
    df: pd.DataFrame,
    start_row: int = 5000,
    dos_rows: int = 1000,
    attack_payload_str: str = "002#0000000000000000",
    physical_delay: float = 0.00025,
) -> pd.DataFrame:
    """Inserts a set amount of high-priority DoS frames into the CAN stream."""
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    part_before = work_df.iloc[:start_row].copy()
    part_after = work_df.iloc[start_row:].copy()

    last_ts = float(str(part_before.iloc[-1, 0]).strip("()"))
    next_ts = float(str(part_after.iloc[0, 0]).strip("()"))

    t_start = last_ts + physical_delay
    dos_times = [t_start + i * physical_delay for i in range(dos_rows)]

    dos_df = pd.DataFrame({
        0: [f"({t:.6f})" for t in dos_times],
        1: "can0",
        2: attack_payload_str,
        "label": np.int8(DOS_LABEL),
    })

    time_shift = (dos_times[-1] + physical_delay) - next_ts
    after_ts = part_after[0].astype(str).str.strip("()").astype(float) + time_shift
    part_after[0] = "(" + after_ts.map("{:.6f}".format) + ")"

    return pd.concat([part_before, dos_df, part_after], ignore_index=True)
