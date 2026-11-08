import numpy as np
import pandas as pd

from config import DOS_LABEL
from src.utils.attacks.common import format_timestamps, load_test_raw_df

__all__ = ["load_test_raw_df", "inject_dos_df"]


def inject_dos_df(
    df: pd.DataFrame,
    start_row: int = 5000,
    dos_rows: int = 1000,
    attack_payload_str: str = "002#0000000000000000",
    physical_delay: float = 0.00025,
) -> pd.DataFrame:
    """Inserts a burst of high-priority DoS frames into the CAN stream, delaying subsequent traffic."""
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if dos_rows <= 0 or start_row >= len(work_df):
        return work_df

    start_row = max(1, min(start_row, len(work_df) - 1))
    part_before = work_df.iloc[:start_row].copy()
    part_after = work_df.iloc[start_row:].copy()

    last_ts = float(str(part_before.iloc[-1, 0]).strip("()"))
    next_ts = float(str(part_after.iloc[0, 0]).strip("()"))

    t_start = last_ts + physical_delay
    dos_times = t_start + np.arange(dos_rows, dtype=np.float64) * physical_delay

    dos_df = pd.DataFrame({
        0: format_timestamps(dos_times),
        1: str(part_before.iloc[-1, 1]),
        2: attack_payload_str.strip().upper(),
        "label": np.int8(DOS_LABEL),
    })

    time_shift = max(0.0, (float(dos_times[-1]) + physical_delay) - next_ts)
    after_ts = part_after[0].astype(str).str.strip("()").astype(float).to_numpy() + time_shift
    part_after[0] = format_timestamps(after_ts)

    return pd.concat([part_before, dos_df, part_after], ignore_index=True)
