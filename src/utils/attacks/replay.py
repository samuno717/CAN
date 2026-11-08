import numpy as np
import pandas as pd

from config import REPLAY_LABEL
from src.utils.attacks.common import extract_timestamps, merge_and_arbitrate


def inject_replay_df(
    df: pd.DataFrame,
    capture_start_row: int = 500,
    start_row: int = 10000,
    duration_rows: int = 1000,
    target_can_id_hex: str = "all",
    speed_factor: float = 1.0,
    min_bus_gap: float = 0.00020,
) -> pd.DataFrame:
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if duration_rows <= 0 or start_row >= len(work_df):
        return work_df

    clean_pool = work_df[work_df["label"] == 0]
    target = target_can_id_hex.strip().upper()

    if target == "ALL":
        captured = clean_pool.iloc[capture_start_row : capture_start_row + duration_rows].copy()
    else:
        ecu_pool = clean_pool.iloc[capture_start_row:]
        captured = ecu_pool[ecu_pool[2].astype(str).str.upper().str.startswith(f"{target}#")].iloc[:duration_rows].copy()

    if captured.empty:
        return work_df

    live_ts = extract_timestamps(work_df)
    cap_ts = extract_timestamps(captured)

    anchor_ts = float(live_ts[min(start_row, len(live_ts) - 1)]) + min_bus_gap
    captured["_ts"] = anchor_ts + (cap_ts - cap_ts[0]) / max(speed_factor, 1e-6)
    captured["label"] = np.int8(REPLAY_LABEL)
    work_df["_ts"] = live_ts

    return merge_and_arbitrate(work_df, captured, min_bus_gap=min_bus_gap)
