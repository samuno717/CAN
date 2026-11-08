import numpy as np
import pandas as pd

from config import FUZZING_LABEL
from src.utils.attacks.common import extract_timestamps, merge_and_arbitrate


def inject_fuzzing_df(
    df: pd.DataFrame,
    start_row: int = 25000,
    num_frames: int = 800,
    id_mode: str = "random",
    interval_min: float = 0.0005,
    interval_max: float = 0.0030,
    min_bus_gap: float = 0.00020,
    seed: int = 42,
) -> pd.DataFrame:
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if num_frames <= 0 or start_row >= len(work_df):
        return work_df

    rng = np.random.default_rng(seed)
    live_ts = extract_timestamps(work_df)
    work_df["_ts"] = live_ts

    known_ids = (
        work_df[2]
        .astype(str)
        .str.split("#")
        .str[0]
        .str.upper()
        .unique()
    )
    iface = str(work_df.iloc[0, 1])

    anchor_ts = float(live_ts[min(start_row, len(live_ts) - 1)]) + min_bus_gap
    intervals = rng.uniform(interval_min, interval_max, size=num_frames)
    fuzz_times = anchor_ts + np.cumsum(intervals)

    mode = id_mode.strip().lower()
    fuzz_rows = []
    for t_inj in fuzz_times:
        if mode == "legit":
            can_id_hex = str(rng.choice(known_ids))
        elif mode == "mixed":
            if rng.random() < 0.5:
                can_id_hex = str(rng.choice(known_ids))
            else:
                can_id_hex = f"{int(rng.integers(0, 0x800)):03X}"
        else:
            # Standard 11-bit CAN ID fuzzing (0x000..0x7FF)
            can_id_hex = f"{int(rng.integers(0, 0x800)):03X}"

        rand_bytes = rng.integers(0, 256, size=8, dtype=np.uint8)
        payload_hex = "".join(f"{int(b):02X}" for b in rand_bytes)

        fuzz_rows.append({
            0: f"({t_inj:.6f})",
            1: iface,
            2: f"{can_id_hex}#{payload_hex}",
            "label": np.int8(FUZZING_LABEL),
            "_ts": float(t_inj),
        })

    fuzz_df = pd.DataFrame(fuzz_rows)
    return merge_and_arbitrate(work_df, fuzz_df, min_bus_gap=min_bus_gap)
