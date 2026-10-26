import numpy as np
import pandas as pd

from config import REPLAY_LABEL


def inject_replay_df(
    df: pd.DataFrame,
    capture_start_row: int = 500,
    start_row: int = 10000,
    duration_rows: int = 1000,
    target_can_id_hex: str = "all",
    speed_factor: float = 1.0,
    min_bus_gap: float = 0.00020,
) -> pd.DataFrame:
    """Captures normal CAN frames from an earlier segment and replays them at start_row."""
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    clean_pool = work_df[work_df["label"] == 0]
    target = target_can_id_hex.strip().upper()

    if target == "ALL":
        captured = clean_pool.iloc[capture_start_row : capture_start_row + duration_rows].copy()
    else:
        ecu_pool = clean_pool.iloc[capture_start_row:]
        captured = ecu_pool[ecu_pool[2].astype(str).str.upper().str.startswith(f"{target}#")].iloc[:duration_rows].copy()

    live_ts = work_df[0].astype(str).str.strip("()").astype(float)
    cap_ts = captured[0].astype(str).str.strip("()").astype(float).to_numpy()

    anchor_ts = float(live_ts.iloc[start_row]) + min_bus_gap
    captured["_ts"] = anchor_ts + (cap_ts - cap_ts[0]) / speed_factor
    captured["label"] = np.int8(REPLAY_LABEL)
    work_df["_ts"] = live_ts.to_numpy()

    merged = (
        pd.concat([work_df, captured], ignore_index=True)
        .sort_values("_ts", kind="mergesort")
        .reset_index(drop=True)
    )

    # Ensure minimum physical time gap between frames on the CAN bus
    ts = merged["_ts"].to_numpy(copy=True)
    for i in range(1, len(ts)):
        ts[i] = max(ts[i], ts[i - 1] + min_bus_gap)

    merged[0] = "(" + pd.Series(ts).map("{:.6f}".format) + ")"
    return merged.drop(columns=["_ts"])
