import numpy as np
import pandas as pd

from config import FREQUENCY_LABEL
from src.utils.attacks.common import extract_timestamps, merge_and_arbitrate


def inject_frequency_anomaly_df(
    df: pd.DataFrame,
    start_row: int = 15000,
    duration_rows: int = 2000,
    target_can_id_hex: str = "130",
    freq_multiplier: int = 5,
    min_bus_gap: float = 0.00020,
) -> pd.DataFrame:
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if freq_multiplier <= 1 or duration_rows <= 0 or start_row >= len(work_df):
        return work_df

    target = target_can_id_hex.strip().upper()
    end_row = min(len(work_df), start_row + duration_rows)
    window_df = work_df.iloc[start_row:end_row]

    target_mask = window_df[2].astype(str).str.upper().str.startswith(f"{target}#")
    target_frames = window_df[target_mask].copy()

    if len(target_frames) < 2:
        return work_df

    work_df["_ts"] = extract_timestamps(work_df)
    target_ts = extract_timestamps(target_frames)
    target_ifaces = target_frames[1].astype(str).to_numpy()
    target_payloads = target_frames[2].astype(str).to_numpy()

    injected_rows = []
    for i in range(len(target_frames) - 1):
        t_curr = target_ts[i]
        t_next = target_ts[i + 1]
        dt = (t_next - t_curr) / freq_multiplier

        for k in range(1, freq_multiplier):
            t_inj = float(t_curr + k * dt)
            injected_rows.append({
                0: f"({t_inj:.6f})",
                1: target_ifaces[i],
                2: target_payloads[i],
                "label": np.int8(FREQUENCY_LABEL),
                "_ts": t_inj,
            })

    injected_df = pd.DataFrame(injected_rows)
    return merge_and_arbitrate(work_df, injected_df, min_bus_gap=min_bus_gap)
