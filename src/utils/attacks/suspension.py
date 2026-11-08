import numpy as np
import pandas as pd

from config import SUSPENSION_LABEL


def inject_suspension_df(
    df: pd.DataFrame,
    start_row: int = 30000,
    duration_rows: int = 2000,
    target_can_id_hex: str = "130",
) -> pd.DataFrame:
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if duration_rows <= 0 or start_row >= len(work_df):
        return work_df

    target = target_can_id_hex.strip().upper()
    end_row = min(len(work_df), start_row + duration_rows)

    window_slice = work_df.iloc[start_row:end_row]
    target_mask = window_slice[2].astype(str).str.upper().str.startswith(f"{target}#")
    drop_indices = window_slice[target_mask].index

    if len(drop_indices) == 0:
        return work_df

    surviving_window_indices = window_slice[~target_mask].index
    work_df.loc[surviving_window_indices, "label"] = np.int8(SUSPENSION_LABEL)

    after_slice = work_df.iloc[end_row:]
    recovery_matches = after_slice[after_slice[2].astype(str).str.upper().str.startswith(f"{target}#")]
    if not recovery_matches.empty:
        first_recovery_idx = recovery_matches.index[0]
        work_df.at[first_recovery_idx, "label"] = np.int8(SUSPENSION_LABEL)

    return work_df.drop(index=drop_indices).reset_index(drop=True)
