from pathlib import Path

import numpy as np
import pandas as pd


def load_test_raw_df(raw_data: Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> pd.DataFrame:
    """Loads clean CAN frames from the test split without modifying the raw file."""
    with open(raw_data, "r", encoding="utf-8") as logfile:
        lines = [line.strip() for line in logfile if line.strip()]

    val_end = int(len(lines) * (train_ratio + val_ratio))
    test_rows = [line.split() for line in lines[val_end:]]
    df = pd.DataFrame(test_rows)
    df["label"] = np.int8(0)
    return df


def extract_timestamps(df: pd.DataFrame) -> np.ndarray:
    """Extracts float timestamps from column 0 formatted as '(1777386242.014469)'."""
    return df[0].astype(str).str.strip("()").astype(float).to_numpy()


def format_timestamps(ts: np.ndarray) -> pd.Series:
    """Formats float timestamps back into candump '(1777386242.014469)' format."""
    return "(" + pd.Series(ts).map("{:.6f}".format) + ")"


def merge_and_arbitrate(
    base_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    min_bus_gap: float = 0.00020,
) -> pd.DataFrame:
    """Merges injected CAN frames into the stream, sorts, and forces minimum physical bus gap."""
    if injected_df.empty:
        return base_df.drop(columns=["_ts"], errors="ignore")

    merged = (
        pd.concat([base_df, injected_df], ignore_index=True)
        .sort_values("_ts", kind="mergesort")
        .reset_index(drop=True)
    )

    ts = merged["_ts"].to_numpy(copy=True)
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1] + min_bus_gap:
            ts[i] = ts[i - 1] + min_bus_gap

    merged[0] = format_timestamps(ts)
    return merged.drop(columns=["_ts"])


def save_attack_dataset(df: pd.DataFrame, log_path: Path, labels_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    labels = df["label"].to_numpy(dtype=np.int8)
    np.save(labels_path, labels)

    lines = (
        df[0].astype(str) + " " + df[1].astype(str) + " " + df[2].astype(str) + "\n"
    ).tolist()
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
