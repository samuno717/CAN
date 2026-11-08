import numpy as np
import pandas as pd

from config import PAYLOAD_LABEL


def _mutate_hex_payload(
    raw_payload: str,
    mode: str,
    target_bytes: list[int],
    custom_payload_hex: str,
    rng: np.random.Generator,
    step_idx: int = 0,
    drift_step: int = 2,
) -> str:
    n_bytes = max(1, len(raw_payload) // 2)
    bytes_arr = [int(raw_payload[i : i + 2], 16) for i in range(0, 2 * n_bytes, 2)]

    if mode == "constant":
        padded = custom_payload_hex.strip().upper().ljust(2 * n_bytes, "0")[: 2 * n_bytes]
        for idx in target_bytes:
            if 0 <= idx < n_bytes:
                bytes_arr[idx] = int(padded[2 * idx : 2 * idx + 2], 16)
    elif mode == "random":
        for idx in target_bytes:
            if 0 <= idx < n_bytes:
                bytes_arr[idx] = int(rng.integers(0, 256))
    elif mode == "invert":
        for idx in target_bytes:
            if 0 <= idx < n_bytes:
                bytes_arr[idx] = bytes_arr[idx] ^ 0xFF
    elif mode == "drift":
        for idx in target_bytes:
            if 0 <= idx < n_bytes:
                bytes_arr[idx] = (bytes_arr[idx] + step_idx * drift_step) % 256
    else:
        raise ValueError(
            f"Unsupported masquerade payload mode: '{mode}'. Choose from: random, invert, constant, drift."
        )

    return "".join(f"{b:02X}" for b in bytes_arr)


def inject_payload_manipulation_df(
    df: pd.DataFrame,
    start_row: int = 15000,
    duration_rows: int = 2000,
    target_can_id_hex: str = "130",
    mode: str = "random",
    target_bytes: list[int] | None = None,
    custom_payload_hex: str = "FFFFFFFFFFFFFFFF",
    drift_step: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    work_df = df.copy()
    if "label" not in work_df.columns:
        work_df["label"] = np.int8(0)

    if target_bytes is None:
        target_bytes = list(range(8))
    else:
        target_bytes = [int(b) for b in target_bytes]

    rng = np.random.default_rng(seed)
    target = target_can_id_hex.strip().upper()
    mode_norm = mode.strip().lower()

    end_row = min(len(work_df), start_row + duration_rows)
    segment_indices = work_df.index[start_row:end_row]

    step_counter = 0
    for idx in segment_indices:
        frame_str = str(work_df.at[idx, 2])
        can_id_str, raw_payload = frame_str.split("#", 1)

        if target != "ALL" and can_id_str.upper() != target:
            continue

        step_counter += 1
        new_payload = _mutate_hex_payload(
            raw_payload=raw_payload,
            mode=mode_norm,
            target_bytes=target_bytes,
            custom_payload_hex=custom_payload_hex,
            rng=rng,
            step_idx=step_counter,
            drift_step=drift_step,
        )
        work_df.at[idx, 2] = f"{can_id_str}#{new_payload}"
        work_df.at[idx, "label"] = np.int8(PAYLOAD_LABEL)

    return work_df
