from collections import deque
import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np

from config import BYTES_SCALER_PATH, FEATURE_COLUMNS_PATH, TIME_SCALER_PATH


class StreamingFeatureExtractor:
    """
    Stateful O(1) feature extractor and sliding window ring buffer for real-time CAN bus streams.

    Converts incoming (timestamp, can_id, payload_bytes) frames into the exact 26-dimensional
    scaled feature vector used during model training:
      - index 0:     scaled delta_t (per-ECU inter-arrival time)
      - index 1..8:  scaled byte1..byte8 (payload bytes 0..255)
      - index 9..25: 17 one-hot encoded known ECU arbitration IDs
    """

    def __init__(
        self,
        bytes_scaler_path: Path = BYTES_SCALER_PATH,
        time_scaler_path: Path = TIME_SCALER_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
        seq_len: int = 64,
    ) -> None:
        self.seq_len = seq_len

        bytes_scaler = joblib.load(bytes_scaler_path)
        time_scaler = joblib.load(time_scaler_path)
        with open(feature_columns_path, "r", encoding="utf-8") as f:
            self.feature_columns: list[str] = json.load(f)

        self.num_features = len(self.feature_columns)

        # Extract linear scaling parameters: scaled = x * scale_ + min_
        self.time_scale = float(time_scaler.scale_[0])
        self.time_min = float(time_scaler.min_[0])

        self.bytes_scale = np.asarray(bytes_scaler.scale_, dtype=np.float32)
        self.bytes_min = np.asarray(bytes_scaler.min_, dtype=np.float32)

        # Map known CAN IDs (int) -> feature column index (9..25)
        self.can_id_to_col_idx: dict[int, int] = {}
        for idx, col_name in enumerate(self.feature_columns):
            if col_name.startswith("can_id_"):
                ecu_id = int(col_name.removeprefix("can_id_"))
                self.can_id_to_col_idx[ecu_id] = idx

        self.last_timestamp_per_id: dict[int, float] = {}
        self.last_global_timestamp: float | None = None
        self.window_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
        self.total_frames: int = 0
        self.unknown_id_frames: int = 0

        logger.info(
            f"StreamingFeatureExtractor initialized: {self.num_features} features, "
            f"{len(self.can_id_to_col_idx)} known ECUs, window seq_len={self.seq_len}"
        )

    def reset(self) -> None:
        """Clears state for a new stream session."""
        self.last_timestamp_per_id.clear()
        self.last_global_timestamp = None
        self.window_buffer.clear()
        self.total_frames = 0
        self.unknown_id_frames = 0

    def process_frame(
        self,
        timestamp: float,
        can_id: int,
        payload_bytes: list[int] | bytes | bytearray,
    ) -> tuple[np.ndarray, float, bool]:
        """
        Processes a single CAN frame, updates the ring buffer, and returns:
          (feature_vector_26d, raw_delta_t, is_known_ecu)
        """
        # Detect stream loop reset or long bus pause (> 0.5s gap on whole bus)
        if self.last_global_timestamp is not None and (
            timestamp < self.last_global_timestamp or (timestamp - self.last_global_timestamp) > 0.5
        ):
            self.last_timestamp_per_id.clear()
        self.last_global_timestamp = timestamp

        if can_id in self.last_timestamp_per_id:
            delta_t = round(max(0.0, timestamp - self.last_timestamp_per_id[can_id]), 6)
        else:
            delta_t = 0.0
        self.last_timestamp_per_id[can_id] = timestamp

        vec = np.zeros(self.num_features, dtype=np.float32)

        # 0: scaled delta_t - identical to the offline pipeline (no clipping; long bus pauses are handled
        #    by the per-ID state reset above, stream loop restarts by the timestamp-decrease check)
        vec[0] = np.float32(delta_t * self.time_scale + self.time_min)

        # 1..8: scaled payload bytes (padded with zeros if DLC < 8)
        n_bytes = min(len(payload_bytes), 8)
        if n_bytes > 0:
            raw_b = np.fromiter(payload_bytes[:n_bytes], dtype=np.float32, count=n_bytes)
            if n_bytes < 8:
                padded = np.zeros(8, dtype=np.float32)
                padded[:n_bytes] = raw_b
                raw_b = padded
        else:
            raw_b = np.zeros(8, dtype=np.float32)

        vec[1:9] = raw_b * self.bytes_scale + self.bytes_min

        # 9..25: one-hot CAN ID
        col_idx = self.can_id_to_col_idx.get(can_id)
        is_known = col_idx is not None
        if is_known:
            vec[col_idx] = 1.0
        else:
            self.unknown_id_frames += 1

        self.window_buffer.append(vec)
        self.total_frames += 1
        return vec, delta_t, is_known

    def is_ready(self) -> bool:
        return len(self.window_buffer) == self.seq_len

    def get_current_window(self) -> np.ndarray:
        """Returns current [seq_len, num_features] window array."""
        return np.asarray(self.window_buffer, dtype=np.float32)


def parse_candump_line(line: str) -> tuple[float, int, list[int]] | None:
    """
    Parses a standard candump line:
      '(1777386242.000215) can0 130#08F913801A0100A0'
    Returns (timestamp, can_id, payload_list) or None if invalid.
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None

    try:
        timestamp = float(parts[0].strip("()"))
        frame_spec = parts[2]
        if "#" not in frame_spec:
            return None
        raw_id, raw_payload = frame_spec.split("#", 1)
        can_id = int(raw_id, 16)
        payload = [int(raw_payload[i : i + 2], 16) for i in range(0, min(len(raw_payload), 16), 2)]
        if len(payload) < 8:
            payload.extend([0] * (8 - len(payload)))
        return timestamp, can_id, payload
    except (ValueError, IndexError):
        return None
