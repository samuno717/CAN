import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger

from config import METRICS_DIR, PROJECT_ROOT, WEIGHTS_DIR
from src.models.ae_lstm import AELSTM
from src.models.isolation_forest import CANIsolationForest

ModelType = Literal["ae_lstm", "iforest"]


def load_model_params(params_path: Path = PROJECT_ROOT / "params.yaml") -> dict:
    params = {
        "ae_weights": "ae_lstm_can_model_h64_e400.pth",
        "iforest_weights": "isolation_forest_n200_summary.pkl",
        "hidden_size": 64,
        "latent_dim": 16,
        "seq_len": 64,
        "num_features": 26,
    }
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(params_path)
        params.update(
            {
                "ae_weights": str(cfg.model.path),
                "iforest_weights": str(cfg.iforest.path),
                "hidden_size": int(cfg.model.hidden_size),
                "latent_dim": int(cfg.model.latent_dim),
                "seq_len": int(cfg.model.sequence_length),
                "num_features": int(cfg.model.num_features),
            }
        )
    except Exception as exc:  # params.yaml missing or omegaconf unavailable -> keep defaults
        logger.warning(f"[IDSDetector] Could not read {params_path} ({exc}); using built-in defaults.")

    if os.getenv("AE_WEIGHTS"):
        params["ae_weights"] = os.environ["AE_WEIGHTS"]
    if os.getenv("IFOREST_WEIGHTS"):
        params["iforest_weights"] = os.environ["IFOREST_WEIGHTS"]
    return params


@dataclass
class BatchScoreResult:
    scores: np.ndarray               # [B] primary anomaly score (MSE or IForest score)
    timing_errors: np.ndarray        # [B] error/deviation on feature 0 (delta_t)
    payload_errors: np.ndarray       # [B] error/deviation on features 1..8 (bytes)
    id_errors: np.ndarray            # [B] error/deviation on features 9..25 (one-hot IDs)
    inference_ms: float              # Total batch inference latency in ms
    per_window_ms: float             # Latency per window in ms


class IDSDetector:

    def __init__(
        self,
        model_type: ModelType = "ae_lstm",
        persistence_k: int = 10,
        seq_len: int | None = None,
        num_features: int | None = None,
        hidden_size: int | None = None,
        latent_dim: int | None = None,
        ae_weights_path: Path | None = None,
        iforest_weights_path: Path | None = None,
        ae_threshold_path: Path = METRICS_DIR / "threshold.json",
        iforest_threshold_path: Path = METRICS_DIR / "iforest_threshold.json",
        custom_threshold: float | None = None,
        device_str: str = "auto",
        torch_threads: int = 4,
    ) -> None:
        self.model_type: ModelType = model_type.strip().lower()  # type: ignore[assignment]
        if self.model_type not in ("ae_lstm", "iforest"):
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'ae_lstm' or 'iforest'.")

        mp = load_model_params()
        seq_len = seq_len or mp["seq_len"]
        num_features = num_features or mp["num_features"]
        hidden_size = hidden_size or mp["hidden_size"]
        latent_dim = latent_dim or mp["latent_dim"]
        ae_weights_path = ae_weights_path or (WEIGHTS_DIR / mp["ae_weights"])
        iforest_weights_path = iforest_weights_path or (WEIGHTS_DIR / mp["iforest_weights"])

        self.persistence_k = persistence_k
        self.seq_len = seq_len
        self.num_features = num_features
        self.weights_path: Path = ae_weights_path if self.model_type == "ae_lstm" else iforest_weights_path
        self.threshold_path: Path = ae_threshold_path if self.model_type == "ae_lstm" else iforest_threshold_path

        self.consecutive_anomalies: int = 0
        self.in_alarm: bool = False

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        if self.device.type == "cpu" and torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self.ae_model: AELSTM | None = None
        self.iforest_model: CANIsolationForest | None = None

        if self.model_type == "ae_lstm":
            with open(ae_threshold_path, "r", encoding="utf-8") as f:
                default_thresh = float(json.load(f)["threshold"])
            self.threshold = float(custom_threshold) if custom_threshold is not None else default_thresh

            self.ae_model = AELSTM(
                num_features=num_features,
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                sequence_length=seq_len,
            ).to(self.device)
            state_dict = torch.load(ae_weights_path, map_location=self.device, weights_only=True)
            self.ae_model.load_state_dict(state_dict)
            self.ae_model.eval()

            # Warmup pass for accurate latency measurement on rpi CPU
            with torch.no_grad():
                dummy = torch.zeros((1, seq_len, num_features), dtype=torch.float32, device=self.device)
                _ = self.ae_model(dummy)

            logger.success(
                f"[IDSDetector] Loaded AE-LSTM from '{ae_weights_path.name}' on {self.device} "
                f"(threshold={self.threshold:.5f}, persistence_k={self.persistence_k})"
            )
        else:
            with open(iforest_threshold_path, "r", encoding="utf-8") as f:
                default_thresh = float(json.load(f)["threshold"])
            self.threshold = float(custom_threshold) if custom_threshold is not None else default_thresh

            self.iforest_model = CANIsolationForest.load(iforest_weights_path)
            self.iforest_model.model.n_jobs = 1

            dummy_feat = np.zeros((1, num_features * 4), dtype=np.float32)
            _ = self.iforest_model.score_windows(dummy_feat)

            logger.success(
                f"[IDSDetector] Loaded Isolation Forest from '{iforest_weights_path.name}' "
                f"(threshold={self.threshold:.5f}, persistence_k={self.persistence_k})"
            )

    def reset_state(self) -> None:
        self.consecutive_anomalies = 0
        self.in_alarm = False

    def score_batch(self, windows_3d: np.ndarray) -> BatchScoreResult:
        batch_size = len(windows_3d)
        t0 = time.perf_counter()

        if self.model_type == "ae_lstm":
            assert self.ae_model is not None
            with torch.no_grad():
                tensor_in = torch.from_numpy(windows_3d).to(self.device)
                recon = self.ae_model(tensor_in)
                sq_err = (recon - tensor_in) ** 2  # [B, seq_len, 26]

                scores = torch.mean(sq_err, dim=(1, 2)).cpu().numpy()
                timing_err = torch.mean(sq_err[:, :, 0], dim=1).cpu().numpy()
                payload_err = torch.mean(sq_err[:, :, 1:9], dim=(1, 2)).cpu().numpy()
                id_err = torch.mean(sq_err[:, :, 9:], dim=(1, 2)).cpu().numpy()
        else:
            assert self.iforest_model is not None
            mode = self.iforest_model.feature_mode.strip().lower()
            if mode == "summary":
                w_mean = windows_3d.mean(axis=1)
                w_std = windows_3d.std(axis=1)
                w_min = windows_3d.min(axis=1)
                w_max = windows_3d.max(axis=1)
                feats = np.concatenate([w_mean, w_std, w_min, w_max], axis=1).astype(np.float32)
            else:
                w_std = windows_3d.std(axis=1)
                feats = windows_3d.reshape(batch_size, -1).astype(np.float32)

            scores = self.iforest_model.score_windows(feats)
            timing_err = w_std[:, 0].astype(np.float32)
            payload_err = w_std[:, 1:9].mean(axis=1).astype(np.float32)
            id_err = w_std[:, 9:].mean(axis=1).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_win_ms = elapsed_ms / max(1, batch_size)

        return BatchScoreResult(
            scores=scores.astype(np.float32),
            timing_errors=timing_err.astype(np.float32),
            payload_errors=payload_err.astype(np.float32),
            id_errors=id_err.astype(np.float32),
            inference_ms=elapsed_ms,
            per_window_ms=per_win_ms,
        )

    def update_persistence(self, score: float) -> tuple[bool, bool, int]:
        is_raw = bool(score > self.threshold)
        if is_raw:
            self.consecutive_anomalies += 1
            if self.consecutive_anomalies >= self.persistence_k:
                self.in_alarm = True
        else:
            self.consecutive_anomalies = 0
            self.in_alarm = False

        return is_raw, self.in_alarm, self.consecutive_anomalies
