"""
Online vs offline consistency check.

Re-computes, with the OFFLINE pipeline (parse_can_data -> transform_attack_features -> model), the anomaly
score of exactly the windows that the online IDS evaluated during a recorded run, and compares the two.

  * IDS_SOURCE=file (log timestamps)   -> scores should be practically identical (differences only from
                                          delta_t clipping / gap reset in StreamingFeatureExtractor)
  * IDS_SOURCE=socketcan (vcan0)       -> differences come from real arrival-time jitter of the stream

Usage (PC, needs torch + scikit-learn):
    uv run python -m scripts.online.check_consistency --run-dir reports/online/<run_id>

Output: <run_dir>/consistency.json and <run_dir>/consistency.png
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    BYTES_SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    PROCESSED_ATTACKS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    TIME_SCALER_PATH,
    WEIGHTS_DIR,
)
from src.ids.detector import load_model_params
from src.models.ae_lstm import AELSTM
from src.models.isolation_forest import CANIsolationForest
from src.utils.feature_prep.scale_features import transform_attack_features
from src.utils.metrics import apply_persistence_filter
from src.utils.raw_data_parser import parse_can_data


def offline_features(scenario: str) -> np.ndarray:
    scenario = scenario.strip().lower()
    if scenario in ("raw", "clean"):
        raw_log = sorted(RAW_DATA_DIR.glob("candump-*.log"))[-1]
        with open(raw_log, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(PROJECT_ROOT / "params.yaml")
            ratio = float(cfg.preprocessing.train_ratio) + float(cfg.preprocessing.val_ratio)
        except Exception:
            ratio = 0.85
        with tempfile.TemporaryDirectory() as tmp:
            tmp_log = Path(tmp) / "raw_test_split.log"
            tmp_log.write_text("".join(lines[int(len(lines) * ratio):]), encoding="utf-8")
            df = parse_can_data(tmp_log)
    else:
        df = parse_can_data(PROCESSED_ATTACKS_DIR / f"{scenario}_attack.log")

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    df = transform_attack_features(df, joblib.load(BYTES_SCALER_PATH), joblib.load(TIME_SCALER_PATH), feature_columns)
    return df.to_numpy(dtype=np.float32)


def score_windows(features: np.ndarray, starts: np.ndarray, model_type: str, weights: Path, seq_len: int) -> np.ndarray:
    scores = np.empty(len(starts), dtype=np.float32)
    chunk = 4096
    if model_type == "ae_lstm":
        mp = load_model_params()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AELSTM(
            num_features=features.shape[1],
            hidden_size=mp["hidden_size"],
            latent_dim=mp["latent_dim"],
            sequence_length=seq_len,
        ).to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            for i in range(0, len(starts), chunk):
                s = starts[i : i + chunk]
                batch = np.stack([features[j : j + seq_len] for j in s]).astype(np.float32)
                x = torch.from_numpy(batch).to(device)
                scores[i : i + len(s)] = torch.mean((model(x) - x) ** 2, dim=(1, 2)).cpu().numpy()
    else:
        forest = CANIsolationForest.load(weights)
        for i in range(0, len(starts), chunk):
            s = starts[i : i + chunk]
            w = np.stack([features[j : j + seq_len] for j in s]).astype(np.float32)
            if forest.feature_mode.strip().lower() == "summary":
                feats = np.concatenate([w.mean(axis=1), w.std(axis=1), w.min(axis=1), w.max(axis=1)], axis=1)
            else:
                feats = w.reshape(len(w), -1)
            scores[i : i + len(s)] = forest.score_windows(feats.astype(np.float32))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    seq_len = int(cfg.get("seq_len", 64))
    k = int(cfg["persistence_k"])
    threshold = float(cfg["threshold"])
    weights = WEIGHTS_DIR / cfg["weights_file"]

    win = pd.read_csv(run_dir / "windows.csv.gz")
    features = offline_features(cfg["scenario"])
    win = win[win["end_frame"] <= len(features)].reset_index(drop=True)
    starts = win["end_frame"].to_numpy() - seq_len

    offline = score_windows(features, starts, cfg["model_type"], weights, seq_len)
    online = win["score"].to_numpy(dtype=np.float32)

    diff = online - offline
    rel = np.abs(diff) / np.maximum(np.abs(offline), 1e-12)
    raw_on = online > threshold
    raw_off = offline > threshold
    alarm_on = win["alarm"].to_numpy().astype(bool)
    alarm_off = apply_persistence_filter(raw_off.astype(np.int8), k).astype(bool)

    result = {
        "run_id": cfg["run_id"],
        "source_mode": cfg["source_mode"],
        "model_type": cfg["model_type"],
        "windows_compared": int(len(win)),
        "pearson_r": float(np.corrcoef(online, offline)[0, 1]) if len(win) > 1 else None,
        "abs_diff": {"mean": float(np.mean(np.abs(diff))), "max": float(np.max(np.abs(diff)))},
        "rel_diff": {
            "p50": float(np.percentile(rel, 50)),
            "p99": float(np.percentile(rel, 99)),
            "share_below_1e-4": float(np.mean(rel < 1e-4)),
        },
        "raw_decision_agreement": float(np.mean(raw_on == raw_off)),
        "raw_decision_disagreements": int(np.sum(raw_on != raw_off)),
        "alarm_agreement": float(np.mean(alarm_on == alarm_off)),
        "alarm_disagreements": int(np.sum(alarm_on != alarm_off)),
    }
    with open(run_dir / "consistency.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(offline, online, s=2, alpha=0.3)
    lo = max(1e-6, float(min(offline.min(), online.min())))
    hi = float(max(offline.max(), online.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.axhline(threshold, color="tab:red", lw=1)
    ax.axvline(threshold, color="tab:red", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wynik offline")
    ax.set_ylabel("Wynik online")
    ax.set_title(f"r = {result['pearson_r']:.5f}")
    ax = axes[1]
    n_show = min(len(win), 5000)
    ax.plot(offline[:n_show], lw=1, label="offline")
    ax.plot(online[:n_show], lw=1, alpha=0.7, label="online")
    ax.axhline(threshold, color="tab:red", lw=1, label="próg")
    ax.set_yscale("log")
    ax.set_xlabel("Okno")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "consistency.png", dpi=150)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
