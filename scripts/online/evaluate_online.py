"""
Evaluates recorded online IDS runs (reports/online/<run_id>/) against ground-truth attack labels.

Usage (PC with uv, or on the Pi inside the can-ids image):
    uv run python -m scripts.online.evaluate_online --all
    uv run python -m scripts.online.evaluate_online --run-dir reports/online/<run_id>
    docker compose run --rm --no-deps --entrypoint python can-ids -m scripts.online.evaluate_online --all

Output: <run_dir>/evaluation.json with
  - alignment     : how online windows were mapped to log frames (index / time) and a verification rate
  - window_metrics: Precision / Recall / F1 / FPR for raw decisions and persistence-filtered alarms
                    (same definitions as the offline pipeline: a window is "attack" if it contains >= 1 attack frame)
  - per_attack    : recall per attack class
  - events        : event-level detection (was each attack episode detected, and how fast)
  - false_alarms  : alarm onsets outside attack episodes, per hour of benign traffic
  - realtime      : processing lag, inference time, dropped frames, real-time factor
  - system        : CPU / RAM / temperature summary (copied from summary.json)

Metric definitions are imported from src.utils.metrics (numpy only), so they are identical to the
offline pipeline and the script needs no torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ATTACK_NAMES,
    PROCESSED_ATTACKS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from src.utils.metrics import (
    align_window_labels,
    apply_persistence_filter,
    calculate_metrics,
)

ONLINE_DIR = REPORTS_DIR / "online"
EVENT_MERGE_GAP_FRAMES = 2000  # label-c frames closer than this belong to the same attack episode


# ----------------------------------------------------------------------------- helpers
def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "min": float(values.min()),
        "n": int(values.size),
    }


# ----------------------------------------------------------------------------- ground truth
def load_ground_truth(scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (labels[int8], log_timestamps[float64], log_can_ids[int]) for the replayed log."""
    scenario = scenario.strip().lower()
    if scenario in ("raw", "clean"):
        raw_logs = sorted(RAW_DATA_DIR.glob("candump-*.log"))
        log_path = raw_logs[-1]
        df = pd.read_csv(log_path, sep=r"\s+", header=None, usecols=[0, 2], names=["ts", "frame"], engine="c")
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(PROJECT_ROOT / "params.yaml")
            ratio = float(cfg.preprocessing.train_ratio) + float(cfg.preprocessing.val_ratio)
        except Exception:
            ratio = 0.85
        df = df.iloc[int(len(df) * ratio):].reset_index(drop=True)
        labels = np.zeros(len(df), dtype=np.int8)
    else:
        log_path = PROCESSED_ATTACKS_DIR / f"{scenario}_attack.log"
        labels = np.load(PROCESSED_ATTACKS_DIR / f"{scenario}_labels.npy").astype(np.int8)
        df = pd.read_csv(log_path, sep=r"\s+", header=None, usecols=[0, 2], names=["ts", "frame"], engine="c")
        if len(df) != len(labels):
            raise ValueError(f"{log_path.name}: {len(df)} frames but {len(labels)} labels - regenerate the attack")
    ts = df["ts"].str.strip("()").astype(np.float64).to_numpy()
    can_ids = df["frame"].str.split("#", n=1).str[0].map(lambda h: int(h, 16)).to_numpy()
    return labels, ts, can_ids


def find_events(labels: np.ndarray) -> list[dict]:
    """Attack episodes: frames of the same class merged when gaps are < EVENT_MERGE_GAP_FRAMES."""
    events = []
    for cls in [int(c) for c in np.unique(labels) if c > 0]:
        idx = np.flatnonzero(labels == cls)
        splits = np.flatnonzero(np.diff(idx) > EVENT_MERGE_GAP_FRAMES) + 1
        for chunk in np.split(idx, splits):
            events.append({"class": cls, "name": ATTACK_NAMES.get(cls, str(cls)), "start": int(chunk[0]), "end": int(chunk[-1])})
    return sorted(events, key=lambda e: e["start"])


# ----------------------------------------------------------------------------- evaluation
def evaluate_run(run_dir: Path) -> dict:
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    scenario = cfg["scenario"]
    seq_len = int(cfg.get("seq_len", 64))
    k = int(cfg["persistence_k"])
    speed = float(cfg.get("speed", 1.0)) or 1.0

    win = pd.read_csv(run_dir / "windows.csv.gz")
    labels, log_ts, log_ids = load_ground_truth(scenario)
    n_log = len(labels)
    frames_received = int(summary.get("frames_received", 0))
    dropped = int(summary.get("socket_rx_dropped", 0))

    # Only the first pass over the log is evaluated (LOOPS=1 in experiments)
    win = win[win["end_frame"] <= n_log].reset_index(drop=True)

    # --- alignment: online window end_frame (1-based count) <-> log frame index end_frame-1
    end_idx = win["end_frame"].to_numpy() - 1
    id_match = float(np.mean(log_ids[end_idx] == win["last_can_id"].to_numpy())) if len(win) else 0.0
    alignment = "index"
    paced = bool(cfg.get("pacing", True)) or cfg["source_mode"] == "socketcan"
    if (id_match < 0.999 or dropped > 0) and paced:
        # Frames were lost (or reordered): map windows to log frames by relative arrival time instead
        alignment = "time"
        log_rel = (log_ts - log_ts[0]) / speed
        end_idx = np.clip(np.searchsorted(log_rel, win["t_arrival"].to_numpy(), side="right") - 1, seq_len - 1, n_log - 1)
    start_idx = end_idx - (seq_len - 1)

    # Stride-1 labels indexed by window START frame, same rule as the offline pipeline
    y_cls_all = align_window_labels(labels, seq_len, stride=1)
    y_cls = y_cls_all[start_idx]
    y_true = (y_cls > 0).astype(np.int8)
    y_raw = win["raw"].to_numpy().astype(np.int8)
    y_alarm = win["alarm"].to_numpy().astype(np.int8)

    # Persistence filter re-applied offline on the online raw decisions (sanity check of the online filter)
    y_alarm_recomputed = apply_persistence_filter(y_raw, k)

    window_metrics = {
        "raw": calculate_metrics(y_true, y_raw),
        "alarm": calculate_metrics(y_true, y_alarm),
        "alarm_filter_consistent": bool(np.array_equal(y_alarm, y_alarm_recomputed)),
        "windows": int(len(win)),
        "attack_windows": int(y_true.sum()),
    }

    per_attack = {}
    for cls in [int(c) for c in np.unique(y_cls) if c > 0]:
        m = y_cls == cls
        per_attack[ATTACK_NAMES.get(cls, str(cls)).lower()] = {
            "windows": int(m.sum()),
            "raw_recall": float(y_raw[m].mean()),
            "alarm_recall": float(y_alarm[m].mean()),
            "mean_score": float(win["score"].to_numpy()[m].mean()),
        }

    # --- event-level detection & detection delay
    t_arr = win["t_arrival"].to_numpy()
    t_scored = win["t_scored"].to_numpy()
    events = find_events(labels)
    # Partial runs (stream stopped early): ignore episodes the IDS never received
    last_seen = int(end_idx[-1]) if len(end_idx) else -1
    events = [ev for ev in events if ev["start"] <= last_seen]
    for ev in events:
        overlap = (end_idx >= ev["start"]) & (start_idx <= ev["end"])
        alarm_pos = np.flatnonzero(overlap & (y_alarm == 1))
        ev["duration_frames"] = ev["end"] - ev["start"] + 1
        ev["duration_s"] = float(log_ts[ev["end"]] - log_ts[ev["start"]])
        ev["detected"] = bool(alarm_pos.size)
        ev["alarm_coverage"] = float(y_alarm[overlap].mean()) if overlap.any() else 0.0
        if alarm_pos.size:
            j = int(alarm_pos[0])
            first_attack_win = np.flatnonzero(end_idx >= ev["start"])
            ref = int(first_attack_win[0]) if first_attack_win.size else j
            ev["delay_frames"] = int(end_idx[j] - ev["start"])
            ev["delay_log_ms"] = float((log_ts[end_idx[j]] - log_ts[ev["start"]]) * 1000.0)
            # Wall-clock: from arrival of the first attack frame (exact for stride 1) to the moment the alarm was raised
            ev["delay_wall_ms"] = float((t_scored[j] - t_arr[ref]) * 1000.0)
        else:
            ev["delay_frames"] = ev["delay_log_ms"] = ev["delay_wall_ms"] = None

    # --- false alarm onsets outside attack episodes
    onsets = np.flatnonzero(np.diff(np.concatenate([[0], y_alarm])) == 1)
    in_event = np.zeros(len(win), dtype=bool)
    for ev in events:
        # alarms may start up to k windows after the episode ends because of the persistence filter
        in_event |= (end_idx >= ev["start"]) & (start_idx <= ev["end"] + k * int(cfg["stride"]))
    false_onsets = [int(o) for o in onsets if not in_event[o]]
    covered_ts = log_ts[min(end_idx[-1], n_log - 1)] - log_ts[0] if len(win) else 0.0
    attack_s = sum(max(0.0, e["duration_s"]) for e in events)
    benign_h = max(1e-9, (covered_ts - attack_s) / 3600.0)
    false_alarms = {
        "count": len(false_onsets),
        "per_hour": len(false_onsets) / benign_h,
        "benign_hours": benign_h,
        "at_log_seconds": [float(log_ts[end_idx[o]] - log_ts[0]) for o in false_onsets[:50]],
    }

    lag_ms = (t_scored - t_arr) * 1000.0
    log_duration = float(log_ts[-1] - log_ts[0])
    wall = float(summary.get("wall_duration_s", 0.0))
    realtime = {
        "frames_expected": n_log,
        "frames_received": frames_received,
        "frames_missing": max(0, n_log - frames_received),
        "socket_rx_dropped": dropped,
        "processing_lag_ms": _stats(lag_ms),
        "inference_ms_per_window": summary.get("inference_ms_per_window", {}),
        "throughput": summary.get("throughput", {}),
        "log_duration_s": log_duration,
        "wall_duration_s": wall,
        # >= 1 / speed means the IDS kept pace with the replayed stream
        "realtime_factor": (log_duration / wall) if wall > 0 else None,
        "keeps_up": bool(dropped == 0 and frames_received >= n_log),
    }

    result = {
        "run_id": cfg["run_id"],
        "model_type": cfg["model_type"],
        "scenario": scenario,
        "source_mode": cfg["source_mode"],
        "pacing": cfg.get("pacing", True),
        "speed": speed,
        "stride": cfg["stride"],
        "persistence_k": k,
        "threshold": cfg["threshold"],
        "weights_file": cfg.get("weights_file"),
        "run_tag": cfg.get("run_tag", ""),
        "alignment": {"method": alignment, "last_id_match_rate": id_match},
        "window_metrics": window_metrics,
        "per_attack": per_attack,
        "events": events,
        "event_detection_rate": float(np.mean([e["detected"] for e in events])) if events else None,
        "false_alarms": false_alarms,
        "realtime": realtime,
        "system": summary.get("system", {}),
    }
    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _print_short(res: dict) -> None:
    wm = res["window_metrics"]["alarm"]
    rt = res["realtime"]
    lag = rt["processing_lag_ms"].get("p95", float("nan"))
    print(
        f"{res['run_id']:<55} align={res['alignment']['method']:<5} "
        f"F1={wm['f1_score']:.3f} P={wm['precision']:.3f} R={wm['recall']:.3f} "
        f"FA/h={res['false_alarms']['per_hour']:.2f} lag_p95={lag:.1f}ms "
        f"dropped={rt['socket_rx_dropped']} missing={rt['frames_missing']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=str, help="Single run directory")
    ap.add_argument("--all", action="store_true", help="Evaluate every run in reports/online")
    ap.add_argument("--force", action="store_true", help="Re-evaluate runs that already have evaluation.json")
    ap.add_argument("--root", type=str, default=str(ONLINE_DIR))
    args = ap.parse_args()

    if args.run_dir:
        runs = [Path(args.run_dir)]
    elif args.all:
        runs = sorted(p for p in Path(args.root).iterdir() if (p / "summary.json").exists())
        if not args.force:
            runs = [p for p in runs if not (p / "evaluation.json").exists()]
    else:
        ap.error("use --run-dir or --all")

    for run in runs:
        try:
            _print_short(evaluate_run(run))
        except Exception as exc:  # keep going over the remaining runs
            print(f"[ERROR] {run.name}: {exc}")


if __name__ == "__main__":
    main()
