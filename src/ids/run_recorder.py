"""
Experiment run recorder for the real-time CAN IDS.

Writes everything needed to evaluate an online run offline, without relying on Grafana:

reports/online/<run_id>/
    config.json      - full run configuration (model, threshold, k, stride, source, weights hash, host info)
    windows.csv.gz   - one row per evaluated sliding window (score, raw flag, alarm flag, timings)
    batches.csv      - one row per inference micro-batch (batch size, inference time)
    system.csv       - 1 Hz samples of process/system CPU, RSS memory, SoC temperature and CPU frequency
    alerts.jsonl     - confirmed incidents (INTRUSION_STARTED / INTRUSION_ENDED)
    summary.json     - aggregated statistics written when the run finishes

"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
from loguru import logger

WINDOW_COLUMNS = [
    "win_idx",        # sequential index of the evaluated window (0-based)
    "end_frame",      # number of frames received when the window was closed (1-based) -> frames [end_frame-64, end_frame-1]
    "last_can_id",    # arbitration ID of the last frame in the window (used to verify frame alignment)
    "t_arrival",      # arrival time of the last frame of the window [s since first frame]
    "t_scored",       # time when the window score became available [s since first frame]
    "score",          # anomaly score (MSE for AE-LSTM, IForest score)
    "raw",            # score > threshold
    "alarm",          # persistence-filtered alarm state after this window
    "timing_err",     # diagnostic component: delta_t
    "payload_err",    # diagnostic component: payload bytes
    "id_err",         # diagnostic component: one-hot IDs
]

BATCH_COLUMNS = ["batch_idx", "t_scored", "batch_size", "batch_ms", "per_window_ms"]

SYSTEM_COLUMNS = [
    "t",                 # [s since recorder start]
    "proc_cpu_pct",      # IDS process CPU, % of ONE core (can exceed 100 with several threads)
    "proc_cpu_pct_norm", # IDS process CPU, % of the whole machine (0..100)
    "sys_cpu_pct",       # whole-system CPU utilisation (0..100)
    "rss_mb",            # resident memory of the IDS process [MiB]
    "temp_c",            # SoC temperature [deg C] (Raspberry Pi: thermal_zone0), empty if unavailable
    "cpu_freq_mhz",      # current CPU0 frequency [MHz] (throttling indicator), empty if unavailable
]


def _read_sys_value(path: str, scale: float) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(f.read().strip()) / scale
    except (OSError, ValueError):
        return None


def _sha256_short(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


class _SystemSampler(threading.Thread):
    """Samples CPU / memory / temperature at a fixed interval into system.csv."""

    def __init__(self, csv_path: Path, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True, name="ids-system-sampler")
        self.csv_path = csv_path
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        # Dedicated Process object: psutil keeps cpu_percent() state per object, so this does not
        # interfere with the Prometheus exporter's own measurements.
        self._proc = psutil.Process(os.getpid())
        self._cpu_count = max(1, psutil.cpu_count() or 1)
        self.rows: list[list] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        t0 = time.perf_counter()
        self._proc.cpu_percent(interval=None)
        prev_sys = psutil.cpu_times()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(SYSTEM_COLUMNS)
            while not self._stop_event.wait(self.interval_s):
                try:
                    proc_cpu = self._proc.cpu_percent(interval=None)
                    cur_sys = psutil.cpu_times()
                    busy = (sum(cur_sys) - cur_sys.idle - getattr(cur_sys, "iowait", 0.0)) - (
                        sum(prev_sys) - prev_sys.idle - getattr(prev_sys, "iowait", 0.0)
                    )
                    total = sum(cur_sys) - sum(prev_sys)
                    sys_cpu = 100.0 * busy / total if total > 0 else 0.0
                    prev_sys = cur_sys
                    rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                except psutil.Error:
                    continue
                temp = _read_sys_value("/sys/class/thermal/thermal_zone0/temp", 1000.0)
                freq = _read_sys_value("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0)
                row = [
                    round(time.perf_counter() - t0, 3),
                    round(proc_cpu, 2),
                    round(proc_cpu / self._cpu_count, 2),
                    round(sys_cpu, 2),
                    round(rss_mb, 2),
                    "" if temp is None else round(temp, 1),
                    "" if freq is None else round(freq, 0),
                ]
                self.rows.append(row)
                writer.writerow(row)
                f.flush()


class RunRecorder:
    def __init__(self, root_dir: Path, run_id: str, config: dict) -> None:
        self.run_id = run_id
        self.run_dir = Path(root_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_path = self.run_dir / "alerts.jsonl"
        self._windows_path = self.run_dir / "windows.csv"
        self._batches_path = self.run_dir / "batches.csv"
        self._system_path = self.run_dir / "system.csv"

        self._win_file = open(self._windows_path, "w", newline="", encoding="utf-8", buffering=1 << 20)
        self._win_writer = csv.writer(self._win_file)
        self._win_writer.writerow(WINDOW_COLUMNS)
        self._batch_file = open(self._batches_path, "w", newline="", encoding="utf-8", buffering=1 << 16)
        self._batch_writer = csv.writer(self._batch_file)
        self._batch_writer.writerow(BATCH_COLUMNS)

        self.t0: float | None = None          # perf_counter of the first frame
        self.start_epoch: float = time.time()
        self.first_frame_epoch: float | None = None
        self.n_windows = 0
        self.n_batches = 0
        self._lags: list[np.ndarray] = []
        self._per_window_ms: list[float] = []
        self._batch_sizes: list[int] = []
        self._batch_ms: list[float] = []
        self._finalized = False

        self.config = dict(config)
        self.config.update(
            {
                "run_id": run_id,
                "started_utc": datetime.now(UTC).isoformat(),
                "host": {
                    "hostname": platform.node(),
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "cpu_count": psutil.cpu_count(),
                    "total_ram_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
                },
            }
        )
        for key in ("weights_path", "threshold_path"):
            p = self.config.get(key)
            if p:
                self.config[key.replace("_path", "_sha256")] = _sha256_short(Path(p))
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        self._sampler = _SystemSampler(self._system_path)
        self._sampler.start()
        logger.success(f"[RunRecorder] Recording online run '{run_id}' to {self.run_dir}")

    # ------------------------------------------------------------------ timing helpers
    def mark_first_frame(self, perf_ts: float) -> None:
        if self.t0 is None:
            self.t0 = perf_ts
            self.first_frame_epoch = time.time()

    def rel(self, perf_ts: float) -> float:
        return perf_ts - (self.t0 if self.t0 is not None else perf_ts)

    # ------------------------------------------------------------------ recording
    def record_batch(
        self,
        end_frames: list[int],
        last_can_ids: list[int],
        arrival_perf: list[float],
        scored_perf: float,
        scores: np.ndarray,
        raw_flags: list[bool],
        alarm_flags: list[bool],
        timing_err: np.ndarray,
        payload_err: np.ndarray,
        id_err: np.ndarray,
        batch_ms: float,
    ) -> None:
        n = len(end_frames)
        if n == 0:
            return
        t_scored = self.rel(scored_perf)
        t_arr = np.asarray(arrival_perf, dtype=np.float64) - (self.t0 or 0.0)
        rows = [
            (
                self.n_windows + i,
                end_frames[i],
                last_can_ids[i],
                f"{t_arr[i]:.6f}",
                f"{t_scored:.6f}",
                f"{float(scores[i]):.7g}",
                int(raw_flags[i]),
                int(alarm_flags[i]),
                f"{float(timing_err[i]):.6g}",
                f"{float(payload_err[i]):.6g}",
                f"{float(id_err[i]):.6g}",
            )
            for i in range(n)
        ]
        self._win_writer.writerows(rows)
        self._lags.append(t_scored - t_arr)

        per_win = batch_ms / n
        self._batch_writer.writerow([self.n_batches, f"{t_scored:.6f}", n, f"{batch_ms:.4f}", f"{per_win:.4f}"])
        self._batch_sizes.append(n)
        self._batch_ms.append(batch_ms)
        self._per_window_ms.append(per_win)
        self.n_windows += n
        self.n_batches += 1

    # ------------------------------------------------------------------ finalize
    def finalize(self, stats: dict, keep_plain_csv: bool = False) -> Path:
        if self._finalized:
            return self.run_dir / "summary.json"
        self._finalized = True
        end_epoch = time.time()

        self._sampler.stop()
        self._sampler.join(timeout=3.0)
        self._win_file.close()
        self._batch_file.close()

        # Compress the per-window log AFTER measurement is finished (does not affect measured overhead)
        gz_path = self._windows_path.with_suffix(".csv.gz")
        with open(self._windows_path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        if not keep_plain_csv:
            self._windows_path.unlink(missing_ok=True)

        lags_ms = np.concatenate(self._lags) * 1000.0 if self._lags else np.array([])
        batch_sizes = np.asarray(self._batch_sizes, dtype=np.float64)
        batch_ms = np.asarray(self._batch_ms, dtype=np.float64)
        # Per-window inference time weighted by batch size (= total inference time / total windows)
        weighted_per_win = float(batch_ms.sum() / batch_sizes.sum()) if batch_sizes.size else 0.0

        sys_rows = np.asarray(
            [[np.nan if v == "" else float(v) for v in r] for r in self._sampler.rows], dtype=np.float64
        ) if self._sampler.rows else np.empty((0, len(SYSTEM_COLUMNS)))

        def _col_stats(idx: int) -> dict:
            if sys_rows.size == 0:
                return {}
            col = sys_rows[:, idx]
            col = col[~np.isnan(col)]
            return _percentiles(col)

        wall_s = (end_epoch - self.first_frame_epoch) if self.first_frame_epoch else 0.0
        frames = int(stats.get("frames_received", 0))

        summary = {
            "run_id": self.run_id,
            "start_epoch": self.start_epoch,
            "first_frame_epoch": self.first_frame_epoch,
            "end_epoch": end_epoch,
            "wall_duration_s": wall_s,
            **stats,
            "windows_evaluated": self.n_windows,
            "batches": self.n_batches,
            "throughput": {
                "frames_per_s": frames / wall_s if wall_s > 0 else 0.0,
                "windows_per_s": self.n_windows / wall_s if wall_s > 0 else 0.0,
            },
            "inference_ms_per_window": {
                "weighted_mean": weighted_per_win,
                **{f"batch_{k}": v for k, v in _percentiles(np.asarray(self._per_window_ms)).items()},
            },
            "batch_size": _percentiles(batch_sizes),
            "batch_inference_ms": _percentiles(batch_ms),
            "processing_lag_ms": _percentiles(lags_ms),
            "system": {
                "proc_cpu_pct_one_core": _col_stats(1),
                "proc_cpu_pct_machine": _col_stats(2),
                "sys_cpu_pct": _col_stats(3),
                "rss_mb": _col_stats(4),
                "temp_c": _col_stats(5),
                "cpu_freq_mhz": _col_stats(6),
                "samples": int(sys_rows.shape[0]),
            },
        }
        out = self.run_dir / "summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.success(
            f"[RunRecorder] Run '{self.run_id}' finished: {frames:,} frames, {self.n_windows:,} windows, "
            f"lag p95={summary['processing_lag_ms'].get('p95', 0):.2f} ms, "
            f"inference={weighted_per_win:.3f} ms/window -> {out}"
        )
        return out
