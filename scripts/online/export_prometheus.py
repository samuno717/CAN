"""
Exports Prometheus time series of a recorded run to <run_dir>/prometheus.csv (1 s resolution).

Standard library only, so it runs directly on the Raspberry Pi host (python3) without a virtualenv:
    python3 scripts/online/export_prometheus.py --run-dir reports/online/<run_id>

The time range is taken from <run_dir>/summary.json (first frame -> end of run, +/- margin).
This complements windows.csv.gz / system.csv recorded by the IDS itself; it is mainly useful to plot
the same curves that Grafana shows (e.g. for figures in the thesis).
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

METRICS = [
    "can_ids_anomaly_score_batch_max",
    "can_ids_anomaly_score_rolling_mean",
    "can_ids_threshold",
    "can_ids_alarm_active",
    "can_ids_raw_anomaly_active",
    "can_ids_consecutive_anomalies",
    "can_ids_component_timing_error",
    "can_ids_component_payload_error",
    "can_ids_component_id_error",
    "can_ids_inference_latency_ms",
    "can_ids_batch_inference_latency_ms",
    "can_ids_bus_frames_per_second",
    "can_ids_windows_per_second",
    "can_ids_process_cpu_percent",
    "can_ids_system_cpu_percent",
    "can_ids_process_memory_mb",
    "can_ids_socket_rx_dropped_frames",
    "can_ids_frames_processed_total",
    "can_ids_attack_incidents_total",
]


def query_range(base_url: str, metric: str, start: float, end: float, step: str) -> dict[float, float]:
    # max() collapses the duplicate scrape targets (localhost / host.docker.internal) into one series
    params = urllib.parse.urlencode({"query": f"max({metric})", "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{base_url}/api/v1/query_range?{params}", timeout=30) as resp:
        payload = json.load(resp)
    out: dict[float, float] = {}
    for series in payload.get("data", {}).get("result", []):
        for ts, val in series.get("values", []):
            out[float(ts)] = float(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--step", default="1s")
    ap.add_argument("--margin", type=float, default=5.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    start = (summary.get("first_frame_epoch") or summary["start_epoch"]) - args.margin
    end = summary["end_epoch"] + args.margin

    columns: dict[str, dict[float, float]] = {}
    for metric in METRICS:
        try:
            columns[metric] = query_range(args.prometheus, metric, start, end, args.step)
        except Exception as exc:  # metric missing or Prometheus unreachable
            print(f"[WARN] {metric}: {exc}")
            columns[metric] = {}

    timestamps = sorted({ts for col in columns.values() for ts in col})
    if not timestamps:
        print("[WARN] Prometheus returned no data for this run (was the prometheus container running?)")
        return
    t0 = summary.get("first_frame_epoch") or timestamps[0]
    out = run_dir / "prometheus.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "t_rel_s", *METRICS])
        for ts in timestamps:
            writer.writerow([ts, round(ts - t0, 3), *[columns[m].get(ts, "") for m in METRICS]])
    print(f"[OK] {len(timestamps)} samples -> {out}")


if __name__ == "__main__":
    main()
