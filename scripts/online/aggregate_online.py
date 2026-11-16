"""
Aggregates all evaluated online runs into thesis-ready tables and plots.

Usage (PC):
    uv run python -m scripts.online.aggregate_online
    uv run python -m scripts.online.aggregate_online --tag main --tag stress

Outputs:
    reports/online/_aggregate/runs.csv               - one row per run (flat)
    reports/online/_aggregate/summary.csv            - mean / std per (tag, model, scenario, source, stride, speed)
    reports/online/_aggregate/detection_table.tex    - LaTeX: detection quality per scenario and model
    reports/online/_aggregate/performance_table.tex  - LaTeX: latency / CPU / RAM per model and stride
    reports/online/_aggregate/events.csv             - every attack episode with detection delay
    plots/online/*.svg                               - latency, CPU and detection-delay figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIR, REPORTS_DIR

ONLINE_DIR = REPORTS_DIR / "online"
OUT_DIR = ONLINE_DIR / "_aggregate"
PLOT_DIR = PLOTS_DIR / "online"
MODEL_LABEL = {"ae_lstm": "AE-LSTM", "iforest": "Isolation Forest"}
SCENARIO_ORDER = ["raw", "dos", "replay", "payload", "frequency", "fuzzing", "suspension", "multi"]
SCENARIO_LABEL = {
    "raw": "Brak ataku", "dos": "DoS", "replay": "Replay", "payload": "Payload", "frequency": "Frequency",
    "fuzzing": "Fuzzing", "suspension": "Suspension", "multi": "Multi-attack",
}


def _get(d: dict, path: str, default=np.nan):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d or d[key] is None:
            return default
        d = d[key]
    return d


def load_runs(root: Path, tags: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, events = [], []
    for ev_path in sorted(root.glob("*/evaluation.json")):
        with open(ev_path, encoding="utf-8") as f:
            ev = json.load(f)
        if tags and ev.get("run_tag") not in tags:
            continue
        cons_path = ev_path.parent / "consistency.json"
        cons = json.load(open(cons_path, encoding="utf-8")) if cons_path.exists() else {}
        rows.append(
            {
                "run_id": ev["run_id"],
                "tag": ev.get("run_tag", ""),
                "model": ev["model_type"],
                "scenario": ev["scenario"],
                "source": ev["source_mode"],
                "pacing": ev.get("pacing", True),
                "stride": ev["stride"],
                "speed": ev["speed"],
                "weights": ev.get("weights_file"),
                "alignment": _get(ev, "alignment.method", ""),
                "precision": _get(ev, "window_metrics.alarm.precision"),
                "recall": _get(ev, "window_metrics.alarm.recall"),
                "f1": _get(ev, "window_metrics.alarm.f1_score"),
                "fpr": _get(ev, "window_metrics.alarm.false_positive_rate"),
                "raw_f1": _get(ev, "window_metrics.raw.f1_score"),
                "raw_fpr": _get(ev, "window_metrics.raw.false_positive_rate"),
                "event_detection_rate": ev.get("event_detection_rate", np.nan),
                "false_alarms_per_h": _get(ev, "false_alarms.per_hour"),
                "lag_p50_ms": _get(ev, "realtime.processing_lag_ms.p50"),
                "lag_p95_ms": _get(ev, "realtime.processing_lag_ms.p95"),
                "lag_p99_ms": _get(ev, "realtime.processing_lag_ms.p99"),
                "lag_max_ms": _get(ev, "realtime.processing_lag_ms.max"),
                "infer_ms_per_window": _get(ev, "realtime.inference_ms_per_window.weighted_mean"),
                "frames_per_s": _get(ev, "realtime.throughput.frames_per_s"),
                "windows_per_s": _get(ev, "realtime.throughput.windows_per_s"),
                "realtime_factor": _get(ev, "realtime.realtime_factor"),
                "dropped": _get(ev, "realtime.socket_rx_dropped", 0),
                "frames_missing": _get(ev, "realtime.frames_missing", 0),
                "keeps_up": _get(ev, "realtime.keeps_up", False),
                "cpu_one_core_mean": _get(ev, "system.proc_cpu_pct_one_core.mean"),
                "cpu_one_core_p95": _get(ev, "system.proc_cpu_pct_one_core.p95"),
                "cpu_machine_mean": _get(ev, "system.proc_cpu_pct_machine.mean"),
                "sys_cpu_mean": _get(ev, "system.sys_cpu_pct.mean"),
                "rss_max_mb": _get(ev, "system.rss_mb.max"),
                "temp_max_c": _get(ev, "system.temp_c.max"),
                "cpu_freq_min_mhz": _get(ev, "system.cpu_freq_mhz.min"),
                "consistency_r": cons.get("pearson_r", np.nan),
                "consistency_alarm_agreement": cons.get("alarm_agreement", np.nan),
            }
        )
        for e in ev.get("events", []):
            events.append(
                {
                    "run_id": ev["run_id"], "tag": ev.get("run_tag", ""), "model": ev["model_type"],
                    "scenario": ev["scenario"], "stride": ev["stride"], "speed": ev["speed"],
                    "attack": e["name"], "detected": e["detected"], "duration_s": e["duration_s"],
                    "alarm_coverage": e["alarm_coverage"], "delay_frames": e["delay_frames"],
                    "delay_log_ms": e["delay_log_ms"], "delay_wall_ms": e["delay_wall_ms"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(events)


def mean_std(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std) or std == 0:
        return f"{mean:.{digits}f}".replace(".", ",")
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}".replace(".", ",")


def detection_table(summary: pd.DataFrame) -> str:
    s = summary[(summary["speed"] == 1)]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Skuteczność detekcji w trybie online (Raspberry Pi 5, vcan0, filtr persystencji)}",
        "\\label{tab:online_detection}",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Scenariusz & Model & Precision & Recall & F1 & Fałszywe alarmy/h \\\\",
        "\\hline",
    ]
    for scen in SCENARIO_ORDER:
        for model in ("ae_lstm", "iforest"):
            r = s[(s["scenario"] == scen) & (s["model"] == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"{SCENARIO_LABEL[scen]} & {MODEL_LABEL[model]} & {fmt(r.precision_mean, r.precision_std)} & "
                f"{fmt(r.recall_mean, r.recall_std)} & {fmt(r.f1_mean, r.f1_std)} & "
                f"{fmt(r.false_alarms_per_h_mean, r.false_alarms_per_h_std, 2)} \\\\"
            )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def performance_table(summary: pd.DataFrame) -> str:
    s = summary[(summary["speed"] == 1)].groupby(["model", "stride"], as_index=False).agg(
        infer=("infer_ms_per_window_mean", "mean"),
        lag95=("lag_p95_ms_mean", "mean"),
        lag99=("lag_p99_ms_mean", "mean"),
        cpu=("cpu_one_core_mean_mean", "mean"),
        rss=("rss_max_mb_mean", "max"),
        wps=("windows_per_s_mean", "mean"),
        dropped=("dropped_mean", "max"),
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wydajność systemu IDS na Raspberry Pi 5 (średnia ze wszystkich scenariuszy)}",
        "\\label{tab:online_performance}",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "Model & Krok & Inferencja [ms/okno] & Opóźn. p95 [ms] & Opóźn. p99 [ms] & CPU [\\% rdzenia] & RAM [MiB] & Okna/s \\\\",
        "\\hline",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"{MODEL_LABEL.get(r.model, r.model)} & {int(r.stride)} & {r.infer:.3f} & {r.lag95:.1f} & {r.lag99:.1f} & "
            f"{r.cpu:.1f} & {r.rss:.0f} & {r.wps:.0f} \\\\".replace(".", ",")
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def plots(runs: pd.DataFrame, events: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    base = runs[runs["speed"] == 1]

    # 1. Processing lag & inference time per model / stride
    if not base.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        groups = base.groupby(["model", "stride"])
        labels = [f"{MODEL_LABEL.get(m, m)}\nkrok={s}" for (m, s) in groups.groups]
        axes[0].bar(labels, groups["infer_ms_per_window"].mean(), yerr=groups["infer_ms_per_window"].std(), capsize=4)
        axes[0].set_ylabel("Czas inferencji [ms / okno]")
        p50 = groups["lag_p50_ms"].mean().to_numpy()
        p95 = groups["lag_p95_ms"].mean().to_numpy()
        p99 = groups["lag_p99_ms"].mean().to_numpy()
        x = np.arange(len(labels))
        axes[1].bar(x - 0.25, p50, 0.25, label="p50")
        axes[1].bar(x, p95, 0.25, label="p95")
        axes[1].bar(x + 0.25, p99, 0.25, label="p99")
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("Opóźnienie przetwarzania okna [ms]")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "online_latency.svg")
        plt.close(fig)

        # 2. CPU and RAM
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(labels, groups["cpu_one_core_mean"].mean(), yerr=groups["cpu_one_core_mean"].std(), capsize=4)
        axes[0].set_ylabel("CPU procesu IDS [% jednego rdzenia]")
        axes[1].bar(labels, groups["rss_max_mb"].max())
        axes[1].set_ylabel("Maks. pamięć RSS [MiB]")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "online_resources.svg")
        plt.close(fig)

    # 3. Stress test: throughput vs replay speed
    stress = runs[runs["speed"] != 1]
    if not stress.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (model, stride), g in pd.concat([base, stress]).groupby(["model", "stride"]):
            g = g.groupby("speed", as_index=False).agg(fps=("frames_per_s", "mean"), lag=("lag_p95_ms", "mean"))
            ax.plot(g["speed"], g["lag"], marker="o", label=f"{MODEL_LABEL.get(model, model)}, krok={stride}")
        ax.set_xlabel("Przyspieszenie odtwarzania [x]")
        ax.set_ylabel("Opóźnienie p95 [ms]")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "online_stress.svg")
        plt.close(fig)

    # 4. Detection delay per attack and model
    ev = events[(events["speed"] == 1) & events["detected"]] if not events.empty else events
    if not ev.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        attacks = sorted(ev["attack"].unique())
        x = np.arange(len(attacks))
        models = sorted(ev["model"].unique())
        w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            vals = [ev[(ev["model"] == m) & (ev["attack"] == a)]["delay_wall_ms"].mean() for a in attacks]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=MODEL_LABEL.get(m, m))
        ax.set_xticks(x, attacks)
        ax.set_ylabel("Czas do wykrycia [ms]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "online_detection_delay.svg")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ONLINE_DIR))
    ap.add_argument("--tag", action="append", help="Only runs with this RUN_TAG (repeatable)")
    args = ap.parse_args()

    runs, events = load_runs(Path(args.root), args.tag)
    if runs.empty:
        print("No evaluated runs found (run scripts.online.evaluate_online first).")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    events.to_csv(OUT_DIR / "events.csv", index=False)

    metric_cols = [
        "precision", "recall", "f1", "fpr", "raw_f1", "event_detection_rate", "false_alarms_per_h",
        "lag_p50_ms", "lag_p95_ms", "lag_p99_ms", "infer_ms_per_window", "frames_per_s", "windows_per_s",
        "dropped", "cpu_one_core_mean", "cpu_machine_mean", "rss_max_mb", "temp_max_c",
    ]
    summary = mean_std(runs, ["tag", "model", "scenario", "source", "stride", "speed"], metric_cols)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    main_runs = summary[summary["source"] == "socketcan"]
    if not main_runs.empty:
        (OUT_DIR / "detection_table.tex").write_text(detection_table(main_runs), encoding="utf-8")
        (OUT_DIR / "performance_table.tex").write_text(performance_table(main_runs), encoding="utf-8")
    plots(runs, events)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["tag", "model", "scenario", "stride", "speed", "n_runs", "f1_mean", "recall_mean",
                       "false_alarms_per_h_mean", "lag_p95_ms_mean", "infer_ms_per_window_mean",
                       "cpu_one_core_mean_mean", "dropped_mean"]].round(4).to_string(index=False))
    print(f"\nTables -> {OUT_DIR}\nPlots  -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
