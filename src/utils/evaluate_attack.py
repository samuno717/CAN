import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from config import ATTACK_COLORS, ATTACK_NAMES
from src.utils.evaluator import apply_persistence_filter, calculate_metrics


def align_window_labels(labels: np.ndarray, sequence_length: int = 64, stride: int = 1) -> np.ndarray:
    """Assigns an attack label to a window if any frame inside the window is an attack."""
    windows = sliding_window_view(labels, window_shape=sequence_length)[::stride]
    return windows.max(axis=1).astype(np.int8)


def compute_detection_metrics(
    errors: np.ndarray,
    y_true_cls: np.ndarray,
    threshold: float,
    persistence_k: int = 10,
    scenario_name: str = "attack",
    metrics_save_path: Path | None = None,
) -> dict:
    """Computes global and per-attack detection metrics and saves them to JSON."""
    y_true_bin = (y_true_cls > 0).astype(np.int8)
    y_pred_raw = (errors > threshold).astype(np.int8)
    y_pred_pers = apply_persistence_filter(y_pred_raw, k=persistence_k)

    raw_stats = calculate_metrics(y_true_bin, y_pred_raw, threshold)
    pers_stats = calculate_metrics(y_true_bin, y_pred_pers, threshold)

    per_attack = {}
    for cid in [int(c) for c in np.unique(y_true_cls) if c > 0]:
        name = ATTACK_NAMES.get(cid, f"Attack #{cid}")
        mask = y_true_cls == cid
        total = int(np.sum(mask))
        raw_recall = float(np.mean(y_pred_raw[mask])) if total > 0 else 0.0
        pers_recall = float(np.mean(y_pred_pers[mask])) if total > 0 else 0.0

        per_attack[name.lower()] = {
            "display_name": name,
            "attack_windows": total,
            "mean_mse": float(np.mean(errors[mask])),
            "raw_recall": raw_recall,
            "persistence_recall": pers_recall,
        }
        logger.info(
            f"  -> [{name}] windows={total:,} | raw_recall={raw_recall:.4f} | persist_recall={pers_recall:.4f}"
        )

    logger.success(
        f"[RAW]         Precision: {raw_stats['precision']:.4f} | "
        f"Recall: {raw_stats['recall']:.4f} | F1: {raw_stats['f1_score']:.4f}"
    )
    logger.success(
        f"[PERSIST k={persistence_k}] Precision: {pers_stats['precision']:.4f} | "
        f"Recall: {pers_stats['recall']:.4f} | F1: {pers_stats['f1_score']:.4f}"
    )

    benign_mask = y_true_bin == 0
    attack_mask = y_true_bin == 1
    metrics = {
        "scenario": scenario_name,
        "threshold": float(threshold),
        "persistence_k": int(persistence_k),
        "total_windows": len(errors),
        "attack_windows": int(np.sum(attack_mask)),
        "benign_windows": int(np.sum(benign_mask)),
        "error_summary": {
            "benign_mean_mse": float(np.mean(errors[benign_mask])) if np.any(benign_mask) else 0.0,
            "attack_mean_mse": float(np.mean(errors[attack_mask])) if np.any(attack_mask) else 0.0,
            "attack_max_mse": float(np.max(errors[attack_mask])) if np.any(attack_mask) else 0.0,
        },
        "per_attack": per_attack,
        "raw": raw_stats,
        "persistence_filtered": pers_stats,
    }

    if metrics_save_path:
        metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.success(f"Metrics saved to: {metrics_save_path}")

    return metrics


def _shade_regions(
    ax,
    x_indices: np.ndarray,
    color: str,
    label: str,
    ymin: float = 0.0,
    ymax: float = 1.0,
    alpha: float = 0.23,
    zorder: int | None = None,
) -> None:
    if len(x_indices) == 0:
        return
    splits = np.where(np.diff(x_indices) > 1)[0] + 1
    for i, region in enumerate(np.split(x_indices, splits)):
        ax.axvspan(
            region[0],
            region[-1],
            ymin=ymin,
            ymax=ymax,
            color=color,
            alpha=alpha,
            zorder=zorder,
            label=label if i == 0 else "",
        )


def plot_attack_evaluation(
    errors: np.ndarray,
    y_true_cls: np.ndarray,
    threshold: float,
    save_path: Path,
    persistence_k: int = 10,
    metrics: dict | None = None,
    scenario_title: str = "Attack Evaluation",
) -> None:
    """Plots zoomed view around injected attacks (top) and full test stream overview (bottom)."""
    y_pred_pers = apply_persistence_filter((errors > threshold).astype(np.int8), k=persistence_k)
    attack_ids = [int(c) for c in np.unique(y_true_cls) if c > 0]

    # Zoom bounds covering all attacks with a 2000-window margin
    atk_idx = np.where(y_true_cls > 0)[0]
    z_start = max(0, int(atk_idx[0]) - 2000) if len(atk_idx) > 0 else 0
    z_end = min(len(errors), int(atk_idx[-1]) + 2000) if len(atk_idx) > 0 else min(10000, len(errors))

    fig, (ax_zoom, ax_full) = plt.subplots(
        2, 1, figsize=(16, 10), dpi=300, gridspec_kw={"height_ratios": [1.8, 1.0], "hspace": 0.32}
    )

    # 1. Top panel: Zoomed attack zone
    x_zoom = np.arange(z_start, z_end)
    err_zoom = errors[z_start:z_end]
    cls_zoom = y_true_cls[z_start:z_end]
    pers_zoom = y_pred_pers[z_start:z_end]

    ax_zoom.plot(x_zoom, err_zoom, color="#1C3D5A", linewidth=1.1, zorder=3, label="Reconstruction MSE")
    ax_zoom.axhline(
        threshold, color="#E74C3C", linestyle="--", linewidth=1.5, zorder=4, label=f"Threshold ({threshold:.5f})"
    )

    # Reserve top 12% of the axis for the alarm status bar so it never touches the MSE curve
    max_y = max(float(np.max(err_zoom)), threshold)
    ax_zoom.set_ylim(bottom=0.0, top=max_y * 1.16)

    for cid in attack_ids:
        name = ATTACK_NAMES.get(cid, f"Attack #{cid}")
        color = ATTACK_COLORS.get(cid, "#D97706")
        _shade_regions(ax_zoom, x_zoom[cls_zoom == cid], color, f"Ground-Truth: {name}", ymax=0.93)

    # Top status bar strip (ymin=0.94..0.99) showing active persistence-filtered alarm
    ax_zoom.axhline(max_y * 1.16 * 0.935, color="#CBD5E1", linestyle=":", linewidth=0.9, zorder=2)
    alarms = x_zoom[pers_zoom == 1]
    if len(alarms) > 0:
        _shade_regions(
            ax_zoom,
            alarms,
            color="#D97706",
            label=f"Active Alarm Bar (k={persistence_k})",
            ymin=0.94,
            ymax=0.99,
            alpha=0.92,
            zorder=5,
        )

    ax_zoom.set_title(f"IDS Response — {scenario_title} (Zoomed)", fontsize=13, fontweight="bold")
    ax_zoom.set_xlabel("Sliding Window Index")
    ax_zoom.set_ylabel("Mean Squared Error (MSE)")
    ax_zoom.set_xlim(z_start, z_end)
    ax_zoom.grid(True, linestyle="--", alpha=0.5)

    leg = ax_zoom.legend(bbox_to_anchor=(1.015, 1.0), loc="upper left", borderaxespad=0.0, framealpha=0.95)

    # 2. Bottom panel: Downsampled full test stream
    n_bins = min(2000, len(errors))
    edges = np.linspace(0, len(errors), n_bins + 1, dtype=int)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_max = [np.max(errors[edges[i] : edges[i + 1]]) for i in range(n_bins)]
    bin_mean = [np.mean(errors[edges[i] : edges[i + 1]]) for i in range(n_bins)]

    ax_full.plot(centers, bin_max, color="#64748B", alpha=0.55, linewidth=0.9, label="Peak MSE (Full Stream)")
    ax_full.plot(centers, bin_mean, color="#1E3A8A", linewidth=1.2, label="Mean MSE (Full Stream)")
    ax_full.axhline(threshold, color="#E74C3C", linestyle="--", linewidth=1.4, label="Threshold")

    for cid in attack_ids:
        name = ATTACK_NAMES.get(cid, f"Attack #{cid}")
        color = ATTACK_COLORS.get(cid, "#D97706")
        first_idx = int(np.where(y_true_cls == cid)[0][0])
        ax_full.axvline(first_idx, color=color, linestyle=":", linewidth=1.6, label=f"Injected {name}")

    ax_full.set_title(f"Full Test Stream Overview (N = {len(errors):,} Windows)", fontsize=11, fontweight="bold")
    ax_full.set_xlabel("Sliding Window Index")
    ax_full.set_ylabel("MSE")
    ax_full.set_xlim(0, len(errors))
    ax_full.grid(True, linestyle=":", alpha=0.5)
    ax_full.legend(loc="upper right", framealpha=0.92, fontsize=8.8)

    # Summary box on the right
    if metrics is not None:
        raw_m = metrics["raw"]
        pers_m = metrics["persistence_filtered"]
        err_s = metrics["error_summary"]

        per_atk_text = ""
        for info in metrics.get("per_attack", {}).values():
            per_atk_text += (
                f"  • {info['display_name']:<9}: "
                f"{info['raw_recall'] * 100:>5.1f}% / {info['persistence_recall'] * 100:>5.1f}%\n"
            )

        info_text = (
            "Detection Summary:\n"
            f"  • Threshold       : {threshold:>9.5f}\n"
            f"  • Attack windows  : {metrics['attack_windows']:>9,}\n"
            f"  • Benign mean MSE : {err_s['benign_mean_mse']:>9.5f}\n"
            f"  • Attack mean MSE : {err_s['attack_mean_mse']:>9.5f}\n\n"
            "Per-Attack Recall (Raw / Persist):\n"
            f"{per_atk_text}\n"
            "Global Raw:\n"
            f"  • Precision       : {raw_m['precision']:>9.4f}\n"
            f"  • Recall          : {raw_m['recall']:>9.4f}\n"
            f"  • F1-Score        : {raw_m['f1_score']:>9.4f}\n\n"
            f"Global Persistence (k={persistence_k}):\n"
            f"  • Precision       : {pers_m['precision']:>9.4f}\n"
            f"  • Recall          : {pers_m['recall']:>9.4f}\n"
            f"  • F1-Score        : {pers_m['f1_score']:>9.4f}"
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_bbox = leg.get_window_extent(renderer).transformed(ax_zoom.transAxes.inverted())

        ax_zoom.text(
            1.015,
            leg_bbox.ymin - 0.04,
            info_text,
            transform=ax_zoom.transAxes,
            fontsize=8.6,
            fontfamily="monospace",
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.6,rounding_size=0.2",
                "facecolor": "#FFFFFF",
                "edgecolor": "#CBD5E1",
                "alpha": 0.95,
            },
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    logger.success(f"Plot saved to: {save_path}")
