import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from config import METRICS_DIR, PLOTS_DIR

SCENARIOS = ["dos", "replay", "payload", "frequency", "fuzzing", "suspension", "multi"]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_model_comparison(comparison: dict, save_path: Path, persistence_k: int) -> None:
    scenarios = list(comparison["scenarios"].keys())
    labels = [s.upper() for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35

    ae_raw_f1 = [comparison["scenarios"][s]["ae_lstm"]["raw"]["f1_score"] for s in scenarios]
    if_raw_f1 = [comparison["scenarios"][s]["isolation_forest"]["raw"]["f1_score"] for s in scenarios]

    ae_pers_f1 = [comparison["scenarios"][s]["ae_lstm"]["persistence_filtered"]["f1_score"] for s in scenarios]
    if_pers_f1 = [
        comparison["scenarios"][s]["isolation_forest"]["persistence_filtered"]["f1_score"] for s in scenarios
    ]

    ae_pers_rec = [comparison["scenarios"][s]["ae_lstm"]["persistence_filtered"]["recall"] for s in scenarios]
    if_pers_rec = [
        comparison["scenarios"][s]["isolation_forest"]["persistence_filtered"]["recall"] for s in scenarios
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=300, gridspec_kw={"hspace": 0.36})

    panels = [
        (axes[0], ae_raw_f1, if_raw_f1, "Raw Window-Level F1-Score", "F1-Score"),
        (axes[1], ae_pers_f1, if_pers_f1, f"Persistence-Filtered F1-Score (k={persistence_k})", "F1-Score"),
        (axes[2], ae_pers_rec, if_pers_rec, f"Persistence-Filtered Recall (k={persistence_k})", "Recall"),
    ]

    for ax, ae_vals, if_vals, title, ylabel in panels:
        rects1 = ax.bar(x - width / 2, ae_vals, width, label="AE-LSTM", color="#1E3A8A", alpha=0.9)
        rects2 = ax.bar(x + width / 2, if_vals, width, label="Isolation Forest", color="#0D9488", alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold")
        ax.set_ylim(0.0, 1.12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.95)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    logger.success(f"Comparison plot saved to: {save_path}")


@hydra.main(version_base=None, config_path="../..", config_name="params")
def main(cfg: DictConfig) -> None:
    persistence_k = int(cfg.detection.persistence_k)
    comparison = {
        "persistence_k": persistence_k,
        "scenarios": {},
    }

    for sc in SCENARIOS:
        ae_path = METRICS_DIR / f"{sc}_metrics.json"
        if_path = METRICS_DIR / f"iforest_{sc}_metrics.json"

        if not ae_path.exists() or not if_path.exists():
            logger.warning(f"Skipping scenario '{sc}' in comparison (missing metrics file).")
            continue

        ae_data = _load_json(ae_path)
        if_data = _load_json(if_path)

        comparison["scenarios"][sc] = {
            "attack_windows": ae_data["attack_windows"],
            "benign_windows": ae_data["benign_windows"],
            "ae_lstm": {
                "threshold": ae_data["threshold"],
                "raw": ae_data["raw"],
                "persistence_filtered": ae_data["persistence_filtered"],
                "per_attack": ae_data.get("per_attack", {}),
            },
            "isolation_forest": {
                "threshold": if_data["threshold"],
                "raw": if_data["raw"],
                "persistence_filtered": if_data["persistence_filtered"],
                "per_attack": if_data.get("per_attack", {}),
            },
            "delta_f1_persistence": round(
                ae_data["persistence_filtered"]["f1_score"] - if_data["persistence_filtered"]["f1_score"], 5
            ),
        }

    out_json = METRICS_DIR / "model_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    logger.success(f"Model comparison summary saved to: {out_json}")

    out_plot = PLOTS_DIR / "model_comparison_ae_lstm_vs_iforest.svg"
    if comparison["scenarios"]:
        plot_model_comparison(comparison, save_path=out_plot, persistence_k=persistence_k)


if __name__ == "__main__":
    main()
