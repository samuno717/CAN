import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from loguru import logger

from config import METRICS_DIR, PLOTS_DIR


def plot_seeds_comparison(metrics_dir: Path, save_path: Path, hidden_size: int = 64):

    save_path.parent.mkdir(parents=True, exist_ok=True)

    history_files = sorted(metrics_dir.glob(f"history_h{hidden_size}_seed_*.json"))
    if not history_files:
        logger.error(f"No history files found in {metrics_dir} for h{hidden_size}!")
        return

    all_losses = []
    seeds = []
    min_epochs = []
    min_losses = []

    for hf in history_files:
        with open(hf, "r") as f:
            data = json.load(f)
            losses = data["history"]["train"]
            all_losses.append(losses)
            seeds.append(data["seed"])
            min_epochs.append(int(np.argmin(losses)) + 1)
            min_losses.append(float(np.min(losses)))

    all_losses = np.array(all_losses)
    num_seeds, num_epochs = all_losses.shape
    epochs = np.arange(1, num_epochs + 1)
    mean_losses = np.mean(all_losses, axis=0)

    fig, ax = plt.subplots(figsize=(15, 7.0), dpi=300)

    seed_colors = [
        "#2563EB",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#DB2777",
        "#0891B2",
    ]

    for idx in range(num_seeds):
        color = seed_colors[idx % len(seed_colors)]
        ax.plot(
            epochs,
            all_losses[idx],
            color=color,
            alpha=0.35,
            linewidth=1.2,
            linestyle="-",
            marker="o",
            markersize=3.5,
            label=f"Seed {seeds[idx]} (Min Ep {min_epochs[idx]}: {min_losses[idx]:.5f})",
            zorder=2,
        )

    ax.plot(
        epochs,
        mean_losses,
        color="#1E3A8A",
        linewidth=2.2,
        marker="o",
        markersize=6.0,
        markerfacecolor="#3B82F6",
        markeredgecolor="#1E3A8A",
        markeredgewidth=1.2,
        label=r"Mean",
        zorder=4,
    )

    best_mean_epoch = int(np.argmin(mean_losses)) + 1
    best_mean_val = float(np.min(mean_losses))
    ax.scatter(
        [best_mean_epoch],
        [best_mean_val],
        color="#DC2626",
        s=90,
        zorder=5,
        edgecolor="#7F1D1D",
        linewidth=1.5,
        label=f"Mean minimum (ep {best_mean_epoch}: {best_mean_val:.5f})",
    )

    ax.set_title(
        f"AE-LSTM seed influence on MSE change over each epoch (h={hidden_size}, N={num_seeds})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10.5, labelpad=8)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=1)

    y_min = max(0.0, (np.min(all_losses)) * 0.9)
    y_max = np.max(all_losses) * 1.06
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0.8, num_epochs + 0.2)

    leg = ax.legend(
        bbox_to_anchor=(1.015, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=9.0,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
    )

    avg_min_loss = float(np.mean(min_losses))
    std_min_loss = float(np.std(min_losses))
    avg_min_ep = float(np.mean(min_epochs))
    min_epochs_range = f"{min(min_epochs)} - {max(min_epochs)}"

    info_text = (
        "Statistics\n"
        f" • Analyzed seeds   : {num_seeds:>8}\n"
        f" • Avg best epoch   : {avg_min_ep:>8.1f}\n"
        f" • Min epochs range : {min_epochs_range:>8}\n"
        f" • Avg lowest MSE   : {avg_min_loss:>8.5f}\n"
        f" • Best MSE std (σ) : {std_min_loss:>8.5f}\n\n"
    )

    plt.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bbox = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    y_pos = leg_bbox.ymin - 0.04

    ax.text(
        1.015,
        y_pos,
        info_text,
        transform=ax.transAxes,
        fontsize=8.8,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.6,rounding_size=0.2",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=leg.get_frame().get_linewidth(),
            alpha=0.95,
        ),
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    logger.success(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    plot_seeds_comparison(
        metrics_dir=METRICS_DIR,
        save_path=PLOTS_DIR / "seeds_convergence_comparison.png",
        hidden_size=64,
    )