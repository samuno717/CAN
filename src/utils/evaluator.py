from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    logger.info("Evaluating MSE for all windows...")
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Windows evaluation (MSE)", unit="batch"):
            batch = batch.to(device)
            reconstruction = model(batch)

            loss_per_window = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            errors.extend(loss_per_window.cpu().numpy())

    return np.array(errors)

def plot_reconstruction_errors(errors: np.ndarray, save_path: Path):
    mean_val = np.mean(errors)
    std_val = np.std(errors)
    threshold_99 = np.percentile(errors, 99.0)
    threshold_99_9 = np.percentile(errors, 99.9)
    threshold_3std = mean_val + 3 * std_val

    logger.success(f"MSE: {mean_val:.6f}")
    logger.success(f"Threshold (99. percentile): {threshold_99:.6f}")
    logger.success(f"Threshold (99.9. percentile): {threshold_99_9:.6f}")
    logger.success(f"Threshold (Mean + 3*STD): {threshold_3std:.6f}")

    subset_len = min(10000, len(errors))
    errors_subset = errors[:subset_len]
    rolling_mean = pd.Series(errors_subset).rolling(window=50, min_periods=1).mean().to_numpy()

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1,
        figsize=(20, 10),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False
    )

    ax_main.plot(
        errors_subset, color="#4A90E2", alpha=0.35, linewidth=0.8,
        label="Raw MSE (Window)"
    )
    ax_main.plot(
        rolling_mean, color="#1C3D5A", linewidth=1.8,
        label="Rolling average (SMA, window=50)"
    )

    ax_main.axhline(y=threshold_99, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"99.0% threshold ({threshold_99:.5f})")
    ax_main.axhline(y=threshold_99_9, color="#9B59B6", linestyle="-.", linewidth=1.5, label=f"99.9% threshold ({threshold_99_9:.5f})")
    ax_main.axhline(y=threshold_3std, color="#E67E22", linestyle=":", linewidth=1.5, label=f"$\mu + 3\sigma$ threshold ({threshold_3std:.5f})")

    anomalies_idx = np.where(errors_subset > threshold_99_9)[0]
    if len(anomalies_idx) > 0:
        ax_main.scatter(
            anomalies_idx, errors_subset[anomalies_idx],
            color="#C0392B", s=15, zorder=5,
            label=f"Above >99.9% (N={len(anomalies_idx)})"
        )

    ax_main.set_title("AE-LSTM reconstruction error analysis on CAN bus", fontsize=14, fontweight="bold", pad=12)
    ax_main.set_xlabel("Time Window Index", fontsize=11)
    ax_main.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_main.set_xlim(0, subset_len)

    ax_main.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=10
    )

    ax_hist.hist(errors, bins=150, color="#7F8C8D", alpha=0.7, density=True, edgecolor="none")
    ax_hist.axvline(threshold_99, color="#E74C3C", linestyle="--", linewidth=1.5)
    ax_hist.axvline(threshold_99_9, color="#9B59B6", linestyle="-.", linewidth=1.5)
    ax_hist.axvline(threshold_3std, color="#E67E22", linestyle=":", linewidth=1.5)

    ax_hist.set_title("Distribution of MSE across the entire dataset", fontsize=11, fontweight="bold", pad=8)
    ax_hist.set_xlabel("MSE value", fontsize=11)
    ax_hist.set_ylabel("Distribution", fontsize=10)
    ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.success(f"Chart saved in: {save_path}")