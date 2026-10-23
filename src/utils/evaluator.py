from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


def evaluate_model(model, dataloader, device) -> np.ndarray:
    logger.info("Evaluating MSE for all windows...")
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Windows evaluation (MSE)", unit="batch"):
            batch = batch.contiguous().to(device, non_blocking=True)
            reconstruction = model(batch)

            loss_per_window = torch.mean(
                (reconstruction - batch) ** 2, dim=(1, 2)
            )
            errors.extend(loss_per_window.cpu().numpy())

    return np.array(errors)


def load_sliding_windows(file_path: Path, seq_len: int, device: torch.device, stride: int) -> torch.Tensor:
    logger.info(f"Loading {file_path.name} to {device}...")
    raw_array = np.load(file_path)
    tensor_2d = torch.from_numpy(raw_array).float().to(device)

    windows = tensor_2d.unfold(0, seq_len, stride).transpose(1, 2)
    logger.info(f"Created {windows.shape[0]:,} windows of shape ({seq_len}, {tensor_2d.shape[1]}).")
    return windows


def apply_persistence_filter(raw_preds: np.ndarray, k: int = 3) -> np.ndarray:
    filtered = np.zeros_like(raw_preds, dtype=np.int8)
    consecutive = 0
    in_alarm = False

    for i, is_anomaly in enumerate(raw_preds):
        if is_anomaly == 1:
            consecutive += 1
            if consecutive >= k:
                in_alarm = True
        else:
            consecutive = 0
            in_alarm = False

        if in_alarm:
            filtered[i] = 1

    return filtered


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def plot_reconstruction_errors(errors: np.ndarray, save_path: Path, cfg: DictConfig) -> None:
    mean_val = float(np.mean(errors))
    std_val = float(np.std(errors))
    threshold_99 = float(np.percentile(errors, 99.0))
    threshold_99_9 = float(np.percentile(errors, 99.9))
    threshold_3std = mean_val + 3 * std_val

    subset_len = min(10000, len(errors))
    errors_subset = errors[:subset_len]
    rolling_mean = (
        pd.Series(errors_subset)
        .rolling(window=50, min_periods=1)
        .mean()
        .to_numpy()
    )

    fig, (ax_main, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        dpi=300,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.28},
        sharex=False,
    )

    ax_main.plot(errors_subset, color="#4A90E2", alpha=0.35, linewidth=0.8, label="Raw MSE (Window)")
    ax_main.plot(rolling_mean, color="#1C3D5A", linewidth=1.8, label="Rolling average (SMA, window=50)")
    ax_main.axhline(y=threshold_99, color="#E74C3C", linestyle="--", linewidth=1.3, label=f"99.0% threshold ({threshold_99:.5f})")
    ax_main.axhline(y=threshold_99_9, color="#9B59B6", linestyle="-.", linewidth=1.3, label=f"99.9% threshold ({threshold_99_9:.5f})")
    ax_main.axhline(y=threshold_3std, color="#E67E22", linestyle=":", linewidth=1.4, label=rf"$\mu + 3\sigma$ threshold ({threshold_3std:.5f})")

    anomalies_idx = np.where(errors_subset > threshold_99_9)[0]
    if len(anomalies_idx) > 0:
        ax_main.scatter(
            anomalies_idx,
            errors_subset[anomalies_idx],
            color="#C0392B",
            s=18,
            zorder=5,
            label=f"Above >99.9% (N={len(anomalies_idx)})",
        )

    ax_main.set_title("AE-LSTM Reconstruction Error Analysis on CAN Bus", fontsize=13, fontweight="bold", pad=10)
    ax_main.set_xlabel("Time Window Index", fontsize=10)
    ax_main.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_main.set_xlim(0, subset_len)

    leg = ax_main.legend(
        bbox_to_anchor=(1.015, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=9.2,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
    )

    ax_hist.hist(errors, bins=150, color="#64748B", alpha=0.75, density=True, edgecolor="none")
    ax_hist.axvline(threshold_99, color="#E74C3C", linestyle="--", linewidth=1.3)
    ax_hist.axvline(threshold_99_9, color="#9B59B6", linestyle="-.", linewidth=1.3)
    ax_hist.axvline(threshold_3std, color="#E67E22", linestyle=":", linewidth=1.4)

    ax_hist.set_title("Distribution of MSE across Full Dataset", fontsize=11, fontweight="bold", pad=8)
    ax_hist.set_xlabel("MSE Value", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    info_text = (
        "Pipeline Configuration:\n"
        f"  • Sequence length   : {cfg.model.sequence_length:>8}\n"
        f"  • Bottleneck size   : {cfg.model.hidden_size:>8}\n"
        f"  • Input features    : {cfg.model.num_features:>8}\n"
        f"  • Batch             : {cfg.training.batch_size:>8}\n"
        f"  • Learning rate     : {cfg.training.learning_rate:>8}\n\n"
        "Evaluation Statistics:\n"
        f"  • Total windows     : {len(errors):>8,}\n"
        f"  • Threshold (99.9%) : {threshold_99_9:>8.5f}\n"
        f"  • Threshold (μ+3σ)  : {threshold_3std:>8.5f}\n"
        f"  • Mean MSE (μ)      : {mean_val:>8.5f}\n"
        f"  • Std Dev (σ)       : {std_val:>8.5f}\n"
    )

    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_bbox = leg.get_window_extent(renderer).transformed(ax_main.transAxes.inverted())
    y_pos = leg_bbox.ymin - 0.035

    ax_main.text(
        1.015,
        y_pos,
        info_text,
        transform=ax_main.transAxes,
        fontsize=9.0,
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

    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    logger.success(f"Chart saved in: {save_path}")