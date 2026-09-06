import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm


def evaluate_attack_stream(model: torch.nn.Module, dataloader, device: torch.device) -> np.ndarray:
    """Computes mean squared reconstruction error across all sliding windows."""
    logger.info("Computing reconstruction MSE across all sliding windows...")
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference on attacked stream", unit="batch"):
            batch = batch.to(device)
            reconstruction = model(batch)
            loss_per_window = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            errors.extend(loss_per_window.cpu().numpy())

    return np.array(errors, dtype=np.float32)


def align_window_labels(labels: np.ndarray, sequence_length: int = 64) -> np.ndarray:
    """
    Aligns raw frame-level labels with sliding windows.
    A window is marked as an anomaly (1) if its closing frame is an attack frame.
    """
    logger.info(f"Aligning frame labels with sequence length: {sequence_length}...")
    return labels[sequence_length - 1 :]


def compute_detection_metrics(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    metrics_save_path: str | Path | None = None,
) -> dict:
    """Evaluates classification performance and optionally dumps report to JSON."""
    if len(errors) != len(y_true):
        raise ValueError(
            f"Shape mismatch: {len(errors)} error scores vs {len(y_true)} target labels."
        )

    logger.info(f"Evaluating classification performance at threshold: {threshold:.5f}")
    y_pred = (errors > threshold).astype(np.int8)

    # Force 2x2 matrix shape via labels=[0, 1] to prevent unpacking failure
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    logger.info(f"Confusion Matrix: [TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}]")
    logger.success(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | FPR: {fpr:.4f}")

    metrics = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if metrics_save_path:
        metrics_path = Path(metrics_save_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.success(f"Metrics artifact exported to: {metrics_path}")

    return metrics


def plot_attack_evaluation(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    save_path: str | Path,
    attack_name: str = "Denial of Service (DoS)",
    display_limit: int = 12000,
):
    """Plots reconstruction MSE, decision threshold, and ground-truth attack regions."""
    logger.info(f"Generating diagnostic plot for {attack_name}...")

    limit = min(display_limit, len(errors))
    errors_sub = errors[:limit]
    labels_sub = y_true[:limit]

    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)

    # 1. Reconstruction MSE Curve
    ax.plot(
        errors_sub,
        color="#1C3D5A",
        alpha=0.85,
        linewidth=1.2,
        label="Reconstruction MSE (Window)",
    )

    # 2. Decision Threshold
    ax.axhline(
        y=threshold,
        color="#E74C3C",
        linestyle="--",
        linewidth=1.8,
        label=f"Decision Threshold ({threshold:.5f})",
    )

    # 3. Ground Truth Attack Shading
    attack_indices = np.where(labels_sub == 1)[0]
    if len(attack_indices) > 0:
        diffs = np.diff(attack_indices)
        split_points = np.where(diffs > 1)[0] + 1
        attack_ranges = np.split(attack_indices, split_points)

        for i, r in enumerate(attack_ranges):
            ax.axvspan(
                r[0],
                r[-1],
                color="#C0392B",
                alpha=0.22,
                label="Ground Truth Attack Zone" if i == 0 else "",
            )

    ax.set_title(
        f"Intrusion Detection System Response — {attack_name}",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Time Window Index", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, limit)

    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=10,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches='tight' safely encompasses external legends without requiring tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Attack reaction plot saved to: {save_path}")