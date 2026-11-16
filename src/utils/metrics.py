"""
Window-level detection metrics shared by the offline DVC pipeline and the online IDS evaluation.

Depends on numpy only, so it can be imported anywhere (PC, can-ids container on the Raspberry Pi)
without pulling torch / matplotlib / tqdm.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def align_window_labels(labels: np.ndarray, sequence_length: int = 64, stride: int = 1) -> np.ndarray:
    """
    Assigns the dominant non-zero attack label to each window (or 0 if all frames are normal).
    Windows are indexed by their START frame: window i covers labels[i * stride : i * stride + sequence_length].
    """
    windows = sliding_window_view(labels, window_shape=sequence_length)[::stride]
    has_attack = np.any(windows > 0, axis=1)
    result = np.zeros(len(windows), dtype=np.int8)

    if not np.any(has_attack):
        return result

    attack_windows = windows[has_attack]
    unique_attack_ids = [int(c) for c in np.unique(attack_windows) if c > 0]

    if len(unique_attack_ids) == 1:
        result[has_attack] = np.int8(unique_attack_ids[0])
        return result

    counts = np.stack([(attack_windows == cid).sum(axis=1) for cid in unique_attack_ids], axis=1)
    dominant_idx = np.argmax(counts, axis=1)
    result[has_attack] = np.array(unique_attack_ids, dtype=np.int8)[dominant_idx]
    return result


def apply_persistence_filter(raw_preds: np.ndarray, k: int) -> np.ndarray:
    """Raises an alarm on a window only once it is the k-th (or later) consecutive anomalous window."""
    filtered = np.zeros_like(raw_preds, dtype=np.int8)
    consecutive = 0

    for i, is_anomaly in enumerate(raw_preds):
        consecutive = consecutive + 1 if is_anomaly == 1 else 0
        if consecutive >= k:
            filtered[i] = 1

    return filtered


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float | None = None) -> dict:
    """Precision / Recall / F1 / FPR and the confusion matrix; 'threshold' is included only when given."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {} if threshold is None else {"threshold": float(threshold)}
    metrics.update(
        {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        }
    )
    return metrics
