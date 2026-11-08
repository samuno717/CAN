from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from omegaconf import DictConfig
from sklearn.ensemble import IsolationForest


def extract_iforest_window_features(
    file_path: Path,
    seq_len: int = 64,
    stride: int = 1,
    feature_mode: str = "summary",
) -> np.ndarray:
    raw_array = np.load(file_path).astype(np.float32)
    # sliding_window_view on (N, F) with window_shape=(seq_len, F) -> (N - seq_len + 1, 1, seq_len, F)
    windows = sliding_window_view(raw_array, window_shape=seq_len, axis=0)[::stride]
    # windows shape: (num_windows, F, seq_len) -> transpose to (num_windows, seq_len, F)
    windows = np.transpose(windows, (0, 2, 1))

    mode = feature_mode.strip().lower()
    if mode == "flatten":
        features = windows.reshape(len(windows), -1)
    elif mode == "summary":
        w_mean = windows.mean(axis=1)
        w_std = windows.std(axis=1)
        w_min = windows.min(axis=1)
        w_max = windows.max(axis=1)
        features = np.concatenate([w_mean, w_std, w_min, w_max], axis=1)
    else:
        raise ValueError(f"Unsupported feature_mode: '{feature_mode}'. Choose 'summary' or 'flatten'.")

    logger.info(
        f"Extracted {len(features):,} Isolation Forest windows [mode='{mode}', shape={features.shape}] from {file_path.name}"
    )
    return features.astype(np.float32)


class CANIsolationForest:
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | float | str = "auto",
        contamination: str | float = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        feature_mode: str = "summary",
    ):
        self.feature_mode = feature_mode
        self.score_offset: float = 0.0
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, x_train: np.ndarray) -> "CANIsolationForest":
        logger.info(
            f"Fitting IsolationForest on {len(x_train):,} windows "
            f"(n_estimators={self.model.n_estimators}, max_samples={self.model.max_samples})..."
        )
        self.model.fit(x_train)
        raw_train_scores = -self.model.score_samples(x_train)
        self.score_offset = float(np.min(raw_train_scores))
        logger.success("IsolationForest training completed!")
        return self

    def score_windows(self, x_windows: np.ndarray, batch_size: int = 32768) -> np.ndarray:
        scores = np.empty(len(x_windows), dtype=np.float32)
        for start in range(0, len(x_windows), batch_size):
            end = min(start + batch_size, len(x_windows))
            raw = -self.model.score_samples(x_windows[start:end])
            scores[start:end] = np.maximum(0.0, raw - self.score_offset)
        return scores

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.success(f"IsolationForest model saved to: {path}")

    @staticmethod
    def load(path: Path) -> "CANIsolationForest":
        if not path.exists():
            raise FileNotFoundError(f"IsolationForest model not found at: {path}")
        return joblib.load(path)


def plot_iforest_validation_scores(scores: np.ndarray, save_path: Path, cfg: DictConfig) -> None:
    mean_val = float(np.mean(scores))
    std_val = float(np.std(scores))
    threshold_99 = float(np.percentile(scores, 99.0))
    threshold_99_9 = float(np.percentile(scores, 99.9))
    threshold_3std = mean_val + 3 * std_val

    subset_len = min(10000, len(scores))
    scores_subset = scores[:subset_len]
    rolling_mean = pd.Series(scores_subset).rolling(window=50, min_periods=1).mean().to_numpy()

    fig, (ax_main, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        dpi=300,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.28},
        sharex=False,
    )

    ax_main.plot(scores_subset, color="#0D9488", alpha=0.35, linewidth=0.8, label="Raw Anomaly Score (Window)")
    ax_main.plot(rolling_mean, color="#115E59", linewidth=1.8, label="Rolling average (SMA, window=50)")
    ax_main.axhline(
        y=threshold_99, color="#E74C3C", linestyle="--", linewidth=1.3, label=f"99.0% threshold ({threshold_99:.5f})"
    )
    ax_main.axhline(
        y=threshold_99_9,
        color="#9B59B6",
        linestyle="-.",
        linewidth=1.3,
        label=f"99.9% threshold ({threshold_99_9:.5f})",
    )
    ax_main.axhline(
        y=threshold_3std,
        color="#E67E22",
        linestyle=":",
        linewidth=1.4,
        label=rf"$\mu + 3\sigma$ threshold ({threshold_3std:.5f})",
    )

    anomalies_idx = np.where(scores_subset > threshold_99_9)[0]
    if len(anomalies_idx) > 0:
        ax_main.scatter(
            anomalies_idx,
            scores_subset[anomalies_idx],
            color="#C0392B",
            s=18,
            zorder=5,
            label=f"Above >99.9% (N={len(anomalies_idx)})",
        )

    ax_main.set_title("Isolation Forest Anomaly Score Analysis on CAN Bus", fontsize=13, fontweight="bold", pad=10)
    ax_main.set_xlabel("Time Window Index", fontsize=10)
    ax_main.set_ylabel("Anomaly Score", fontsize=10)
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

    ax_hist.hist(scores, bins=150, color="#0F766E", alpha=0.75, density=True, edgecolor="none")
    ax_hist.axvline(threshold_99, color="#E74C3C", linestyle="--", linewidth=1.3)
    ax_hist.axvline(threshold_99_9, color="#9B59B6", linestyle="-.", linewidth=1.3)
    ax_hist.axvline(threshold_3std, color="#E67E22", linestyle=":", linewidth=1.4)

    ax_hist.set_title("Distribution of Anomaly Scores across Validation Dataset", fontsize=11, fontweight="bold", pad=8)
    ax_hist.set_xlabel("Anomaly Score", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    info_text = (
        "Isolation Forest Config:\n"
        f"  \u2022 Sequence length   : {cfg.model.sequence_length:>8}\n"
        f"  \u2022 Feature mode      : {cfg.iforest.feature_mode:>8}\n"
        f"  \u2022 n_estimators      : {cfg.iforest.n_estimators:>8}\n"
        f"  \u2022 max_samples       : {str(cfg.iforest.max_samples):>8}\n"
        f"  \u2022 Seed              : {cfg.training.seed:>8}\n\n"
        "Validation Statistics:\n"
        f"  \u2022 Total windows     : {len(scores):>8,}\n"
        f"  \u2022 Threshold (99.9%) : {threshold_99_9:>8.5f}\n"
        f"  \u2022 Threshold (\u03bc+3\u03c3)  : {threshold_3std:>8.5f}\n"
        f"  \u2022 Mean Score (\u03bc)    : {mean_val:>8.5f}\n"
        f"  \u2022 Std Dev (\u03c3)       : {std_val:>8.5f}\n"
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
        bbox={
            "boxstyle": "round,pad=0.6,rounding_size=0.2",
            "facecolor": "#FFFFFF",
            "edgecolor": "#CBD5E1",
            "linewidth": leg.get_frame().get_linewidth(),
            "alpha": 0.95,
        },
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    logger.success(f"Isolation Forest validation chart saved in: {save_path}")
