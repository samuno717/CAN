import copy
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch
from loguru import logger
from matplotlib import ticker
from tqdm import tqdm


class GPUMonitor:
    def __init__(self, device_idx: int = 0):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.available = True
        except Exception:  # noqa: BLE001
            self.available = False

    def get_stats_str(self) -> str:
        if not self.available:
            return ""

        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        temp = pynvml.nvmlDeviceGetTemperature(
            self.handle, pynvml.NVML_TEMPERATURE_GPU
        )
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        vram_used_gb = mem.used / (1024**3)

        return (
            f"GPU: {util:>2}% | {temp}°C | {power:>3.0f}W | VRAM: {vram_used_gb:.1f}GB"
        )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs: int,
    device: torch.device,
    save_path: Path | None = None,
    max_grad_norm: float = 1.0
):

    lr_patience = 15  # Liczba epok bez poprawy przed redukcją
    lr_factor = 0.5  # Mnożnik (obcięcie o połowę)
    min_lr = 1e-5  # Bezpieczny próg dolny
    lr_counter = 0  # Licznik epok bez poprawy

    logger.info(f"Begin training loop ({epochs} epochs)...")
    global_start_time = time.perf_counter()
    is_cuda = device.type == "cuda"

    scaler = torch.amp.GradScaler(device=device.type, enabled=is_cuda)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_weights = None

    gpu_mon = GPUMonitor(device_idx=0) if is_cuda else None

    if is_cuda:
        logger.info("Warming up CUDA & cuDNN kernels...")
        model.train()

        dummy_batch = next(iter(train_loader)).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
            dummy_out = model(dummy_batch)
            dummy_loss = criterion(dummy_out, dummy_batch)
        scaler.scale(dummy_loss).backward()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        logger.info("Warmup complete. Starting measured training loop.")

    for epoch in range(epochs):
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            epoch_start_time = time.perf_counter()

        model.train()
        train_loss_total = 0.0

        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d}/{epochs} [Train]",
            unit="batch",
            leave=False
        )
        gpu_stats = gpu_mon.get_stats_str() if gpu_mon else ""

        for step, batch in enumerate(train_iterator):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            train_loss_total += current_loss

            if is_cuda and gpu_mon and step % 30 == 0:
                gpu_stats = gpu_mon.get_stats_str()

            train_iterator.set_postfix_str(
                f"loss: {current_loss:.5f} | {gpu_stats}"
            )

        avg_train_loss = train_loss_total / len(train_loader)
        history["train"].append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            val_iterator = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1:03d}/{epochs} [Val]",
                unit="batch",
                leave=False,
            )
            for batch in val_iterator:
                batch = batch.to(device, non_blocking=True)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        history["val"].append(avg_val_loss)

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            epoch_duration = start_event.elapsed_time(end_event) / 1000.0
        else:
            epoch_duration = time.perf_counter() - epoch_start_time

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_weights = copy.deepcopy(model.state_dict())

            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_weights, save_path)

        if is_best:
            lr_counter = 0
        else:
            lr_counter += 1
            if lr_counter >= lr_patience:
                current_lr = optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr < current_lr:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    logger.warning(
                        f"Epoch {epoch+1:03d}: No improvement for {lr_patience} epochs. "
                        f"Decreasing LR: {current_lr:.6f} -> {new_lr:.6f}"
                    )
                lr_counter = 0

        active_lr = optimizer.param_groups[0]["lr"]

        marker = "[BEST]" if is_best else ""
        logger.info(
            f"Epoch {epoch+1:03d}/{epochs} finished in {epoch_duration:.2f}s | "
            f"Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | LR: {active_lr} | {marker} "
        )

    if is_cuda:
        torch.cuda.synchronize()

    global_end_time = time.perf_counter()
    total_seconds = global_end_time - global_start_time
    formatted_time = str(timedelta(seconds=int(total_seconds)))

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.success(
            f"Model restored to best state from Epoch {best_epoch} (lowest Val MSE: {best_val_loss:.6f})"
        )

    logger.success(
        f"Training finished! Total time: {formatted_time} ({total_seconds:.2f} s)"
    )

    return model, history


def plot_training_loss(
    history: dict[str, list[float]] | list[float],
    save_path: Path,
    params: dict[str, Any] | None = None,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"

    if isinstance(history, dict):
        train_losses = history["train"]
        val_losses = history.get("val", [])
    else:
        train_losses = history
        val_losses = []

    total_epochs = len(train_losses)
    epochs = np.arange(1, total_epochs + 1)

    eval_target = val_losses if len(val_losses) > 0 else train_losses
    min_loss_idx = int(np.argmin(eval_target))
    min_loss_epoch = epochs[min_loss_idx]
    min_loss_val = eval_target[min_loss_idx]
    final_loss_val = eval_target[-1]
    init_loss_val = eval_target[0]

    loss_reduction_pct = (
        ((init_loss_val - min_loss_val) / init_loss_val) * 100.0
        if init_loss_val > 0
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(26, 10), dpi=300)
    show_markers = total_epochs <= 30

    ax.plot(
        epochs,
        train_losses,
        color="#1E3A8A",
        linewidth=2.0,
        marker="o" if show_markers else None,
        markersize=4.0,
        markerfacecolor="#3B82F6",
        markeredgecolor="#1E3A8A",
        label="Training loss (MSE)",
        zorder=3,
    )

    if len(val_losses) > 0:
        ax.plot(
            epochs,
            val_losses,
            color="#D97706",
            linewidth=2.0,
            linestyle="--",
            marker="s" if show_markers else None,
            markersize=4.0,
            markerfacecolor="#FBBF24",
            markeredgecolor="#D97706",
            label="Validation loss (MSE)",
            zorder=3,
        )

    best_label = (
        f"Best Val (Ep {min_loss_epoch}: {min_loss_val:.6f})"
        if len(val_losses) > 0
        else f"Min Train (Ep {min_loss_epoch}: {min_loss_val:.6f})"
    )
    ax.scatter(
        [min_loss_epoch],
        [min_loss_val],
        color="#DC2626",
        s=80,
        zorder=5,
        edgecolor="#7F1D1D",
        linewidth=1.5,
        label=best_label,
    )

    p = params or {}
    run_timestamp = p.get(
        "timestamp", datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

    ax.set_title(
        "AE-LSTM Training & Validation Convergence",
        loc="center",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_title(
        run_timestamp,
        loc="right",
        fontsize=8.5,
        color="#64748B",
        pad=14,
    )

    ax.set_xlabel("Epoch", fontsize=10, labelpad=8)
    ax.set_ylabel("MSE", fontsize=10, labelpad=8)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    ax.set_xlim(1, total_epochs)

    all_losses = train_losses + val_losses
    y_min = max(0.0, min_loss_val * 0.9)
    y_max = max(all_losses) * 1.05
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=1)

    ax.legend(
        loc="upper right",
        framealpha=0.95,
        fontsize=9,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
    )

    target_name = "Validation" if len(val_losses) > 0 else "Training"
    info_text = (
        "Training parameters:\n"
        f"  • Hidden size (dim) : {p.get('hidden_size', 64):>8}\n"
        f"  • Latent dim        : {p.get('latent_dim', '-'):>8}\n"
        f"  • Sequence length   : {p.get('sequence_length', 64):>8}\n"
        f"  • Batch size        : {p.get('batch_size', 2048):>8}\n"
        f"  • Learning rate     : {p.get('learning_rate', 0.001):>8}\n\n"
        f"Results ({target_name}):\n"
        f"  • Total epochs      : {total_epochs:>8}\n"
        f"  • Initial MSE (Ep 1): {init_loss_val:>8.6f}\n"
        f"  • Final MSE         : {final_loss_val:>8.6f}\n"
        f"  • Lowest MSE        : {min_loss_val:>8.6f}\n"
        f"  • Loss reduction    : {loss_reduction_pct:>7.1f}%"
    )

    ax.text(
        1.025,
        1.0,
        info_text,
        transform=ax.transAxes,
        fontsize=8.8,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.7,rounding_size=0.2",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=1.0,
            alpha=0.95,
        ),
    )

    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)