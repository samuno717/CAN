import json
import os
import time
from collections import deque
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
import psutil
import torch

from config import ATTACKS_DIR, REPORTS_DIR, WEIGHTS_DIR
from src.models.ae_lstm import AELSTM


@hydra.main(version_base=None, config_path="..", config_name="params")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    process = psutil.Process(os.getpid())

    logger.info("Initializing Edge IDS Stream Demo on CPU...")

    hidden_size = cfg.model.hidden_size
    sequence_length = cfg.model.sequence_length
    num_features = cfg.model.num_features
    persistence_k = 3
    frame_delay_s = 0.0008

    model_path = WEIGHTS_DIR / f"ae_lstm_can_model_h{hidden_size}.pth"
    attack_data_path = ATTACKS_DIR / "dos_tensors.npy"
    threshold_path = REPORTS_DIR / "metrics" / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    if not attack_data_path.exists():
        raise FileNotFoundError(f"Attacked stream tensor not found at: {attack_data_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold artifact not found at: {threshold_path}")

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
        threshold = float(threshold_data["threshold"])

    logger.info(f"Loaded decision threshold: {threshold:.5f}")
    logger.info(f"Persistence filter: k={persistence_k} consecutive windows")

    model = AELSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    stream_data = np.load(attack_data_path)[:12000]
    total_frames = len(stream_data)
    logger.success(f"Stream loaded ({total_frames} frames). Commencing real-time processing...\n")

    window_buffer = deque(maxlen=sequence_length)

    consecutive_anomalies = 0
    latencies_ms = []

    print("=" * 82)
    print(f"{'FRAME':<7} | {'MSE':<9} | {'LATENCY':<10} | {'CPU %':<7} | {'RAM (MB)':<9} | {'DETECTION STATUS'}")
    print("=" * 82)

    with torch.no_grad():
        for frame_idx in range(total_frames):
            frame = stream_data[frame_idx]
            window_buffer.append(frame)

            if len(window_buffer) < sequence_length:
                continue

            input_tensor = torch.tensor(
                np.array(window_buffer), dtype=torch.float32
            ).unsqueeze(0).to(device)

            t_start = time.perf_counter()
            reconstruction = model(input_tensor)
            mse_loss = torch.mean((reconstruction - input_tensor) ** 2).item()
            t_end = time.perf_counter()

            infer_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(infer_ms)

            if mse_loss > threshold:
                consecutive_anomalies += 1
            else:
                consecutive_anomalies = 0

            if consecutive_anomalies >= persistence_k:
                status = f"\033[91mATTACK (k={consecutive_anomalies})\033[0m"
            elif mse_loss > threshold:
                status = "\033[93mALERT\033[0m"
            else:
                status = "\033[92mOK\033[0m"

            if frame_idx % 150 == 0 or consecutive_anomalies >= 1:
                cpu_usage = process.cpu_percent()
                ram_mb = process.memory_info().rss / (1024 * 1024)
                print(
                    f"{frame_idx:<7} | {mse_loss:<9.5f} | {infer_ms:>6.2f} ms | {cpu_usage:>5.1f}% | {ram_mb:>7.1f} MB | {status}"
                )

            if frame_delay_s > 0:
                time.sleep(frame_delay_s)

    print("=" * 82)
    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    peak_ram = float(process.memory_info().rss / (1024 * 1024))

    logger.success("Hardware Benchmark Summary (CPU Single-Core):")
    logger.info(f"Average Inference Latency: {avg_latency:.2f} ms")
    logger.info(f"95th Percentile Latency:  {p95_latency:.2f} ms")
    logger.info(f"Peak RAM Allocation:       {peak_ram:.1f} MB")
    logger.info(f"Theoretical Max Throughput: {1000.0 / avg_latency:.1f} windows/second")


if __name__ == "__main__":
    main()