from collections import deque
import time

from loguru import logger
from prometheus_client import Counter, Gauge, Info, start_http_server
import psutil


class IDSMetricsExporter:
    """
    Prometheus metrics exporter for the Real-Time CAN IDS.
    Exposes anomaly scores, alarm state, persistence streaks, feature error decomposition,
    and Raspberry Pi 5 hardware/inference benchmarks.
    """

    def __init__(
        self,
        port: int = 8000,
        model_type: str = "ae_lstm",
        threshold: float = 0.04144,
        persistence_k: int = 10,
        stride: int = 5,
        interface: str = "vcan0",
    ) -> None:
        self.port = port
        self.model_type = model_type
        self.threshold = threshold
        self.persistence_k = persistence_k
        self.stride = stride
        self.interface = interface

        self.cpu_count = max(1, psutil.cpu_count() or 1)
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        self.recent_scores: deque[float] = deque(maxlen=50)
        self.last_rate_ts = time.perf_counter()
        self.frames_since_last = 0
        self.windows_since_last = 0

        # 1-second peak-hold buffer so Prometheus (1s scrape interval) never misses sub-second spikes
        self.hold_reset_ts = time.perf_counter()
        self.held_batch_max: float = 0.0
        self.held_latest_score: float = 0.0
        self.held_timing_err: float = 0.0
        self.held_payload_err: float = 0.0
        self.held_id_err: float = 0.0
        self.held_in_alarm: bool = False
        self.held_raw_anomaly: bool = False
        self.held_consecutive: int = 0

        # Static configuration info
        self.info = Info(
            "can_ids_build",
            "Active configuration of the Real-Time CAN Bus IDS",
        )
        self.info.info(
            {
                "model_type": model_type.upper(),
                "interface": interface,
                "persistence_k": str(persistence_k),
                "stride": str(stride),
            }
        )

        # Core detection gauges
        self.g_score_latest = Gauge(
            "can_ids_anomaly_score_latest",
            "Latest sliding window anomaly score (MSE for AE-LSTM or Score for IForest)",
        )
        self.g_score_batch_max = Gauge(
            "can_ids_anomaly_score_batch_max",
            "Peak anomaly score in the current 1s Prometheus scrape window",
        )
        self.g_score_sma50 = Gauge(
            "can_ids_anomaly_score_rolling_mean",
            "Rolling mean of anomaly scores over the last 50 evaluated windows",
        )
        self.g_threshold = Gauge(
            "can_ids_threshold",
            "Calibrated 99.9th percentile anomaly detection threshold",
        )
        self.g_threshold.set(threshold)

        self.g_threshold_ratio = Gauge(
            "can_ids_score_to_threshold_ratio",
            "Ratio of current anomaly score to calibrated threshold (> 1.0 indicates raw anomaly)",
        )

        # Alarm & persistence state
        self.g_alarm_active = Gauge(
            "can_ids_alarm_active",
            "Confirmed intrusion alarm status after persistence filter (0 = normal, 1 = under attack)",
        )
        self.g_raw_anomaly = Gauge(
            "can_ids_raw_anomaly_active",
            "Raw single-window anomaly flag before persistence filter (0 or 1)",
        )
        self.g_consecutive = Gauge(
            "can_ids_consecutive_anomalies",
            "Number of consecutive anomalous windows currently observed",
        )
        self.g_persistence_k = Gauge(
            "can_ids_persistence_k",
            "Configured persistence filter window count k",
        )
        self.g_persistence_k.set(persistence_k)

        # Feature component breakdown (diagnostic)
        self.g_err_timing = Gauge(
            "can_ids_component_timing_error",
            "Window reconstruction error / variance on delta_t timing feature",
        )
        self.g_err_payload = Gauge(
            "can_ids_component_payload_error",
            "Window reconstruction error / variance on payload bytes (byte1..byte8)",
        )
        self.g_err_id = Gauge(
            "can_ids_component_id_error",
            "Window reconstruction error / variance on one-hot CAN arbitration IDs",
        )

        # Counters
        self.c_frames_total = Counter(
            "can_ids_frames_processed_total",
            "Total number of CAN bus frames captured and processed",
        )
        self.c_unknown_id_frames = Counter(
            "can_ids_unknown_id_frames_total",
            "Total number of CAN frames with arbitration IDs outside the 17 known ECUs",
        )
        self.c_windows_total = Counter(
            "can_ids_windows_evaluated_total",
            "Total number of 64-frame sliding windows evaluated by the model",
        )
        self.c_raw_anomalies_total = Counter(
            "can_ids_raw_anomalous_windows_total",
            "Total number of windows exceeding the raw anomaly threshold",
        )
        self.c_alarm_windows_total = Counter(
            "can_ids_alarm_windows_total",
            "Total number of windows in confirmed alarm state (after persistence filter)",
        )
        self.c_incidents_total = Counter(
            "can_ids_attack_incidents_total",
            "Total number of distinct confirmed intrusion incidents detected",
        )

        # Performance / Raspberry Pi 5 benchmark metrics
        self.g_infer_window_ms = Gauge(
            "can_ids_inference_latency_ms",
            "Model inference time per sliding window in milliseconds",
        )
        self.g_infer_batch_ms = Gauge(
            "can_ids_batch_inference_latency_ms",
            "Total model inference time for the micro-batch in milliseconds",
        )
        self.g_fps = Gauge(
            "can_ids_bus_frames_per_second",
            "Current incoming CAN bus frame rate (frames/sec)",
        )
        self.g_wps = Gauge(
            "can_ids_windows_per_second",
            "Current IDS evaluation throughput (windows/sec)",
        )
        self.g_cpu_percent = Gauge(
            "can_ids_process_cpu_percent",
            "Normalized CPU utilization percentage (0-100%) of the IDS Python process",
        )
        self.g_sys_cpu_percent = Gauge(
            "can_ids_system_cpu_percent",
            "Total system-wide CPU utilization percentage (0-100%) on Raspberry Pi 5",
        )
        self.g_rx_dropped = Gauge(
            "can_ids_socket_rx_dropped_frames",
            "Frames dropped by the kernel because the IDS socket receive queue was full (SO_RXQ_OVFL)",
        )
        self.g_memory_mb = Gauge(
            "can_ids_process_memory_mb",
            "Resident memory (RSS) used by the IDS process in MiB",
        )

    def start_server(self) -> None:
        start_http_server(self.port)
        logger.success(f"[Prometheus] Metrics server listening on http://0.0.0.0:{self.port}/metrics")

    def record_batch_metrics(
        self,
        latest_score: float,
        batch_max_score: float,
        sma50: float,
        in_alarm: bool,
        raw_anomaly: bool,
        consecutive: int,
        timing_err: float,
        payload_err: float,
        id_err: float,
        per_window_ms: float,
        batch_ms: float,
    ) -> None:
        now = time.perf_counter()
        if (now - self.hold_reset_ts) >= 1.05:
            # Start fresh 1-second peak-hold interval
            self.held_batch_max = batch_max_score
            self.held_latest_score = latest_score
            self.held_timing_err = timing_err
            self.held_payload_err = payload_err
            self.held_id_err = id_err
            self.held_in_alarm = in_alarm
            self.held_raw_anomaly = raw_anomaly
            self.held_consecutive = consecutive
            self.hold_reset_ts = now
        else:
            self.held_batch_max = max(self.held_batch_max, batch_max_score)
            self.held_latest_score = max(self.held_latest_score, latest_score) if in_alarm else latest_score
            self.held_timing_err = max(self.held_timing_err, timing_err)
            self.held_payload_err = max(self.held_payload_err, payload_err)
            self.held_id_err = max(self.held_id_err, id_err)
            self.held_in_alarm = self.held_in_alarm or in_alarm
            self.held_raw_anomaly = self.held_raw_anomaly or raw_anomaly
            self.held_consecutive = max(self.held_consecutive, consecutive) if in_alarm else consecutive

        self.g_score_latest.set(self.held_latest_score)
        self.g_score_batch_max.set(self.held_batch_max)
        self.g_score_sma50.set(sma50)
        self.g_threshold_ratio.set(self.held_batch_max / max(1e-9, self.threshold))

        self.g_alarm_active.set(1.0 if self.held_in_alarm else 0.0)
        self.g_raw_anomaly.set(1.0 if self.held_raw_anomaly else 0.0)
        self.g_consecutive.set(float(self.held_consecutive))

        self.g_err_timing.set(self.held_timing_err)
        self.g_err_payload.set(self.held_payload_err)
        self.g_err_id.set(self.held_id_err)

        self.g_infer_window_ms.set(per_window_ms)
        self.g_infer_batch_ms.set(batch_ms)

    def update_hardware_And_rates(self, new_frames: int, new_windows: int) -> None:
        self.frames_since_last += new_frames
        self.windows_since_last += new_windows

        now = time.perf_counter()
        dt = now - self.last_rate_ts
        if dt >= 0.5:
            self.g_fps.set(self.frames_since_last / dt)
            self.g_wps.set(self.windows_since_last / dt)
            self.frames_since_last = 0
            self.windows_since_last = 0
            self.last_rate_ts = now

            try:
                proc_cpu_norm = self.process.cpu_percent(interval=None) / self.cpu_count
                self.g_cpu_percent.set(proc_cpu_norm)
                self.g_sys_cpu_percent.set(psutil.cpu_percent(interval=None))
                mem_mb = self.process.memory_info().rss / (1024 * 1024)
                self.g_memory_mb.set(mem_mb)
            except psutil.Error as exc:
                logger.debug(f"psutil telemetry warning: {exc}")
