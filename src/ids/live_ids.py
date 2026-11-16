import argparse
from collections.abc import Iterator
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time

from loguru import logger
import numpy as np

from config import PROCESSED_ATTACKS_DIR, PROJECT_ROOT, RAW_DATA_DIR, REPORTS_DIR
from src.ids.detector import IDSDetector, ModelType
from src.ids.feature_extractor import StreamingFeatureExtractor, parse_candump_line
from src.ids.metrics_exporter import IDSMetricsExporter
from src.ids.run_recorder import RunRecorder

# Linux SocketCAN constants (PF_CAN = 29, AF_CAN = 29, CAN_RAW = 1, SOL_CAN_RAW = 101, CAN_RAW_RECV_OWN_MSGS = 4)
AF_CAN = getattr(socket, "AF_CAN", 29)
CAN_RAW = getattr(socket, "CAN_RAW", 1)
SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", 101)
CAN_RAW_RECV_OWN_MSGS = getattr(socket, "CAN_RAW_RECV_OWN_MSGS", 4)
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_EFF_MASK = 0x1FFFFFFF
# struct can_frame: can_id (uint32), can_dlc (uint8), __pad (3 bytes), data (8 bytes) = 16 bytes
CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)


SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)  # Linux: per-socket counter of frames dropped on a full rx queue
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)  # Linux: kernel receive timestamp (struct timespec)


def iter_socketcan_frames(
    interface: str,
    stop_flag: list[bool],
    socket_stats: dict | None = None,
    rcvbuf_bytes: int = 0,
) -> Iterator[tuple[float | None, int, list[int]]]:
    """
    Captures live CAN frames directly from a Linux SocketCAN interface (e.g. vcan0)
    using Python's built-in raw socket API with high-resolution monotonic timestamps.
    Yields (None, 0, []) on timeout so the main loop can periodically flush
    pending windows and update Prometheus metrics.

    Timestamps are KERNEL receive times (SO_TIMESTAMPNS, epoch seconds), not the moment Python reads the
    frame: if the IDS falls behind and frames queue up in the socket, their inter-arrival times (delta_t
    feature) stay correct. Fallback (no kernel timestamp): time.time() at read.

    socket_stats (optional dict) is updated in place with:
      - 'rx_dropped':   frames dropped by the kernel because the socket receive queue was full
                        (SO_RXQ_OVFL); > 0 means the IDS did not keep up with the bus
      - 'rcvbuf_bytes': effective socket receive buffer size
    """
    if socket_stats is None:
        socket_stats = {}
    socket_stats.setdefault("rx_dropped", 0)
    socket_stats.setdefault("rx_dropped_supported", False)

    sock = socket.socket(AF_CAN, socket.SOCK_RAW, CAN_RAW)
    try:
        sock.setsockopt(SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, 1)
    except OSError:
        pass
    if rcvbuf_bytes > 0:
        try:
            # Kernel caps this at net.core.rmem_max (raise it on the host with sysctl if needed)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
        except OSError as exc:
            logger.warning(f"Could not set SO_RCVBUF={rcvbuf_bytes}: {exc}")
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
        socket_stats["rx_dropped_supported"] = True
    except OSError:
        logger.warning("SO_RXQ_OVFL not supported - dropped frame counter unavailable.")
    socket_stats.setdefault("kernel_timestamps", False)
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        socket_stats["kernel_timestamps"] = True
    except OSError:
        logger.warning("SO_TIMESTAMPNS not supported - using user-space read time as frame timestamp.")
    try:
        socket_stats["rcvbuf_bytes"] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    except OSError:
        pass
    sock.settimeout(0.05)
    anc_size = (socket.CMSG_SPACE(4) + socket.CMSG_SPACE(16)) if hasattr(socket, "CMSG_SPACE") else 0

    # Wait until interface is brought up by host or can-player container
    bound = False
    while not bound and not stop_flag[0]:
        try:
            sock.bind((interface,))
            bound = True
        except OSError as exc:
            logger.warning(f"Waiting for SocketCAN interface '{interface}' to become available ({exc})...")
            time.sleep(1.0)

    logger.success(f"Listening for live CAN traffic on SocketCAN interface '{interface}'...")

    try:
        while not stop_flag[0]:
            kernel_ts: float | None = None
            try:
                if anc_size:
                    raw_packet, ancdata, _flags, _addr = sock.recvmsg(CAN_FRAME_SIZE, anc_size)
                    for cmsg_level, cmsg_type, cmsg_data in ancdata:
                        if cmsg_level != socket.SOL_SOCKET:
                            continue
                        if cmsg_type == SO_RXQ_OVFL and len(cmsg_data) >= 4:
                            socket_stats["rx_dropped"] = struct.unpack("=I", cmsg_data[:4])[0]
                        elif cmsg_type == SO_TIMESTAMPNS:
                            # struct timespec: 2 x 64-bit on aarch64 / x86_64, 2 x 32-bit on 32-bit ARM
                            fmt = "=qq" if len(cmsg_data) >= 16 else "=ii"
                            sec, nsec = struct.unpack(fmt, cmsg_data[: struct.calcsize(fmt)])
                            kernel_ts = sec + nsec * 1e-9
                else:
                    raw_packet = sock.recv(CAN_FRAME_SIZE)
            except TimeoutError:
                # Yield heartbeat tick so caller can flush buffered windows and update metrics
                yield None, 0, []
                continue
            except OSError as exc:
                logger.warning(f"SocketCAN read error on '{interface}': {exc}. Retrying in 1s...")
                time.sleep(1.0)
                continue

            if len(raw_packet) < CAN_FRAME_SIZE:
                continue

            ts = kernel_ts if kernel_ts is not None else time.time()
            raw_id, dlc, data_bytes = struct.unpack(CAN_FRAME_FMT, raw_packet[:CAN_FRAME_SIZE])

            # Ignore error or remote request frames
            if raw_id & (CAN_ERR_FLAG | CAN_RTR_FLAG):
                continue

            if raw_id & CAN_EFF_FLAG:
                can_id = raw_id & CAN_EFF_MASK
            else:
                can_id = raw_id & CAN_SFF_MASK

            dlc = min(dlc, 8)
            payload = list(data_bytes[:dlc])
            if dlc < 8:
                payload.extend([0] * (8 - dlc))

            yield ts, can_id, payload
    finally:
        sock.close()


def raw_test_split_start_line(raw_log: Path) -> int:
    """First line of the clean 15% test split inside the raw candump log (same rule as load_test_raw_df)."""
    train_ratio, val_ratio = 0.70, 0.15
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "params.yaml")
        train_ratio = float(cfg.preprocessing.train_ratio)
        val_ratio = float(cfg.preprocessing.val_ratio)
    except Exception:
        pass
    with open(raw_log, "rb") as f:
        n_lines = sum(1 for line in f if line.strip())
    return int(n_lines * (train_ratio + val_ratio))


def resolve_scenario_log_path(scenario: str) -> tuple[Path, int]:
    """Returns (log_path, start_line). For 'raw' the clean test split of the raw log is used."""
    scenario_clean = scenario.strip().lower()
    if scenario_clean in ("raw", "clean"):
        logs = sorted(RAW_DATA_DIR.glob("candump-*.log"))
        if not logs:
            raise FileNotFoundError(f"No raw candump logs found in {RAW_DATA_DIR}")
        return logs[-1], raw_test_split_start_line(logs[-1])
    attack_log = PROCESSED_ATTACKS_DIR / f"{scenario_clean}_attack.log"
    if not attack_log.exists():
        raise FileNotFoundError(f"Attack log not found: {attack_log}")
    return attack_log, 0


def iter_log_replay_frames(
    log_path: Path,
    loops: str = "1",
    realtime_pacing: bool = True,
    stop_flag: list[bool] | None = None,
    start_line: int = 0,
    speed: float = 1.0,
) -> Iterator[tuple[float | None, int, list[int]]]:
    """
    Streams CAN frames directly from a candump .log file with accurate inter-frame timing.
    Useful for direct replay testing or benchmarking when SocketCAN is bypassed.

    realtime_pacing=False streams frames as fast as possible (pure processing throughput benchmark).
    speed > 1 accelerates paced replay. Yielded timestamps are always the ORIGINAL log timestamps,
    so features (delta_t) are identical to the offline pipeline.
    """
    speed = max(1e-6, float(speed))
    if stop_flag is None:
        stop_flag = [False]

    infinite = loops.strip().lower() in ("i", "inf", "infinite", "0", "-1")
    max_loops = 0 if infinite else max(1, int(loops))
    loop_idx = 0

    while not stop_flag[0] and (infinite or loop_idx < max_loops):
        loop_idx += 1
        logger.info(f"[File Replay] Starting pass #{loop_idx} from '{log_path.name}'...")
        first_log_ts: float | None = None
        first_wall_ts: float = time.perf_counter()

        with open(log_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if stop_flag[0]:
                    break
                if line_no < start_line:
                    continue
                parsed = parse_candump_line(line)
                if parsed is None:
                    continue
                log_ts, can_id, payload = parsed

                if realtime_pacing:
                    if first_log_ts is None:
                        first_log_ts = log_ts
                        first_wall_ts = time.perf_counter()
                    else:
                        target_elapsed = (log_ts - first_log_ts) / speed
                        actual_elapsed = time.perf_counter() - first_wall_ts
                        wait_s = target_elapsed - actual_elapsed
                        if wait_s > 0.002:
                            time.sleep(wait_s)

                yield log_ts, can_id, payload


class LiveCANIDSService:
    """
    Main Real-Time CAN IDS Coordinator:
      - Ingests frames from SocketCAN (`vcan0`/`can0`) or candump log replay
      - Maintains O(1) sliding window ring buffer (`seq_len=64`)
      - Evaluates windows in micro-batches (`max_batch_size` or `flush_interval_ms`)
      - Applies persistence filter (`k=10`) and logs intrusion incidents
      - Publishes live metrics to Prometheus for Grafana visualization
    """

    def __init__(
        self,
        model_type: ModelType = "ae_lstm",
        interface: str = "vcan0",
        source_mode: str = "socketcan",
        scenario: str = "multi",
        log_file: Path | None = None,
        loops: str = "1",
        stride: int = 5,
        persistence_k: int = 10,
        max_batch_size: int = 32,
        flush_interval_ms: float = 50.0,
        metrics_port: int = 8000,
        custom_threshold: float | None = None,
        torch_threads: int = 2,
        alerts_log_path: Path = REPORTS_DIR / "live_alerts.jsonl",
        record: bool = False,
        run_id: str | None = None,
        runs_root: Path = REPORTS_DIR / "online",
        run_tag: str = "",
        idle_exit_s: float = 0.0,
        pacing: bool = True,
        speed: float = 1.0,
        rcvbuf_bytes: int = 0,
    ) -> None:
        self.model_type = model_type
        self.interface = interface
        self.source_mode = source_mode
        self.scenario = scenario
        self.log_file = log_file
        self.loops = loops
        self.stride = max(1, stride)
        self.persistence_k = max(1, persistence_k)
        self.max_batch_size = max(1, max_batch_size)
        self.flush_interval_s = max(0.005, flush_interval_ms / 1000.0)
        self.idle_exit_s = max(0.0, idle_exit_s)
        self.pacing = pacing
        self.speed = speed
        self.rcvbuf_bytes = rcvbuf_bytes
        self.socket_stats: dict = {}
        self.alerts_log_path = alerts_log_path
        self.alerts_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.extractor = StreamingFeatureExtractor(seq_len=64)
        self.detector = IDSDetector(
            model_type=model_type,
            persistence_k=self.persistence_k,
            custom_threshold=custom_threshold,
            torch_threads=torch_threads,
        )
        self.exporter = IDSMetricsExporter(
            port=metrics_port,
            model_type=self.detector.model_type,
            threshold=self.detector.threshold,
            persistence_k=self.persistence_k,
            stride=self.stride,
            interface=interface if source_mode == "socketcan" else f"file:{scenario}",
        )

        # Incident tracking state
        self.incident_count: int = 0
        self.active_incident_start_ts: float | None = None
        self.active_incident_start_frame: int = 0
        self.active_incident_start_window: int = 0
        self.active_incident_peak_score: float = 0.0
        self.active_incident_windows: int = 0
        self.total_windows_evaluated: int = 0
        self.last_heartbeat_log_ts: float = time.perf_counter()

        # Optional experiment recorder (per-window log + system metrics + summary)
        self.recorder: RunRecorder | None = None
        if record:
            if not run_id:
                run_id = (
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.detector.model_type}_{scenario}"
                    f"_s{self.stride}_x{speed:g}"
                )
            self.recorder = RunRecorder(
                root_dir=runs_root,
                run_id=run_id,
                config={
                    "model_type": self.detector.model_type,
                    "scenario": scenario,
                    "run_tag": run_tag,
                    "source_mode": source_mode,
                    "interface": interface,
                    "log_file": str(log_file) if log_file else "",
                    "loops": loops,
                    "pacing": pacing,
                    "speed": speed,
                    "stride": self.stride,
                    "seq_len": self.extractor.seq_len,
                    "persistence_k": self.persistence_k,
                    "threshold": self.detector.threshold,
                    "max_batch_size": self.max_batch_size,
                    "flush_interval_ms": self.flush_interval_s * 1000.0,
                    "torch_threads": torch_threads,
                    "weights_path": str(self.detector.weights_path),
                    "weights_file": self.detector.weights_path.name,
                    "threshold_path": str(self.detector.threshold_path),
                    "idle_exit_s": self.idle_exit_s,
                    "rcvbuf_bytes_requested": rcvbuf_bytes,
                },
            )
            self.alerts_log_path = self.recorder.alerts_path

    def _record_incident_event(self, event_type: str, details: dict) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event_type,
            "model": self.detector.model_type.upper(),
            "threshold": round(self.detector.threshold, 6),
            "persistence_k": self.persistence_k,
            **details,
        }
        with open(self.alerts_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _flush_pending_windows(
        self,
        pending_windows: list[np.ndarray],
        pending_frame_indices: list[int],
        new_frames_count: int,
        new_unknown_ids: int,
        pending_arrivals: list[float] | None = None,
        pending_last_ids: list[int] | None = None,
    ) -> None:
        if self.socket_stats.get("rx_dropped_supported"):
            self.exporter.g_rx_dropped.set(float(self.socket_stats.get("rx_dropped", 0)))
        if new_frames_count > 0:
            self.exporter.c_frames_total.inc(new_frames_count)
        if new_unknown_ids > 0:
            self.exporter.c_unknown_id_frames.inc(new_unknown_ids)

        if not pending_windows:
            self.exporter.update_hardware_And_rates(new_frames=new_frames_count, new_windows=0)
            return

        batch_3d = np.stack(pending_windows, axis=0)
        res = self.detector.score_batch(batch_3d)
        scored_perf = time.perf_counter()
        n_win = len(pending_windows)
        raw_flags: list[bool] = []
        alarm_flags: list[bool] = []

        raw_anom_batch = 0
        alarm_win_batch = 0
        latest_score = float(res.scores[-1])
        batch_max_score = float(np.max(res.scores))

        for i in range(n_win):
            score = float(res.scores[i])
            frame_no = pending_frame_indices[i]
            self.total_windows_evaluated += 1
            self.exporter.recent_scores.append(score)

            was_in_alarm = self.detector.in_alarm
            is_raw, is_alarm, _streak = self.detector.update_persistence(score)
            raw_flags.append(is_raw)
            alarm_flags.append(is_alarm)

            if is_raw:
                raw_anom_batch += 1

            if is_alarm:
                alarm_win_batch += 1
                self.active_incident_windows += 1
                self.active_incident_peak_score = max(self.active_incident_peak_score, score)

            # Transition: Normal -> Confirmed Intrusion Alarm
            if not was_in_alarm and is_alarm:
                self.incident_count += 1
                self.exporter.c_incidents_total.inc()
                self.active_incident_start_ts = time.time()
                self.active_incident_start_frame = frame_no
                self.active_incident_start_window = self.total_windows_evaluated
                self.active_incident_peak_score = score
                self.active_incident_windows = 1

                logger.error(
                    f"[INTRUSION DETECTED - Incident #{self.incident_count}] "
                    f"Model={self.detector.model_type.upper()} | "
                    f"Score={score:.5f} > Threshold={self.detector.threshold:.5f} "
                    f"(k={self.persistence_k} consecutive windows) | Frame #{frame_no:,}"
                )
                self._record_incident_event(
                    "INTRUSION_STARTED",
                    {
                        "incident_id": self.incident_count,
                        "frame_index": frame_no,
                        "window_index": self.total_windows_evaluated,
                        "trigger_score": round(score, 6),
                    },
                )

            # Transition: Confirmed Intrusion Alarm -> Normal
            elif was_in_alarm and not is_alarm:
                duration_s = (
                    time.time() - self.active_incident_start_ts
                    if self.active_incident_start_ts is not None
                    else 0.0
                )
                logger.success(
                    f"[INTRUSION ENDED - Incident #{self.incident_count}] "
                    f"Duration={duration_s:.2f}s | Affected Windows={self.active_incident_windows:,} | "
                    f"Peak Score={self.active_incident_peak_score:.5f} | Frame #{frame_no:,}"
                )
                self._record_incident_event(
                    "INTRUSION_ENDED",
                    {
                        "incident_id": self.incident_count,
                        "start_frame": self.active_incident_start_frame,
                        "end_frame": frame_no,
                        "affected_windows": self.active_incident_windows,
                        "peak_score": round(self.active_incident_peak_score, 6),
                        "duration_seconds": round(duration_s, 3),
                    },
                )
                self.active_incident_start_ts = None

        if self.recorder is not None:
            self.recorder.record_batch(
                end_frames=pending_frame_indices,
                last_can_ids=pending_last_ids or [-1] * n_win,
                arrival_perf=pending_arrivals or [scored_perf] * n_win,
                scored_perf=scored_perf,
                scores=res.scores,
                raw_flags=raw_flags,
                alarm_flags=alarm_flags,
                timing_err=res.timing_errors,
                payload_err=res.payload_errors,
                id_err=res.id_errors,
                batch_ms=res.inference_ms,
            )

        # Update Prometheus metrics with 1s peak-hold so short bursts are never missed between scrapes
        sma50 = float(np.mean(self.exporter.recent_scores)) if self.exporter.recent_scores else latest_score
        self.exporter.record_batch_metrics(
            latest_score=latest_score,
            batch_max_score=batch_max_score,
            sma50=sma50,
            in_alarm=self.detector.in_alarm,
            raw_anomaly=bool(batch_max_score > self.detector.threshold),
            consecutive=self.detector.consecutive_anomalies,
            timing_err=float(np.max(res.timing_errors)),
            payload_err=float(np.max(res.payload_errors)),
            id_err=float(np.max(res.id_errors)),
            per_window_ms=res.per_window_ms,
            batch_ms=res.inference_ms,
        )

        self.exporter.c_windows_total.inc(n_win)
        if raw_anom_batch > 0:
            self.exporter.c_raw_anomalies_total.inc(raw_anom_batch)
        if alarm_win_batch > 0:
            self.exporter.c_alarm_windows_total.inc(alarm_win_batch)

        self.exporter.update_hardware_And_rates(new_frames=new_frames_count, new_windows=n_win)

        now = time.perf_counter()
        if (now - self.last_heartbeat_log_ts) >= 5.0:
            status_tag = "ALARM" if self.detector.in_alarm else "SECURE"
            logger.info(
                f"[{status_tag}] Frames={self.extractor.total_frames:,} | "
                f"Windows={self.total_windows_evaluated:,} | "
                f"LatestScore={latest_score:.5f} (Peak={batch_max_score:.5f}, Thresh={self.detector.threshold:.5f}) | "
                f"Latency={res.per_window_ms:.2f}ms/win | Incidents={self.incident_count}"
            )
            self.last_heartbeat_log_ts = now

    def run(self) -> None:
        stop_flag = [False]

        def _handle_signal(signum, _frame):
            logger.info(f"Received signal {signum}, shutting down CAN IDS gracefully...")
            stop_flag[0] = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        self.exporter.start_server()

        logger.info("============================================================")
        logger.info(" Real-Time CAN Bus Intrusion Detection System (IDS)")
        logger.info("------------------------------------------------------------")
        logger.info(f" Active Model     : {self.detector.model_type.upper()}")
        logger.info(f" Anomaly Threshold: {self.detector.threshold:.5f} (99.9th percentile)")
        logger.info(f" Persistence (k)  : {self.persistence_k} consecutive windows")
        logger.info(f" Window Length    : {self.extractor.seq_len} frames (stride={self.stride})")
        logger.info(f" Source Mode      : {self.source_mode} ({self.interface})")
        logger.info(f" Alerts Log File  : {self.alerts_log_path}")
        logger.info("============================================================")

        if self.source_mode == "file":
            if self.log_file:
                log_path, start_line = self.log_file, 0
            else:
                log_path, start_line = resolve_scenario_log_path(self.scenario)
            frame_iter = iter_log_replay_frames(
                log_path,
                loops=self.loops,
                realtime_pacing=self.pacing,
                stop_flag=stop_flag,
                start_line=start_line,
                speed=self.speed,
            )
        else:
            frame_iter = iter_socketcan_frames(
                self.interface,
                stop_flag=stop_flag,
                socket_stats=self.socket_stats,
                rcvbuf_bytes=self.rcvbuf_bytes,
            )

        pending_windows: list[np.ndarray] = []
        pending_frame_indices: list[int] = []
        pending_arrivals: list[float] = []
        pending_last_ids: list[int] = []
        new_frames_since_flush = 0
        new_unknown_since_flush = 0
        last_flush_ts = time.perf_counter()
        last_frame_wall: float | None = None
        exit_reason = "stream_end"

        def _flush() -> None:
            self._flush_pending_windows(
                pending_windows=pending_windows,
                pending_frame_indices=pending_frame_indices,
                new_frames_count=new_frames_since_flush,
                new_unknown_ids=new_unknown_since_flush,
                pending_arrivals=pending_arrivals,
                pending_last_ids=pending_last_ids,
            )
            pending_windows.clear()
            pending_frame_indices.clear()
            pending_arrivals.clear()
            pending_last_ids.clear()

        try:
            for ts, can_id, payload in frame_iter:
                if stop_flag[0]:
                    exit_reason = "signal"
                    break

                now = time.perf_counter()

                # Heartbeat tick when socket timed out waiting for frames
                if ts is None:
                    if (now - last_flush_ts) >= self.flush_interval_s:
                        _flush()
                        new_frames_since_flush = 0
                        new_unknown_since_flush = 0
                        last_flush_ts = time.perf_counter()
                    # Experiment mode: stop automatically once the replayed stream has finished
                    if (
                        self.idle_exit_s > 0
                        and last_frame_wall is not None
                        and (now - last_frame_wall) >= self.idle_exit_s
                    ):
                        logger.info(f"No CAN frames for {self.idle_exit_s:.1f}s - stream finished, stopping IDS.")
                        exit_reason = "idle_timeout"
                        break
                    continue

                last_frame_wall = now
                arrival = now
                if self.source_mode != "file":
                    # ts is the kernel receive time (epoch): subtract the time the frame waited in the socket queue
                    arrival = now - max(0.0, time.time() - ts)
                if self.recorder is not None:
                    self.recorder.mark_first_frame(arrival)

                _, _, is_known = self.extractor.process_frame(ts, can_id, payload)
                new_frames_since_flush += 1
                if not is_known:
                    new_unknown_since_flush += 1

                frame_idx = self.extractor.total_frames
                if self.extractor.is_ready() and ((frame_idx - self.extractor.seq_len) % self.stride == 0):
                    pending_windows.append(self.extractor.get_current_window())
                    pending_frame_indices.append(frame_idx)
                    pending_arrivals.append(arrival)
                    pending_last_ids.append(can_id)

                if len(pending_windows) >= self.max_batch_size or (now - last_flush_ts) >= self.flush_interval_s:
                    _flush()
                    new_frames_since_flush = 0
                    new_unknown_since_flush = 0
                    # Measured AFTER inference: with 'now' (taken before the flush) a flush slower than
                    # flush_interval made every following frame trigger a new flush with a batch of 1 window
                    last_flush_ts = time.perf_counter()

            # Final flush on shutdown
            if pending_windows or new_frames_since_flush > 0:
                _flush()
        finally:
            logger.info(
                f"IDS stopped ({exit_reason}). Total frames processed: {self.extractor.total_frames:,} | "
                f"Total windows evaluated: {self.total_windows_evaluated:,} | "
                f"Total incidents detected: {self.incident_count}"
            )
            if self.recorder is not None:
                self.recorder.finalize(
                    stats={
                        "exit_reason": exit_reason,
                        "frames_received": self.extractor.total_frames,
                        "unknown_id_frames": self.extractor.unknown_id_frames,
                        "incidents": self.incident_count,
                        "socket_rx_dropped": int(self.socket_stats.get("rx_dropped", 0)),
                        "socket_rx_dropped_supported": bool(self.socket_stats.get("rx_dropped_supported", False)),
                        "socket_rcvbuf_bytes": self.socket_stats.get("rcvbuf_bytes"),
                        "socket_kernel_timestamps": bool(self.socket_stats.get("kernel_timestamps", False)),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time CAN Bus Intrusion Detection System (IDS)")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=os.getenv("MODEL_TYPE", "ae_lstm"),
        choices=["ae_lstm", "iforest"],
        help="Detection model to run: 'ae_lstm' or 'iforest' (env: MODEL_TYPE)",
    )
    parser.add_argument(
        "--interface",
        "-i",
        type=str,
        default=os.getenv("CAN_IFACE", "vcan0"),
        help="SocketCAN interface to monitor, e.g. vcan0 or can0 (env: CAN_IFACE)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("IDS_SOURCE", "socketcan"),
        choices=["socketcan", "file"],
        help="Frame source: 'socketcan' (Linux vcan0/can0) or 'file' (direct candump log replay)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default=os.getenv("SCENARIO", "multi"),
        help="Scenario name when --source=file (raw, dos, replay, payload, frequency, fuzzing, suspension, multi)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=os.getenv("CUSTOM_LOG_PATH", ""),
        help="Explicit path to a candump .log file when --source=file",
    )
    parser.add_argument(
        "--loops",
        "-l",
        type=str,
        default=os.getenv("LOOPS", "1"),
        help="Loops when --source=file ('1' or 'i' for infinite)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(os.getenv("IDS_STRIDE", "5")),
        help="Sliding window step in frames (default: 5, env: IDS_STRIDE)",
    )
    parser.add_argument(
        "--persistence-k",
        type=int,
        default=int(os.getenv("PERSISTENCE_K", "10")),
        help="Number of consecutive anomalous windows required to trigger alarm (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("IDS_BATCH_SIZE", "32")),
        help="Maximum micro-batch size for model inference (default: 32)",
    )
    parser.add_argument(
        "--flush-ms",
        type=float,
        default=float(os.getenv("IDS_FLUSH_MS", "50")),
        help="Maximum time in ms before flushing accumulated windows to model (default: 50ms)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.getenv("METRICS_PORT", "8000")),
        help="Prometheus metrics HTTP port (default: 8000, env: METRICS_PORT)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ["CUSTOM_THRESHOLD"]) if "CUSTOM_THRESHOLD" in os.environ else None,
        help="Optional override for anomaly detection threshold",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.getenv("TORCH_THREADS", "2")),
        help="Number of CPU threads for PyTorch on Raspberry Pi 5 (default: 2)",
    )

    # ---- Experiment / measurement options ----
    parser.add_argument(
        "--record",
        action="store_true",
        default=os.getenv("RECORD_RUN", "0") == "1",
        help="Record the run (per-window scores, timings, system metrics) to reports/online/<run_id> (env: RECORD_RUN=1)",
    )
    parser.add_argument("--run-id", type=str, default=os.getenv("RUN_ID", ""), help="Run directory name (env: RUN_ID)")
    parser.add_argument(
        "--runs-root",
        type=str,
        default=os.getenv("RUNS_ROOT", str(REPORTS_DIR / "online")),
        help="Root directory for recorded runs (env: RUNS_ROOT)",
    )
    parser.add_argument("--run-tag", type=str, default=os.getenv("RUN_TAG", ""), help="Free-text experiment tag (env: RUN_TAG)")
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=float(os.getenv("IDLE_EXIT_SEC", "0")),
        help="Stop after N seconds without frames once the stream has started (0 = run forever, env: IDLE_EXIT_SEC)",
    )
    parser.add_argument(
        "--no-pacing",
        action="store_true",
        default=os.getenv("IDS_PACING", "1") == "0",
        help="--source=file only: process frames as fast as possible (throughput benchmark, env: IDS_PACING=0)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=float(os.getenv("SPEED", "1")),
        help="Replay speed factor (file pacing; recorded in run config for socketcan runs, env: SPEED)",
    )
    parser.add_argument(
        "--rcvbuf",
        type=int,
        default=int(os.getenv("IDS_RCVBUF", "0")),
        help="Requested SocketCAN receive buffer size in bytes (0 = kernel default, env: IDS_RCVBUF)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.file) if args.file else None

    service = LiveCANIDSService(
        model_type=args.model,
        interface=args.interface,
        source_mode=args.source,
        scenario=args.scenario,
        log_file=log_file,
        loops=args.loops,
        stride=args.stride,
        persistence_k=args.persistence_k,
        max_batch_size=args.batch_size,
        flush_interval_ms=args.flush_ms,
        metrics_port=args.port,
        custom_threshold=args.threshold,
        torch_threads=args.threads,
        record=args.record,
        run_id=args.run_id or None,
        runs_root=Path(args.runs_root),
        run_tag=args.run_tag,
        idle_exit_s=args.idle_exit,
        pacing=not args.no_pacing,
        speed=args.speed,
        rcvbuf_bytes=args.rcvbuf,
    )
    service.run()


if __name__ == "__main__":
    main()
