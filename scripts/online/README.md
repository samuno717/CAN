# Online IDS experiments (Raspberry Pi 5, vcan0)

Measurement toolkit for the real-time IDS. Grafana is for live viewing and screenshots; all numbers for the
thesis come from files written by the IDS itself.

## What gets recorded

With `RECORD_RUN=1` the IDS writes `reports/online/<RUN_ID>/`:

| File | Content |
|---|---|
| `config.json` | model, weights file + SHA-256, threshold, k, stride, batch size, source, speed, host info |
| `windows.csv.gz` | one row per evaluated window: `end_frame`, `last_can_id`, `t_arrival`, `t_scored`, `score`, `raw`, `alarm`, error components |
| `batches.csv` | one row per inference micro-batch (size, time) |
| `system.csv` | 1 Hz: process CPU, system CPU, RSS, SoC temperature, CPU frequency |
| `alerts.jsonl` | confirmed incidents |
| `summary.json` | aggregates: throughput, inference time, processing lag percentiles, dropped frames, CPU/RAM |
| `prometheus.csv` | (optional) the same series as in Grafana, exported from Prometheus |
| `evaluation.json` | written by `evaluate_online.py` (see below) |

`processing lag = t_scored - t_arrival` is the time from the arrival of a window's last frame to its score being
available (queueing + batching + inference). The kernel drop counter (`SO_RXQ_OVFL`) shows frames lost because the
IDS did not keep up.

## Running on the Pi

```bash
make exp-main      # all scenarios x both models, stride 1, real time            (~3 h)
make exp-perf      # raw + multi, stride 1 and 5, 3 repetitions                    (~4 h)
make exp-stress    # raw at 2x/4x/8x/16x speed, stride 1 and 5                     (~1 h)
make exp-bench     # file source, no pacing: max throughput + consistency source
```

Each target calls `scripts/online/run_experiments.sh`; any variable can be overridden, e.g.

```bash
MODELS=ae_lstm SCENARIOS=multi REPS=1 RUN_TAG=try bash scripts/online/run_experiments.sh
```

The script rebuilds the images (so the current weights/thresholds are used), replays each scenario once,
stops the IDS automatically `IDLE_EXIT_SEC` after the last frame, exports Prometheus data and finally runs
`evaluate_online.py --all` inside the `can-ids` image.

Optional: `sudo sysctl -w net.core.rmem_max=8388608` and `IDS_RCVBUF=8388608` to enlarge the socket buffer.

## Evaluation (PC)

Copy `reports/online/` from the Pi, then:

```bash
uv run python -m scripts.online.evaluate_online --all          # if not done on the Pi
uv run python -m scripts.online.check_consistency --run-dir reports/online/<bench run>
uv run python -m scripts.online.aggregate_online               # tables (.csv, .tex) + plots/online/*.svg
```

`evaluate_online.py` uses the same window labelling and persistence filter as the offline pipeline, so online
and offline Precision/Recall/F1/FPR are directly comparable. In addition it computes event-level detection,
detection delay (frames, log time, wall-clock), false alarms per hour of benign traffic and real-time indicators.
