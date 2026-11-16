FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

# lightweight system libraries and SocketCAN utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    can-utils \
    iproute2 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-optimized PyTorch and IDS dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "numpy<2.0.0" \
    "scikit-learn>=1.5.0" \
    "joblib>=1.4.0" \
    "pandas==2.2.3" \
    "matplotlib>=3.9.0" \
    "omegaconf>=2.3.0" \
    "loguru>=0.7.3" \
    "psutil>=7.0.0" \
    "prometheus-client>=0.21.0" \
    "torch"

WORKDIR /workspace

# Copy project configuration, source code, trained weights, scalers, and thresholds
COPY config.py params.yaml /workspace/
COPY src/ /workspace/src/
COPY reports/metrics/threshold.json /workspace/reports/metrics/threshold.json
COPY reports/metrics/iforest_threshold.json /workspace/reports/metrics/iforest_threshold.json

EXPOSE 8000

ENTRYPOINT ["python", "-m", "src.ids.live_ids"]
