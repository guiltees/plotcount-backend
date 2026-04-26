# ─── PlotCount AI — Backend Dockerfile ──────────────────────────────────────
# Multi-stage build: slim final image with pre-cached YOLO weights.

# ── Stage 1: dependency layer ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/pkgs -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/pkgs /usr/local

# Copy application source
COPY . .

# Pre-download YOLOv8n-seg weights at build time to avoid cold-start delays
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" || true

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user for security
RUN useradd -m plotcount
USER plotcount

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
