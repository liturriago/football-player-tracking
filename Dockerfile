# ─────────────────────────────────────────────────────────────────────────────
# Flag Football Tracking Pipeline
# Base: CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
# Includes: YOLO tracking + ONNX Runtime GPU + DeepSORT + TorchReID
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── system deps ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# make python3 the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3    /usr/bin/python  && \
    pip3 install --upgrade pip

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy project ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── default command: show help ────────────────────────────────────────────────
CMD ["python", "sport_tracker_deepsort.py", "--help"]
