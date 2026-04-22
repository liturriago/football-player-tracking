# Flag Football Tactical Tracker

A computer vision pipeline that transforms raw smartphone footage of flag football matches into a synchronized **top-down tactical radar view** with persistent player IDs.

Built as a solution to a technical Computer Vision assignment, this project covers multi-object tracking (MOT), field registration via homography, and Triton-ready ONNX deployment.

---

## Demo

> Left: original video with bounding boxes and IDs — Right: synchronized 2D radar map

*(output video: `data/tactical_result.mp4`)*

---

## Architecture

```
YOLO Detection
      ↓
DeepSORT (Kalman filter + cosine Re-ID)
      ↓
Jersey Embedder  ←─── HSV histogram  (fast, no GPU)
                 ←─── TorchReID OSNet (deep features, GPU)
                 ←─── Combined        (both, best quality)
      ↓
Radar Gate (jump veto in homography space)
      ↓
FieldTransformer (homography → 2D radar template)
```

---

## Tasks

### Task A — Multi-Object Tracking

**Tracker:** [DeepSORT](https://github.com/levan92/deep_sort_realtime) with a custom domain-specific embedder instead of the default MobileNetV2.

**Embedder options:**

| Mode | Description | Dim | GPU |
|------|-------------|-----|-----|
| `hsv` | HSV color histogram on jersey region [30%–70%] | 48 | No |
| `torchreid` | OSNet-AIN Re-ID model (domain-generalized) | 512 | Yes |
| `torchreid+hsv` | Both combined — deep features + team color | 560 | Yes |

**ID persistence strategy** — inspired by [SportSORT (Pham et al., 2025)](https://doi.org/10.1007/s00138-025-01756-y), the state-of-the-art MOT method for sports:

- DeepSORT's Kalman filter handles frame-to-frame motion prediction
- Cosine distance on jersey embeddings gates Re-ID associations
- `RadarGate` vetoes assignments where the jump in **radar space** (post-homography) exceeds a physical threshold — this prevents cross-field ID switches that are geometrically impossible between consecutive frames
- `max_age` keeps lost tracks alive for Out-of-View re-association

**Known limitation:** flag football players on the same team wear nearly identical uniforms. Without legible jersey numbers (as used in SportSORT's OCR pipeline), appearance-based Re-ID has an irreducible ambiguity ceiling. SportSORT itself — the current state of the art — reports 81.3% HOTA on SportsMOT; real-world flag football with uniform jerseys is a harder scenario.

---

### Task B — Field Registration

A homography matrix maps the player's feet coordinates (video pixels) to the 2D flag football pitch template.

**Calibration** is done interactively once per camera angle:

```bash
python sport_tracker_deepsort.py --calibrate \
    --input data/test_1.mp4 --model weights/best.pt --radar data/radar_template.png
```

Click the 4 field corners in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left. The pipeline starts automatically with the calibrated points.

**Trapezoid correction** (`--field_bot_visible`) handles cameras with a `\ /` perspective where the bottom corners of the field are cut off, mapping src_pts to a trapezoid-shaped dst region inside the radar template rather than a rectangle.

---

### Task C — Triton-Ready Optimization

**ONNX export settings:**
```python
model.export(
    format="onnx",
    imgsz=512,
    half=True,       # FP16 weights
    dynamic=True,    # dynamic batch and spatial dims
    simplify=True,
    nms=False,       # NMS applied manually in postprocessing
)
```

**Standalone ONNX inference** (no Ultralytics dependency):
```bash
pip install onnxruntime-gpu
python onnx_inference.py --model weights/best.onnx --input data/test_1.mp4
```

**Triton deployment** — see `model_repository/yolo_players/config.pbtxt`:

```
model_repository/
└── yolo_players/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

Key config choices for maximizing throughput:

- **Dynamic batching** — aggregates requests from multiple camera feeds into a single GPU batch (5ms queue window)
- **TensorRT accelerator** — compiles ONNX to TRT FP16 on first load, achieving 2–3× additional speedup over plain ONNX Runtime
- **Model warmup** — sends a dummy batch at startup to avoid cold-start latency on the first real request

Launch Triton:
```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

---

## Project Structure

```
football-player-tracking/
├── sport_tracker_deepsort.py   # Main pipeline (Tasks A + B)
├── onnx_inference.py           # Standalone ONNX Runtime inference (Task C)
├── evaluate_yolo.py            # YOLO model evaluation script
├── yolo-usafootball.ipynb      # Training notebook (Kaggle)
├── Dockerfile                  # Containerized environment
├── docker-compose.yml          # Run tracker or ONNX inference
├── requirements.txt            # Python dependencies
├── model_repository/
│   └── yolo_players/
│       ├── config.pbtxt        # Triton server config (Task C)
│       └── 1/
│           └── model.onnx      # Exported ONNX model
├── weights/
│   ├── best.pt                 # PyTorch model
│   └── best.onnx               # ONNX model (FP16, dynamic)
└── data/
    ├── test_1.mp4              # Input video
    ├── radar_template.png      # 2D pitch template
    └── tactical_result.mp4     # Output video (generated)
```

---

## Quickstart

### Without Docker

```bash
pip install -r requirements.txt

# Step 1 — calibrate field corners (once per camera angle)
python sport_tracker_deepsort.py --calibrate \
    --input data/test_1.mp4 \
    --model weights/best.pt \
    --radar data/radar_template.png

# Step 2 — run tracker
python sport_tracker_deepsort.py \
    --input  data/test_1.mp4 \
    --model  weights/best.pt \
    --radar  data/radar_template.png \
    --output data/tactical_result.mp4 \
    --num_players 10 \
    --embedder torchreid+hsv
```

### With Docker

```bash
docker build -t flag-football-tracker .
docker compose run tracker
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedder` | `hsv` | `hsv` / `torchreid` / `torchreid+hsv` |
| `--num_players` | `0` | Expected players on field. IDs remapped to 1..N |
| `--max_cosine_distance` | `0.4` | Re-ID strictness. Lower = harder to switch IDs |
| `--max_age` | `30` | Frames a lost track stays alive before deletion |
| `--n_init` | `3` | Detections required to confirm a new track |
| `--max_jump` | `80` | Max radar-pixel jump per frame before dot is hidden |
| `--field_bot_visible` | `1.0,1.0` | Bottom-corner inward cut for `\ /` camera angle |
| `--src_pts` | *(calibrated)* | 8 numbers: field corners in video pixels |

---

## References

- Pham, D.T., Nguyen, T.T.T., Tran, L.Q. (2025). *SportSORT: Overcoming challenges of multi-object tracking in sports through domain-specific features and out-of-view re-association.* Machine Vision and Applications, 36, 128. https://doi.org/10.1007/s00138-025-01756-y
- Zhang, Y. et al. (2022). *ByteTrack: Multi-object tracking by associating every detection box.*
- Wojke, N. et al. (2017). *Simple online and realtime tracking with a deep association metric (DeepSORT).*
- Zhou, K. et al. (2019). *Omni-Scale Feature Learning for Person Re-Identification (OSNet/TorchReID).*
