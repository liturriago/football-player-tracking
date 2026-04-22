"""
Task C — ONNX Inference Script
================================
Runs the exported YOLO model (best.onnx) directly via ONNX Runtime,
bypassing Ultralytics entirely. This is what Triton will do internally.

Export settings used:
    format="onnx", imgsz=512, half=True, dynamic=True,
    simplify=True, nms=False

Because nms=False, we apply NMS manually after inference.

Usage:
    python onnx_inference.py --model weights/best.onnx --input data/test_1.mp4
    python onnx_inference.py --model weights/best.onnx --input data/frame.jpg

Install:
    pip install onnxruntime-gpu   # GPU (recommended)
    pip install onnxruntime       # CPU fallback
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Run: pip install onnxruntime-gpu  (or onnxruntime for CPU)")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    new_shape: int = 512,
    color: tuple = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize keeping aspect ratio, pad to square.
    Returns (padded_img, scale, (pad_w, pad_h)).
    """
    h, w = img.shape[:2]
    scale = new_shape / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right  = pad_w // 2, pad_w - pad_w // 2

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, scale, (left, top)


def preprocess(img_bgr: np.ndarray, imgsz: int = 512) -> np.ndarray:
    """BGR frame → ONNX input tensor (1, 3, H, W) float16."""
    img, _, _ = letterbox(img_bgr, imgsz)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose(2, 0, 1)           # HWC → CHW
    tensor = np.expand_dims(tensor, 0)             # → (1,3,H,W)
    tensor = tensor.astype(np.float16) / 255.0     # half precision, normalize
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Postprocessing (manual NMS since nms=False at export)
# ─────────────────────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_thresh: float = 0.45) -> np.ndarray:
    """CPU NMS. Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int32)


def postprocess(
    output: np.ndarray,       # raw model output (1, 5+nc, num_anchors)
    orig_shape: tuple,        # (H, W) of original frame
    imgsz: int  = 512,
    conf_thresh: float = 0.3,
    iou_thresh:  float = 0.45,
    player_cls:  int   = 0,   # class index for "player"
) -> list[dict]:
    """
    Returns list of dicts: {bbox:[x1,y1,x2,y2], conf:float, cls:int}
    Coordinates are in original image pixels.
    """
    orig_h, orig_w = orig_shape

    # output shape: (1, 5+nc, N) — transpose to (N, 5+nc)
    preds = output[0].T.astype(np.float32)   # (N, 5+nc)

    # cx, cy, w, h in letterbox space; class scores start at index 4
    boxes_lbox = preds[:, :4]
    class_scores = preds[:, 4:]               # (N, nc)
    cls_ids  = class_scores.argmax(axis=1)
    confs    = class_scores.max(axis=1)

    # filter by confidence and player class
    mask = (confs >= conf_thresh) & (cls_ids == player_cls)
    if mask.sum() == 0:
        return []

    boxes_lbox = boxes_lbox[mask]
    confs      = confs[mask]
    cls_ids    = cls_ids[mask]

    # xywh → xyxy in letterbox space
    boxes_xyxy = xywh2xyxy(boxes_lbox)

    # NMS
    keep = nms(boxes_xyxy, confs, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    confs      = confs[keep]
    cls_ids    = cls_ids[keep]

    # Scale back to original image coords
    # letterbox scale factor
    scale = imgsz / max(orig_h, orig_w)
    pad_w = (imgsz - int(orig_w * scale)) // 2
    pad_h = (imgsz - int(orig_h * scale)) // 2

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
    boxes_xyxy = np.clip(boxes_xyxy,
                         [0, 0, 0, 0],
                         [orig_w, orig_h, orig_w, orig_h])

    results = []
    for box, conf, cls in zip(boxes_xyxy, confs, cls_ids):
        results.append({
            "bbox": box.tolist(),
            "conf": float(conf),
            "cls":  int(cls),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime session
# ─────────────────────────────────────────────────────────────────────────────

def build_session(model_path: str) -> ort.InferenceSession:
    """
    Creates ONNX Runtime session.
    Prefers CUDAExecutionProvider (GPU), falls back to CPU.
    """
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )
    provider_used = session.get_providers()[0]
    print(f"[ONNX] Session ready — provider: {provider_used}")
    print(f"[ONNX] Input  : {session.get_inputs()[0].name} "
          f"{session.get_inputs()[0].shape}")
    print(f"[ONNX] Output : {session.get_outputs()[0].name} "
          f"{session.get_outputs()[0].shape}")
    return session


def run_inference(
    session: ort.InferenceSession,
    tensor: np.ndarray,
) -> np.ndarray:
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: tensor})[0]


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["conf"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_on_video(session, input_path: str, output_path: str,
                 imgsz: int, conf: float, start_frame: int) -> None:
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    frame_idx  = 0
    total_time = 0.0
    print(f"[INFO] {input_path} → {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < start_frame:
            out.write(frame)
            continue

        t0 = time.perf_counter()
        tensor = preprocess(frame, imgsz)
        raw    = run_inference(session, tensor)
        dets   = postprocess(raw, frame.shape[:2], imgsz, conf)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        draw_detections(frame, dets)
        cv2.putText(frame, f"ONNX {1/elapsed:.1f} FPS  {len(dets)} players",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        out.write(frame)

        if frame_idx % 100 == 0:
            print(f"[INFO] Frame {frame_idx} | "
                  f"{len(dets)} detections | "
                  f"{elapsed*1000:.1f} ms/frame")

    processed = max(frame_idx - start_frame, 1)
    print(f"\n[DONE] Avg inference: {total_time/processed*1000:.1f} ms/frame  "
          f"({processed/total_time:.1f} FPS)")
    cap.release()
    out.release()


def run_on_image(session, input_path: str,
                 imgsz: int, conf: float) -> None:
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(input_path)

    t0     = time.perf_counter()
    tensor = preprocess(frame, imgsz)
    raw    = run_inference(session, tensor)
    dets   = postprocess(raw, frame.shape[:2], imgsz, conf)
    print(f"[ONNX] {len(dets)} detections in {(time.perf_counter()-t0)*1000:.1f} ms")

    out = draw_detections(frame.copy(), dets)
    out_path = Path(input_path).with_stem(Path(input_path).stem + "_onnx")
    cv2.imwrite(str(out_path), out)
    print(f"[SAVED] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="ONNX Runtime inference for YOLO player detector"
    )
    p.add_argument("--model",       required=True,
                   help="Path to best.onnx")
    p.add_argument("--input",       required=True,
                   help="Video or image path")
    p.add_argument("--output",      default="onnx_result.mp4",
                   help="Output video path (ignored for images)")
    p.add_argument("--imgsz",       type=int,   default=512)
    p.add_argument("--conf",        type=float, default=0.3)
    p.add_argument("--start_frame", type=int,   default=27)
    args = p.parse_args()

    session = build_session(args.model)

    ext = Path(args.input).suffix.lower()
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        run_on_video(session, args.input, args.output,
                     args.imgsz, args.conf, args.start_frame)
    else:
        run_on_image(session, args.input, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
