"""
Sport Tracker - Flag Football MOT Pipeline
==========================================
Task A: DeepSORT (deep-sort-realtime) with custom HSV jersey embedder
Task B: Field Registration — homography + trapezoid-aware dst_pts

Install:
    pip install deep-sort-realtime ultralytics opencv-python numpy
    pip install torchreid gdown tensorboard   # optional but recommended

Usage:
    python sport_tracker.py --calibrate --input video.mp4 --model m.pt --radar r.png
    python sport_tracker.py --input video.mp4 --model m.pt --radar r.png --num_players 15

    # Use TorchReID embedder (stronger Re-ID, slower):
    python sport_tracker.py --input video.mp4 --model m.pt --radar r.png \
        --embedder torchreid --num_players 15
"""

import argparse
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# TASK B — Field Registration
# ─────────────────────────────────────────────────────────────────────────────

class FieldTransformer:
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray):
        self.h_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if self.h_matrix is None:
            raise ValueError("Homography failed — check src_pts are non-collinear.")

    def project_point(self, x: int, y: int) -> Tuple[int, int]:
        pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pt, self.h_matrix)
        return int(out[0][0][0]), int(out[0][0][1])

    @staticmethod
    def calibrate_from_video(video_path: str, frame_idx: int = 30) -> np.ndarray:
        WIN = "Calibration - click 4 corners (TL, TR, BR, BL)"
        points = []
        frame_display = [None]

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                labels = ["TOP-LEFT", "TOP-RIGHT", "BOT-RIGHT", "BOT-LEFT"]
                cv2.circle(frame_display[0], (x, y), 8, (0, 0, 255), -1)
                cv2.putText(frame_display[0],
                            f"{labels[len(points)-1]} ({x},{y})",
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(WIN, frame_display[0])
                if len(points) == 4:
                    flat = ",".join(str(int(v)) for row in points for v in row)
                    print(f"\n[CALIBRATION] src_pts: {flat}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_idx}")

        frame_display[0] = frame.copy()
        cv2.putText(frame_display[0],
                    "Click: 1=TOP-LEFT  2=TOP-RIGHT  3=BOT-RIGHT  4=BOT-LEFT",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.imshow(WIN, frame_display[0])
        cv2.waitKey(1)
        cv2.setMouseCallback(WIN, on_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return np.array(points, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TASK A — Custom Jersey Embedder
# ─────────────────────────────────────────────────────────────────────────────

class JerseyEmbedder:
    """
    Two embedding modes selectable via --embedder:

    'hsv'  (default, fast):
        HSV color histogram on jersey region [30%-70%] of bbox.
        48-dim, no GPU needed. Good for same-team color distinction.

    'torchreid' (stronger, slower):
        OSNet-AIN Re-ID model from TorchReID, trained on multiple
        person Re-ID datasets. 512-dim deep features. Much better at
        distinguishing players with identical jersey colors.
        Requires: pip install torchreid gdown tensorboard

    In both modes the output is L2-normalized and fed directly to
    DeepSORT's cosine distance matching.
    """
    BINS = 16

    def __init__(self, mode: str = "hsv"):
        self.mode = mode
        self._reid = None

        if mode in ("torchreid", "torchreid+hsv"):
            self._init_torchreid()

    def _init_torchreid(self) -> None:
        try:
            import torchreid
            import torch
            print("[INFO] Loading TorchReID model (osnet_ain_x1_0)...")
            self._reid = torchreid.utils.FeatureExtractor(
                model_name="osnet_ain_x1_0",
                model_path="",          # uses default pretrained weights
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            )
            print("[INFO] TorchReID ready.")
        except ImportError:
            print("[WARN] torchreid not installed — falling back to HSV embedder.")
            print("       Install with: pip install torchreid gdown tensorboard")
            self.mode = "hsv"
            self._reid = None

    def embed(self, frame: np.ndarray, raw_dets: list) -> np.ndarray:
        if self.mode == "torchreid+hsv" and self._reid is not None:
            # Combine deep Re-ID features with HSV — best of both worlds:
            # TorchReID handles structure/texture, HSV handles team color
            reid = self._embed_torchreid(frame, raw_dets)  # (N, 512)
            hsv  = self._embed_hsv(frame, raw_dets)         # (N, 48)
            combined = np.concatenate([reid, hsv], axis=1)  # (N, 560)
            norms = np.linalg.norm(combined, axis=1, keepdims=True) + 1e-6
            return (combined / norms).astype(np.float32)
        if self.mode in ("torchreid", "torchreid+hsv") and self._reid is not None:
            return self._embed_torchreid(frame, raw_dets)
        return self._embed_hsv(frame, raw_dets)

    # ── HSV histogram embedder ────────────────────────────────────────────

    def _embed_hsv(self, frame: np.ndarray, raw_dets: list) -> np.ndarray:
        embeddings = []
        fh, fw = frame.shape[:2]
        for det in raw_dets:
            ltwh = det[0]
            x1, y1 = int(ltwh[0]), int(ltwh[1])
            x2, y2 = int(ltwh[0] + ltwh[2]), int(ltwh[1] + ltwh[3])
            bh = y2 - y1
            y_top = y1 + int(bh * 0.30)
            y_bot = y1 + int(bh * 0.70)
            crop = frame[max(0,y_top):min(fh,y_bot), max(0,x1):min(fw,x2)]
            if crop.size == 0:
                embeddings.append(np.zeros(self.BINS * 3, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hh = cv2.calcHist([hsv],[0],None,[self.BINS],[0,180]).flatten()
            hs = cv2.calcHist([hsv],[1],None,[self.BINS],[0,256]).flatten()
            hv = cv2.calcHist([hsv],[2],None,[self.BINS],[0,256]).flatten()
            feat = np.concatenate([hh, hs, hv]).astype(np.float32)
            norm = np.linalg.norm(feat) + 1e-6
            embeddings.append(feat / norm)
        return np.array(embeddings, dtype=np.float32)

    # ── TorchReID embedder ────────────────────────────────────────────────

    def _embed_torchreid(self, frame: np.ndarray, raw_dets: list) -> np.ndarray:
        import torch
        fh, fw = frame.shape[:2]
        crops = []
        for det in raw_dets:
            ltwh = det[0]
            x1, y1 = int(ltwh[0]), int(ltwh[1])
            x2, y2 = int(ltwh[0] + ltwh[2]), int(ltwh[1] + ltwh[3])
            crop = frame[max(0,y1):min(fh,y2), max(0,x1):min(fw,x2)]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            # TorchReID expects RGB, 256x128
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (128, 256))
            crops.append(crop_resized)

        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        with torch.no_grad():
            feats = self._reid(crops)   # returns (N, 512) tensor

        feats_np = feats.cpu().numpy().astype(np.float32)
        # L2-normalize each row
        norms = np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-6
        return feats_np / norms


# ─────────────────────────────────────────────────────────────────────────────
# TASK A — Radar jump gate
# ─────────────────────────────────────────────────────────────────────────────

class RadarGate:
    """
    Vetoes radar dots whose position jumps more than max_jump radar-pixels
    from the smoothed recent history. Operates in radar space (not video
    pixels) so perspective distortion does not affect the threshold.
    """
    def __init__(self, max_jump: float = 80.0, history: int = 5):
        self.max_jump    = max_jump
        self.history_len = history
        self._hist: Dict[int, deque] = {}

    def check_and_update(self, tid: int, rxy: Tuple[int,int], frames_lost: int = 0) -> bool:
        allowed = self.max_jump + frames_lost * 10
        if tid not in self._hist or len(self._hist[tid]) == 0:
            self._push(tid, rxy)
            return True
        avg = np.mean(list(self._hist[tid]), axis=0)
        if float(np.linalg.norm(np.array(rxy) - avg)) > allowed:
            return False
        self._push(tid, rxy)
        return True

    def _push(self, tid: int, rxy: Tuple[int,int]) -> None:
        if tid not in self._hist:
            self._hist[tid] = deque(maxlen=self.history_len)
        self._hist[tid].append(rxy)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class SportPipeline:
    COLOR = (0, 200, 255)

    def __init__(
        self,
        model_path:  str,
        radar_path:  str,
        conf:        float,
        start_frame: int,
        src_pts:     np.ndarray,
        dst_pts:     Optional[np.ndarray] = None,
        max_age:     int   = 30,
        n_init:      int   = 3,
        max_cosine_distance: float = 0.4,
        max_jump:    float = 80.0,
        num_players: int   = 0,
        embedder:    str   = "hsv",    # "hsv" or "torchreid"
        field_visible:     tuple = (1.0, 1.0, 1.0, 1.0),
        field_bot_visible: tuple = (1.0, 1.0),
    ):
        self.model       = YOLO(model_path)
        self.conf        = conf
        self.start_frame = start_frame
        self.num_players = num_players

        self.radar_template = cv2.imread(radar_path)
        if self.radar_template is None:
            raise FileNotFoundError(f"Radar template not found: {radar_path}")
        self.radar_h, self.radar_w = self.radar_template.shape[:2]

        if dst_pts is None:
            dst_pts = self._build_dst_pts(field_visible, field_bot_visible)
            print("[INFO] dst_pts computed:")
            for name, pt in zip(["TL","TR","BR","BL"], dst_pts):
                print(f"       {name}: ({pt[0]:.1f}, {pt[1]:.1f})")

        self.transformer = FieldTransformer(src_pts, dst_pts)

        self.embedder = JerseyEmbedder(mode=embedder)
        self.tracker  = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder=None,   # we supply embeddings manually
            half=False,
        )
        self.radar_gate = RadarGate(max_jump=max_jump)

    def _build_dst_pts(self, fv, fbv) -> np.ndarray:
        W, H = self.radar_w, self.radar_h
        vl, vr, vt, vb = fv
        vbl, vbr = fbv
        pt  = (1-vt)*H;  pb  = (1-vb)*H
        pl  = (1-vl)*W;  pr  = (1-vr)*W
        pbl = pl + (1-vbl)*W
        pbr = pr + (1-vbr)*W
        return np.array([
            [pl,    pt   ],
            [W-pr,  pt   ],
            [W-pbr, H-pb ],
            [pbl,   H-pb ],
        ], dtype=np.float32)

    def _draw_radar(self, radar_view, fx, fy, tid):
        rx, ry = self.transformer.project_point(fx, fy)
        R = 18
        rx = int(np.clip(rx, R+4, self.radar_w-R-4))
        ry = int(np.clip(ry, R+4, self.radar_h-R-4))
        cv2.circle(radar_view, (rx,ry), R+3, (0,0,0),       -1)
        cv2.circle(radar_view, (rx,ry), R,   self.COLOR,     -1)
        cv2.circle(radar_view, (rx,ry), R,   (255,255,255),   2)
        lbl = str(tid)
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(radar_view, lbl, (rx-tw//2, ry+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(radar_view, lbl, (rx-tw//2, ry+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        return rx, ry

    def run(self, input_path: str, output_path: str) -> None:
        cap   = cv2.VideoCapture(input_path)
        fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out   = cv2.VideoWriter(output_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (vid_w+600, vid_h))
        frame_idx = 0
        print(f"[INFO] {input_path}  ->  {output_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            radar_view = self.radar_template.copy()

            if frame_idx >= self.start_frame:
                results = self.model(frame, conf=self.conf, verbose=False)[0]

                raw_dets = []
                for box in results.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1,y1,x2,y2 = map(float, box.xyxy[0])
                    raw_dets.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), 0))

                if raw_dets:
                    embeds = self.embedder.embed(frame, raw_dets)
                    tracks = self.tracker.update_tracks(raw_dets, embeds=embeds)
                    confirmed = [t for t in tracks if t.is_confirmed()]

                    # compact ID remap
                    if self.num_players > 0 and confirmed:
                        srt = sorted(confirmed, key=lambda t: t.time_since_update)
                        id_remap = {t.track_id: i+1
                                    for i,t in enumerate(srt[:self.num_players])}
                    else:
                        id_remap = None

                    for track in confirmed:
                        raw_id = track.track_id
                        if id_remap is not None:
                            if raw_id not in id_remap:
                                continue
                            did = id_remap[raw_id]
                        else:
                            did = raw_id

                        ltrb = track.to_ltrb(orig=True)
                        if ltrb is None:
                            ltrb = track.to_ltrb()
                        x1,y1,x2,y2 = map(int, ltrb)
                        fx, fy = (x1+x2)//2, y2

                        rxy = self.transformer.project_point(fx, fy)
                        ok  = self.radar_gate.check_and_update(
                            did, rxy, track.time_since_update)

                        cv2.rectangle(frame,(x1,y1),(x2,y2),self.COLOR,2)
                        cv2.putText(frame, f"ID:{did}", (x1,y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR, 2)
                        if ok:
                            self._draw_radar(radar_view, fx, fy, did)

            radar_resized = cv2.resize(radar_view, (600, vid_h))
            out.write(np.hstack((frame, radar_resized)))
            if frame_idx % 100 == 0:
                print(f"[INFO] Frame {frame_idx}")

        cap.release()
        out.release()
        print(f"[SUCCESS] -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_pts(s):
    v = list(map(float, s.split(",")))
    if len(v) != 8:
        raise argparse.ArgumentTypeError("Need 8 comma-separated numbers.")
    return np.array(v, dtype=np.float32).reshape(4,2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--model",        required=True)
    p.add_argument("--radar",        required=True)
    p.add_argument("--output",       default="tactical_result.mp4")
    p.add_argument("--conf",         type=float, default=0.3)
    p.add_argument("--start_frame",  type=int,   default=27)
    p.add_argument("--src_pts",      type=parse_pts,
                   default=parse_pts("236,262,812,255,1145,631,2,632"))
    p.add_argument("--dst_pts",      type=parse_pts, default=None)
    p.add_argument("--field_visible",     default="1.0,1.0,1.0,1.0")
    p.add_argument("--field_bot_visible", default="1.0,1.0")
    p.add_argument("--calibrate",    action="store_true")
    p.add_argument("--max_age",      type=int,   default=30)
    p.add_argument("--n_init",       type=int,   default=3)
    p.add_argument("--max_cosine_distance", type=float, default=0.4)
    p.add_argument("--max_jump",     type=float, default=80.0)
    p.add_argument("--num_players",  type=int,   default=0)
    p.add_argument("--embedder",     default="hsv",
                   choices=["hsv", "torchreid"],
                   help="'hsv' = fast color histogram (default). "
                        "'torchreid' = OSNet-AIN deep Re-ID (stronger, needs GPU).")
    args = p.parse_args()

    if args.calibrate:
        print("[CALIBRATION] Click 4 field corners then press any key...")
        found = FieldTransformer.calibrate_from_video(
            args.input, frame_idx=args.start_frame)
        flat = ",".join(str(int(v)) for v in found.flatten())
        print(f"[INFO] src_pts: {flat}")
        print("[INFO] Continuing with these points...\n")
        args.src_pts = found

    SportPipeline(
        model_path           = args.model,
        radar_path           = args.radar,
        conf                 = args.conf,
        start_frame          = args.start_frame,
        src_pts              = args.src_pts,
        dst_pts              = args.dst_pts,
        max_age              = args.max_age,
        n_init               = args.n_init,
        max_cosine_distance  = args.max_cosine_distance,
        max_jump             = args.max_jump,
        num_players          = args.num_players,
        embedder             = args.embedder,
        field_visible        = tuple(float(x) for x in args.field_visible.split(",")),
        field_bot_visible    = tuple(float(x) for x in args.field_bot_visible.split(",")),
    ).run(args.input, args.output)

if __name__ == "__main__":
    main()