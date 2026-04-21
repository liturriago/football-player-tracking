"""
Multi-Object Tracking Pipeline for Sports Analytics.

Utilizing BoT-SORT natively via Ultralytics with a custom YAML configuration.
BoT-SORT combines ReID (visuals) to prevent identity swaps between different uniforms,
GMC (camera motion compensation), and low-confidence matching for crowding.
"""

import argparse
import sys
import cv2
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO + Custom BoT-SORT pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input video")
    parser.add_argument("--output", type=str, default="output_tactical_view.mp4")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt")
    parser.add_argument("--tracker", type=str, default="custom_botsort.yaml", help="Custom tracker config")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Base confidence")
    return parser.parse_args()

def process_video(video_path: str, output_path: str, model_path: str, tracker_cfg: str, conf_thresh: float) -> None:
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"[INFO] Processing with BoT-SORT at {fps} FPS...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Ultralytics native tracking using BoT-SORT
        results = model.track(
            frame, 
            conf=conf_thresh, 
            iou=0.6, 
            persist=True, 
            tracker=tracker_cfg, 
            verbose=False
        )[0]

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            class_ids = results.boxes.cls.int().cpu().numpy()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # Filter for Class 0 (Player)
                if class_id == 0:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Dibujo de resultados
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + 50, y1), (255, 0, 0), -1)
                    cv2.putText(
                        frame, f"#{track_id}", (x1 + 5, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )

        out.write(frame)
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"[SUCCESS] Video saved to {output_path}")

def main() -> None:
    args = parse_arguments()
    process_video(args.input, args.output, args.model, args.tracker, args.conf_thresh)

if __name__ == "__main__":
    main()
