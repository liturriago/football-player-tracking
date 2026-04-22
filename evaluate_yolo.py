"""
YOLO Raw Detection Evaluation Script.

This module processes a video using only the YOLO model to visualize
the raw bounding boxes, classes, and confidence scores. It is intended
for debugging model accuracy before passing detections to a tracker.
"""

import argparse
import sys

import cv2
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate raw YOLO detections on a video.")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="yolo_raw_eval.mp4", 
        help="Path to save the annotated output video"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to the trained YOLO weights (.pt)"
    )
    parser.add_argument(
        "--conf-thresh", 
        type=float, 
        default=0.25, 
        help="Confidence threshold for detections"
    )
    
    return parser.parse_args()


def evaluate_video(video_path: str, output_path: str, model_path: str, conf_thresh: float) -> None:
    """Runs YOLO inference and saves the annotated video."""
    
    # 1. Load Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # 2. Setup Video I/O
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Evaluating video at {fps} FPS. Resolution: {width}x{height}")

    # 3. Processing Loop
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Inference with confidence threshold applied
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        # Use Ultralytics native plotting to draw boxes, classes, and confidences
        annotated_frame = results.plot()

        out.write(annotated_frame)
        
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    # 4. Cleanup
    cap.release()
    out.release()
    print(f"[SUCCESS] Evaluation video saved to {output_path}")


def main() -> None:
    args = parse_arguments()
    evaluate_video(args.input, args.output, args.model, args.conf_thresh)


if __name__ == "__main__":
    main()
