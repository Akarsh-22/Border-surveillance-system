"""
detector.py
Core detection logic for IBVAP prototype:
 - Reads video frames
 - Runs YOLOv8 for human/vehicle detection
 - Checks a "virtual fence" zone for intrusion
 - Logs alerts to a CSV file with timestamp + snapshot

Run standalone for testing:
    python detector.py
"""

import cv2
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ---------- CONFIG ----------
VIDEO_SOURCE = "test.mp4"          # change to 0 for webcam, or an RTSP url for real CCTV
MODEL_NAME = "yolov8n.pt"          # smallest/fastest YOLOv8 model, auto-downloads first run
CONFIDENCE_THRESHOLD = 0.4
ALERT_LOG_FILE = "alerts.csv"
SNAPSHOT_DIR = "snapshots"

# Virtual fence zone: a polygon of (x, y) points in frame pixel coordinates.
# NOTE: these are placeholder coordinates — adjust them to match your video's resolution.
# Easiest way to find good points: print frame.shape once and eyeball a rectangle
# over the area you want to treat as "restricted".
VIRTUAL_FENCE_ZONE = [(400, 200), (900, 200), (900, 600), (400, 600)]

# Classes we care about (COCO class ids): 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "object_class", "confidence", "snapshot_path"])


def log_alert(event_type, object_class, confidence, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{event_type}_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, frame)
    with open(ALERT_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, object_class, f"{confidence:.2f}", snapshot_path])
    print(f"[ALERT] {event_type} | {object_class} | conf={confidence:.2f} | saved={snapshot_path}")


def point_in_zone(point, zone):
    """Check if a point lies inside the virtual fence polygon."""
    import numpy as np
    contour = np.array(zone, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0


def draw_fence(frame, zone):
    import numpy as np
    pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "RESTRICTED ZONE", (zone[0][0], zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def process_frame(model, frame, frame_count, alert_cooldown):
    """Runs detection on a single frame, draws boxes, checks fence, returns annotated frame."""
    results = model(frame, verbose=False)[0]
    draw_fence(frame, VIRTUAL_FENCE_ZONE)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id not in TARGET_CLASSES or conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = TARGET_CLASSES[cls_id]
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Default box color: green
        color = (0, 255, 0)

        # Check virtual fence intrusion
        if point_in_zone(center, VIRTUAL_FENCE_ZONE):
            color = (0, 0, 255)  # red = intrusion
            cooldown_key = f"{label}_{cls_id}"
            last_alert_time = alert_cooldown.get(cooldown_key, 0)
            if time.time() - last_alert_time > 5:  # avoid spamming alerts every frame
                log_alert("INTRUSION", label, conf, frame)
                alert_cooldown[cooldown_key] = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, center, 4, color, -1)

    return frame


def run():
    ensure_dirs()
    print("Loading YOLOv8 model (first run downloads weights)...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: could not open video source '{VIDEO_SOURCE}'")
        return

    frame_count = 0
    alert_cooldown = {}
    print("Press 'q' to quit the preview window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1
        if frame_count % 3 != 0:   # skip frames to keep things fast (process ~1 of every 3)
            continue

        annotated = process_frame(model, frame, frame_count, alert_cooldown)
        cv2.imshow("IBVAP Prototype - Border Video Analytics", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
