"""
detector.py
<<<<<<< HEAD
Core detection logic for IBVAP prototype.

Supports TWO modes automatically:
 1. Custom trained model (runs/detect/fence_model/weights/best.pt) —
    if this exists, it's used. Its classes (fence, person-climbing-fence,
    person-standing) let us alert directly on "person-climbing-fence"
    detections — no manual zone needed.
 2. Fallback: pretrained YOLOv8 (yolov8n.pt) with a manual virtual-fence
    polygon zone, for general person/vehicle detection.
=======
Core detection logic for IBVAP prototype:
 - Reads video frames
 - Runs YOLOv8 for human/vehicle detection
 - Checks a "virtual fence" zone for intrusion
 - Logs alerts to a CSV file with timestamp + snapshot
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa

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
<<<<<<< HEAD
VIDEO_SOURCE = "test.mp4"
CUSTOM_MODEL_PATH = "runs/detect/fence_model-4/weights/best.pt"
PRETRAINED_MODEL_NAME = "yolov8n.pt"
=======
VIDEO_SOURCE = "test.mp4"          # change to 0 for webcam, or an RTSP url for real CCTV
MODEL_NAME = "yolov8n.pt"          # smallest/fastest YOLOv8 model, auto-downloads first run
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa
CONFIDENCE_THRESHOLD = 0.4
NIGHT_LUMA_THRESHOLD = 70       # mean grayscale brightness below this = low light
MOTION_PIXEL_THRESHOLD = 0.01   # fraction of changed pixels needed for movement
NIGHT_ALERT_COOLDOWN = 10       # seconds between night movement alerts
ALERT_LOG_FILE = "alerts.csv"
SNAPSHOT_DIR = "snapshots"

<<<<<<< HEAD
# Only used in fallback (pretrained) mode
VIRTUAL_FENCE_ZONE = [(400, 200), (900, 200), (900, 600), (400, 600)]
PRETRAINED_TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Classes that should trigger an alert immediately when detected,
# regardless of zone — used in custom model mode.
CUSTOM_ALERT_CLASSES = {"person-climbing-fence"}


def get_active_model():
    """Returns (model, mode) — mode is 'custom' or 'pretrained'."""
    if os.path.exists(CUSTOM_MODEL_PATH):
        return YOLO(CUSTOM_MODEL_PATH), "custom"
    return YOLO(PRETRAINED_MODEL_NAME), "pretrained"
=======
# Virtual fence zone: a polygon of (x, y) points in frame pixel coordinates.
# NOTE: these are placeholder coordinates — adjust them to match your video's resolution.
# Easiest way to find good points: print frame.shape once and eyeball a rectangle
# over the area you want to treat as "restricted".
VIRTUAL_FENCE_ZONE = [(400, 200), (900, 200), (900, 600), (400, 600)]

# Classes we care about (COCO class ids): 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa


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
<<<<<<< HEAD
    import numpy as np
    contour = np.array(zone, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0
=======
    """Check if a point lies inside the virtual fence polygon."""
    import numpy as np
    contour = np.array(zone, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa


def draw_fence(frame, zone):
    import numpy as np
    pts = np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, "RESTRICTED ZONE", (zone[0][0], zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


<<<<<<< HEAD
<<<<<<< HEAD
def process_frame(model, mode, frame, alert_cooldown, log_alerts=True):
    """Runs detection on a single frame, draws boxes/alerts, returns annotated frame."""
    results = model(frame, verbose=False)[0]
    names = results.names  # class id -> name, works for both custom and pretrained

    if mode == "pretrained":
        draw_fence(frame, VIRTUAL_FENCE_ZONE)
=======
def process_frame(model, frame, frame_count, alert_cooldown):
=======
def detect_night_motion(frame, motion_state):
    """Return (is_low_light, motion_ratio) using brightness and frame change."""
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    is_low_light = float(np.mean(gray)) < NIGHT_LUMA_THRESHOLD

    previous = motion_state.get("previous_gray")
    motion_state["previous_gray"] = gray
    if previous is None:
        return is_low_light, 0.0

    difference = cv2.absdiff(previous, gray)
    _, changed = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.dilate(changed, np.ones((5, 5), np.uint8), iterations=1)
    return is_low_light, float(np.count_nonzero(changed)) / changed.size


def process_frame(model, frame, frame_count, alert_cooldown, motion_state=None):
>>>>>>> 2bff324c2e3bfae4ce74ea1e35dcd9884ff6a226
    """Runs detection on a single frame, draws boxes, checks fence, returns annotated frame."""
    if motion_state is None:
        motion_state = {}

    is_low_light, motion_ratio = detect_night_motion(frame, motion_state)
    if is_low_light and motion_ratio >= MOTION_PIXEL_THRESHOLD:
        last_alert_time = motion_state.get("last_night_alert", 0)
        if time.time() - last_alert_time > NIGHT_ALERT_COOLDOWN:
            log_alert("NIGHT_MOVEMENT", "unknown", motion_ratio, frame)
            motion_state["last_night_alert"] = time.time()

    results = model(frame, verbose=False)[0]
    draw_fence(frame, VIRTUAL_FENCE_ZONE)
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa

    light_label = "NIGHT / MOVEMENT" if is_low_light and motion_ratio >= MOTION_PIXEL_THRESHOLD else \
        ("NIGHT / CLEAR" if is_low_light else "DAY")
    light_color = (0, 165, 255) if is_low_light else (255, 255, 255)
    cv2.putText(frame, f"MODE: {light_label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, light_color, 2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
<<<<<<< HEAD
        label = names.get(cls_id, str(cls_id))

        if mode == "pretrained" and cls_id not in PRETRAINED_TARGET_CLASSES:
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        color = (0, 255, 0)
        is_alert = False

        if mode == "custom":
            if label in CUSTOM_ALERT_CLASSES:
                color = (0, 0, 255)
                is_alert = True
        else:  # pretrained mode uses the zone
            if point_in_zone(center, VIRTUAL_FENCE_ZONE):
                color = (0, 0, 255)
                is_alert = True

        if is_alert and log_alerts:
            cooldown_key = f"{label}_{cls_id}"
            last_alert_time = alert_cooldown.get(cooldown_key, 0)
            if time.time() - last_alert_time > 3:
=======
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
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa
                log_alert("INTRUSION", label, conf, frame)
                alert_cooldown[cooldown_key] = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, center, 4, color, -1)

    return frame


def run():
<<<<<<< HEAD
    """Standalone OpenCV preview (not the Streamlit dashboard)."""
    ensure_dirs()
    model, mode = get_active_model()
    print(f"Loaded model in '{mode}' mode.")
=======
    ensure_dirs()
    print("Loading YOLOv8 model (first run downloads weights)...")
    model = YOLO(MODEL_NAME)
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: could not open video source '{VIDEO_SOURCE}'")
        return

    frame_count = 0
    alert_cooldown = {}
    motion_state = {}
    print("Press 'q' to quit the preview window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
<<<<<<< HEAD
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        annotated = process_frame(model, mode, frame, alert_cooldown)
        cv2.imshow("IBVAP Prototype", annotated)
=======

        frame_count += 1
        if frame_count % 3 != 0:   # skip frames to keep things fast (process ~1 of every 3)
            continue

        annotated = process_frame(model, frame, frame_count, alert_cooldown, motion_state)
        cv2.imshow("IBVAP Prototype - Border Video Analytics", annotated)

>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
<<<<<<< HEAD
    run()
=======
    run()
>>>>>>> 0e0a59a5e78392cebce2e8962b615db9340359aa
