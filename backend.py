"""
backend.py
IBVAP — FastAPI backend, built on top of the current detector.py
(which includes Akshay's night-vision / motion detection additions).

Replaces the Streamlit app with a REST API + static HTML5/JS frontend —
much easier to deploy than Streamlit (Docker, Render, Railway, a VPS, etc.)

Endpoints:
    GET  /                          -> serves the frontend (static/index.html)
    GET  /api/videos                -> list of available videos (default + uploaded)
    POST /api/upload                -> upload a new video
    GET  /api/video_info/{video_id} -> total_frames, fps for a video
    GET  /api/frame/{video_id}/{n}  -> annotated JPEG frame at index n
    GET  /api/alerts                -> JSON list of logged alerts
    DELETE /api/alerts              -> clear the alert log
    GET  /api/model_info            -> which model is active + feature flags

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import csv as csvmod

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from detector import (
    VIDEO_SOURCE, MODEL_NAME, ALERT_LOG_FILE,
    load_authorized_people, process_frame, ensure_dirs,
)

app = FastAPI(title="IBVAP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ensure_dirs()

# video_id -> file path + display name. "default" always points at test.mp4
VIDEO_REGISTRY = {}
if os.path.exists(VIDEO_SOURCE):
    VIDEO_REGISTRY["default"] = {"path": VIDEO_SOURCE, "name": f"Default: {VIDEO_SOURCE}"}

# Load the YOLO model once at startup (shared across requests)
MODEL = YOLO(MODEL_NAME)

# Per-video state: alert cooldown dict + night-motion tracking dict.
# Kept in memory; resets if the server restarts.
VIDEO_STATE = {}


def get_state(video_id: str):
    if video_id not in VIDEO_STATE:
        VIDEO_STATE[video_id] = {"alert_cooldown": {}, "motion_state": {}}
    return VIDEO_STATE[video_id]


def get_video_path(video_id: str) -> str:
    entry = VIDEO_REGISTRY.get(video_id)
    if entry is None or not os.path.exists(entry["path"]):
        raise HTTPException(status_code=404, detail="video_id not found")
    return entry["path"]


@app.get("/api/videos")
def list_videos():
    return [{"video_id": vid, "name": info["name"]} for vid, info in VIDEO_REGISTRY.items()]


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    VIDEO_REGISTRY[video_id] = {"path": save_path, "name": file.filename}
    VIDEO_STATE[video_id] = {"alert_cooldown": {}, "motion_state": {}}

    cap = cv2.VideoCapture(save_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.release()

    return {"video_id": video_id, "total_frames": total_frames, "fps": fps, "filename": file.filename}


@app.get("/api/video_info/{video_id}")
def video_info(video_id: str):
    path = get_video_path(video_id)
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.release()
    return {"video_id": video_id, "total_frames": total_frames, "fps": fps}


@app.get("/api/frame/{video_id}/{frame_idx}")
def get_frame(video_id: str, frame_idx: int, log_alerts: bool = True):
    path = get_video_path(video_id)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=404, detail="Frame not available (end of video?)")

    state = get_state(video_id)
    cooldown = state["alert_cooldown"] if log_alerts else {}
    motion_state = state["motion_state"]

    annotated = process_frame(MODEL, frame, frame_idx, cooldown, motion_state, load_authorized_people())

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")

    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    if not os.path.exists(ALERT_LOG_FILE):
        return JSONResponse([])
    rows = []
    with open(ALERT_LOG_FILE, "r", newline="") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append(row)
    return JSONResponse(rows[-limit:][::-1])


@app.delete("/api/alerts")
def clear_alerts():
    if os.path.exists(ALERT_LOG_FILE):
        os.remove(ALERT_LOG_FILE)
    ensure_dirs()
    return {"status": "cleared"}


@app.get("/api/model_info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "features": {
            "virtual_fence_intrusion": True,
            "night_motion_detection": True,
        },
    }


# Serve the frontend (index.html, CSS, JS, logo) from ./static
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
