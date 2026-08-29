"""Streamlit dashboard for multi-video border surveillance analysis."""

import hashlib
import os
import re
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from detector import ALERT_LOG_FILE, MODEL_NAME, ensure_dirs, process_frame
from alpr import get_alpr

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
DEFAULT_VIDEO = PROJECT_DIR / "test.mp4"

st.set_page_config(page_title="IBVAP - Border Video Analytics", layout="wide")
st.title("🛡️ IBVAP — Intelligent Border Video Analytics Platform")
st.caption("Upload one or more videos for detection, fence alerts, and night-movement analysis.")
ensure_dirs()
UPLOAD_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)


@st.cache_resource
def load_alpr():
    return get_alpr()


def save_upload(uploaded_file):
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:10]
    original = Path(uploaded_file.name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original.stem)[:80] or "video"
    destination = UPLOAD_DIR / f"{safe_stem}_{digest}{original.suffix.lower()}"
    if not destination.exists():
        destination.write_bytes(data)
    return destination


def show_alerts():
    if os.path.exists(ALERT_LOG_FILE):
        alerts = pd.read_csv(ALERT_LOG_FILE)
        st.dataframe(alerts.tail(10).iloc[::-1], width="stretch")


def analyze_video(video_path, model, alpr_model, video_placeholder, progress_placeholder):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_count = 0
    alert_cooldown = {}
    motion_state = {}
    plate_cooldown = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        annotated = process_frame(model, frame, frame_count, alert_cooldown, motion_state, alpr_model, plate_cooldown)
        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        if total_frames:
            progress_placeholder.progress(min(frame_count / total_frames, 1.0), text=f"{video_path.name}: frame {frame_count}/{total_frames}")
        time.sleep(0.01)
    cap.release()
    progress_placeholder.progress(1.0, text=f"Finished: {video_path.name}")


uploaded_files = st.sidebar.file_uploader(
    "Upload videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True,
    help="Select multiple CCTV or night videos at once.",
)
video_paths = {}
uploaded_names = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        video_paths[uploaded_file.name] = save_upload(uploaded_file)
        uploaded_names.append(uploaded_file.name)
if DEFAULT_VIDEO.exists():
    video_paths.setdefault("Default: test.mp4", DEFAULT_VIDEO)

if not video_paths:
    st.info("Upload one or more videos in the sidebar to begin.")
else:
    selected_name = st.sidebar.selectbox("Video to analyze", list(video_paths))
    analyze_selected = st.sidebar.button("Analyze selected video", type="primary")
    analyze_all = st.sidebar.button("Analyze all uploaded videos")
    st.subheader("Live Feed")
    video_placeholder = st.empty()
    progress_placeholder = st.empty()
    if analyze_selected or analyze_all:
        model = load_model()
        alpr_model = load_alpr()
        paths = ([video_paths[name] for name in uploaded_names] or [DEFAULT_VIDEO]) if analyze_all else [video_paths[selected_name]]
        for path in paths:
            analyze_video(path, model, alpr_model, video_placeholder, progress_placeholder)
    st.subheader("🚨 Recent Alerts")
    show_alerts()
