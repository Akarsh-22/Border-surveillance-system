"""
app.py
Streamlit dashboard for the IBVAP prototype.
Wraps detector.py's logic so you can see the annotated video feed
and live alert log in a browser instead of an OpenCV popup window.

Run with:
    streamlit run app.py
"""

import time
import os
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from detector import (
    VIDEO_SOURCE, MODEL_NAME, ALERT_LOG_FILE,
    process_frame, ensure_dirs
)

st.set_page_config(page_title="IBVAP - Border Video Analytics", layout="wide")
st.title("🛡️ IBVAP — Intelligent Border Video Analytics Platform")
st.caption("Prototype dashboard: human/vehicle detection + virtual fence intrusion alerts")

ensure_dirs()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("🚨 Alerts")
    alerts_placeholder = st.empty()

start = st.sidebar.button("Start Analysis")
stop = st.sidebar.button("Stop")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

if st.session_state.running:
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_count = 0
    alert_cooldown = {}

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or source unavailable.")
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        annotated = process_frame(model, frame, frame_count, alert_cooldown)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

        if os.path.exists(ALERT_LOG_FILE):
            df = pd.read_csv(ALERT_LOG_FILE)
            alerts_placeholder.dataframe(df.tail(10)[::-1], use_container_width=True)

        time.sleep(0.03)  # small delay so UI doesn't overload

    cap.release()
else:
    st.info("Click 'Start Analysis' in the sidebar to begin processing the video.")
    if os.path.exists(ALERT_LOG_FILE):
        df = pd.read_csv(ALERT_LOG_FILE)
        st.subheader("Past Alerts")
        st.dataframe(df[::-1], use_container_width=True)
