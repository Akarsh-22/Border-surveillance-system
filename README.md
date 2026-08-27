# IBVAP Prototype — Border Video Analytics

Minimal working prototype: human/vehicle detection + virtual fence
intrusion alerts, using YOLOv8 (pretrained, no training needed).

## Setup (run inside your WSL Ubuntu terminal, in VS Code)

```bash
mkdir -p ~/border-ai-project && cd ~/border-ai-project
# copy detector.py, app.py, requirements.txt into this folder

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place a test video in this folder named `test.mp4`
(any street/CCTV-style footage works — download one or record on your phone).

## Run — option 1: simple OpenCV preview window

```bash
python detector.py
```

Press `q` to quit the preview window. Alerts get logged to `alerts.csv`
and snapshots saved into `snapshots/`.

## Run — option 2: Streamlit dashboard (recommended for demo)

```bash
streamlit run app.py
```

This opens a browser tab. Click "Start Analysis" in the sidebar to
begin processing `test.mp4`. You'll see the annotated video feed and
a live-updating alerts table.

## Tuning the virtual fence

Open `detector.py` and edit `VIRTUAL_FENCE_ZONE` — it's a list of
`(x, y)` pixel coordinates defining a polygon on the frame. Adjust
these to match wherever you want the "restricted zone" to be for
your test video's resolution.

## Next steps to extend

- Add ANPR: crop detected vehicle regions, run EasyOCR on the plate area
- Add face detection: integrate MediaPipe or RetinaFace on person crops
- Swap `test.mp4` for a live RTSP camera URL to simulate a real feed
- Add object tracking (ByteTrack) so the same object isn't re-alerted every frame
