"""
pages/5_Batch_Attendance.py
Batch mode attendance for large classrooms (30-60+ students).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, logging
from datetime import datetime
import cv2, numpy as np, pandas as pd
import streamlit as st

from database.mongo_client import get_db, MongoConnectionError
from batch_module.batch_capture import capture_snapshot, frame_from_bytes
from batch_module.batch_processor import process_batch_frame
from batch_module import batch_config as bcfg

logger = logging.getLogger(__name__)

def _check_dlib() -> bool:
    try:
        import face_recognition; return True
    except ImportError:
        return False

st.set_page_config(page_title="Batch Attendance", page_icon="📸", layout="wide")

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

st.title("📸 Batch Attendance Mode")
st.markdown("Designed for **large classrooms (30–60+ students)**. Captures periodic high-resolution snapshots.")
st.info("ℹ️ Each capture may take 3–15s. Optimised for accuracy over a full classroom, not real-time.", icon="📊")
st.markdown("---")

with st.sidebar:
    st.markdown("## ⚙️ Batch Settings")
    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=bcfg.CAMERA_INDEX)
    capture_interval = st.slider("Capture interval (s)", 10, 300, bcfg.CAPTURE_INTERVAL_SECONDS, step=5)
    upscale_factor = st.select_slider("Tile upscale factor", options=[1, 2, 3], value=bcfg.UPSCALE_FACTOR)
    threshold = st.slider("Recognition threshold", 0.40, 0.70, bcfg.BATCH_RECOGNITION_THRESHOLD, step=0.01, format="%.2f")
    st.markdown("---")
    st.markdown(f"**Backend:** {'dlib' if _check_dlib() else 'OpenCV Haar (fallback)'}")

if "batch_session_log" not in st.session_state:
    st.session_state.batch_session_log = []
if "continuous_running" not in st.session_state:
    st.session_state.continuous_running = False

def run_one_batch(source="webcam", uploaded_bytes=None):
    with st.spinner("📷 Capturing snapshot…"):
        frame_rgb = frame_from_bytes(uploaded_bytes) if (source == "upload" and uploaded_bytes) else \
            capture_snapshot(camera_index=int(camera_index), width=bcfg.CAPTURE_WIDTH, height=bcfg.CAPTURE_HEIGHT)
    if frame_rgb is None:
        st.error("❌ Could not capture frame."); return
    with st.spinner("🔍 Processing…"):
        try:
            annotated_bgr, summary = process_batch_frame(frame_rgb)
        except Exception as exc:
            st.error(f"❌ {exc}"); return
    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Annotated snapshot", use_container_width=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Faces", summary["unique_faces"])
    c2.metric("Recognised", summary["recognized_count"])
    c3.metric("Unknown", summary["unknown_count"])
    c4.metric("New marks", summary["newly_marked"])
    c5.metric("Time", f"{summary['processing_time_sec']}s")
    if summary["students"]:
        df = pd.DataFrame(summary["students"])[["name","student_id","emotion","distance"]]
        df.columns = ["Name","Student ID","Emotion","Distance"]
        df["Distance"] = df["Distance"].apply(lambda x: f"{x:.3f}")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No enrolled students recognised.")
    st.session_state.batch_session_log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Faces": summary["unique_faces"], "Recognised": summary["recognized_count"],
        "Unknown": summary["unknown_count"], "New Marks": summary["newly_marked"],
        "Process (s)": summary["processing_time_sec"], "Source": source,
    })

tab_webcam, tab_upload = st.tabs(["📷 Webcam Capture", "🖼️ Upload Photo"])
with tab_webcam:
    if st.button("📷 Capture Now", type="primary", key="capture_btn"):
        run_one_batch(source="webcam")
with tab_upload:
    uploaded = st.file_uploader("Upload classroom photo", type=["jpg","jpeg","png"], key="batch_upload")
    if uploaded and st.button("🔍 Process Uploaded Photo", type="primary", key="process_btn"):
        run_one_batch(source="upload", uploaded_bytes=uploaded.read())

st.markdown("---")
st.markdown("### 🔄 Continuous Batch Session")
col_toggle, _ = st.columns([1, 3])
with col_toggle:
    continuous = st.toggle("▶ Run continuous session", value=st.session_state.continuous_running, key="continuous_toggle")
    st.session_state.continuous_running = continuous

if continuous:
    status_ph = st.empty(); image_ph = st.empty(); metrics_ph = st.empty(); students_ph = st.empty()
    count = 0
    while st.session_state.continuous_running:
        count += 1
        status_ph.info(f"🔄 Running — capture #{count}")
        frame_rgb = capture_snapshot(camera_index=int(camera_index), width=bcfg.CAPTURE_WIDTH, height=bcfg.CAPTURE_HEIGHT)
        if frame_rgb is None:
            status_ph.error("❌ Camera error. Retrying in 10s…"); time.sleep(10); continue
        try:
            annotated_bgr, summary = process_batch_frame(frame_rgb)
        except Exception as exc:
            status_ph.error(f"❌ {exc}"); time.sleep(10); continue
        image_ph.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption=f"Capture #{count}", use_container_width=True)
        with metrics_ph.container():
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Faces", summary["unique_faces"]); c2.metric("Recognised", summary["recognized_count"])
            c3.metric("Unknown", summary["unknown_count"]); c4.metric("New", summary["newly_marked"])
            c5.metric("Took", f"{summary['processing_time_sec']}s")
        if summary["students"]:
            with students_ph.container():
                df = pd.DataFrame(summary["students"])[["name","student_id","emotion","distance"]]
                df.columns = ["Name","Student ID","Emotion","Distance"]
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.session_state.batch_session_log.append({
            "Time": datetime.now().strftime("%H:%M:%S"), "Faces": summary["unique_faces"],
            "Recognised": summary["recognized_count"], "Unknown": summary["unknown_count"],
            "New Marks": summary["newly_marked"], "Process (s)": summary["processing_time_sec"], "Source": "webcam (auto)",
        })
        for _ in range(capture_interval):
            if not st.session_state.get("continuous_running", False): break
            time.sleep(1)
    status_ph.success("✅ Continuous session stopped.")

st.markdown("---")
st.markdown("### 📋 Session Capture Log")
if not st.session_state.batch_session_log:
    st.info("No captures yet.")
else:
    log_df = pd.DataFrame(st.session_state.batch_session_log)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export CSV", log_df.to_csv(index=False).encode(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    if st.button("🗑️ Clear Log"): st.session_state.batch_session_log = []; st.rerun()
