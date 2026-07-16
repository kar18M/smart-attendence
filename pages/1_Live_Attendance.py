"""
pages/1_Live_Attendance.py
---------------------------
Live webcam feed with real-time face recognition and emotion display.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from database.mongo_client import get_db, MongoConnectionError
from database.db_operations import get_all_students
from video_processor import VideoProcessor

st.set_page_config(page_title="Live Attendance — Smart Attendance", page_icon="🎥", layout="wide")
st.title("🎥 Live Attendance")
st.markdown("The webcam feed runs face recognition every frame. Recognised students are marked present automatically.")

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

students = get_all_students()
if not students:
    st.warning("⚠️ No students enrolled yet. Go to **➕ Enroll Student** to register students first.")
    st.stop()

col_video, col_sidebar = st.columns([3, 1])
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col_video:
    st.markdown("### 📡 Live Feed")
    ctx = webrtc_streamer(
        key="smart-attendance-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": {"min": 640, "ideal": 1280}, "height": {"min": 480, "ideal": 720},
                      "frameRate": {"ideal": 15, "max": 30},
                      "noiseSuppression": False, "echoCancellation": False, "autoGainControl": False},
            "audio": False,
        },
        async_processing=True,
    )
    if ctx.state.playing:
        st.success("🟢 Stream active — face recognition running")
    else:
        st.info("⚪ Click **START** above to begin the live session.")
    st.caption("💡 If stream fails to start, check webcam permission.")

with col_sidebar:
    st.markdown("### 📊 Session Status")
    st.success("✅ MongoDB connected")
    st.metric("Enrolled students", len(students))
    st.markdown("---")
    st.markdown("### ✅ Marked Present This Session")
    if ctx.video_processor:
        processor = ctx.video_processor
        marked = processor.marked_this_session
        if marked:
            for entry in marked:
                st.markdown(f"- 🧑‍🎓 **{entry['name']}** `{entry['student_id']}`")
        else:
            st.info("No one marked yet.")
        if not processor.mongo_ok:
            st.error(f"DB error: {processor.mongo_error[:120]}…")
    else:
        st.info("Stream not active.")
    st.markdown("---")
    if st.button("🔄 Refresh Student List"):
        if ctx.video_processor:
            ctx.video_processor.refresh_encodings()
            st.success("Encodings refreshed from DB.")
        else:
            st.warning("Stream not active.")
