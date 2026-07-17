"""
app.py
------
Main entrypoint for the Smart Attendance System.
Run with:  streamlit run app.py

This file defines the landing/home page and verifies MongoDB connectivity
on startup so every other page can rely on the connection being up.
"""

import streamlit as st

# Page must be configured before any other st.* calls
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- MongoDB connectivity check on every page load --------------------------
from database.mongo_client import get_db, MongoConnectionError

@st.cache_resource(show_spinner=False)
def _check_mongo():
    try:
        db = get_db()
        return True, db.name
    except MongoConnectionError as exc:
        return False, str(exc)

mongo_ok, mongo_info = _check_mongo()

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/student-male--v1.png",
        width=60,
    )
    st.title("🎓 Smart Attendance")
    st.markdown("---")

    if mongo_ok:
        st.success(f"✅ MongoDB: connected\n\n`{mongo_info}`")
    else:
        st.error("❌ MongoDB: **not connected**")

    st.markdown("---")
    st.markdown(
        """
        **Navigation**
        - 🎥 Live Attendance
        - 📋 Attendance Log
        - 📊 Analytics
        - ➕ Enroll Student
        """
    )

# ---- Error banner (shown on every page if MongoDB is down) ------------------
if not mongo_ok:
    st.error(
        "⚠️ **MongoDB is not running.**\n\n"
        "Please start it before using this app:\n\n"
        "```bash\n"
        "sudo systemctl start mongod\n"
        "# or\n"
        "mongod --dbpath /data/db\n"
        "```\n\n"
        f"Details: `{mongo_info}`",
        icon="🔴",
    )
    st.stop()

# ---- Home page content -------------------------------------------------------
st.title("🎓 Smart Attendance System")
st.markdown("### Face Recognition · Emotion Tracking · Engagement Analytics")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px; padding:20px; text-align:center; color:white;">
            <h1 style="margin:0;font-size:2.5rem;">🎥</h1>
            <h3>Live Attendance</h3>
            <p style="font-size:0.85rem;">Real-time webcam feed with face recognition and emotion overlay</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg,#11998e,#38ef7d);
                    border-radius:12px; padding:20px; text-align:center; color:white;">
            <h1 style="margin:0;font-size:2.5rem;">📋</h1>
            <h3>Attendance Log</h3>
            <p style="font-size:0.85rem;">Browse and export daily attendance records</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg,#f093fb,#f5576c);
                    border-radius:12px; padding:20px; text-align:center; color:white;">
            <h1 style="margin:0;font-size:2.5rem;">📊</h1>
            <h3>Analytics</h3>
            <p style="font-size:0.85rem;">Engagement trends, heatmaps, and session metrics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg,#4facfe,#00f2fe);
                    border-radius:12px; padding:20px; text-align:center; color:white;">
            <h1 style="margin:0;font-size:2.5rem;">➕</h1>
            <h3>Enroll Student</h3>
            <p style="font-size:0.85rem;">Register new students with face photos</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown(
    """
    ## Quick Start Guide

    | Step | Action |
    |------|--------|
    | **1** | Go to **➕ Enroll Student** and register at least one student with a clear photo |
    | **2** | Go to **🎥 Live Attendance** and click **Start** to begin the webcam session |
    | **3** | The system will recognise enrolled students and mark them as present |
    | **4** | View today's records in **📋 Attendance Log** |
    | **5** | Explore engagement trends in **📊 Analytics** |

    ### Emotion Model Status
    """
)

import config
weights_exist = os.path.exists(config.EMOTION_WEIGHTS_PATH)
if weights_exist:
    st.success("✅ Emotion model weights found — emotion tracking is **active**.")
else:
    st.warning(
        "⚠️ Emotion model weights **not found**.  "
        "The system will work in attendance-only mode.\n\n"
        "To enable emotion tracking, download `fer2013.csv` from Kaggle and run:\n"
        "```bash\npython -m emotion_module.train\n```"
    )

# Enrolled students count
try:
    from database.db_operations import get_all_students
    students = get_all_students()
    st.info(f"👥 **{len(students)}** student(s) currently enrolled.")
except Exception:
    pass
