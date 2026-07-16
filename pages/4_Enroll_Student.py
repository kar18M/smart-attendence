"""
pages/4_Enroll_Student.py
--------------------------
Register new students with face photos.

Three tabs:
  📷 Webcam Capture   — snap a photo live from webcam
  📁 Upload Photos    — upload 1-3 JPEG/PNG files
  👥 Manage Students  — view enrolled students, delete records
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from database.mongo_client import get_db, MongoConnectionError
from database.db_operations import (
    get_all_students, delete_student, delete_student_full,
    get_all_attendance, get_engagement_logs,
)
from face_recognition_module.enroll import enroll_student, RECOGNITION_BACKEND
import config as cfg

st.set_page_config(page_title="Enroll Student — Smart Attendance", page_icon="➕", layout="wide")

st.markdown("""
<style>
.backend-badge {
    display: inline-block; background: #0f3460; color: #e94560;
    border-radius: 8px; padding: 4px 12px; font-size: 0.78rem;
    font-family: monospace; margin-bottom: 16px;
}
.tip-box {
    background: #0d2137; border-left: 4px solid #4facfe;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.87rem; color: #a8c8e8; margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("➕ Enroll Student")
st.markdown("Register a new student using your **webcam** or by **uploading photos**. Each photo must contain exactly **one person**.")
st.markdown(f'<span class="backend-badge">Recognition backend: {RECOGNITION_BACKEND}</span>', unsafe_allow_html=True)

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

tab_cam, tab_upload, tab_manage = st.tabs(["📷 Webcam Capture", "📁 Upload Photos", "👥 Manage Students"])

# =============================================================================
# TAB 1 — Webcam Capture
# =============================================================================
with tab_cam:
    st.markdown("### 📷 Capture from Webcam")
    st.markdown('<div class="tip-box">💡 Face the camera directly in good lighting. Click <strong>Take photo</strong> when ready.</div>', unsafe_allow_html=True)
    col_cam, col_form = st.columns([1, 1])
    with col_cam:
        captured = st.camera_input("Live camera preview", key="enrollment_camera")
    with col_form:
        st.markdown("#### Student Details")
        cam_student_id = st.text_input("Student ID *", placeholder="e.g. CS101", key="cam_student_id")
        cam_name = st.text_input("Full Name *", placeholder="e.g. Arun Kumar", key="cam_name")
        if captured:
            st.image(captured, width=220)
        enroll_btn = st.button("✅ Enroll from Webcam", type="primary", disabled=(captured is None), key="enroll_cam_btn")
    if enroll_btn:
        errors = []
        if not cam_student_id.strip(): errors.append("Student ID is required.")
        if not cam_name.strip(): errors.append("Full Name is required.")
        if errors:
            for e in errors: st.error(f"❌ {e}")
        else:
            with st.spinner("Processing face…"):
                ok, message = enroll_student(captured.getvalue(), cam_student_id.strip(), cam_name.strip())
            if ok:
                st.success(message)
                st.balloons()
            else:
                st.error(f"❌ {message}")

# =============================================================================
# TAB 2 — Upload Photos
# =============================================================================
with tab_upload:
    st.markdown("### 📁 Upload Face Photos")
    st.markdown('<div class="tip-box">💡 Upload 1 to 3 clear, front-facing photos.</div>', unsafe_allow_html=True)
    with st.form("enrollment_upload_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            upload_student_id = st.text_input("Student ID *", placeholder="e.g. CS102", key="upload_student_id")
        with col2:
            upload_name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma", key="upload_name")
        uploaded_files = st.file_uploader("Upload 1 to 3 face photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("✅ Enroll Student", type="primary")
    if submitted:
        errors = []
        if not upload_student_id.strip(): errors.append("Student ID required.")
        if not upload_name.strip(): errors.append("Full Name required.")
        if not uploaded_files: errors.append("Upload at least one photo.")
        elif len(uploaded_files) > 3: errors.append("Maximum 3 photos.")
        if errors:
            for e in errors: st.error(f"❌ {e}")
        else:
            first_success = False
            for uf in uploaded_files:
                ok, message = enroll_student(uf.read(), upload_student_id.strip(), upload_name.strip())
                if ok and not first_success:
                    st.success(f"**{uf.name}**: {message}")
                    first_success = True
                elif not ok and "already enrolled" in message:
                    break
                elif not ok:
                    st.error(f"**{uf.name}**: {message}")
            if first_success:
                st.balloons()

# =============================================================================
# TAB 3 — Manage Students
# =============================================================================
with tab_manage:
    st.markdown("### 👥 Enrolled Students")
    students = get_all_students()
    if not students:
        st.info("No students enrolled yet.")
    else:
        for s in students:
            sid = s["student_id"]
            sname = s["name"]
            enrolled = s.get("enrolled_on").strftime("%Y-%m-%d %H:%M") if s.get("enrolled_on") else "—"
            photo_path = s.get("photo_path", "")
            full_photo = os.path.join(cfg.BASE_DIR, photo_path) if photo_path else None

            with st.container(border=True):
                col_photo, col_info, col_action = st.columns([1, 4, 2])
                with col_photo:
                    if full_photo and os.path.exists(full_photo):
                        st.image(full_photo, width=80)
                    else:
                        st.markdown("🧑‍🎓")
                with col_info:
                    st.markdown(f"**{sname}**")
                    st.caption(f"ID: `{sid}`  •  Enrolled: {enrolled}")
                with col_action:
                    if st.button("🗑️ Delete", key=f"del_open_{sid}", type="secondary"):
                        current = st.session_state.get(f"del_expand_{sid}", False)
                        st.session_state[f"del_expand_{sid}"] = not current

            if st.session_state.get(f"del_expand_{sid}", False):
                with st.expander(f"⚠️ Delete {sname}  ({sid})", expanded=True):
                    att_count = len([r for r in get_all_attendance() if r["student_id"] == sid])
                    eng_count = len(get_engagement_logs(student_id=sid))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Attendance records", att_count)
                    c2.metric("Engagement logs", eng_count)
                    c3.metric("Face encoding", "1")
                    del_mode = st.radio("What to delete", [
                        "Profile only  (keep attendance history)",
                        "⚠️ Full delete — remove EVERYTHING",
                    ], index=1, key=f"del_mode_{sid}")
                    full_del = "EVERYTHING" in del_mode
                    if full_del:
                        st.error(f"Permanently erases **all {att_count} attendance record(s)** and **all {eng_count} engagement log(s)** for **{sname}**.", icon="🚨")
                    typed = st.text_input(f"Type the student ID `{sid}` to confirm", key=f"del_typed_{sid}", placeholder=sid)
                    ready = typed.strip() == sid
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        label = "🚨 Confirm Full Delete" if full_del else "🗑️ Confirm Delete"
                        if st.button(label, disabled=not ready, type="primary", key=f"del_confirm_{sid}"):
                            if full_del:
                                res = delete_student_full(sid)
                                if res["student_deleted"]:
                                    st.success(f"✅ {sname} fully deleted — {res['attendance_deleted']} attendance, {res['engagement_deleted']} engagement records removed.")
                                else:
                                    st.error("Student not found.")
                            else:
                                if delete_student(sid):
                                    st.success(f"✅ Profile for {sname} removed.")
                                else:
                                    st.error("Deletion failed.")
                            try:
                                from face_recognition_module.encodings_store import get_store
                                get_store().refresh()
                            except Exception:
                                pass
                            st.session_state[f"del_expand_{sid}"] = False
                            st.rerun()
                    with bcol2:
                        if st.button("Cancel", key=f"del_cancel_{sid}"):
                            st.session_state[f"del_expand_{sid}"] = False
                            st.rerun()
