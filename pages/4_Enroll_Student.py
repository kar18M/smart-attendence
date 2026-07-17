"""
pages/4_Enroll_Student.py
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

st.set_page_config(page_title="Enroll Student", page_icon="➕", layout="wide")
st.title("➕ Enroll Student")
st.markdown(f'<span style="background:#0f3460;color:#e94560;border-radius:8px;padding:4px 12px;font-size:0.78rem;font-family:monospace">Backend: {RECOGNITION_BACKEND}</span>', unsafe_allow_html=True)

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

tab_cam, tab_upload, tab_manage = st.tabs(["\U0001f4f7 Webcam Capture", "📁 Upload Photos", "👥 Manage Students"])

with tab_cam:
    st.markdown("### Capture from Webcam")
    col_cam, col_form = st.columns([1, 1])
    with col_cam:
        captured = st.camera_input("Live camera preview", key="enrollment_camera")
    with col_form:
        cam_student_id = st.text_input("Student ID *", placeholder="e.g. CS101", key="cam_student_id")
        cam_name = st.text_input("Full Name *", placeholder="e.g. Arun Kumar", key="cam_name")
        if captured:
            st.image(captured, width=220)
        enroll_btn = st.button("✅ Enroll from Webcam", type="primary", disabled=(captured is None), key="enroll_cam_btn")
    if enroll_btn:
        errors = []
        if not cam_student_id.strip(): errors.append("Student ID is required.")
        if not cam_name.strip(): errors.append("Full Name is required.")
        if not captured: errors.append("Please take a photo first.")
        if errors:
            for e in errors: st.error(f"❌ {e}")
        else:
            with st.spinner("Processing face…"):
                ok, message = enroll_student(image_bytes=captured.getvalue(), student_id=cam_student_id.strip(), name=cam_name.strip())
            if ok:
                st.success(message)
                st.balloons()
            else:
                st.error(f"❌ {message}")

with tab_upload:
    st.markdown("### Upload Face Photos")
    with st.form("enrollment_upload_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            upload_student_id = st.text_input("Student ID *", placeholder="e.g. CS102", key="upload_student_id")
        with col2:
            upload_name = st.text_input("Full Name *", placeholder="e.g. Priya Sharma", key="upload_name")
        uploaded_files = st.file_uploader("Upload 1–3 face photos (JPEG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        submitted = st.form_submit_button("✅ Enroll Student", type="primary")
    if submitted:
        errors = []
        if not upload_student_id.strip(): errors.append("Student ID is required.")
        if not upload_name.strip(): errors.append("Full Name is required.")
        if not uploaded_files: errors.append("Please upload at least one photo.")
        elif len(uploaded_files) > 3: errors.append("Maximum 3 photos allowed.")
        if errors:
            for e in errors: st.error(f"❌ {e}")
        else:
            first_success = False
            for uf in uploaded_files:
                ok, msg = enroll_student(image_bytes=uf.read(), student_id=upload_student_id.strip(), name=upload_name.strip())
                if ok:
                    st.success(f"**{uf.name}**: {msg}")
                    first_success = True
                else:
                    st.error(f"**{uf.name}**: {msg}")
                    if "already enrolled" in msg: break
            if first_success:
                st.balloons()

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
                        st.session_state[f"del_expand_{sid}"] = not st.session_state.get(f"del_expand_{sid}", False)
            if st.session_state.get(f"del_expand_{sid}", False):
                with st.expander(f"⚠️ Delete {sname} ({sid})", expanded=True):
                    att_count = len([r for r in get_all_attendance() if r["student_id"] == sid])
                    eng_count = len(get_engagement_logs(student_id=sid))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Attendance", att_count)
                    c2.metric("Engagement logs", eng_count)
                    c3.metric("Face encoding", "1")
                    del_mode = st.radio("What to delete", ["Profile only", "⚠️ Full delete — remove EVERYTHING"], index=1, key=f"del_mode_{sid}")
                    full_del = "EVERYTHING" in del_mode
                    typed = st.text_input(f"Type `{sid}` to confirm", key=f"del_typed_{sid}", placeholder=sid)
                    ready = typed.strip() == sid
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        label = "🚨 Confirm Full Delete" if full_del else "🗑️ Confirm Delete"
                        if st.button(label, disabled=not ready, type="primary", key=f"del_confirm_{sid}"):
                            if full_del:
                                res = delete_student_full(sid)
                                if res["student_deleted"]:
                                    st.success(f"✅ {sname} fully deleted.")
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
