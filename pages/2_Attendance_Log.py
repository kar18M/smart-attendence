"""
pages/2_Attendance_Log.py
--------------------------
Browse attendance records by date with CSV export.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import date
from database.mongo_client import get_db, MongoConnectionError
from database.db_operations import get_attendance_by_date, get_all_attendance

st.set_page_config(page_title="Attendance Log — Smart Attendance", page_icon="📋", layout="wide")
st.title("📋 Attendance Log")
st.markdown("View, filter, and export daily attendance records.")

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

col_date, col_all, _ = st.columns([2, 1, 3])
with col_date:
    selected_date = st.date_input("📅 Select date", value=date.today())
with col_all:
    st.markdown("<br>", unsafe_allow_html=True)
    show_all = st.checkbox("Show all dates")

if show_all:
    records = get_all_attendance()
    title_str = "All records"
else:
    records = get_attendance_by_date(selected_date.isoformat())
    title_str = f"Records for **{selected_date.strftime('%A, %d %B %Y')}**"

st.markdown(f"### {title_str}")
if not records:
    st.info("💭 No attendance records found.")
else:
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "date" not in df.columns:
        df["date"] = selected_date.isoformat()
    preferred_order = ["name", "student_id", "date", "timestamp", "status"]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Present", len(df))
    if "date" in df.columns: m2.metric("Unique Dates", df["date"].nunique())
    if "name" in df.columns: m3.metric("Unique Students", df["name"].nunique())
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    filename = f"attendance_all_{date.today().isoformat()}.csv" if show_all else f"attendance_{selected_date.isoformat()}.csv"
    st.download_button("⬇️ Download as CSV", data=csv_data, file_name=filename, mime="text/csv", type="primary")
