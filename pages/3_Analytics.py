"""
pages/3_Analytics.py
---------------------
Engagement analytics dashboard: line charts, heatmap, and KPI metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime, timezone
from database.mongo_client import get_db, MongoConnectionError
from database.db_operations import (
    get_all_students, get_attendance_by_date, get_engagement_logs,
    get_engagement_summary_by_student_date,
)
from engagement.scorer import score_to_numeric, SCORE_COLOURS

st.set_page_config(page_title="Analytics — Smart Attendance", page_icon="📊", layout="wide")
st.title("📊 Engagement Analytics")
st.markdown("Visualise engagement trends, per-student breakdowns, and session summaries.")

try:
    get_db()
except MongoConnectionError as exc:
    st.error(f"❌ MongoDB not connected: {exc}")
    st.stop()

students = get_all_students()
if not students:
    st.warning("⚠️ No students enrolled yet.")
    st.stop()

student_names = {s["student_id"]: s["name"] for s in students}
col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
with col_f1:
    options = ["All Students"] + [f"{s['name']} ({s['student_id']})" for s in students]
    selected_option = st.selectbox("👤 Select student", options)
    selected_student_id = None if selected_option == "All Students" else selected_option.split("(")[-1].rstrip(")")
with col_f2:
    start_date = st.date_input("📅 From", value=date.today() - timedelta(days=7))
with col_f3:
    end_date = st.date_input("📅 To", value=date.today())
if start_date > end_date:
    st.error("'From' date must be before 'To' date.")
    st.stop()

start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
logs = get_engagement_logs(student_id=selected_student_id, start_dt=start_dt, end_dt=end_dt)
st.markdown("---")

today_str = date.today().isoformat()
today_attendance = get_attendance_by_date(today_str)
today_logs = get_engagement_logs(date_str=today_str)
mk1, mk2, mk3, mk4 = st.columns(4)
mk1.metric("Present Today", len(today_attendance))
mk2.metric("Enrolled Total", len(students))
if today_logs:
    pct = round(sum(1 for l in today_logs if l["engagement_score"] == "Engaged") / len(today_logs) * 100, 1)
    mk3.metric("% Engaged Today", f"{pct}%")
    mk4.metric("Engagement Logs Today", len(today_logs))
else:
    mk3.metric("% Engaged Today", "—")
    mk4.metric("Engagement Logs Today", 0)

st.markdown("---")
st.subheader("📈 Engagement Over Time")
if not logs:
    st.info("No engagement data for the selected filters.")
else:
    df_logs = pd.DataFrame(logs)
    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
    df_logs["score_numeric"] = df_logs["engagement_score"].apply(score_to_numeric)
    df_logs["name"] = df_logs["student_id"].map(student_names).fillna(df_logs["student_id"])
    fig_line = px.scatter(df_logs, x="timestamp", y="score_numeric", color="name",
                          symbol="engagement_score", hover_data=["emotion", "engagement_score"],
                          labels={"score_numeric": "Engagement", "timestamp": "Time", "name": "Student"},
                          title="Engagement Level Over Time")
    fig_line.update_yaxes(tickvals=[0, 1, 2], ticktext=["Disengaged", "Neutral", "Engaged"], range=[-0.3, 2.3])
    fig_line.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.02)")
    st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.subheader("🗟️ Engagement Heatmap (Student × Day)")
summary = get_engagement_summary_by_student_date()
if not summary:
    st.info("Not enough engagement data for heatmap.")
else:
    df_summary = pd.DataFrame(summary)
    df_summary["name"] = df_summary["student_id"].map(student_names).fillna(df_summary["student_id"])
    df_summary["pct_engaged"] = (df_summary["Engaged"] / df_summary["total"] * 100).round(1)
    pivot = df_summary.pivot_table(index="name", columns="date", values="pct_engaged", fill_value=0)
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0.0, "#e74c3c"], [0.5, "#f39c12"], [1.0, "#2ecc71"]],
        zmin=0, zmax=100, colorbar=dict(title="% Engaged")))
    fig_heat.update_layout(title="% Engaged per Student per Day", xaxis_title="Date",
                           height=max(300, 80 * len(pivot.index)), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_heat, use_container_width=True)
