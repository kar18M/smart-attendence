"""
database/db_operations.py
--------------------------
All CRUD helpers used by the application.
"""

from __future__ import annotations
import logging, os
from datetime import datetime, date, timezone
from typing import List, Optional
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.mongo_client import get_db

logger = logging.getLogger(__name__)

# Students
def insert_student(student_id: str, name: str, face_encoding, photo_path: str) -> bool:
    db = get_db()
    col = db[config.COL_STUDENTS]
    if col.find_one({"student_id": student_id}):
        return False
    col.insert_one({"student_id": student_id, "name": name,
                    "face_encoding": face_encoding.tolist(),
                    "enrolled_on": datetime.now(timezone.utc), "photo_path": photo_path})
    return True

def get_all_students() -> List[dict]:
    db = get_db()
    students = []
    for doc in db[config.COL_STUDENTS].find({}, {"_id": 0}):
        doc["face_encoding"] = np.array(doc["face_encoding"], dtype=np.float64)
        students.append(doc)
    return students

def delete_student(student_id: str) -> bool:
    db = get_db()
    result = db[config.COL_STUDENTS].delete_one({"student_id": student_id})
    return result.deleted_count > 0

def delete_student_full(student_id: str) -> dict:
    """Cascade delete: student + attendance + engagement logs + photo on disk."""
    db = get_db()
    result = {"student_deleted": False, "attendance_deleted": 0,
              "engagement_deleted": 0, "photo_deleted": False, "photo_path": None}
    student_doc = db[config.COL_STUDENTS].find_one({"student_id": student_id}, {"photo_path": 1, "_id": 0})
    if student_doc:
        result["photo_path"] = student_doc.get("photo_path")
    s_res = db[config.COL_STUDENTS].delete_one({"student_id": student_id})
    result["student_deleted"] = s_res.deleted_count > 0
    a_res = db[config.COL_ATTENDANCE].delete_many({"student_id": student_id})
    result["attendance_deleted"] = a_res.deleted_count
    e_res = db[config.COL_ENGAGEMENT].delete_many({"student_id": student_id})
    result["engagement_deleted"] = e_res.deleted_count
    if result["photo_path"]:
        full_path = os.path.join(config.BASE_DIR, result["photo_path"])
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                result["photo_deleted"] = True
        except OSError as exc:
            logger.warning("Could not delete photo %s: %s", full_path, exc)
    return result

# Attendance
def mark_attendance_if_new(student_id: str, name: str) -> bool:
    db = get_db()
    col = db[config.COL_ATTENDANCE]
    today_str = date.today().isoformat()
    if col.find_one({"student_id": student_id, "date": today_str}):
        return False
    col.insert_one({"student_id": student_id, "name": name, "date": today_str,
                    "timestamp": datetime.now(timezone.utc), "status": "Present"})
    return True

def get_attendance_by_date(date_str: str) -> List[dict]:
    db = get_db()
    return list(db[config.COL_ATTENDANCE].find({"date": date_str}, {"_id": 0}))

def get_all_attendance() -> List[dict]:
    db = get_db()
    return list(db[config.COL_ATTENDANCE].find({}, {"_id": 0}).sort("timestamp", -1))

# Engagement
def log_engagement(student_id: str, emotion: str, engagement_score: str) -> None:
    db = get_db()
    db[config.COL_ENGAGEMENT].insert_one({"student_id": student_id,
                                          "timestamp": datetime.now(timezone.utc),
                                          "emotion": emotion, "engagement_score": engagement_score})

def get_engagement_logs(student_id: Optional[str] = None, date_str: Optional[str] = None,
                        start_dt=None, end_dt=None) -> List[dict]:
    db = get_db()
    query: dict = {}
    if student_id:
        query["student_id"] = student_id
    if date_str:
        from datetime import timedelta
        day_start = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        query["timestamp"] = {"$gte": day_start, "$lt": day_start + timedelta(days=1)}
    elif start_dt or end_dt:
        ts_filter = {}
        if start_dt: ts_filter["$gte"] = start_dt
        if end_dt: ts_filter["$lte"] = end_dt
        query["timestamp"] = ts_filter
    return list(db[config.COL_ENGAGEMENT].find(query, {"_id": 0}).sort("timestamp", 1))

def get_engagement_summary_by_student_date() -> List[dict]:
    db = get_db()
    pipeline = [
        {"$group": {"_id": {"student_id": "$student_id",
                             "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                             "score": "$engagement_score"}, "count": {"$sum": 1}}},
        {"$group": {"_id": {"student_id": "$_id.student_id", "date": "$_id.date"},
                    "scores": {"$push": {"score": "$_id.score", "count": "$count"}},
                    "total": {"$sum": "$count"}}},
        {"$sort": {"_id.date": 1, "_id.student_id": 1}},
    ]
    results = []
    for row in db[config.COL_ENGAGEMENT].aggregate(pipeline):
        entry = {"student_id": row["_id"]["student_id"], "date": row["_id"]["date"],
                 "Engaged": 0, "Neutral": 0, "Disengaged": 0, "total": row["total"]}
        for item in row["scores"]:
            entry[item["score"]] = item["count"]
        results.append(entry)
    return results
