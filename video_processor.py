"""
video_processor.py
-------------------
streamlit-webrtc VideoProcessorBase subclass that ties together:
  - Face detection & recognition
  - Emotion classification
  - Per-frame annotation
  - Attendance marking (idempotent, daily)
  - Engagement logging (per-student cooldown timer)

Thread safety note
------------------
streamlit-webrtc calls recv() from a background thread. We initialise all
heavy state in __init__ so there are no race conditions on first use.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import av
import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from database.mongo_client import MongoConnectionError
from database.db_operations import mark_attendance_if_new, log_engagement
from face_recognition_module.encodings_store import get_store
from face_recognition_module.recognizer import recognize_faces, draw_annotations
from emotion_module.predict import predict_emotion
from engagement.scorer import score_engagement

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    WebRTC video processor: detects faces, recognises students, annotates frames,
    marks attendance, and logs engagement snapshots.
    """

    def __init__(self) -> None:
        self._store = get_store()
        try:
            count = self._store.refresh()
            logger.info("VideoProcessor initialised with %d known student(s).", count)
        except MongoConnectionError as exc:
            logger.error("MongoDB unavailable during VideoProcessor init: %s", exc)

        self._frame_counter: int = 0
        self._last_results: list = []
        self._last_emotion_map: Dict[str, str] = {}
        self._last_logged_time: Dict[str, float] = {}
        self.marked_this_session: List[dict] = []
        self.mongo_ok: bool = True
        self.mongo_error: str = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        self._frame_counter += 1
        if self._frame_counter % config.FRAME_SKIP_RATE == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._process_frame(rgb, bgr)
        annotated = draw_annotations(bgr, self._last_results, self._last_emotion_map)
        blurred = cv2.GaussianBlur(annotated, (0, 0), sigmaX=2)
        annotated = cv2.addWeighted(annotated, 1.5, blurred, -0.5, 0)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _process_frame(self, rgb: np.ndarray, bgr: np.ndarray) -> None:
        encodings = self._store.encodings
        metadata = self._store.metadata
        try:
            results = recognize_faces(rgb, encodings, metadata, config.RECOGNITION_THRESHOLD)
        except Exception as exc:
            logger.warning("recognize_faces error: %s", exc)
            results = []
        self._last_results = results
        emotion_map: Dict[str, str] = {}
        for bbox, student_id, name, distance in results:
            if student_id == "Unknown":
                continue
            top, right, bottom, left = bbox
            pad = 10
            h, w = rgb.shape[:2]
            face_crop = rgb[
                max(0, top - pad): min(h, bottom + pad),
                max(0, left - pad): min(w, right + pad),
            ]
            emotion: Optional[str] = None
            if face_crop.size > 0:
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                try:
                    emotion = predict_emotion(face_gray)
                except Exception as exc:
                    logger.debug("predict_emotion error: %s", exc)
            if emotion:
                emotion_map[student_id] = emotion
            try:
                newly_marked = mark_attendance_if_new(student_id, name)
                if newly_marked:
                    logger.info("Attendance marked for %s", name)
                    self.marked_this_session.append({"student_id": student_id, "name": name})
                self.mongo_ok = True
                self.mongo_error = ""
            except MongoConnectionError as exc:
                self.mongo_ok = False
                self.mongo_error = str(exc)
            except Exception as exc:
                logger.warning("Attendance DB error: %s", exc)
            if emotion:
                now = time.time()
                last_ts = self._last_logged_time.get(student_id, 0.0)
                if (now - last_ts) >= config.ENGAGEMENT_COOLDOWN_SECONDS:
                    score = score_engagement(emotion)
                    try:
                        log_engagement(student_id, emotion, score)
                        self._last_logged_time[student_id] = now
                    except Exception as exc:
                        logger.warning("Engagement log error for %s: %s", student_id, exc)
        self._last_emotion_map = emotion_map

    def refresh_encodings(self) -> None:
        self._store.refresh()

    @property
    def student_count(self) -> int:
        return len(self._store.encodings)
