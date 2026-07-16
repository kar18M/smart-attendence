"""
face_recognition_module/enroll.py
----------------------------------
Handles student enrollment: face detection, embedding computation,
photo saving, and database insertion.

FALLBACK: Uses OpenCV Haar cascade + histogram embedding if dlib unavailable.
For production accuracy, install: pip install face-recognition
"""

from __future__ import annotations
import logging
import os
import uuid
from typing import Tuple, List
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.db_operations import insert_student

logger = logging.getLogger(__name__)

try:
    import face_recognition
    _FR_AVAILABLE = True
    _BACKEND = "face_recognition"
except ImportError:
    _FR_AVAILABLE = False
    _BACKEND = "opencv_haar"

_HAAR_CASCADE = None

def _get_haar_cascade():
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        _HAAR_CASCADE = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
    return _HAAR_CASCADE

def _compute_histogram_embedding(face_rgb):
    face_resized = cv2.resize(face_rgb, (64, 64))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
    hists = []
    for ch in range(3):
        h, _ = np.histogram(face_resized[:, :, ch], bins=32, range=(0, 256))
        hists.append(h.astype(np.float64))
    h_gray, _ = np.histogram(gray, bins=32, range=(0, 256))
    hists.append(h_gray.astype(np.float64))
    embedding = np.concatenate(hists)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding

def _detect_faces_haar(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    cascade = _get_haar_cascade()
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    bboxes = []
    if len(detections) > 0:
        for (x, y, w, h) in detections:
            bboxes.append((y, x + w, y + h, x))
    return bboxes

def enroll_student(image_bytes: bytes, student_id: str, name: str) -> Tuple[bool, str]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr_image is None:
        return False, "Could not decode image."
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if _FR_AVAILABLE:
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
    else:
        face_locations = _detect_faces_haar(rgb_image)
    if len(face_locations) == 0:
        return False, "No face detected. Use a clear, front-facing photo."
    if len(face_locations) > 1:
        return False, f"{len(face_locations)} faces detected. Upload a photo with exactly ONE person."
    if _FR_AVAILABLE:
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not encodings:
            return False, "Face detected but encoding failed."
        encoding = encodings[0]
    else:
        top, right, bottom, left = face_locations[0]
        face_crop = rgb_image[top:bottom, left:right]
        encoding = _compute_histogram_embedding(face_crop)
    os.makedirs(config.ENROLLED_FACES_DIR, exist_ok=True)
    filename = f"{student_id}_{uuid.uuid4().hex[:8]}.jpg"
    photo_path = os.path.join(config.ENROLLED_FACES_DIR, filename)
    cv2.imwrite(photo_path, bgr_image)
    relative_path = os.path.relpath(photo_path, config.BASE_DIR)
    inserted = insert_student(student_id=student_id, name=name, face_encoding=encoding, photo_path=relative_path)
    if not inserted:
        return False, f"Student ID '{student_id}' already enrolled."
    backend_note = "" if _FR_AVAILABLE else " (OpenCV Haar fallback)"
    return True, f"✅ Enrolled {name} (ID: {student_id}){backend_note}."

RECOGNITION_BACKEND = _BACKEND
