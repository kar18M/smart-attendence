"""
face_recognition_module/recognizer.py
---------------------------------------
Frame-level face recognition.

Uses face_recognition (dlib) if available; otherwise falls back to
OpenCV Haar cascade detection + cosine-distance histogram embedding comparison.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

try:
    import face_recognition  # type: ignore
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

BoundingBox = Tuple[int, int, int, int]
RecognitionResult = Tuple[BoundingBox, str, str, float]

_HAAR_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_haar_cascade() -> cv2.CascadeClassifier:
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        _HAAR_CASCADE = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
    return _HAAR_CASCADE


def _detect_faces_haar(rgb: np.ndarray) -> List[BoundingBox]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = _get_haar_cascade()
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    bboxes: List[BoundingBox] = []
    if len(dets) > 0:
        for (x, y, w, h) in dets:
            bboxes.append((y, x + w, y + h, x))
    return bboxes


def _compute_histogram_embedding(face_rgb: np.ndarray) -> np.ndarray:
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


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def recognize_faces(
    frame_rgb: np.ndarray,
    known_encodings: List[np.ndarray],
    known_metadata: List[dict],
    threshold: float = config.RECOGNITION_THRESHOLD,
) -> List[RecognitionResult]:
    results: List[RecognitionResult] = []

    if _FR_AVAILABLE:
        face_locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not face_locations:
            return []
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        for bbox, encoding in zip(face_locations, face_encodings):
            if not known_encodings:
                results.append((bbox, "Unknown", "Unknown", 1.0))
                continue
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist <= threshold:
                meta = known_metadata[best_idx]
                results.append((bbox, meta["student_id"], meta["name"], best_dist))
            else:
                results.append((bbox, "Unknown", "Unknown", best_dist))
    else:
        cosine_threshold = threshold * 0.5
        face_locations = _detect_faces_haar(frame_rgb)
        if not face_locations:
            return []
        for bbox in face_locations:
            top, right, bottom, left = bbox
            face_crop = frame_rgb[top:bottom, left:right]
            if face_crop.size == 0:
                continue
            encoding = _compute_histogram_embedding(face_crop)
            if not known_encodings:
                results.append((bbox, "Unknown", "Unknown", 1.0))
                continue
            distances = np.array([_cosine_distance(encoding, k) for k in known_encodings])
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist <= cosine_threshold:
                meta = known_metadata[best_idx]
                results.append((bbox, meta["student_id"], meta["name"], best_dist))
            else:
                results.append((bbox, "Unknown", "Unknown", best_dist))

    return results


def draw_annotations(
    frame_bgr: np.ndarray,
    results: List[RecognitionResult],
    emotion_labels: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    for bbox, student_id, name, distance in results:
        top, right, bottom, left = bbox
        colour = (0, 200, 0) if student_id != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), colour, 2)
        label = name if student_id != "Unknown" else "Unknown"
        if emotion_labels and student_id in emotion_labels:
            label += f" | {emotion_labels[student_id]}"
        label_y = top - 10 if top > 20 else bottom + 20
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            frame_bgr,
            (left, label_y - text_h - 4),
            (left + text_w + 4, label_y + 2),
            colour, cv2.FILLED,
        )
        cv2.putText(
            frame_bgr, label, (left + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame_bgr
