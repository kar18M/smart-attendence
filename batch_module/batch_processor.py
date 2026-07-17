"""
batch_module/batch_processor.py
Full pipeline: tile frame -> detect -> dedup -> recognise -> emotion -> log -> annotate.
"""
from __future__ import annotations
import logging, time
from typing import List, Optional, Dict, Tuple
import cv2, numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from batch_module.batch_config import (
    TILE_SIZE, TILE_OVERLAP, UPSCALE_FACTOR, MIN_FACE_SIZE_PX,
    BATCH_RECOGNITION_THRESHOLD, DEDUP_IOU_THRESHOLD, BATCH_ENGAGEMENT_COOLDOWN_SECONDS,
)
from batch_module.tiling import split_into_tiles, merge_overlapping_detections, Detection
from face_recognition_module.encodings_store import get_store
from emotion_module.predict import predict_emotion
from engagement.scorer import score_engagement
from database.db_operations import mark_attendance_if_new, log_engagement

logger = logging.getLogger(__name__)
try:
    import face_recognition as _fr; _FR_AVAILABLE = True
except ImportError:
    _fr = None; _FR_AVAILABLE = False

_last_logged: Dict[str, float] = {}

def _detect_in_tile(tile_rgb, x_offset, y_offset, upscale, min_face_px):
    if upscale > 1:
        h, w = tile_rgb.shape[:2]
        tile_rgb = cv2.resize(tile_rgb, (w*upscale, h*upscale), interpolation=cv2.INTER_LINEAR)
    if _FR_AVAILABLE:
        locations = _fr.face_locations(tile_rgb, model="hog")
    else:
        gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
        dets = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(min_face_px*upscale, min_face_px*upscale))
        locations = [(y, x+w, y+h, x) for (x, y, w, h) in dets] if len(dets) > 0 else []
    boxes = []
    for (top, right, bottom, left) in locations:
        if (right-left)/upscale < min_face_px or (bottom-top)/upscale < min_face_px: continue
        boxes.append((int(top/upscale)+y_offset, int(right/upscale)+x_offset, int(bottom/upscale)+y_offset, int(left/upscale)+x_offset))
    return boxes

def _recognise_crop(face_rgb, full_frame_rgb, bbox, encodings, metadata, threshold):
    if not encodings: return "Unknown", "Unknown", 1.0
    if _FR_AVAILABLE:
        top, right, bottom, left = bbox
        embs = _fr.face_encodings(full_frame_rgb, known_face_locations=[(top, right, bottom, left)])
        if not embs: return "Unknown", "Unknown", 1.0
        distances = _fr.face_distance(encodings, embs[0])
    else:
        face_resized = cv2.resize(face_rgb, (64, 64))
        gray_r = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
        hists = [np.histogram(face_resized[:,:,ch], bins=32, range=(0,256))[0].astype(np.float64) for ch in range(3)]
        hists.append(np.histogram(gray_r, bins=32, range=(0,256))[0].astype(np.float64))
        encoding = np.concatenate(hists)
        norm = np.linalg.norm(encoding)
        if norm > 0: encoding /= norm
        threshold *= 0.5
        distances = np.array([float(1.0 - np.dot(encoding, k)) for k in encodings])
    best_idx = int(np.argmin(distances)); best_dist = float(distances[best_idx])
    if best_dist <= threshold:
        meta = metadata[best_idx]
        return meta["student_id"], meta["name"], best_dist
    return "Unknown", "Unknown", best_dist

def process_batch_frame(frame_rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
    t0 = time.time()
    store = get_store(); store.ensure_loaded()
    encodings, metadata = store.encodings, store.metadata
    tiles = split_into_tiles(frame_rgb, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    raw: List[Detection] = []
    for tile_img, x_off, y_off in tiles:
        for box in _detect_in_tile(tile_img, x_off, y_off, UPSCALE_FACTOR, MIN_FACE_SIZE_PX):
            raw.append((*box, "?", "?", 1.0, None))
    deduped = merge_overlapping_detections(raw, iou_threshold=DEDUP_IOU_THRESHOLD)
    h, w = frame_rgb.shape[:2]
    results, student_records = [], []
    recognized_count = unknown_count = newly_marked = 0
    now = time.time()
    for det in deduped:
        top = max(0, det[0]); right = min(w, det[1]); bottom = min(h, det[2]); left = max(0, det[3])
        face_crop = frame_rgb[top:bottom, left:right]
        if face_crop.size == 0: continue
        student_id, name, dist = _recognise_crop(face_crop, frame_rgb, (top,right,bottom,left), encodings, metadata, BATCH_RECOGNITION_THRESHOLD)
        emotion = None
        try:
            emotion = predict_emotion(cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY))
        except Exception: pass
        results.append((top, right, bottom, left, student_id, name, dist, emotion))
        if student_id == "Unknown":
            unknown_count += 1
        else:
            recognized_count += 1
            try:
                if mark_attendance_if_new(student_id, name): newly_marked += 1
            except Exception as exc:
                logger.warning("Attendance DB error for %s: %s", student_id, exc)
            if emotion and (now - _last_logged.get(student_id, 0.0)) >= BATCH_ENGAGEMENT_COOLDOWN_SECONDS:
                try:
                    log_engagement(student_id, emotion, score_engagement(emotion))
                    _last_logged[student_id] = now
                except Exception as exc:
                    logger.warning("Engagement log error for %s: %s", student_id, exc)
            student_records.append({"student_id": student_id, "name": name, "distance": round(dist,4), "emotion": emotion or "—", "bbox": (top,right,bottom,left)})
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for top, right, bottom, left, student_id, name, dist, emotion in results:
        colour = (0,200,0) if student_id != "Unknown" else (0,0,220)
        cv2.rectangle(frame_bgr, (left,top), (right,bottom), colour, 2)
        label = (name if student_id != "Unknown" else "Unknown") + (f" | {emotion}" if emotion else "")
        label_y = top-10 if top > 20 else bottom+20
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (left, label_y-th-4), (left+tw+4, label_y+2), colour, cv2.FILLED)
        cv2.putText(frame_bgr, label, (left+2, label_y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    elapsed = round(time.time()-t0, 2)
    return frame_bgr, {
        "total_faces_detected": len(raw), "unique_faces": len(deduped),
        "recognized_count": recognized_count, "unknown_count": unknown_count,
        "newly_marked": newly_marked, "processing_time_sec": elapsed, "students": student_records,
    }
