"""
batch_module/batch_capture.py
------------------------------
Grabs a single high-resolution frame from the webcam using plain OpenCV.
"""

from __future__ import annotations
import logging
from typing import Optional
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_module.batch_config import CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT

logger = logging.getLogger(__name__)

def capture_snapshot(camera_index: int = CAMERA_INDEX, width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera at index %d.", camera_index)
        return None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(3):
            cap.grab()
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

def frame_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
