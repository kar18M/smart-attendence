"""
emotion_module/predict.py
--------------------------
Emotion inference: load the trained EmotionCNN and classify a 48×48 face crop.
Returns None gracefully if model weights are not found.
"""

from __future__ import annotations
import logging, os
from typing import Optional
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from emotion_module.model import build_model

logger = logging.getLogger(__name__)
_model = None
_model_loaded = False

def _load_model_internal():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    if not os.path.exists(config.EMOTION_WEIGHTS_PATH):
        logger.warning("Emotion model weights not found. App continues in attendance-only mode.")
        _model = None
        return None
    try:
        model = build_model(num_classes=7)
        state = torch.load(config.EMOTION_WEIGHTS_PATH, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        return _model
    except Exception as exc:
        logger.error("Failed to load emotion model: %s", exc)
        _model = None
        return None

def predict_emotion(face_gray: np.ndarray) -> Optional[str]:
    model = _load_model_internal()
    if model is None:
        return None
    if face_gray.shape != (48, 48):
        face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    face_float = face_gray.astype(np.float32) / 255.0
    face_float = (face_float - 0.5) / 0.5
    tensor = torch.from_numpy(face_float).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred_idx = int(logits.argmax(dim=1).item())
    return config.EMOTION_LABELS[pred_idx]
