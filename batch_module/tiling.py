"""
batch_module/tiling.py
-----------------------
Splits a large frame into overlapping tiles for small-face detection,
and merges duplicate detections from the overlap regions via IoU NMS.
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

TileInfo = Tuple[np.ndarray, int, int]
Detection = Tuple[int, int, int, int, str, str, float, Optional[str]]

def split_into_tiles(frame: np.ndarray, tile_size: Tuple[int, int] = (640, 640), overlap: float = 0.2) -> List[TileInfo]:
    h, w = frame.shape[:2]
    tile_w, tile_h = tile_size
    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))
    tiles = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = min(x + tile_w, w)
            y_end = min(y + tile_h, h)
            tile = frame[y:y_end, x:x_end]
            if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
                padded = np.zeros((tile_h, tile_w, 3), dtype=frame.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((tile, x, y))
            if x_end == w: break
            x += step_x
        if y_end == h: break
        y += step_y
    return tiles

def _iou(box_a, box_b) -> float:
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b
    inter_top = max(a_top, b_top)
    inter_left = max(a_left, b_left)
    inter_bottom = min(a_bottom, b_bottom)
    inter_right = min(a_right, b_right)
    inter_h = max(0, inter_bottom - inter_top)
    inter_w = max(0, inter_right - inter_left)
    intersection = inter_h * inter_w
    if intersection == 0: return 0.0
    area_a = max(0, a_bottom - a_top) * max(0, a_right - a_left)
    area_b = max(0, b_bottom - b_top) * max(0, b_right - b_left)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def merge_overlapping_detections(detections: List[Detection], iou_threshold: float = 0.35) -> List[Detection]:
    if not detections: return []
    def sort_key(d): return d[6] if d[4] != "Unknown" else 10.0
    sorted_dets = sorted(detections, key=sort_key)
    kept = []
    for det in sorted_dets:
        box = (det[0], det[1], det[2], det[3])
        if not any(_iou(box, (k[0], k[1], k[2], k[3])) >= iou_threshold for k in kept):
            kept.append(det)
    return kept
