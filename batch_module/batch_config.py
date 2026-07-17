"""
batch_module/batch_config.py
-----------------------------
All tunable constants for Batch Mode attendance.
"""

CAPTURE_INTERVAL_SECONDS: int = 30
CAMERA_INDEX: int = 0
CAPTURE_WIDTH:  int = 1920
CAPTURE_HEIGHT: int = 1080

TILE_SIZE: tuple = (640, 640)
TILE_OVERLAP: float = 0.2
UPSCALE_FACTOR: int = 2

MIN_FACE_SIZE_PX: int = 40
BATCH_RECOGNITION_THRESHOLD: float = 0.55
DEDUP_IOU_THRESHOLD: float = 0.35

BATCH_ENGAGEMENT_COOLDOWN_SECONDS: int = 60
