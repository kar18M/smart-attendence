"""
config.py
---------
Central configuration for the Smart Attendance System.
All tuneable constants live here — change them once, they propagate everywhere.
"""

import os

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "smart_attendance"

# Collection names
COL_STUDENTS = "students"
COL_ATTENDANCE = "attendance"
COL_ENGAGEMENT = "engagement_logs"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory where enrollment photos are saved
ENROLLED_FACES_DIR = os.path.join(BASE_DIR, "data", "enrolled_faces")

# Trained emotion model weights
EMOTION_WEIGHTS_PATH = os.path.join(
    BASE_DIR, "emotion_module", "weights", "emotion_cnn.pth"
)

# FER2013 CSV (user must download from Kaggle — see README)
FER2013_CSV_PATH = os.path.join(BASE_DIR, "data", "fer2013.csv")

# OpenCV Haar cascade (bundled in data/ — used as fallback when dlib unavailable)
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------
# Euclidean distance threshold below which a face is considered a match.
# Lower = stricter matching.  face_recognition default is 0.6.
RECOGNITION_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------
# Only run detection on every Nth frame (1 = every frame, 2 = every other, etc.)
FRAME_SKIP_RATE = 2

# ---------------------------------------------------------------------------
# Engagement Logging
# ---------------------------------------------------------------------------
# Minimum seconds between consecutive engagement log entries per student
ENGAGEMENT_COOLDOWN_SECONDS = 8

# ---------------------------------------------------------------------------
# Emotion Classes (must match training label order)
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
APP_TITLE = "Smart Attendance System"
APP_ICON = "🎓"
