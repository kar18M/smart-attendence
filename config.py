"""
config.py
---------
Central configuration for the Smart Attendance System.
All tuneable constants live here — change them once, they propagate everywhere.
"""

import os

# MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "smart_attendance"
COL_STUDENTS = "students"
COL_ATTENDANCE = "attendance"
COL_ENGAGEMENT = "engagement_logs"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLLED_FACES_DIR = os.path.join(BASE_DIR, "data", "enrolled_faces")
EMOTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "emotion_module", "weights", "emotion_cnn.pth")
FER2013_CSV_PATH = os.path.join(BASE_DIR, "data", "fer2013.csv")
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "data", "haarcascade_frontalface_default.xml")

# Face Recognition
RECOGNITION_THRESHOLD = 0.6

# Video Processing
FRAME_SKIP_RATE = 2

# Engagement Logging
ENGAGEMENT_COOLDOWN_SECONDS = 8

# Emotion Classes
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Misc
APP_TITLE = "Smart Attendance System"
APP_ICON = "🎓"
