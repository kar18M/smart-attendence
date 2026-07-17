# Smart Attendance System with Emotion & Engagement Tracking

A locally-run, webcam-based attendance system that recognises enrolled students
via face embeddings, classifies their emotion using a small CNN, logs attendance
and engagement snapshots to MongoDB, and presents everything in a Streamlit
multi-page dashboard.

---

## Features

- 🎥 **Live face recognition** via webcam (streamlit-webrtc, continuous feed)
- 😊 **Emotion classification** (7 classes) using a PyTorch CNN trained on FER2013
- 📋 **Automatic attendance marking** — once per student per day
- 📊 **Engagement analytics** — Engaged / Neutral / Disengaged with Plotly charts
- 🗄️ **MongoDB** storage — local, no cloud required
- 🚧 No GPU required — CPU-friendly throughout
- 📸 **Batch Mode** — tiled processing for large classrooms (30-60+ students)

---

## Prerequisites

### 1. MongoDB

**Install MongoDB Community Edition (Ubuntu/Debian):**
```bash
sudo apt install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
```

**Start MongoDB:**
```bash
sudo systemctl start mongod
sudo systemctl enable mongod   # auto-start on boot
```

Verify:
```bash
mongosh --eval "db.runCommand({ ping: 1 })"
```

### 2. System Build Dependencies

```bash
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev
```

---

## Python Environment Setup

```bash
cd smart-attendance/
python3 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

---

## Emotion Model Training (Optional)

### Step 1 — Download FER2013

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download `fer2013.csv` and place it at: `data/fer2013.csv`

### Step 2 — Train the CNN

```bash
python -m emotion_module.train
```

Training takes ~10-20 minutes on CPU for 25 epochs.
The best checkpoint is saved to `emotion_module/weights/emotion_cnn.pth`.

> **Without training:** The app still works in attendance-only mode.

---

## Running the Application

```bash
streamlit run app.py
```

Open: **http://localhost:8501**

---

## Workflow

1. **Enroll Students** — go to **➕ Enroll Student**, upload clear front-facing photos
2. **Live Attendance** — go to **🎥 Live Attendance**, click START
3. **View Records** — **📋 Attendance Log** for daily records, **📊 Analytics** for engagement
4. **Batch Mode** — **📸 Batch Attendance** for large classrooms with many students

---

## Project Structure

```
smart-attendance/
├── app.py                          # Landing page & MongoDB health check
├── config.py                       # All tunable constants
├── requirements.txt
├── video_processor.py              # WebRTC VideoProcessor (core pipeline)
├── database/
│   ├── mongo_client.py             # Connection singleton + index creation
│   └── db_operations.py            # All CRUD functions
├── face_recognition_module/
│   ├── enroll.py                   # Enrollment: detect, encode, save, store
│   ├── recognizer.py               # Frame recognition + annotation drawing
│   └── encodings_store.py          # In-memory encoding cache
├── emotion_module/
│   ├── model.py                    # PyTorch CNN definition
│   ├── train.py                    # FER2013 training script
│   └── predict.py                  # Inference wrapper
├── engagement/
│   └── scorer.py                   # Emotion -> Engaged/Neutral/Disengaged
├── batch_module/
│   ├── batch_processor.py          # Full batch pipeline
│   ├── tiling.py                   # Frame tiling + IoU dedup
│   ├── batch_capture.py            # Webcam snapshot capture
│   └── batch_config.py             # Batch-mode constants
└── pages/
    ├── 1_Live_Attendance.py
    ├── 2_Attendance_Log.py
    ├── 3_Analytics.py
    ├── 4_Enroll_Student.py
    └── 5_Batch_Attendance.py
```

---

## MongoDB Collections

| Collection | Purpose |
|------------|---------|
| `students` | Enrolled students with face encodings |
| `attendance` | Daily attendance records (one per student per day) |
| `engagement_logs` | Per-student emotion snapshots (every ~8 seconds) |

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| "MongoDB not running" banner | `sudo systemctl start mongod` |
| Webcam not starting | Check browser permissions |
| `dlib` build fails | `sudo apt install cmake build-essential` then retry |
| No emotion labels | Train: `python -m emotion_module.train` |
| "No students enrolled" | Enroll via ➕ Enroll Student first |
| Recognition slow | Increase `FRAME_SKIP_RATE` in `config.py` |

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `RECOGNITION_THRESHOLD` | `0.6` | Lower = stricter face matching |
| `FRAME_SKIP_RATE` | `2` | Process every Nth frame |
| `ENGAGEMENT_COOLDOWN_SECONDS` | `8` | Min seconds between log entries per student |
