# Geuse — AI-Based Hand Rehabilitation Monitoring System

Geuse is a desktop application that uses computer vision and machine learning to monitor hand rehabilitation exercises in real time using a standard webcam. Built as a final year project for BSc (Hons) Data Science at Plymouth University.

---

## What it does

- Detects and classifies hand postures (open palm, mid flexion, full fist, thumb-index pinch) using a webcam and MediaPipe hand landmark detection
- Measures a continuous closure value (0.0 to 1.0) representing hand closure progress between fully open and fully closed
- Guides users through a one-time onboarding assessment to generate a personalised rehabilitation plan
- Tracks real-time exercise performance during sessions — rep counting and hold duration monitoring
- Logs session history and tracks progress over time
- Generates a rehabilitation plan based on clinically informed rules (ROM assessment, pain-directed exercise modification, condition-specific session frequency)

---

## Who it is for

Patients recovering from hand-related injuries or conditions such as stroke, arthritis, post-fracture rehabilitation, or tendon repair who need a low-cost, accessible way to monitor home-based hand exercises without specialist hardware.

---

## Tech stack

- **Python** — core application logic
- **PyWebView** — native desktop window with HTML/CSS/JS frontend
- **MediaPipe** — real-time hand landmark detection
- **PyTorch** — neural network model for gesture classification and closure regression
- **OpenCV** — webcam capture and frame processing
- **SQLite** — local database for user profile, sessions, and progress
- **HTML / CSS / JS** — frontend UI (no framework, vanilla)

---

## Running from source

**Requirements:** Python 3.10+, Windows
```bash
git clone https://github.com/sithuu99/geuse
cd geuse

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

cd geuse
python main.py
```

Make sure `geuse_multitask.pt` is placed in `geuse/assets/models/` before running.

---

## Running the prebuilt exe

Download `Geuse-v1.0.0-windows.zip` from the [Releases](https://github.com/sithuu99/geuse/releases) page, extract the folder, and run `Geuse.exe`. No Python installation required.

---

## Project structure
```
geuse/                  # Desktop app
  app/                  # Python backend
    api.py              # PyWebView JS bridge
    camera.py           # Webcam capture + MediaPipe
    model.py            # Gesture classification + closure inference
    database.py         # SQLite helpers
    plan.py             # Rehabilitation plan generation
  ui/                   # Frontend
    pages/              # One HTML file per screen
    styles/             # Global CSS design system
    js/                 # Bridge utilities
  assets/
    models/             # Trained PyTorch model (.pt)

ml/                     # Machine learning (training only)
  scripts/              # Dataset building + model training
  models/               # Raw model checkpoints
  data/                 # Landmark datasets
```

---

## Model

The gesture recognition model (`GeuseMultiTask`) is a multitask neural network trained on hand landmark features extracted via MediaPipe. It outputs:

- A **classification head** — predicts one of 5 hand states: neutral, palm, grabbing, fist, thumb\_index
- A **regression head** — predicts a continuous closure value (0.0 to 1.0)

Training scripts and datasets are in the `ml/` folder.

---

## Limitations

- The system is not a medical device and does not provide clinical diagnosis or treatment recommendations
- The rehabilitation plan generator is a rule-based prototype and has not been clinically validated
- Currently supports single-hand tracking only
- Tested on Windows only
- Requires adequate lighting and a clear background for reliable hand detection

---

## Author

Hakuru Gunarathne — Plymouth Index Number 10953745
BSc (Hons) Data Science, Plymouth University
Supervisor: Ms. Lakni Peiris