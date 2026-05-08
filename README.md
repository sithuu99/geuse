# Geuse — AI-Based Hand Rehabilitation Monitoring System

Geuse is a desktop application that uses computer vision and machine learning 
to monitor hand rehabilitation exercises in real time using a standard webcam. 
Built as a final year project for BSc (Hons) Data Science at Plymouth University.

---

## What it does

- Detects and classifies hand postures across a five-class gesture space using 
  a webcam and MediaPipe hand landmark detection
- Measures a continuous closure value (0.0 to 1.0) representing hand closure 
  progress between fully open and fully closed
- Guides users through a one-time onboarding assessment to generate a 
  personalised rehabilitation plan
- Tracks real-time exercise performance during sessions — rep counting and hold 
  duration monitoring with live closure feedback
- Adapts the rehabilitation plan over time based on session performance
- Logs session history, tracks progress, and flags inconsistencies between 
  reported pain and performance trends
- Generates a rehabilitation plan based on clinically informed rules (ROM 
  assessment, pain-directed exercise modification, condition-specific session 
  frequency)

---

## Who it is for

Patients recovering from hand-related injuries or conditions such as stroke, 
arthritis, post-fracture rehabilitation, or tendon repair who need a low-cost, 
accessible way to monitor home-based hand exercises without specialist hardware.

---

## Tech stack

- **Python** — core application logic
- **PyWebView** — native desktop window with HTML/CSS/JS frontend
- **MediaPipe** — real-time hand landmark detection
- **PyTorch** — multitask neural network for gesture classification and closure 
  regression
- **OpenCV** — webcam capture and frame processing
- **SQLite** — local database for user profile, sessions, and progress
- **HTML / CSS / JS** — frontend UI (vanilla, no framework)

---

## Model

The gesture recognition model (`GeuseMultiTask`) is a multitask neural network 
trained on hand landmark features extracted via MediaPipe. It outputs:

- A **classification head** — predicts one of 5 hand states: neutral, palm, 
  grabbing, fist, thumb_index
- A **regression head** — predicts a continuous closure value (0.0 to 1.0) 
  using sigmoid activation

**Evaluation results on held-out test set (710 samples, 15% of dataset):**

| Metric | Value |
|---|---|
| Overall accuracy | 93.66% |
| Closure MAE | 0.0615 |
| Closure RMSE | 0.0833 |

Per-class F1 scores: neutral 0.997, palm 0.939, grabbing 0.899, 
fist 0.924, thumb-index 0.921.

---

## Dataset

Gesture training data was sourced and adapted from the 
[HaGRID dataset](https://github.com/hukenovs/hagrid) 
(Hand Gesture Recognition Image Dataset) by Alexander Kapitanov et al. 
HaGRID is a large-scale dataset of 552,992 images across 18 gesture classes 
captured under diverse real-world conditions.

A subset of HaGRID images relevant to hand rehabilitation gestures was selected 
and processed through MediaPipe to extract 21 hand landmark coordinates per 
sample. These landmarks were normalised and used as input features for training 
the GeuseMultiTask model. The final dataset contains 4,731 samples across 
5 classes: neutral, palm, grabbing, fist, and thumb_index.

Training scripts and the processed landmark dataset are in the `ml/` folder.

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


Make sure `geuse_multitask.pt` is placed in `geuse/assets/models/` 
before running.

---
```
## Running the prebuilt exe

Download `Geuse-v1.1.0-windows.zip` from the 
[Releases](https://github.com/sithuu99/geuse/releases) page, extract 
the folder, and run `Geuse.exe`. No Python installation required.

**Prerequisite:** .NET 8 Desktop Runtime (x64) must be installed.
Download from: https://dotnet.microsoft.com/en-us/download/dotnet/8.0 
— select ".NET Desktop Runtime 8.0" column, x64 installer.

---

## Project structure


geuse/                    # Desktop app
  app/                    # Python backend
    api.py                # PyWebView JS bridge
    camera.py             # Webcam capture + MediaPipe
    model.py              # Gesture classification + closure inference
    database.py           # SQLite helpers
    plan.py               # Rehabilitation plan generation + progression
  ui/                     # Frontend
    pages/                # One HTML file per screen
    styles/               # Global CSS design system
    js/                   # Bridge utilities
  assets/
    models/               # Trained PyTorch model (.pt)

ml/                       # Machine learning (training only)
  scripts/                # Dataset building, model training, evaluation
  models/                 # Raw model checkpoints
  data/                   # Processed landmark dataset


---

## Limitations

- The system is not a medical device and does not provide clinical diagnosis 
  or treatment recommendations
- The rehabilitation plan generator is a rule-based prototype and has not 
  been clinically validated
- Currently supports single-hand tracking only
- Tested on Windows only
- Requires adequate lighting and a clear background for reliable hand detection
- Patients unable to perform basic hand movements due to severe pain are 
  advised to consult a physiotherapist before using this app

---

## Author

Hakuru Gunarathne — Plymouth Index Number 10953745
BSc (Hons) Data Science, Plymouth University
Supervisor: Ms. Lakni Peiris
