# Air Math Vision

A local real-time computer vision project that combines face verification, hand-gesture recognition, air-writing digit recognition, and gesture-controlled arithmetic in a FastAPI dashboard.

The project is designed for Windows 10/11 and runs locally with a webcam.

## Demo Capabilities

- Live people/face counting from the webcam
- Local face enrollment and verification
- MediaPipe hand tracking
- Air-writing recognition for digits `0-9`
- Gesture-controlled arithmetic operators
- Multi-digit number buffering
- Undo and token commit gestures
- Real-time expression evaluation
- Live camera stream and event dashboard
- Separate recognition worker so air-writing classification does not block the camera loop
- Drawing-grace logic to avoid ending a stroke too quickly

## Gesture Controls

| Gesture | Action |
| --- | --- |
| Thumbs up | Start / Ready |
| Index finger | Draw a digit in the air |
| Open palm | Commit the current number |
| Thumbs down | Undo |
| Pinky only | `+` |
| Pinky + ring | `-` |
| Pinky + ring + middle | `×` |
| Four fingers, thumb folded | `÷` |
| Closed fist | `=` |

Air writing is used for numbers. Arithmetic operators are entered with hand gestures.

## Tech Stack

- Python 3.11 / 3.12
- OpenCV
- OpenCV YuNet face detector
- OpenCV SFace face recognizer
- MediaPipe Hands
- NumPy
- FastAPI
- Uvicorn
- Jinja2
- Vanilla JavaScript / CSS

## Architecture

```text
air-math-vision/
├── app/
│   ├── airwrite.py       # Stroke normalization + template recognition
│   ├── config.py         # Camera, face, gesture and drawing thresholds
│   ├── expression.py     # Restricted arithmetic expression evaluator
│   ├── gesture.py        # Hand-pose classification
│   ├── main.py           # FastAPI application and local API
│   ├── models.py         # Session state / snapshot models
│   ├── session.py        # Interaction state machine
│   ├── vision.py         # Camera, face, hand and drawing pipeline
│   ├── static/
│   └── templates/
├── docs/
│   └── ARCHITECTURE.md
├── diagnose_windows.bat
├── requirements.txt
├── run.py
├── self_test.py
├── setup_windows.bat
├── start_windows.bat
└── README.md
```

## How It Works

### 1. Face stage

The application detects faces using YuNet. On first use, a face feature vector is enrolled locally. Later sessions compare the current face with the enrolled feature using SFace cosine similarity.

### 2. Gesture stage

MediaPipe provides 21 hand landmarks. `gesture.py` converts landmark geometry into application gestures such as thumbs up, open palm, operator poses, and a closed fist.

### 3. Air-writing stage

While the index-finger drawing gesture is active, fingertip points are collected into a stroke.

The recognizer:

1. resamples the stroke to a fixed number of points,
2. normalizes rotation, scale, and translation,
3. compares it with digit templates,
4. returns the closest accepted character.

Recognition runs on a separate worker thread so the camera loop remains responsive.

### 4. Expression stage

Recognized digits are combined into number tokens. Operators come from gestures. Expressions are evaluated using a restricted Python AST evaluator that accepts only numeric constants and the supported arithmetic operators.

## Windows Setup

### Requirements

- Windows 10/11 64-bit
- Python 3.11 or 3.12 64-bit
- Webcam
- Internet access during initial setup

### Install

Run:

```text
setup_windows.bat
```

The setup script:

- creates `.venv`,
- installs Python dependencies,
- verifies the MediaPipe version,
- downloads the YuNet face-detection model,
- downloads the SFace face-recognition model.

The model binaries are intentionally not committed to this repository.

### Start

Run:

```text
start_windows.bat
```

The local dashboard opens at:

```text
http://127.0.0.1:8000
```

For diagnostics:

```text
diagnose_windows.bat
```

## Self Test

After setup:

```bash
.venv\Scripts\python.exe self_test.py
```

The self-test covers template recognition, token semantics, operator gestures, and safe arithmetic evaluation.

## Drawing Configuration

Important drawing-related settings are in `app/config.py`:

```python
DRAW_IDLE_SECONDS = 3.50
DRAW_MIN_POINTS = 12
DRAW_MIN_DISTANCE_PX = 3.0
DRAW_GESTURE_GRACE_SECONDS = 0.90
```

The grace period allows brief gesture-classification loss without prematurely ending a stroke.

## Privacy and Security

This repository does not contain an enrolled face or other user biometric data.

At runtime, the application stores the enrolled face **feature vector** locally in:

```text
data/enrolled_face.npy
```

That file is excluded by `.gitignore` and should never be committed.

The application does not intentionally save webcam frames to disk.

The server binds to:

```text
127.0.0.1
```

and is intended as a **local demo application**. The API and video stream are not designed for direct public internet exposure. If this project is adapted for network deployment, authentication, authorization, transport security, rate limiting, and endpoint hardening should be added first.

## Limitations

- Air-writing recognition is template-based rather than a trained handwriting model.
- Recognition accuracy depends on drawing style, camera position, lighting, and hand tracking.
- Face verification thresholds are demo-oriented and should not be treated as a production biometric security system.
- Gesture rules are geometric heuristics and can require tuning for different users/cameras.

## Portfolio Notes

This project demonstrates:

- real-time computer vision pipelines,
- OpenCV face processing,
- MediaPipe landmark processing,
- state-machine design,
- background worker threads,
- safe expression parsing,
- local API/dashboard development,
- gesture-based human-computer interaction,
- practical debugging and recovery tooling for Windows.

## Author

Farid Farahani

Python Developer | Web Developer

Website: https://faridfarahani.ir
