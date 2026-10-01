# Architecture

## Runtime flow

```text
Webcam
  |
  v
VisionEngine
  |----> YuNet face detection ----> SFace enrollment / verification
  |
  |----> MediaPipe Hands ---------> gesture classifier
  |                                   |
  |                                   +----> application controls/operators
  |
  +----> fingertip stroke --------> recognition worker
                                      |
                                      v
                              AirWritingRecognizer
                                      |
                                      v
                               SessionController
                                      |
                                      v
                               expression evaluator
                                      |
                                      v
                              FastAPI JSON status
                                      |
                                      v
                                local dashboard
```

## Camera thread

`VisionEngine` owns webcam capture and the real-time vision loop.

Air-writing recognition is deliberately moved to a separate worker queue. Completed strokes are submitted to the recognition worker and results are consumed by the camera loop later. This prevents template matching from freezing live video.

## Session state machine

`SessionController` maintains the interaction state:

- monitoring,
- face positioning,
- face verification,
- waiting for start gesture,
- ready,
- drawing,
- token committed,
- result.

The controller also owns the current character buffer, expression tokens, result, and bounded event history.

## Safe arithmetic

`expression.py` parses arithmetic with Python's AST module but evaluates only:

- numbers,
- addition,
- subtraction,
- multiplication,
- division,
- unary plus/minus.

Arbitrary Python expressions and function calls are rejected.

## Local biometric state

The enrolled face is stored as an SFace feature vector in `data/enrolled_face.npy`.

That path is runtime-only and excluded from Git. The public repository contains no enrolled biometric template.

## Deployment boundary

The included Uvicorn configuration binds to loopback (`127.0.0.1`). The project is a local computer-vision application, not a hardened public web service.
