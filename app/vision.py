import threading
import time
import queue
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .airwrite import AirWritingRecognizer
from .config import (
    CAMERA_INDEX,
    DRAW_IDLE_SECONDS,
    DRAW_GESTURE_GRACE_SECONDS,
    DRAW_MIN_DISTANCE_PX,
    DRAW_MIN_POINTS,
    ENROLLED_FACE_FILE,
    FACE_DETECT_MODEL,
    FACE_GUIDE_H_RATIO,
    FACE_GUIDE_W_RATIO,
    FACE_GUIDE_X_RATIO,
    FACE_GUIDE_Y_RATIO,
    FACE_NMS_THRESHOLD,
    FACE_RECOGNITION_MODEL,
    FACE_SCORE_THRESHOLD,
    FACE_STABLE_SECONDS,
    FACE_TOP_K,
    FACE_VERIFY_COSINE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from .gesture import classify_hand
from .models import SessionState


class VisionEngine:
    def __init__(self, session):
        self.session = session
        self.running = False
        self.thread = None
        self.cap = None

        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()

        self.face_detector = cv2.FaceDetectorYN.create(
            str(FACE_DETECT_MODEL), "", (FRAME_WIDTH, FRAME_HEIGHT),
            FACE_SCORE_THRESHOLD, FACE_NMS_THRESHOLD, FACE_TOP_K
        )
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            str(FACE_RECOGNITION_MODEL), ""
        )

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        )

        self.air_recognizer = AirWritingRecognizer()

        # Air-writing recognition MUST NOT run on the camera thread.
        # The camera only submits completed strokes and consumes results.
        self.recognition_jobs = queue.Queue(maxsize=1)
        self.recognition_results = queue.Queue(maxsize=4)
        self.recognition_thread = None
        self.recognition_busy = False
        self.recognition_epoch = 0
        self.recognition_lock = threading.Lock()

        self.stroke = []
        self.last_stroke_move = 0.0
        self.last_point = None
        self.last_pointing_seen = 0.0

        self.face_inside_since = None
        self.last_face_verify_attempt = 0.0

        self.enrolled_feature = self._load_enrolled_face()
        self.session.set_identity_enrolled(self.enrolled_feature is not None)

    def _load_enrolled_face(self):
        if ENROLLED_FACE_FILE.exists():
            try:
                arr = np.load(str(ENROLLED_FACE_FILE))
                if arr.size:
                    return arr.astype(np.float32)
            except Exception:
                return None
        return None

    def reset_identity(self):
        self.enrolled_feature = None
        if ENROLLED_FACE_FILE.exists():
            ENROLLED_FACE_FILE.unlink()
        self.session.set_identity_enrolled(False)
        self.session.snapshot.identity_verified = False
        self.session.snapshot.face_similarity = None
        self.session.add_event("Enrolled face removed")

    def start(self):
        if self.running:
            return
        self.running = True
        self.recognition_thread = threading.Thread(
            target=self._recognition_worker,
            daemon=True,
            name="air-writing-recognizer",
        )
        self.recognition_thread.start()

        self.thread = threading.Thread(target=self._loop, daemon=True, name="camera-loop")
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            self.recognition_jobs.put_nowait(None)
        except queue.Full:
            pass

        if self.thread:
            self.thread.join(timeout=2.0)
        if self.recognition_thread:
            self.recognition_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.hands.close()

    def get_jpeg(self):
        with self.jpeg_lock:
            return self.latest_jpeg

    def cancel_pending_input(self):
        """Clear current drawing and invalidate any in-flight recognition."""
        with self.recognition_lock:
            self.recognition_epoch += 1

        self._reset_current_stroke()

        while True:
            try:
                self.recognition_jobs.get_nowait()
                self.recognition_jobs.task_done()
            except queue.Empty:
                break

        while True:
            try:
                self.recognition_results.get_nowait()
                self.recognition_results.task_done()
            except queue.Empty:
                break

        self.recognition_busy = False

    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        self.cap = self._open_camera()

        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.session.set_camera_ok(False)
                time.sleep(0.1)
                continue

            self.session.set_camera_ok(True)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
                self.face_detector.setInputSize((w, h))

            faces = self._detect_faces(frame)
            self.session.set_people_count(len(faces))

            self._draw_faces(frame, faces)
            self._process_identity(frame, faces)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = self.hands.process(rgb)

            gesture_name = None
            primary_landmarks = None

            if hand_result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
                    g = classify_hand(hand_landmarks)
                    if primary_landmarks is None or g.name in {
                        "THUMBS_UP",
                        "THUMBS_DOWN",
                        "OPEN_PALM",
                        "POINTING",
                        "OP_ADD",
                        "OP_SUBTRACT",
                        "OP_MULTIPLY",
                        "OP_DIVIDE",
                        "OP_EQUALS",
                    }:
                        primary_landmarks = hand_landmarks
                        gesture_name = g.name

            self.session.set_gesture(gesture_name)
            self._consume_recognition_results()

            if self.session.state not in {
                SessionState.MONITORING,
                SessionState.FACE_POSITIONING,
                SessionState.VERIFYING_FACE,
                SessionState.WAITING_FOR_THUMBS_UP,
            }:
                self._process_drawing(frame, primary_landmarks, gesture_name)
            else:
                self._clear_stroke_if_needed()

            session_gestures = {
                "THUMBS_UP",
                "THUMBS_DOWN",
                "OPEN_PALM",
                "OP_ADD",
                "OP_SUBTRACT",
                "OP_MULTIPLY",
                "OP_DIVIDE",
                "OP_EQUALS",
            }

            # Do not execute a control/operator gesture while a written digit
            # is still waiting to finish or be recognized.
            if (
                gesture_name in session_gestures
                and not self.stroke
                and not self.recognition_busy
            ):
                self.session.process_gesture(gesture_name)
            else:
                self.session.process_gesture(None)

            self._draw_guide(frame)
            self._draw_stroke(frame)
            self._draw_hud(frame)

            ok_jpg, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            if ok_jpg:
                with self.jpeg_lock:
                    self.latest_jpeg = jpg.tobytes()

        self.session.set_camera_ok(False)

    def _detect_faces(self, frame):
        _retval, detected = self.face_detector.detect(frame)
        if detected is None:
            return []
        return [row for row in detected]

    def _draw_faces(self, frame, faces):
        for i, face in enumerate(faces, start=1):
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 210, 110), 2)
            cv2.putText(
                frame, f"Face {i}", (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 210, 110), 2, cv2.LINE_AA
            )

    def _guide_rect(self, frame):
        h, w = frame.shape[:2]
        gw = int(w * FACE_GUIDE_W_RATIO)
        gh = int(h * FACE_GUIDE_H_RATIO)
        cx = int(w * FACE_GUIDE_X_RATIO)
        cy = int(h * FACE_GUIDE_Y_RATIO)
        return cx - gw // 2, cy - gh // 2, gw, gh

    def _face_in_guide(self, frame, faces):
        if not faces:
            return None

        gx, gy, gw, gh = self._guide_rect(frame)
        gcx, gcy = gx + gw / 2, gy + gh / 2

        best = None
        best_d = float("inf")

        for face in faces:
            x, y, w, h = face[:4]
            cx, cy = x + w / 2, y + h / 2

            center_inside = gx <= cx <= gx + gw and gy <= cy <= gy + gh
            size_ok = 0.34 * gw <= w <= 1.25 * gw and 0.34 * gh <= h <= 1.25 * gh

            if not (center_inside and size_ok):
                continue

            d = (cx - gcx) ** 2 + (cy - gcy) ** 2
            if d < best_d:
                best_d = d
                best = face

        return best

    def _feature_for_face(self, frame, face):
        try:
            aligned = self.face_recognizer.alignCrop(frame, face.astype(np.float32))
            feature = self.face_recognizer.feature(aligned)
            return feature.astype(np.float32)
        except Exception:
            return None

    def _process_identity(self, frame, faces):
        if self.session.state != SessionState.FACE_POSITIONING:
            self.face_inside_since = None
            return

        selected = self._face_in_guide(frame, faces)
        now = time.monotonic()

        if selected is None:
            self.face_inside_since = None
            return

        if self.face_inside_since is None:
            self.face_inside_since = now
            return

        if now - self.face_inside_since < FACE_STABLE_SECONDS:
            return

        if now - self.last_face_verify_attempt < 1.25:
            return

        self.last_face_verify_attempt = now
        feature = self._feature_for_face(frame, selected)
        if feature is None:
            return

        if self.enrolled_feature is None:
            ENROLLED_FACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(ENROLLED_FACE_FILE), feature)
            self.enrolled_feature = feature
            self.session.identity_ok(similarity=1.0, enrolled_now=True)
            return

        similarity = float(
            self.face_recognizer.match(
                self.enrolled_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

        if similarity >= FACE_VERIFY_COSINE_THRESHOLD:
            self.session.identity_ok(similarity=similarity, enrolled_now=False)
        else:
            self.session.identity_failed(similarity)
            self.face_inside_since = now

    def _process_drawing(self, frame, hand_landmarks, gesture_name):
        now = time.monotonic()
        # IMPORTANT:
        # Keep the exact DrawingGrace behaviour while a number is being drawn.
        # Operator gestures are intentionally NOT treated as control gestures
        # here, because MediaPipe may transiently classify a writing hand as
        # OP_ADD / OP_SUBTRACT / OP_MULTIPLY / OP_DIVIDE / OP_EQUALS.
        # Treating those classifications as controls was prematurely stopping
        # fingertip tracking and made the stroke feel "too fast".
        control_gestures = {"THUMBS_UP", "THUMBS_DOWN", "OPEN_PALM"}

        # Air Writing is digits-only. Once a number token is committed, the
        # system waits for an operator hand gesture before another number.
        # This restriction applies only BEFORE a stroke starts.
        if not self.stroke and self.session.expected_character_type() != "NUMBER":
            return

        # POINTING is required only to START a new stroke.
        # Once a stroke has started, transient MediaPipe gesture instability
        # (including accidental operator classifications) must not stop
        # tracking the fingertip.
        can_start = (
            hand_landmarks is not None
            and gesture_name == "POINTING"
            and not self.stroke
        )

        operator_gestures = {
            "OP_ADD",
            "OP_SUBTRACT",
            "OP_MULTIPLY",
            "OP_DIVIDE",
            "OP_EQUALS",
        }

        can_continue = (
            hand_landmarks is not None
            and bool(self.stroke)
            and gesture_name not in control_gestures
            and (
                gesture_name == "POINTING"
                or gesture_name in operator_gestures
                or (now - self.last_pointing_seen) <= DRAW_GESTURE_GRACE_SECONDS
                or gesture_name is None
            )
        )

        if gesture_name == "POINTING":
            self.last_pointing_seen = now

        if can_start or can_continue:
            h, w = frame.shape[:2]
            tip = hand_landmarks.landmark[8]
            p = np.array([tip.x * w, tip.y * h], dtype=np.float32)

            if self.last_point is None:
                self.stroke = [tuple(p)]
                self.last_point = p
                self.last_stroke_move = now
                self.last_pointing_seen = now
                self.session.drawing_started()
            else:
                d = float(np.linalg.norm(p - self.last_point))

                # A smaller movement threshold allows slow, deliberate writing.
                if d >= DRAW_MIN_DISTANCE_PX:
                    self.stroke.append(tuple(p))
                    self.last_point = p
                    self.last_stroke_move = now
                    self.session.drawing_started()

        # Character completion is based on REAL fingertip inactivity.
        # DRAW_IDLE_SECONDS is intentionally generous so the user can draw
        # slowly without the recognizer stealing the stroke mid-character.
        if self.stroke and (now - self.last_stroke_move) >= DRAW_IDLE_SECONDS:
            completed_stroke = list(self.stroke)
            self._reset_current_stroke()

            if len(completed_stroke) < DRAW_MIN_POINTS:
                self.session.add_event("Stroke cleared -> too short")
                return

            expected = "NUMBER"
            with self.recognition_lock:
                epoch = self.recognition_epoch

            try:
                self.recognition_jobs.put_nowait((epoch, completed_stroke, expected))
                self.recognition_busy = True
                self.session.add_event("Stroke captured -> analyzing")
            except queue.Full:
                self.session.add_event("Stroke cleared -> recognizer busy")

    def _recognition_worker(self):
        while self.running:
            try:
                job = self.recognition_jobs.get(timeout=0.25)
            except queue.Empty:
                continue

            if job is None:
                break

            epoch, points, expected = job
            started = time.monotonic()

            try:
                char, score = self.air_recognizer.predict(points, expected=expected)
                elapsed = time.monotonic() - started
                self.recognition_results.put(
                    ("ok", epoch, char, float(score), elapsed),
                    timeout=0.25,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                try:
                    self.recognition_results.put(
                        ("error", epoch, None, 0.0, elapsed, str(exc)),
                        timeout=0.25,
                    )
                except queue.Full:
                    pass
            finally:
                self.recognition_jobs.task_done()

    def _consume_recognition_results(self):
        while True:
            try:
                result = self.recognition_results.get_nowait()
            except queue.Empty:
                break

            self.recognition_busy = False

            with self.recognition_lock:
                current_epoch = self.recognition_epoch

            if result[0] == "ok":
                _, result_epoch, char, score, elapsed = result

                # Ignore an old result after Clear / Stop / Start.
                if result_epoch != current_epoch:
                    self.recognition_results.task_done()
                    continue

                if char:
                    self.session.add_character(char, confidence=score)
                else:
                    self.session.add_event(
                        f"Stroke not recognized -> cleared ({score:.2f}, {elapsed:.2f}s)"
                    )
            else:
                _, result_epoch, _, _, elapsed, error = result

                if result_epoch != current_epoch:
                    self.recognition_results.task_done()
                    continue

                self.session.add_event(
                    f"Stroke analysis failed -> cleared ({elapsed:.2f}s)"
                )

            self.recognition_results.task_done()

    def _reset_current_stroke(self):
        self.stroke = []
        self.last_point = None
        self.last_stroke_move = 0.0
        self.last_pointing_seen = 0.0

    def _clear_stroke_if_needed(self):
        if self.session.state in {
            SessionState.MONITORING,
            SessionState.FACE_POSITIONING,
            SessionState.VERIFYING_FACE,
            SessionState.WAITING_FOR_THUMBS_UP,
        }:
            self._reset_current_stroke()

    def _draw_stroke(self, frame):
        if len(self.stroke) < 2:
            return
        pts = np.array(self.stroke, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 220, 255), 5, cv2.LINE_AA)

    def _draw_guide(self, frame):
        if self.session.state != SessionState.FACE_POSITIONING:
            return

        x, y, w, h = self._guide_rect(frame)
        color = (0, 220, 130)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            frame, "Place face inside guide",
            (x, max(28, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA
        )

    def _draw_hud(self, frame):
        state = self.session.state.value
        count = self.session.snapshot.people_count
        gesture = self.session.snapshot.gesture or "-"
        buffer_text = self.session.snapshot.buffer or "-"
        tokens = " ".join(self.session.snapshot.tokens) or "-"

        if self.stroke and self.last_stroke_move > 0:
            idle_elapsed = max(0.0, time.monotonic() - self.last_stroke_move)
            remaining = max(0.0, DRAW_IDLE_SECONDS - idle_elapsed)
            stroke_status = f"DRAWING - closes in {remaining:.1f}s"
        elif self.recognition_busy:
            stroke_status = "ANALYZING"
        else:
            stroke_status = "IDLE"

        lines = [
            f"People: {count}",
            f"State: {state}",
            f"Gesture: {gesture}",
            f"Stroke: {stroke_status}",
            f"Buffer: {buffer_text}",
            f"Tokens: {tokens}",
        ]

        y = 28
        for text in lines:
            cv2.putText(
                frame, text, (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
            )
            y += 28
