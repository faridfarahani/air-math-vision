import time
from collections import deque

from .config import EVENT_LOG_LIMIT, GESTURE_COOLDOWN_SECONDS, GESTURE_HOLD_SECONDS
from .expression import evaluate_tokens
from .models import SessionState, VisionSnapshot


class SessionController:
    def __init__(self):
        self.state = SessionState.MONITORING
        self.snapshot = VisionSnapshot()
        self.events = deque(maxlen=EVENT_LOG_LIMIT)

        self.tokens = []
        self.buffer_chars = []
        self.result = None

        self._gesture_candidate = None
        self._gesture_since = 0.0
        self._gesture_last_fire = 0.0

        self.add_event("System started")
        self._sync()

    def start(self):
        self.tokens = []
        self.buffer_chars = []
        self.result = None
        self.snapshot.identity_verified = False
        self.snapshot.face_similarity = None
        self.state = SessionState.FACE_POSITIONING
        self.add_event("Face positioning started")
        self._sync()

    def stop(self):
        self.state = SessionState.MONITORING
        self.tokens = []
        self.buffer_chars = []
        self.result = None
        self.snapshot.identity_verified = False
        self.add_event("Interaction stopped")
        self._sync()

    def clear(self):
        self.tokens = []
        self.buffer_chars = []
        self.result = None
        if self.state not in {SessionState.MONITORING, SessionState.FACE_POSITIONING}:
            self.state = SessionState.READY
        self.add_event("Expression cleared")
        self._sync()

    def set_identity_enrolled(self, value):
        self.snapshot.identity_enrolled = bool(value)

    def identity_ok(self, similarity=None, enrolled_now=False):
        self.snapshot.identity_enrolled = True
        self.snapshot.identity_verified = True
        self.snapshot.face_similarity = similarity
        self.state = SessionState.WAITING_FOR_THUMBS_UP
        if enrolled_now:
            self.add_event("Face enrolled successfully")
        else:
            self.add_event(f"Face verified ({similarity:.3f})" if similarity is not None else "Face verified")
        self._sync()

    def identity_failed(self, similarity=None):
        self.snapshot.identity_verified = False
        self.snapshot.face_similarity = similarity
        if similarity is not None:
            self.add_event(f"Face verification failed ({similarity:.3f})")
        self._sync()

    def set_people_count(self, count):
        self.snapshot.people_count = int(count)

    def set_camera_ok(self, ok):
        self.snapshot.camera_ok = bool(ok)

    def set_gesture(self, gesture):
        self.snapshot.gesture = gesture

    def process_gesture(self, gesture):
        now = time.monotonic()

        if gesture is None:
            self._gesture_candidate = None
            self._gesture_since = 0.0
            return

        if gesture != self._gesture_candidate:
            self._gesture_candidate = gesture
            self._gesture_since = now
            return

        if now - self._gesture_since < GESTURE_HOLD_SECONDS:
            return
        if now - self._gesture_last_fire < GESTURE_COOLDOWN_SECONDS:
            return

        self._gesture_last_fire = now
        self._gesture_since = now

        if gesture == "THUMBS_UP" and self.state == SessionState.WAITING_FOR_THUMBS_UP:
            self.state = SessionState.READY
            self.add_event("THUMBS UP -> READY")

        elif gesture == "OPEN_PALM" and self.state in {
            SessionState.READY,
            SessionState.DRAWING,
            SessionState.TOKEN_COMMITTED,
            SessionState.RESULT,
        }:
            self.commit_token()

        elif gesture == "THUMBS_DOWN" and self.state not in {
            SessionState.MONITORING,
            SessionState.FACE_POSITIONING,
            SessionState.VERIFYING_FACE,
        }:
            self.undo()

        elif gesture == "OP_ADD" and self._operator_gesture_allowed():
            self.apply_operator("+")

        elif gesture == "OP_SUBTRACT" and self._operator_gesture_allowed():
            self.apply_operator("−")

        elif gesture == "OP_MULTIPLY" and self._operator_gesture_allowed():
            self.apply_operator("×")

        elif gesture == "OP_DIVIDE" and self._operator_gesture_allowed():
            self.apply_operator("÷")

        elif gesture == "OP_EQUALS" and self._operator_gesture_allowed():
            self.apply_operator("=")

        self._sync()

    def _operator_gesture_allowed(self):
        return self.state not in {
            SessionState.MONITORING,
            SessionState.FACE_POSITIONING,
            SessionState.VERIFYING_FACE,
            SessionState.WAITING_FOR_THUMBS_UP,
        }

    def apply_operator(self, operator):
        """
        Add an operator directly from a hand gesture.

        If a number is still in the character buffer, commit the whole number
        first. This preserves token semantics: 156 remains one token.
        """
        operators = {"+", "−", "×", "÷"}

        if self.buffer_chars:
            if all(c.isdigit() for c in self.buffer_chars):
                self.commit_token()
            else:
                self.add_event(f"Operator {operator} ignored -> invalid buffer")
                return

        if operator == "=":
            if not self.tokens:
                self.add_event("= ignored -> expression is empty")
                return

            if self.tokens[-1] in operators or self.tokens[-1] == "=":
                self.add_event("= ignored -> expression is incomplete")
                return

            self.tokens.append("=")
            self.state = SessionState.TOKEN_COMMITTED
            self.add_event("Gesture operator -> =")
            self.calculate()
            return

        if not self.tokens:
            self.add_event(f"Operator {operator} ignored -> enter a number first")
            return

        if self.tokens[-1] in operators:
            self.add_event(f"Operator {operator} ignored -> operator already selected")
            return

        if self.tokens[-1] == "=":
            self.add_event(f"Operator {operator} ignored -> clear or start a new expression")
            return

        self.result = None
        self.tokens.append(operator)
        self.state = SessionState.TOKEN_COMMITTED
        self.add_event(f"Gesture operator -> {operator}")
        self._sync()

    def expected_character_type(self):
        if not self.tokens:
            return "NUMBER"

        if self.buffer_chars:
            if all(c.isdigit() for c in self.buffer_chars):
                return "NUMBER"
            return "OPERATOR"

        last = self.tokens[-1]
        if last == "=":
            return "NUMBER"
        if last in {"+", "−", "-", "×", "*", "÷", "/"}:
            return "NUMBER"
        if last.replace(".", "", 1).isdigit():
            return "OPERATOR"

        return "ANY"

    def drawing_started(self):
        if self.state in {
            SessionState.READY,
            SessionState.TOKEN_COMMITTED,
            SessionState.RESULT,
        }:
            self.state = SessionState.DRAWING
            self._sync()

    def add_character(self, char, confidence=None):
        if not char:
            return

        operators = {"+", "−", "×", "÷", "="}

        # Prevent impossible mixed tokens.
        if char in operators and self.buffer_chars:
            if all(c.isdigit() for c in self.buffer_chars):
                self.commit_token()

        if char.isdigit() and self.buffer_chars and any(c in operators for c in self.buffer_chars):
            self.commit_token()

        self.result = None
        self.buffer_chars.append(char)
        self.state = SessionState.READY
        suffix = f" ({confidence:.2f})" if confidence is not None else ""
        self.add_event(f"Character -> {char}{suffix}")
        self._sync(current_character=char)

    def commit_token(self):
        if not self.buffer_chars:
            self.add_event("OPEN PALM -> nothing to commit")
            return

        token = "".join(self.buffer_chars)
        operators = {"+", "−", "×", "÷", "="}

        if token in operators:
            self.tokens.append(token)
        elif token.isdigit():
            self.tokens.append(token)
        else:
            # Safe fallback if a mixed buffer ever occurs.
            number = ""
            for c in self.buffer_chars:
                if c.isdigit():
                    number += c
                else:
                    if number:
                        self.tokens.append(number)
                        number = ""
                    self.tokens.append(c)
            if number:
                self.tokens.append(number)

        self.buffer_chars = []
        self.state = SessionState.TOKEN_COMMITTED
        self.add_event(f"Token committed -> {self.tokens[-1]}")

        if self.tokens and self.tokens[-1] == "=":
            self.calculate()

        self._sync()

    def undo(self):
        self.result = None

        if self.buffer_chars:
            removed = self.buffer_chars.pop()
            self.add_event(f"UNDO current char -> {removed}")
        elif self.tokens:
            removed = self.tokens.pop()
            self.add_event(f"UNDO token -> {removed}")
        else:
            self.add_event("UNDO -> nothing to remove")

        if self.state == SessionState.RESULT:
            self.state = SessionState.READY
        self._sync()

    def calculate(self):
        try:
            expression, value = evaluate_tokens(self.tokens)
            self.result = str(value)
            self.state = SessionState.RESULT
            self.add_event(f"Result -> {value}")
        except Exception as exc:
            self.result = f"ERROR: {exc}"
            self.state = SessionState.RESULT
            self.add_event(f"Calculation error -> {exc}")
        self._sync()

    def add_event(self, message):
        self.events.appendleft({
            "time": time.strftime("%H:%M:%S"),
            "message": message,
        })

    def _sync(self, current_character=None):
        self.snapshot.state = self.state.value
        self.snapshot.tokens = list(self.tokens)
        self.snapshot.buffer = "".join(self.buffer_chars)
        self.snapshot.expression = " ".join(self.tokens)
        self.snapshot.result = self.result
        if current_character is not None:
            self.snapshot.current_character = current_character
