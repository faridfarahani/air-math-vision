from dataclasses import dataclass
from typing import Optional


@dataclass
class GestureResult:
    name: Optional[str]
    confidence: float = 0.0


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def classify_hand(hand_landmarks) -> GestureResult:
    if hand_landmarks is None:
        return GestureResult(None, 0.0)

    lm = hand_landmarks.landmark

    wrist = lm[0]
    thumb_tip, thumb_ip = lm[4], lm[3]
    index_pip, index_tip = lm[6], lm[8]
    middle_pip, middle_tip = lm[10], lm[12]
    ring_pip, ring_tip = lm[14], lm[16]
    pinky_pip, pinky_tip = lm[18], lm[20]

    index_open = _dist(index_tip, wrist) > _dist(index_pip, wrist) * 1.13
    middle_open = _dist(middle_tip, wrist) > _dist(middle_pip, wrist) * 1.13
    ring_open = _dist(ring_tip, wrist) > _dist(ring_pip, wrist) * 1.13
    pinky_open = _dist(pinky_tip, wrist) > _dist(pinky_pip, wrist) * 1.13

    thumb_extended = _dist(thumb_tip, wrist) > _dist(thumb_ip, wrist) * 1.10
    fingers_open = sum([index_open, middle_open, ring_open, pinky_open])
    folded_others = fingers_open == 0

    # Keep thumbs-up / thumbs-down higher priority than the fist (=) gesture.
    if folded_others and thumb_extended:
        vertical_delta = thumb_tip.y - wrist.y
        horizontal_delta = abs(thumb_tip.x - wrist.x)

        if vertical_delta < -0.10 and abs(vertical_delta) > horizontal_delta * 0.55:
            return GestureResult("THUMBS_UP", 0.92)

        if vertical_delta > 0.13 and abs(vertical_delta) > horizontal_delta * 0.55:
            return GestureResult("THUMBS_DOWN", 0.92)

    # = : closed fist. A thumb that is not clearly UP/DOWN is allowed because
    # it may naturally rest across the palm.
    if folded_others:
        return GestureResult("OP_EQUALS", 0.90)

    # Operator gestures. Keep the thumb folded for these poses.
    if not thumb_extended:
        # + : pinky only
        if pinky_open and not ring_open and not middle_open and not index_open:
            return GestureResult("OP_ADD", 0.93)

        # - : pinky + ring
        if pinky_open and ring_open and not middle_open and not index_open:
            return GestureResult("OP_SUBTRACT", 0.93)

        # × : pinky + ring + middle
        if pinky_open and ring_open and middle_open and not index_open:
            return GestureResult("OP_MULTIPLY", 0.93)

        # ÷ : all four fingers, thumb folded
        if pinky_open and ring_open and middle_open and index_open:
            return GestureResult("OP_DIVIDE", 0.93)

    # ✋ remains the number-token commit gesture: all fingers including thumb.
    if fingers_open == 4 and thumb_extended:
        return GestureResult("OPEN_PALM", 0.95)

    # ☝ index finger only = Air Writing.
    if index_open and not middle_open and not ring_open and not pinky_open:
        return GestureResult("POINTING", 0.93)

    return GestureResult(None, 0.0)
