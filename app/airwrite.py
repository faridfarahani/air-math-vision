import math
import numpy as np


NUM_POINTS = 64
SQUARE_SIZE = 250.0
HALF_DIAGONAL = 0.5 * math.sqrt(SQUARE_SIZE * SQUARE_SIZE * 2)
ANGLE_RANGE = math.radians(45.0)
ANGLE_PRECISION = math.radians(2.0)
PHI = 0.5 * (-1.0 + math.sqrt(5.0))


def _distance(a, b):
    return math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))


def _path_length(points):
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _resample(points, n=NUM_POINTS):
    """
    Deterministic arc-length resampling.

    IMPORTANT:
    The previous implementation inserted points back into the same list while
    iterating. On some strokes it could repeatedly insert the same point and
    stall the camera thread. This implementation never mutates the input list
    and always finishes in O(number_of_points + n).
    """
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float64)

    if len(pts) == 1:
        return np.repeat(pts[:1], n, axis=0)

    # Remove consecutive duplicate / near-duplicate points.
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > 1e-6))
    pts = pts[keep]

    if len(pts) == 1:
        return np.repeat(pts[:1], n, axis=0)

    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])

    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, cumulative, pts[:, 0])
    y = np.interp(targets, cumulative, pts[:, 1])
    return np.column_stack((x, y))


def _centroid(points):
    return np.mean(points, axis=0)


def _indicative_angle(points):
    c = _centroid(points)
    return math.atan2(c[1] - points[0][1], c[0] - points[0][0])


def _rotate(points, angle):
    c = _centroid(points)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    shifted = points - c
    result = np.empty_like(shifted)
    result[:, 0] = shifted[:, 0] * cos_a - shifted[:, 1] * sin_a
    result[:, 1] = shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
    return result + c


def _scale_to_square(points, size=SQUARE_SIZE):
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    dims = np.maximum(max_xy - min_xy, 1e-6)
    return (points - min_xy) * (size / dims)


def _translate_to_origin(points):
    return points - _centroid(points)


def _normalize(points):
    p = _resample(points)
    p = _rotate(p, -_indicative_angle(p))
    p = _scale_to_square(p)
    p = _translate_to_origin(p)
    return p


def _path_distance(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _distance_at_angle(points, template, angle):
    return _path_distance(_rotate(points, angle), template)


def _distance_at_best_angle(points, template):
    a = -ANGLE_RANGE
    b = ANGLE_RANGE
    x1 = PHI * a + (1.0 - PHI) * b
    f1 = _distance_at_angle(points, template, x1)
    x2 = (1.0 - PHI) * a + PHI * b
    f2 = _distance_at_angle(points, template, x2)

    iterations = 0
    while abs(b - a) > ANGLE_PRECISION and iterations < 32:
        iterations += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = PHI * a + (1.0 - PHI) * b
            f1 = _distance_at_angle(points, template, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - PHI) * a + PHI * b
            f2 = _distance_at_angle(points, template, x2)

    return min(f1, f2)


def _line(x1, y1, x2, y2, n=20):
    return [(x1 + (x2-x1)*i/(n-1), y1 + (y2-y1)*i/(n-1)) for i in range(n)]


def _arc(cx, cy, rx, ry, start_deg, end_deg, n=40):
    out = []
    for i in range(n):
        a = math.radians(start_deg + (end_deg-start_deg) * i/(n-1))
        out.append((cx + rx*math.cos(a), cy + ry*math.sin(a)))
    return out


def _join(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


def _templates():
    t = {}

    t["0"] = _arc(50, 50, 30, 45, -90, 270, 70)
    t["1"] = _join(_line(38, 24, 52, 10, 10), _line(52, 10, 52, 92, 45))
    t["2"] = _join(_arc(48, 32, 30, 22, 205, 520, 48), _line(72, 48, 22, 91, 26), _line(22, 91, 80, 91, 22))
    t["3"] = _join(_arc(43, 30, 30, 22, 225, 455, 36), _arc(43, 70, 30, 23, 265, 495, 36))
    t["4"] = _join(_line(72, 92, 72, 8, 36), _line(72, 8, 24, 66, 28), _line(24, 66, 88, 66, 28))
    t["5"] = _join(_line(80, 12, 27, 12, 20), _line(27, 12, 27, 49, 18), _arc(49, 67, 28, 24, 210, 520, 42))
    t["6"] = _join(_arc(55, 52, 31, 42, 40, 330, 58), _arc(49, 69, 23, 20, 0, 360, 34))
    t["7"] = _join(_line(20, 12, 84, 12, 25), _line(84, 12, 40, 94, 40))
    t["8"] = _join(_arc(50, 31, 23, 23, -90, 270, 42), _arc(50, 72, 27, 25, -90, 270, 44))
    t["9"] = _join(_arc(49, 30, 25, 23, -90, 270, 42), _line(74, 30, 42, 94, 34))

    t["−"] = _line(15, 50, 85, 50, 45)
    t["+"] = _join(_line(50, 12, 50, 88, 30), _line(50, 50, 15, 50, 15), _line(15, 50, 85, 50, 28))
    t["×"] = _join(_line(18, 18, 82, 82, 30), _line(82, 82, 82, 18, 8), _line(82, 18, 18, 82, 30))
    t["="] = _join(_line(15, 34, 85, 34, 28), _line(85, 34, 15, 68, 18), _line(15, 68, 85, 68, 28))
    t["÷"] = _join(
        _arc(50, 18, 3, 3, 0, 360, 10),
        _line(50, 18, 17, 52, 10),
        _line(17, 52, 83, 52, 28),
        _line(83, 52, 50, 83, 10),
        _arc(50, 83, 3, 3, 0, 360, 10),
    )
    return t


class AirWritingRecognizer:
    def __init__(self):
        self.templates = {name: _normalize(points) for name, points in _templates().items()}

    def predict(self, points, expected="ANY"):
        if len(points) < 8:
            return None, 0.0

        candidate = _normalize(points)

        allowed = set(self.templates)
        if expected == "NUMBER":
            allowed = set("0123456789")
        elif expected == "OPERATOR":
            allowed = {"+", "−", "×", "÷", "="}

        best_name = None
        best_distance = float("inf")

        for name, template in self.templates.items():
            if name not in allowed:
                continue
            d = _distance_at_best_angle(candidate, template)
            if d < best_distance:
                best_distance = d
                best_name = name

        score = max(0.0, 1.0 - (best_distance / HALF_DIAGONAL))
        if score < 0.38:
            return None, score

        return best_name, score
