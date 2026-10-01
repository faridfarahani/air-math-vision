from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SessionState(str, Enum):
    MONITORING = "MONITORING"
    FACE_POSITIONING = "FACE_POSITIONING"
    VERIFYING_FACE = "VERIFYING_FACE"
    WAITING_FOR_THUMBS_UP = "WAITING_FOR_THUMBS_UP"
    READY = "READY"
    DRAWING = "DRAWING"
    TOKEN_COMMITTED = "TOKEN_COMMITTED"
    RESULT = "RESULT"


@dataclass
class VisionSnapshot:
    people_count: int = 0
    identity_enrolled: bool = False
    identity_verified: bool = False
    face_similarity: Optional[float] = None
    gesture: Optional[str] = None
    state: str = SessionState.MONITORING.value
    current_character: Optional[str] = None
    buffer: str = ""
    tokens: List[str] = field(default_factory=list)
    expression: str = ""
    result: Optional[str] = None
    camera_ok: bool = False
