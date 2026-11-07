"""
SignSpeak - shared helpers.

* HandDetector      : wraps MediaPipe Hands (webcam frame -> 21 landmarks)
* normalize_landmarks: wrist-relative, scale-invariant, hand-side invariant
                       63-value feature vector (synopsis: "Preprocessing")
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

# Standard MediaPipe hand skeleton (kept here so drawing code needs no mediapipe).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]
FINGERTIPS = (4, 8, 12, 16, 20)


def feature_columns() -> List[str]:
    """CSV column names: x0,y0,z0 ... x20,y20,z20 (63 features)."""
    return [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def normalize_landmarks(landmarks, handedness: str = "Right") -> np.ndarray:
    """
    Convert 21 MediaPipe landmarks into a 63-value feature vector.

    1. Translate so the wrist (landmark 0) is the origin   -> position invariant
    2. Mirror left hands so they look like right hands      -> hand-side invariant
    3. Divide by the largest wrist-to-landmark distance     -> distance/size invariant
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    if handedness == "Left":
        pts[:, 0] *= -1.0
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()


@dataclass
class HandResult:
    features: np.ndarray                 # 63 normalised values
    points: List[Tuple[int, int]]        # 21 pixel coordinates (for drawing)
    handedness: str                      # "Left" / "Right"


class HandDetector:
    """Thin wrapper around MediaPipe Hands (single hand, video mode)."""

    def __init__(self):
        try:
            import mediapipe as mp
            hands_module = mp.solutions.hands
        except (ImportError, AttributeError) as exc:
            raise SystemExit(
                "\n[SignSpeak] This project needs the classic MediaPipe Hands API.\n"
                "Newer MediaPipe releases removed it. Fix:\n"
                "    pip install mediapipe==0.10.14\n"
                "(use Python 3.9 - 3.12)\n"
            ) from exc

        self._hands = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=config.MIN_DETECTION_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandResult]:
        """Return the first detected hand, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0].landmark
        handedness = result.multi_handedness[0].classification[0].label
        h, w = frame_bgr.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        return HandResult(normalize_landmarks(landmarks, handedness), points, handedness)

    def close(self):
        self._hands.close()
