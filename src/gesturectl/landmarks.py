"""Thin wrapper around MediaPipe's HandLandmarker (Tasks API, video mode)."""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from .config import MODEL_URL

# MediaPipe hand skeleton, used for drawing
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip


@dataclass
class Hand:
    points: np.ndarray  # (21, 3): x, y normalised to [0, 1] by image size, z relative depth
    handedness: str  # "Left" or "Right", as MediaPipe reports it
    score: float


def ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Tracks one hand across video frames. Video mode reuses the previous frame's hand
    position, which is several times faster than detecting from scratch on every frame."""

    def __init__(self, model_path: Path, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int | None = None) -> Hand | None:
        ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        ts = max(ts, self._last_ts + 1)  # MediaPipe requires strictly increasing timestamps
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        category = result.handedness[0][0]
        return Hand(points, category.category_name, float(category.score))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
