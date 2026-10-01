"""Recording sessions: raw landmarks only (no images), one .npz file per session.

A session is one sitting under fixed conditions (person, place, lighting) in which every
gesture is recorded. Keeping all classes inside each session, and splitting train/test by
session, is what stops the model from learning "which day was it" instead of the gesture.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import LABELS
from .features import features_from_normalized, normalize


@dataclass
class Session:
    id: str
    points: np.ndarray  # (N, 21, 3) raw MediaPipe landmarks
    handedness: np.ndarray  # (N,) "Left" / "Right"
    labels: np.ndarray  # (N,) gesture names
    t: np.ndarray  # (N,) seconds since session start
    meta: dict = field(default_factory=dict)  # person, place, lighting, aspect, ...

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.npz"
        np.savez_compressed(
            path,
            points=self.points.astype(np.float32),
            handedness=self.handedness.astype(str),
            labels=self.labels.astype(str),
            t=self.t.astype(np.float64),
            meta=json.dumps(self.meta),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Session":
        d = np.load(path)
        return cls(
            id=path.stem,
            points=d["points"],
            handedness=d["handedness"],
            labels=d["labels"],
            t=d["t"],
            meta=json.loads(str(d["meta"])),
        )

    def normalized(self) -> np.ndarray:
        aspect = float(self.meta.get("aspect", 16 / 9))
        return np.stack(
            [normalize(p, h, aspect) for p, h in zip(self.points, self.handedness, strict=True)]
        )

    def label_indices(self) -> np.ndarray:
        return np.array([LABELS.index(lab) for lab in self.labels])


def load_sessions(directory: Path) -> list[Session]:
    return [Session.load(p) for p in sorted(directory.glob("*.npz"))]


def to_features(normalized: np.ndarray) -> np.ndarray:
    return np.stack([features_from_normalized(p) for p in normalized])


def augment(normalized: np.ndarray, rng: np.random.Generator, copies: int = 3,
            max_rotation_deg: float = 15, noise: float = 0.02) -> np.ndarray:
    """Small in-plane rotations and landmark jitter. Rotations stay small on purpose:
    orientation is what separates "up" from "down"."""
    out = [normalized]
    for _ in range(copies):
        theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg, len(normalized)))
        c, s = np.cos(theta), np.sin(theta)
        rotated = normalized.copy()
        x, y = normalized[..., 0], normalized[..., 1]
        rotated[..., 0] = c[:, None] * x - s[:, None] * y
        rotated[..., 1] = s[:, None] * x + c[:, None] * y
        out.append(rotated + rng.normal(0, noise, rotated.shape))
    return np.concatenate(out)
