"""Synthetic MediaPipe-style hands, so features, models and evaluation can be tested
without a camera or recorded data."""

import numpy as np

from gesturectl.dataset import Session

# Finger base (MCP) positions for an upright right hand, wrist at the origin, y pointing down.
# Middle MCP is at distance 1 (the normalisation scale).
_MCPS = {5: (-0.32, -0.95), 9: (0.0, -1.0), 13: (0.3, -0.95), 17: (0.55, -0.85)}


def make_hand(extended: tuple[bool, bool, bool, bool], thumb_out: bool = False,
              rotation_deg: float = 0.0) -> np.ndarray:
    """Normalised (21, 3) hand. `extended` = index, middle, ring, pinky."""
    p = np.zeros((21, 3))
    p[1, :2] = (-0.3, -0.25)
    p[2, :2] = (-0.55, -0.45)
    if thumb_out:
        p[3, :2], p[4, :2] = (-0.75, -0.65), (-0.9, -0.85)
    else:  # tucked across the palm
        p[3, :2], p[4, :2] = (-0.45, -0.65), (-0.2, -0.7)
    for (mcp, base), ext in zip(_MCPS.items(), extended, strict=True):
        b = np.array(base)
        p[mcp, :2] = b
        if ext:
            p[mcp + 1, :2], p[mcp + 2, :2], p[mcp + 3, :2] = b + (0, -0.45), b + (0, -0.75), b + (0, -1.0)
        else:  # curled back towards the palm
            p[mcp + 1, :2], p[mcp + 2, :2], p[mcp + 3, :2] = b + (0, -0.35), b + (0, -0.15), b + (0, 0.1)
    theta = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p[:, :2] = p[:, :2] @ rot.T
    return p


GESTURES = {
    "up": {"extended": (True, False, False, False), "rotation_deg": 0},
    "down": {"extended": (True, False, False, False), "rotation_deg": 180},
    "stop": {"extended": (True, True, True, True), "thumb_out": True, "rotation_deg": 0},
    "none": {"extended": (False, False, False, False), "rotation_deg": 0},  # fist
}


def to_image_coords(p: np.ndarray, center=(0.5, 0.6), size=0.15, aspect=4 / 3,
                    left: bool = False) -> tuple[np.ndarray, str]:
    """Inverse of features.normalize: place a normalised hand in MediaPipe image coordinates."""
    q = p.copy()
    if left:
        q[:, 0] *= -1
    q *= size
    q[:, 0] = q[:, 0] / aspect + center[0]
    q[:, 1] = q[:, 1] + center[1]
    q[:, 2] = q[:, 2] / aspect
    return q.astype(np.float32), "Left" if left else "Right"


def sample(label: str, rng: np.random.Generator, aspect=4 / 3) -> tuple[np.ndarray, str]:
    spec = dict(GESTURES[label])
    spec["rotation_deg"] = spec["rotation_deg"] + rng.uniform(-20, 20)
    p = make_hand(**spec) + rng.normal(0, 0.03, (21, 3))
    return to_image_coords(
        p,
        center=(rng.uniform(0.3, 0.7), rng.uniform(0.4, 0.7)),
        size=rng.uniform(0.1, 0.25),
        aspect=aspect,
        left=bool(rng.random() < 0.3),
    )


def make_session(session_id: str, rng: np.random.Generator, per_label: int = 40,
                 labels=("up", "down", "stop", "none"), aspect=4 / 3) -> Session:
    points, hands, labs = [], [], []
    for label in labels:  # contiguous segments, like a real guided recording
        for _ in range(per_label):
            pts, hand = sample(label, rng, aspect)
            points.append(pts)
            hands.append(hand)
            labs.append(label)
    n = len(labs)
    return Session(session_id, np.array(points), np.array(hands), np.array(labs),
                   np.arange(n) / 30.0, {"aspect": aspect})
