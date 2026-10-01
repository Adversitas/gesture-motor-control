"""Turn raw landmarks into features that ignore where the hand is and how big it looks,
but keep its orientation (pointing up vs. down is the signal)."""

import numpy as np

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP = 9
FINGER_PIPS = (6, 10, 14, 18)  # index, middle, ring, pinky
FINGER_TIPS = (8, 12, 16, 20)

FEATURE_DIM = 20 * 3 + 5


def normalize(points: np.ndarray, handedness: str, aspect: float) -> np.ndarray:
    """(21, 3) image-space landmarks -> wrist-centred, palm-size-scaled, right-hand frame.

    `aspect` is width / height: MediaPipe normalises x by width and y by height, so x and z
    are rescaled to make the coordinates isotropic before measuring distances.
    """
    p = np.asarray(points, dtype=np.float64).copy()
    p[:, 0] *= aspect
    p[:, 2] *= aspect
    p -= p[WRIST]
    p /= max(float(np.linalg.norm(p[MIDDLE_MCP, :2])), 1e-6)
    if handedness == "Left":
        p[:, 0] *= -1  # mirror left hands so one model serves both
    return p


def extension_ratios(p: np.ndarray) -> np.ndarray:
    """>1 when a finger is straight, <1 when curled. Wrist is at the origin."""
    fingers = [
        np.linalg.norm(p[tip]) / max(np.linalg.norm(p[pip]), 1e-6)
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS, strict=True)
    ]
    thumb = np.linalg.norm(p[THUMB_TIP] - p[INDEX_MCP]) / max(
        np.linalg.norm(p[THUMB_IP] - p[INDEX_MCP]), 1e-6
    )
    return np.array([thumb, *fingers])


def features_from_normalized(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p[1:].ravel(), extension_ratios(p)]).astype(np.float32)


def featurize(points: np.ndarray, handedness: str, aspect: float) -> np.ndarray:
    return features_from_normalized(normalize(points, handedness, aspect))
