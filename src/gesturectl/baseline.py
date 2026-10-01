"""Hand-written geometric rules. The learned model has to beat this to justify itself."""

import numpy as np

from .features import FINGER_PIPS, FINGER_TIPS, INDEX_MCP, INDEX_TIP


def classify(p: np.ndarray, extended_ratio: float = 1.1, direction_cos: float = 0.6) -> str:
    """Classify one normalised hand (see features.normalize)."""
    extended = [
        np.linalg.norm(p[tip]) > extended_ratio * np.linalg.norm(p[pip])
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS, strict=True)
    ]
    if sum(extended) >= 4:
        return "stop"
    if extended[0] and sum(extended) == 1:
        direction = p[INDEX_TIP, :2] - p[INDEX_MCP, :2]
        dy = direction[1] / max(float(np.linalg.norm(direction)), 1e-6)
        if dy < -direction_cos:  # image y grows downwards
            return "up"
        if dy > direction_cos:
            return "down"
    return "none"
