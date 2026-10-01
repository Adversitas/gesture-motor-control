"""Turns noisy per-frame predictions into stable commands.

A raw classifier flickers between classes on transition frames; sending that straight to
motors would make them stutter. The controller smooths probabilities (EMA), requires a
confidence threshold, and only switches gesture after it has been seen for a few frames
in a row. STOP is the exception: it is acted on after a single confident frame, because
stopping late is worse than stopping early.
"""

from dataclasses import dataclass

import numpy as np

from .config import COMMANDS, LABELS


@dataclass
class ControllerConfig:
    alpha: float = 0.5  # EMA weight of the newest frame
    threshold: float = 0.75  # min smoothed probability to accept a gesture
    confirm_frames: int = 3  # consecutive frames needed to switch to UP / DOWN / HOLD
    stop_confirm_frames: int = 1
    no_hand: str = "stop"  # command when no hand is visible: "stop" or "hold"


class GestureController:
    def __init__(self, config: ControllerConfig | None = None, labels: tuple[str, ...] = LABELS):
        self.cfg = config or ControllerConfig()
        self.labels = labels
        self.reset()

    def reset(self) -> None:
        self._ema: np.ndarray | None = None
        self._candidate = "none"
        self._streak = 0
        self.gesture = "none"

    def update(self, probs: np.ndarray | None) -> str:
        """Feed one frame's class probabilities (None = no hand). Returns the command."""
        if probs is None:
            self.reset()
            return "STOP" if self.cfg.no_hand == "stop" else "HOLD"

        probs = np.asarray(probs, dtype=np.float64)
        a = self.cfg.alpha
        self._ema = probs if self._ema is None else a * probs + (1 - a) * self._ema

        best = int(self._ema.argmax())
        candidate = self.labels[best] if self._ema[best] >= self.cfg.threshold else "none"
        if candidate == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = candidate, 1

        needed = self.cfg.stop_confirm_frames if candidate == "stop" else self.cfg.confirm_frames
        if self._streak >= needed:
            self.gesture = candidate
        return COMMANDS[self.gesture]
