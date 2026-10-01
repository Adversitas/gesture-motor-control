"""OpenCV drawing helpers for the record / run windows."""

import cv2
import numpy as np

from .landmarks import HAND_CONNECTIONS, Hand

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE, GREY = (80, 200, 80), (60, 60, 230), (40, 200, 230), (255, 255, 255), (90, 90, 90)
COMMAND_COLORS = {"UP": GREEN, "DOWN": YELLOW, "STOP": RED, "HOLD": GREY}


def draw_hand(frame: np.ndarray, hand: Hand, color=GREEN) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, WHITE, -1, cv2.LINE_AA)


def draw_text(frame, text, org, scale=0.6, color=WHITE, thickness=1) -> None:
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_banner(frame, title: str, subtitle: str, color=WHITE) -> None:
    draw_text(frame, title, (16, 44), 1.1, color, 2)
    draw_text(frame, subtitle, (16, 76), 0.6)


def draw_probabilities(frame, labels, probs: np.ndarray | None, top: int = 110) -> None:
    for i, lab in enumerate(labels):
        y = top + i * 24
        p = 0.0 if probs is None else float(probs[i])
        cv2.rectangle(frame, (90, y - 14), (90 + int(150 * p), y + 2), GREEN, -1)
        cv2.rectangle(frame, (90, y - 14), (240, y + 2), GREY, 1)
        draw_text(frame, lab, (16, y), 0.5)


def draw_throttle(frame, level: float, label: str, tripped: bool) -> None:
    h, w = frame.shape[:2]
    x0, y0, y1 = w - 50, 110, h - 40
    filled = int((y1 - y0) * min(max(level, 0.0), 1.0))
    cv2.rectangle(frame, (x0, y1 - filled), (x0 + 24, y1), RED if tripped else GREEN, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + 24, y1), WHITE, 1)
    draw_text(frame, label, (x0 - 40, y1 + 24), 0.5)
