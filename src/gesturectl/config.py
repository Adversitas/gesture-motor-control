from pathlib import Path

# Class order is part of the model file format; do not reorder.
LABELS: tuple[str, ...] = ("up", "down", "stop", "none")

# What each gesture means on the wire
COMMANDS: dict[str, str] = {"up": "UP", "down": "DOWN", "stop": "STOP", "none": "HOLD"}

INSTRUCTIONS: dict[str, str] = {
    "up": "Index finger pointing UP, other fingers curled",
    "down": "Index finger pointing DOWN, other fingers curled",
    "stop": "Open palm facing the camera",
    "none": "Anything else: fist, relaxed hand, waving... vary it!",
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
LANDMARKER_PATH = Path("models/hand_landmarker.task")
CLASSIFIER_PATH = Path("models/gesture_mlp.npz")
SESSIONS_DIR = Path("data/sessions")
REPORTS_DIR = Path("reports")

UDP_PORT = 4210
SEND_HZ = 20  # command packets per second; the firmware watchdog expects at least 2 per second
