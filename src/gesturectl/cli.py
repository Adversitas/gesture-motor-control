import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import typer

from .config import (
    CLASSIFIER_PATH,
    INSTRUCTIONS,
    LABELS,
    LANDMARKER_PATH,
    REPORTS_DIR,
    SEND_HZ,
    SESSIONS_DIR,
    UDP_PORT,
)

app = typer.Typer(help="Hand-gesture motor control: record, train, evaluate, run.",
                  no_args_is_help=True)
WINDOW = "gesturectl  (q / Esc to quit)"


def _open_camera(index: int, width: int, height: int):
    import cv2

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY  # DirectShow opens fast on Windows
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise typer.BadParameter(f"Could not open camera {index}")
    return cap


def _read(cap):
    """Read one frame, mirrored so moving your hand left moves it left on screen."""
    import cv2

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera stopped delivering frames")
    return cv2.flip(frame, 1)


def _quit_pressed() -> bool:
    import cv2

    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"


@app.command("download-model")
def download_model():
    """Fetch MediaPipe's hand landmark model (~8 MB) into models/."""
    from .landmarks import ensure_model

    typer.echo(f"Model at {ensure_model(LANDMARKER_PATH)}")


@app.command()
def record(
    person: str = typer.Option(..., help="Who is recording, e.g. 'hamza'"),
    place: str = typer.Option("home", help="Where, e.g. 'desk', 'living-room'"),
    lighting: str = typer.Option("day", help="e.g. 'day', 'lamp', 'backlit'"),
    rounds: int = typer.Option(2, help="Times each gesture is recorded"),
    seconds: float = typer.Option(6.0, help="Recording time per gesture per round"),
    ready_seconds: float = 3.0,
    camera: int = 0,
    width: int = 640,
    height: int = 480,
    out: Path = SESSIONS_DIR,
):
    """Guided recording: prompts each gesture in random order and saves landmarks (no images)."""
    import cv2

    from .dataset import Session
    from .landmarks import HandTracker
    from .overlay import RED, YELLOW, draw_banner, draw_hand, draw_text

    rng = np.random.default_rng()
    plan = [str(lab) for _ in range(rounds) for lab in rng.permutation(LABELS)]
    points, hands, labels, times = [], [], [], []
    cap = _open_camera(camera, width, height)
    tracker = HandTracker(LANDMARKER_PATH)
    frame_shape, start = None, time.monotonic()
    try:
        for step, label in enumerate(plan, 1):
            for phase, duration in (("ready", ready_seconds), ("record", seconds)):
                end = time.monotonic() + duration
                while (now := time.monotonic()) < end:
                    frame = _read(cap)
                    frame_shape = frame.shape
                    hand = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if hand is not None:
                        draw_hand(frame, hand)
                        if phase == "record":
                            points.append(hand.points)
                            hands.append(hand.handedness)
                            labels.append(label)
                            times.append(now - start)
                    title = f"{label.upper()}  ({step}/{len(plan)})"
                    if phase == "ready":
                        draw_banner(frame, f"Get ready: {title}", INSTRUCTIONS[label], YELLOW)
                    else:
                        draw_banner(frame, f"REC {title}  {end - now:.0f}s", INSTRUCTIONS[label], RED)
                        if hand is None:
                            draw_text(frame, "no hand detected", (16, 110), 0.6, YELLOW)
                    cv2.imshow(WINDOW, frame)
                    if _quit_pressed():
                        typer.echo("Aborted - nothing saved.")
                        raise typer.Exit(1)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    h, w = frame_shape[:2]
    session = Session(
        id=f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(person)}_{_slug(place)}_{_slug(lighting)}",
        points=np.array(points),
        handedness=np.array(hands),
        labels=np.array(labels),
        t=np.array(times),
        meta={"person": person, "place": place, "lighting": lighting, "width": w, "height": h,
              "aspect": w / h, "rounds": rounds, "seconds": seconds,
              "created": datetime.now().isoformat(timespec="seconds")},
    )
    path = session.save(out)
    counts = {lab: int((session.labels == lab).sum()) for lab in LABELS}
    typer.echo(f"Saved {len(labels)} frames to {path}: {counts}")


@app.command()
def train(data: Path = SESSIONS_DIR, out: Path = CLASSIFIER_PATH, seed: int = 0):
    """Train the classifier on every recorded session."""
    from .dataset import load_sessions, to_features
    from .evaluate import train_model

    sessions = load_sessions(data)
    if not sessions:
        raise typer.BadParameter(f"No sessions in {data}; run `gesturectl record` first")
    X = np.concatenate([s.normalized() for s in sessions])
    y = np.concatenate([s.label_indices() for s in sessions])
    model = train_model(X, y, seed=seed)
    acc = float((model.predict(to_features(X)) == y).mean())
    model.save(out)
    typer.echo(f"{len(sessions)} sessions, {len(y)} frames; training accuracy {acc:.3f}")
    typer.echo(f"Saved {out}. Training accuracy is optimistic; see `gesturectl evaluate`.")


@app.command()
def evaluate(data: Path = SESSIONS_DIR, seed: int = 0):
    """Compare the MLP with the rule baseline on random-frame vs. leave-one-session-out splits."""
    from .dataset import load_sessions
    from .evaluate import cross_validate, format_confusion

    sessions = load_sessions(data)
    if len(sessions) < 2:
        raise typer.BadParameter("Record at least 2 sessions (ideally 4+) to evaluate")
    typer.echo(f"{len(sessions)} sessions, {sum(len(s.labels) for s in sessions)} frames\n")

    results = [cross_validate(sessions, "frame", seed=seed),
               cross_validate(sessions, "session", seed=seed)]
    typer.echo(f"{'split':<10}{'classifier':<12}{'accuracy':>10}{'macro-F1':>10}")
    for r in results:
        for name, m in (("mlp", r.model), ("rules", r.baseline)):
            typer.echo(f"{r.split:<10}{name:<12}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}")

    session = results[1]
    typer.echo("\nCommands after smoothing (leave-one-session-out):")
    for name, m in (("mlp", session.commands_model), ("rules", session.commands_baseline)):
        typer.echo(f"  {name:<6} wrong-action rate {m['wrong_action_rate']:.3f}"
                   f"   hit rate {m['hit_rate']:.3f}")
    typer.echo("\nMLP confusion matrix (leave-one-session-out):")
    typer.echo(format_confusion(session.model["confusion_matrix"]))

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"sessions": [s.id for s in sessions],
                                "results": [r.__dict__ for r in results]}, indent=2))
    typer.echo(f"\nSaved {path}")


@app.command()
def run(
    host: str = typer.Option("", help="Arduino IP. Leave empty for a dry run."),
    port: int = UDP_PORT,
    token: str = typer.Option("", envvar="GESTURECTL_TOKEN", help="Shared secret (env var)"),
    model_path: Path = typer.Option(CLASSIFIER_PATH, "--model"),
    use_baseline: bool = typer.Option(False, "--baseline", help="Use the rule classifier"),
    threshold: float = 0.75,
    no_hand: str = typer.Option("stop", help="Command when no hand is seen: stop | hold"),
    camera: int = 0,
    width: int = 640,
    height: int = 480,
):
    """Live control. Without --host it is a dry run with a simulated motor on screen."""
    import cv2

    from . import baseline
    from .controller import ControllerConfig, GestureController
    from .features import featurize, normalize
    from .landmarks import HandTracker
    from .model import MLPClassifier
    from .motor_sim import MotorSim
    from .overlay import COMMAND_COLORS, draw_banner, draw_hand, draw_probabilities, draw_throttle
    from .transport import DryRunTransport, UdpTransport

    if use_baseline:
        predict = None
    else:
        if not model_path.exists():
            raise typer.BadParameter(f"{model_path} not found: run `gesturectl train`, "
                                     "or use --baseline")
        predict = MLPClassifier.load(model_path).predict_proba

    transport = UdpTransport(host, port, token) if host else DryRunTransport(token or "dry-run")
    sim = MotorSim(transport.token)
    ctrl = GestureController(ControllerConfig(threshold=threshold, no_hand=no_hand))
    cap = _open_camera(camera, width, height)
    tracker = HandTracker(LANDMARKER_PATH)
    mode = f"UDP -> {host}:{port}" if host else "DRY RUN (no packets sent)"
    next_send, fps, last = 0.0, 0.0, time.monotonic()
    try:
        while True:
            frame = _read(cap)
            h, w = frame.shape[:2]
            hand = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs = None
            if hand is not None:
                if predict is None:
                    p = normalize(hand.points, hand.handedness, w / h)
                    probs = np.eye(len(LABELS))[LABELS.index(baseline.classify(p))]
                else:
                    probs = predict(featurize(hand.points, hand.handedness, w / h))[0]
            command = ctrl.update(probs)

            now = time.monotonic()
            if now >= next_send:
                sim.receive(transport.send(command), now * 1000)
                next_send = now + 1 / SEND_HZ
            sim.tick(now * 1000)
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            if hand is not None:
                draw_hand(frame, hand)
            draw_banner(frame, command, f"{mode}   {fps:.0f} fps", COMMAND_COLORS[command])
            draw_probabilities(frame, LABELS, probs)
            draw_throttle(frame, sim.level, f"{sim.throttle_us:.0f}us", sim.watchdog_tripped)
            cv2.imshow(WINDOW, frame)
            if _quit_pressed():
                break
    finally:
        for _ in range(5):  # make sure the last thing the Arduino hears is STOP
            transport.send("STOP")
        transport.close()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


@app.command()
def bench(frames: int = 300, camera: int = 0, width: int = 640, height: int = 480,
          model_path: Path = typer.Option(CLASSIFIER_PATH, "--model")):
    """Measure per-frame latency of each pipeline stage on the live camera."""
    import cv2

    from .features import featurize
    from .landmarks import HandTracker
    from .model import MLPClassifier

    model = MLPClassifier.load(model_path) if model_path.exists() else None
    cap = _open_camera(camera, width, height)
    tracker = HandTracker(LANDMARKER_PATH)
    timings: dict[str, list[float]] = {"capture": [], "landmarks": [], "classify": []}
    try:
        for _ in range(frames):
            t0 = time.perf_counter()
            frame = _read(cap)
            t1 = time.perf_counter()
            hand = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t2 = time.perf_counter()
            timings["capture"].append(t1 - t0)
            timings["landmarks"].append(t2 - t1)
            if hand is not None and model is not None:
                model.predict_proba(featurize(hand.points, hand.handedness, width / height))
                timings["classify"].append(time.perf_counter() - t2)
    finally:
        cap.release()
        tracker.close()

    typer.echo(f"{'stage':<12}{'p50 ms':>9}{'p95 ms':>9}")
    for stage, values in timings.items():
        if values:
            ms = np.array(values) * 1000
            typer.echo(f"{stage:<12}{np.percentile(ms, 50):>9.2f}{np.percentile(ms, 95):>9.2f}")
    if not timings["classify"]:
        typer.echo("(classify not measured: no trained model or no hand in view)")


if __name__ == "__main__":
    app()
