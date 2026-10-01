# Gesture Motor Control: real-time hand gestures to motors, with honest evaluation

Point your index finger up and the motors spin up; point down and they slow; show an open palm
and they stop. A webcam feeds MediaPipe hand landmarks to a small classifier. The resulting
commands go over Wi-Fi (UDP) to an Arduino UNO R4 driving brushless ESCs.

This is a rebuild of an earlier version that reported **93% validation accuracy but scored 41% on new
photos**, worse than a coin flip. Finding out why shaped every design decision below.

## What went wrong the first time, and what changed

| v1 (CNN on raw 1280×720 photos) | Problem | v2 (this repo) |
|---|---|---|
| "up" photos shot on one day, "down" photos the next | Model could learn *the day* (lighting, background) instead of the gesture | Every recording session contains **all** gestures under the same conditions |
| Random train/val split of burst photos | Near-duplicate frames on both sides: validation measured memorisation | **Leave-one-session-out** evaluation, reported next to the leaky split so the gap is visible |
| Pixels → 116M-parameter CNN (1.4 GB file, ~7 min/epoch) | Huge, slow, sensitive to background | **21 hand landmarks** → 65 features → 4.5k-parameter MLP (trains in seconds, µs per frame) |
| Classification report misaligned with a shuffled generator | Wrong metrics | Metrics implemented and unit-tested |
| HTTP GET with no auth; motors keep last command if Wi-Fi drops | Anyone on the network can drive the motors; no failsafe | UDP with shared token and sequence numbers, **firmware watchdog**, throttle cap and ramp |
| Photos of the user stored and used | Privacy | Only landmark coordinates are saved, never images |

## Pipeline

```
webcam ─► MediaPipe HandLandmarker (video mode) ─► 21 × (x, y, z)
      ─► normalise: wrist-centred, palm-scaled, left hands mirrored, orientation kept
      ─► 65 features (60 coordinates + 5 finger-extension ratios)
      ─► MLP (NumPy) ─► class probabilities {up, down, stop, none}
      ─► controller: EMA smoothing, confidence threshold, N-frame confirmation (STOP: 1 frame)
      ─► UDP "<token> <nonce> <seq> <CMD>" at 20 Hz ─► Arduino: ramp, cap, watchdog ─► ESCs
```

A rule-based classifier (finger-extension geometry) is the **baseline** the MLP has to beat.

## Evaluation

| Metric | Why |
|---|---|
| Accuracy and macro-F1, random-frame split vs. leave-one-session-out | The gap measures how much a leaky split flatters the model |
| Confusion matrix | Which gestures get mixed up |
| **Wrong-action rate** after smoothing | Share of frames where the motors would be told to do something the user didn't ask for: the safety-relevant error |
| Hit rate after smoothing | How responsive control feels |
| Per-stage latency (p50/p95) | Camera, landmarks, classifier |

### Results

Record at least 4 sessions under different conditions, then run `gesturectl evaluate`;
it prints both splits side by side and saves a JSON report to `reports/`.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
gesturectl download-model

# try live control right away with the rule baseline, no hardware, no data needed
gesturectl run --baseline

# record data: each session prompts every gesture in random order (~1 min)
gesturectl record --person me --place desk --lighting day
gesturectl record --person me --place sofa --lighting lamp      # vary conditions!

gesturectl evaluate        # MLP vs rules, frame split vs session split
gesturectl train           # final model -> models/gesture_mlp.npz
gesturectl run             # dry run with the MLP and a simulated motor on screen
gesturectl bench           # latency per stage
```

### With the hardware

1. Copy `firmware/gesture_motor/secrets.example.h` to `secrets.h`; set Wi-Fi and a token.
2. Flash `firmware/gesture_motor/gesture_motor.ino` to an Arduino UNO R4 WiFi (ESCs on pins 9–11).
3. **Props off.** The throttle cap (`US_MAX`) starts at 1200 µs.
4. `set GESTURECTL_TOKEN=<token>` then `gesturectl run --host <arduino-ip>`.

Quitting the app sends STOP. If the app crashes or Wi-Fi drops, the firmware watchdog idles the motors
within 500 ms.

## Tests

`pytest` covers feature invariances, the rule baseline, the MLP and the evaluation code (on synthetic
hands), the controller, packet validation and the firmware logic via its Python twin
(`motor_sim.py`): ramp, cap, watchdog, wrong-token and replay rejection.

## Layout

```
src/gesturectl/
  landmarks.py   MediaPipe wrapper        features.py   normalisation + features
  baseline.py    rule classifier          model.py      NumPy MLP
  dataset.py     sessions, augmentation   evaluate.py   CV, metrics, command replay
  controller.py  smoothing / debouncing   transport.py  UDP protocol
  motor_sim.py   firmware twin            cli.py        record / train / evaluate / run / bench
firmware/gesture_motor/   Arduino sketch + secrets template
tests/                    unit tests and a synthetic hand generator
```
