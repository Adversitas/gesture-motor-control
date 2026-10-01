# Gesture Motor Control

Control motors with hand gestures in front of a webcam.

| Gesture | Command | What the motors do |
|---|---|---|
| ☝️ Index finger pointing **up** | `UP` | Throttle ramps up |
| 👇 Index finger pointing **down** | `DOWN` | Throttle ramps down |
| ✋ **Open palm** | `STOP` | Motors return to idle immediately |
| ✊ Anything else (fist, relaxed hand...) | `HOLD` | Throttle stays where it is |

A Python app on the PC watches the camera, recognises the gesture and sends a command over Wi-Fi
to an **Arduino UNO R4 WiFi**. The Arduino drives the motors through brushless ESCs (electronic
speed controllers).

```
 ┌──────────── PC ──────────────────────────────────────────────┐            ┌──── Arduino ────┐
 │ webcam → hand landmarks → features → classifier → controller │ ── Wi-Fi ─►│ safety → ESCs   │
 └──────────────────────────────────────────────────────────────┘    UDP     └─────────────────┘
```

---

## How it works

### 1. Finding the hand: MediaPipe landmarks

Each camera frame goes through Google's **MediaPipe Hand Landmarker**, which returns 21 keypoints
on the hand: the wrist plus four points along each finger and the thumb.

```
 thumb  index  middle  ring  pinky
   4      8      12     16     20     ← fingertips
   3      7      11     15     19
   2      6      10     14     18
   1      5       9     13     17     ← knuckles
                  0                   ← wrist
```

Working from these 21 points instead of raw pixels is the key design choice:

- **Robust**: the classifier never sees the background, the lighting or your face, so it can't
  learn the wrong cues from them.
- **Small and fast**: 63 numbers per frame instead of almost a million pixels.
- **Private**: recordings store only the coordinates, never images.

MediaPipe runs in *video mode*, so it tracks the hand from one frame to the next instead of
searching the whole image every time.

### 2. Making the hand comparable: normalisation

The same gesture produces very different raw coordinates depending on where the hand is in the
picture, how close it is to the camera, and whether it's a left or right hand. Before
classification, every hand is:

1. **Centred on the wrist**, so its position in the frame doesn't matter;
2. **Scaled by palm size** (wrist to middle knuckle), so its distance from the camera doesn't matter;
3. **Mirrored if it's a left hand**, so one model serves both hands;
4. **Not rotated**, on purpose: the direction the finger points *is* the difference between
   "up" and "down".

### 3. Describing the hand: features

From the normalised hand the app computes **65 features**:

- the 3D positions of the 20 non-wrist points (60 values);
- 5 **finger-extension ratios**: how far each fingertip is from the wrist compared with the middle
  joint of that finger. Above 1 the finger is straight; below 1 it's curled.

### 4. Recognising the gesture: two classifiers

- **Rule-based baseline** (`baseline.py`): hand-written geometry. Four or more straight fingers
  means *stop*; only the index straight means *up* or *down* depending on which way it points;
  anything else is *none*. It needs no training, so it works out of the box.
- **Neural network** (`model.py`): a small MLP with 65 inputs, 64 hidden units and 4 outputs
  (about 4,500 weights), written in NumPy. It is trained on recorded examples and gives a
  probability for each gesture. Training uses light augmentation (small rotations and noise) so
  the model tolerates natural variation.

The MLP has to beat the baseline to be worth using; `gesturectl evaluate` compares the two.

### 5. Turning predictions into steady commands: the controller

A classifier looking at single frames flickers, especially while the hand moves from one gesture
to another. Sending every flicker to the motors would make them stutter, so the controller
(`controller.py`):

- **smooths** the probabilities over recent frames (exponential moving average);
- ignores predictions below a **confidence threshold** (75% by default);
- only switches gesture after it has been seen for **3 frames in a row**;
- makes an exception for **STOP**, which is acted on after a *single* confident frame, because
  stopping late is worse than stopping early;
- sends **STOP when no hand is visible** (configurable).

### 6. Talking to the Arduino: the protocol

The PC sends one small text packet 20 times per second over **UDP**:

```
<token> <nonce> <sequence> <COMMAND>
e.g.   8f3a91c2d4e5b6a7 1c9e2f0a 1532 UP
```

- **token**: a shared secret, so other devices on the network can't drive the motors;
- **nonce**: random for each run of the app, so the Arduino can tell a restarted app from a replay;
- **sequence**: increases with every packet; old or duplicated packets are ignored.

UDP suits this better than HTTP: only the newest command matters, so there's no point setting up
connections or re-sending late packets. Losing one packet is harmless; the next arrives 50 ms later.

### 7. Driving the motors safely: the firmware

The Arduino sketch (`firmware/gesture_motor/gesture_motor.ino`) is built around failing safe:

| Safety feature | Behaviour |
|---|---|
| Arming | ESCs are held at idle while the board boots and connects to Wi-Fi |
| **Watchdog** | No valid packet for 500 ms (app crashed, Wi-Fi dropped, laptop closed) → idle |
| Ramp | UP / DOWN change the throttle gradually (100 µs per second), never in jumps |
| Cap | Throttle can't exceed `US_MAX` (1200 µs by default, a gentle bench-test level) |
| Authentication | Packets with the wrong token or a stale sequence number are ignored |
| Clean exit | Quitting the PC app sends STOP several times |

Wi-Fi credentials and the token live in `secrets.h`, which is never committed (see
`secrets.example.h`).

The firmware logic is mirrored in Python (`motor_sim.py`). This "digital twin" lets the unit
tests check the safety behaviour without hardware, and it shows a simulated throttle bar on screen
during dry runs.

---

## Recording data and evaluating honestly

### Guided recording

`gesturectl record` walks you through the gestures in random order: get ready, then about 6 seconds
of recording each. One run is one **session**, under fixed conditions (place, lighting, person).
Every session contains **every** gesture, so no gesture is tied to a particular room or time of
day.

### Evaluation

`gesturectl evaluate` reports two kinds of split side by side:

- **Random-frame split**: frames are shuffled into train and test sets. Consecutive video frames
  are nearly identical, so the test set contains near-copies of training frames. This flatters
  the model.
- **Leave-one-session-out**: the model is always tested on a whole session it has never seen.
  This matches real use, and it is the number that counts.

The gap between the two shows how much a careless split would overstate performance. It also
reports per-gesture precision and recall, a confusion matrix, and two control-level metrics
obtained by replaying each held-out session through the controller:

- **Wrong-action rate**: how often the motors would be told to do something the user didn't ask
  for. This is the safety-relevant error;
- **Hit rate**: how often the requested action was actually sent (how responsive control feels).

`gesturectl bench` measures the time taken by each stage (camera, landmarks, classifier).

---

## Getting started

Requirements: Python 3.11+, a webcam. The hardware is optional.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install -e ".[dev]"
gesturectl download-model       # MediaPipe hand model, ~8 MB
```

Try it immediately with the rule baseline. No data and no hardware are needed: it's a dry run with
a simulated motor on screen.

```bash
gesturectl run --baseline
```

Train your own model:

```bash
gesturectl record --person me --place desk --lighting day
gesturectl record --person me --place sofa --lighting lamp     # vary the conditions
gesturectl record --person me --place desk --lighting evening
gesturectl record --person friend --place kitchen --lighting day

gesturectl evaluate     # MLP vs rules, random-frame vs leave-one-session-out
gesturectl train        # trains on all sessions -> models/gesture_mlp.npz
gesturectl run          # live dry run with the trained model
gesturectl bench        # latency per stage
```

### With the hardware

Parts: Arduino UNO R4 WiFi, brushless motors with ESCs (signal wires on pins 9, 10, 11), a
battery suited to the motors.

1. Copy `firmware/gesture_motor/secrets.example.h` to `secrets.h` and fill in your Wi-Fi network
   and a token (`python -c "import secrets; print(secrets.token_hex(8))"`).
2. Flash `gesture_motor.ino` with the Arduino IDE and note the IP address shown in the
   Serial Monitor.
3. **Remove the propellers** for the first tests.
4. Run the app against the board:
   ```bash
   set GESTURECTL_TOKEN=<your token>
   gesturectl run --host <arduino-ip>
   ```

---

## Project structure

```
src/gesturectl/
├── landmarks.py    MediaPipe hand tracking
├── features.py     normalisation and the 65 features
├── baseline.py     rule-based classifier
├── model.py        NumPy neural network
├── dataset.py      recording sessions, augmentation
├── evaluate.py     cross-validation, metrics, controller replay
├── controller.py   smoothing and debouncing → commands
├── transport.py    UDP packet protocol
├── motor_sim.py    Python twin of the firmware
├── overlay.py      on-screen drawing
├── config.py       gestures, commands, paths, constants
└── cli.py          record / train / evaluate / run / bench commands
firmware/gesture_motor/
├── gesture_motor.ino      Arduino sketch
└── secrets.example.h      template for Wi-Fi credentials and token
tests/                     unit tests, with a synthetic hand generator
```

Run the tests with `pytest`. They cover the features, both classifiers, the evaluation code, the
controller, the packet protocol, and the firmware's safety logic through its Python twin. They use
synthetic hands, so they need no camera.

---

## Background: lessons from the first version

The first version classified raw 1280×720 webcam photos with a large CNN. It reported **93%
validation accuracy but scored 41% on new photos**. The causes shaped this design:

- each gesture had been photographed on a different day, so the model could learn the lighting
  instead of the hand → **every session now contains every gesture**;
- near-identical burst photos ended up in both training and validation → **leave-one-session-out
  evaluation**;
- the pixel-based model was 1.4 GB and sensitive to the background → **21 landmarks and a tiny
  network**;
- motors were driven by unauthenticated HTTP requests with no failsafe → **token, sequence
  numbers and a watchdog**.
