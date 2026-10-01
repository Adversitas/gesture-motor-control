"""Evaluation.

Two splits are reported side by side on purpose:
- "frame": random K-fold over all frames. Neighbouring video frames are near-duplicates,
  so test frames have twins in the training set. This is the split that made the original
  project look 93% accurate while failing on new photos.
- "session": leave-one-session-out. The model is always tested on a sitting (time, place,
  lighting) it has never seen, which is what happens when someone actually uses it.

Beyond frame accuracy, `command_metrics` replays each held-out session through the
controller and counts what the motors would actually have been told to do.
"""

from dataclasses import dataclass, field

import numpy as np

from . import baseline
from .config import COMMANDS, LABELS
from .controller import ControllerConfig, GestureController
from .dataset import Session, augment, to_features
from .model import MLPClassifier

# --------------------------------------------------------------------------- metrics


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=LABELS) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(labels))
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = cm.sum(axis=1) > 0
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {
            lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lab, p, r, f, s in zip(labels, precision, recall, f1, cm.sum(axis=1), strict=True)
        },
        "confusion_matrix": cm.tolist(),
    }


def command_metrics(true_labels: list[str], commands: list[str]) -> dict:
    """What the motors would have been told, compared with what the user meant.

    wrong_action_rate: share of frames sending an action (UP/DOWN/STOP) that the user
        did not ask for. These are the errors that move motors unexpectedly.
    hit_rate: for frames where the user asked for an action, share where it was sent.
        Misses here are mostly the controller's deliberate confirmation delay.
    """
    actions = {"UP", "DOWN", "STOP"}
    wanted = [COMMANDS[lab] for lab in true_labels]
    sent_action = [c in actions for c in commands]
    wrong = [s and c != w for s, c, w in zip(sent_action, commands, wanted, strict=True)]
    asked = [w in actions for w in wanted]
    hits = [a and c == w for a, c, w in zip(asked, commands, wanted, strict=True)]
    return {
        "wrong_action_rate": float(np.mean(wrong)),
        "hit_rate": float(sum(hits) / max(sum(asked), 1)),
    }


# --------------------------------------------------------------------------- training


def train_model(normalized: np.ndarray, y: np.ndarray, seed: int = 0, **hparams) -> MLPClassifier:
    rng = np.random.default_rng(seed)
    aug = augment(normalized, rng)
    y_aug = np.tile(y, len(aug) // len(y))
    model = MLPClassifier(LABELS, seed=seed, **hparams)
    model.fit(to_features(aug), y_aug)
    return model


# --------------------------------------------------------------------------- cross-validation


@dataclass
class SplitResult:
    split: str
    n_folds: int
    model: dict
    baseline: dict
    commands_model: dict = field(default_factory=dict)
    commands_baseline: dict = field(default_factory=dict)


def _baseline_predict(normalized: np.ndarray) -> np.ndarray:
    return np.array([LABELS.index(baseline.classify(p)) for p in normalized])


def _replay(probs: np.ndarray, config: ControllerConfig) -> list[str]:
    ctrl = GestureController(config)
    return [ctrl.update(p) for p in probs]


def cross_validate(
    sessions: list[Session],
    split: str = "session",
    k: int = 5,
    seed: int = 0,
    controller: ControllerConfig | None = None,
) -> SplitResult:
    controller = controller or ControllerConfig()
    norm = [s.normalized() for s in sessions]
    ys = [s.label_indices() for s in sessions]
    X_all, y_all = np.concatenate(norm), np.concatenate(ys)
    group = np.concatenate([np.full(len(y), i) for i, y in enumerate(ys)])

    if split == "session":
        if len(sessions) < 2:
            raise ValueError("Leave-one-session-out needs at least 2 sessions")
        folds = [np.flatnonzero(group == g) for g in range(len(sessions))]
    elif split == "frame":
        order = np.random.default_rng(seed).permutation(len(y_all))
        folds = np.array_split(order, k)
    else:
        raise ValueError(split)

    pred_m = np.empty_like(y_all)
    pred_b = np.empty_like(y_all)
    commands_m: list[str] = []
    commands_b: list[str] = []
    truth: list[str] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(y_all)), test_idx)
        model = train_model(X_all[train_idx], y_all[train_idx], seed=seed)
        probs = model.predict_proba(to_features(X_all[test_idx]))
        pred_m[test_idx] = probs.argmax(axis=1)
        pred_b[test_idx] = _baseline_predict(X_all[test_idx])

        if split == "session":  # test frames are one session, in recording order
            one_hot_b = np.eye(len(LABELS))[pred_b[test_idx]]
            commands_m += _replay(probs, controller)
            commands_b += _replay(one_hot_b, controller)
            truth += [LABELS[i] for i in y_all[test_idx]]

    result = SplitResult(
        split=split,
        n_folds=len(folds),
        model=classification_metrics(y_all, pred_m),
        baseline=classification_metrics(y_all, pred_b),
    )
    if truth:
        result.commands_model = command_metrics(truth, commands_m)
        result.commands_baseline = command_metrics(truth, commands_b)
    return result


def format_confusion(cm: list[list[int]], labels=LABELS) -> str:
    width = max(6, *(len(lab) for lab in labels)) + 2
    lines = ["true \\ pred".ljust(12) + "".join(lab.rjust(width) for lab in labels)]
    for lab, row in zip(labels, cm, strict=True):
        lines.append(lab.ljust(12) + "".join(str(v).rjust(width) for v in row))
    return "\n".join(lines)
