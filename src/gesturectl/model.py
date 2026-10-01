"""A one-hidden-layer MLP in NumPy.

With 65 landmark features a tiny network is plenty, trains in seconds on CPU, and runs in
microseconds per frame. Written by hand rather than with scikit-learn/PyTorch to keep the
runtime dependencies to NumPy (and because it is short enough to read).
"""

import json
from pathlib import Path

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    def __init__(
        self,
        labels: tuple[str, ...],
        hidden: int = 64,
        lr: float = 3e-3,
        epochs: int = 60,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ):
        self.labels = tuple(labels)
        self.hparams = {
            "hidden": hidden, "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "weight_decay": weight_decay, "seed": seed,
        }  # fmt: skip
        self.params: dict[str, np.ndarray] = {}
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """y holds class indices into self.labels. Returns the training loss per epoch."""
        hp = self.hparams
        rng = np.random.default_rng(hp["seed"])
        X = np.asarray(X, dtype=np.float64)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xs = (X - self.mean) / self.std
        n_in, n_out = X.shape[1], len(self.labels)

        p = {
            "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, hp["hidden"])),
            "b1": np.zeros(hp["hidden"]),
            "W2": rng.normal(0, np.sqrt(1 / hp["hidden"]), (hp["hidden"], n_out)),
            "b2": np.zeros(n_out),
        }
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(val) for k, val in p.items()}
        beta1, beta2, step = 0.9, 0.999, 0
        Y = np.eye(n_out)[y]
        history = []

        for _ in range(hp["epochs"]):
            order = rng.permutation(len(Xs))
            epoch_loss = 0.0
            for start in range(0, len(order), hp["batch_size"]):
                idx = order[start : start + hp["batch_size"]]
                xb, yb = Xs[idx], Y[idx]
                h_pre = xb @ p["W1"] + p["b1"]
                h = np.maximum(h_pre, 0)
                probs = _softmax(h @ p["W2"] + p["b2"])
                epoch_loss += -np.sum(yb * np.log(probs + 1e-12))

                d_logits = (probs - yb) / len(idx)
                grads = {
                    "W2": h.T @ d_logits + hp["weight_decay"] * p["W2"],
                    "b2": d_logits.sum(axis=0),
                }
                d_h = (d_logits @ p["W2"].T) * (h_pre > 0)
                grads["W1"] = xb.T @ d_h + hp["weight_decay"] * p["W1"]
                grads["b1"] = d_h.sum(axis=0)

                step += 1
                for k in p:  # Adam
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                    m_hat = m[k] / (1 - beta1**step)
                    v_hat = v[k] / (1 - beta2**step)
                    p[k] -= hp["lr"] * m_hat / (np.sqrt(v_hat) + 1e-8)
            history.append(epoch_loss / len(Xs))

        self.params = p
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mean) / self.std
        h = np.maximum(X @ self.params["W1"] + self.params["b1"], 0)
        return _softmax(h @ self.params["W2"] + self.params["b2"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"labels": self.labels, "hparams": self.hparams})
        np.savez(path, meta=meta, mean=self.mean, std=self.std, **self.params)

    @classmethod
    def load(cls, path: Path) -> "MLPClassifier":
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        model = cls(tuple(meta["labels"]), **meta["hparams"])
        model.mean, model.std = data["mean"], data["std"]
        model.params = {k: data[k] for k in ("W1", "b1", "W2", "b2")}
        return model
