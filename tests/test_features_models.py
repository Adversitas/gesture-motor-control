import numpy as np
import pytest
from synthetic import GESTURES, make_hand, make_session, sample, to_image_coords

from gesturectl import baseline
from gesturectl.config import LABELS
from gesturectl.dataset import Session, augment, to_features
from gesturectl.evaluate import classification_metrics, cross_validate, train_model
from gesturectl.features import FEATURE_DIM, featurize, normalize
from gesturectl.model import MLPClassifier


def test_normalize_inverts_image_placement():
    p = make_hand((True, False, False, False))
    img, hand = to_image_coords(p, center=(0.3, 0.7), size=0.2, aspect=16 / 9)
    np.testing.assert_allclose(normalize(img, hand, 16 / 9), p, atol=1e-5)


def test_features_ignore_position_and_size_but_not_orientation():
    p = make_hand(**GESTURES["up"])
    a = featurize(*to_image_coords(p, center=(0.2, 0.3), size=0.1), aspect=4 / 3)
    b = featurize(*to_image_coords(p, center=(0.7, 0.6), size=0.3), aspect=4 / 3)
    np.testing.assert_allclose(a, b, atol=1e-4)
    assert a.shape == (FEATURE_DIM,)

    down = featurize(*to_image_coords(make_hand(**GESTURES["down"])), aspect=4 / 3)
    assert np.abs(a - down).max() > 0.5


def test_left_hand_is_mirrored_onto_right():
    p = make_hand(**GESTURES["stop"])
    right = featurize(*to_image_coords(p, left=False), aspect=4 / 3)
    left = featurize(*to_image_coords(p, left=True), aspect=4 / 3)
    np.testing.assert_allclose(right, left, atol=1e-5)


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("tilt", [-20, 0, 20])
def test_rule_baseline_on_clean_hands(label, tilt):
    spec = dict(GESTURES[label])
    spec["rotation_deg"] += tilt
    assert baseline.classify(make_hand(**spec)) == label


def test_augment_keeps_originals_and_adds_copies():
    rng = np.random.default_rng(0)
    x = np.stack([make_hand(**GESTURES["up"])] * 5)
    out = augment(x, rng, copies=3)
    assert out.shape == (20, 21, 3)
    np.testing.assert_array_equal(out[:5], x)


def test_mlp_learns_synthetic_gestures_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    train = make_session("a", rng, per_label=60)
    test = make_session("b", rng, per_label=30)
    model = train_model(train.normalized(), train.label_indices())
    pred = model.predict(to_features(test.normalized()))
    assert classification_metrics(test.label_indices(), pred)["accuracy"] > 0.95

    model.save(tmp_path / "m.npz")
    loaded = MLPClassifier.load(tmp_path / "m.npz")
    X = to_features(test.normalized()[:10])
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.labels == LABELS


def test_session_save_load(tmp_path):
    s = make_session("20260101_test", np.random.default_rng(2), per_label=3)
    loaded = Session.load(s.save(tmp_path))
    np.testing.assert_allclose(loaded.points, s.points)
    assert list(loaded.labels) == list(s.labels)
    assert loaded.meta["aspect"] == pytest.approx(4 / 3)


def test_classification_metrics():
    m = classification_metrics(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 3]))
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["per_class"]["down"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["stop"]["recall"] == 0.0
    assert m["confusion_matrix"][2][3] == 1


def test_cross_validation_runs_both_splits():
    rng = np.random.default_rng(3)
    sessions = [make_session(f"s{i}", rng, per_label=25) for i in range(3)]
    by_session = cross_validate(sessions, "session")
    assert by_session.n_folds == 3
    assert by_session.model["accuracy"] > 0.9
    assert set(by_session.commands_model) == {"wrong_action_rate", "hit_rate"}
    by_frame = cross_validate(sessions, "frame", k=3)
    assert by_frame.n_folds == 3 and not by_frame.commands_model


def test_session_split_needs_two_sessions():
    with pytest.raises(ValueError):
        cross_validate([make_session("only", np.random.default_rng(4), per_label=5)], "session")


def test_sample_produces_valid_image_coordinates():
    pts, hand = sample("stop", np.random.default_rng(5))
    assert pts.shape == (21, 3) and hand in {"Left", "Right"}
