import numpy as np
import pytest

from gesturectl.controller import ControllerConfig, GestureController
from gesturectl.evaluate import command_metrics
from gesturectl.motor_sim import MotorParams, MotorSim
from gesturectl.transport import DryRunTransport, UdpTransport, format_packet, parse_packet

UP, DOWN, STOP, NONE = np.eye(4)


# --------------------------------------------------------------------------- controller


def test_up_needs_confirmation_frames():
    ctrl = GestureController(ControllerConfig(confirm_frames=3))
    assert [ctrl.update(UP) for _ in range(3)] == ["HOLD", "HOLD", "UP"]


def test_stop_acts_on_first_confident_frame():
    ctrl = GestureController(ControllerConfig(alpha=1.0))
    for _ in range(3):
        ctrl.update(UP)
    assert ctrl.update(STOP) == "STOP"


def test_single_flicker_does_not_change_command():
    ctrl = GestureController(ControllerConfig(alpha=1.0, confirm_frames=3))
    for _ in range(3):
        ctrl.update(UP)
    assert ctrl.update(DOWN) == "UP"
    assert ctrl.update(UP) == "UP"


def test_low_confidence_means_hold():
    ctrl = GestureController(ControllerConfig(threshold=0.75, confirm_frames=1))
    assert ctrl.update(np.array([0.5, 0.3, 0.1, 0.1])) == "HOLD"


@pytest.mark.parametrize(("no_hand", "expected"), [("stop", "STOP"), ("hold", "HOLD")])
def test_no_hand_behaviour(no_hand, expected):
    ctrl = GestureController(ControllerConfig(alpha=1.0, confirm_frames=1, no_hand=no_hand))
    ctrl.update(UP)
    assert ctrl.update(None) == expected
    assert ctrl.update(UP) == "UP"  # smoothing state was reset, so no stale history


def test_command_metrics():
    m = command_metrics(["up", "up", "none", "stop"], ["HOLD", "UP", "DOWN", "STOP"])
    assert m["wrong_action_rate"] == pytest.approx(0.25)  # DOWN when nothing was asked
    assert m["hit_rate"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- packets


def test_packet_roundtrip_and_validation():
    pkt = format_packet("tok", "ab12", 7, "UP")
    assert parse_packet(pkt) == ("tok", "ab12", 7, "UP")
    assert parse_packet("tok ab12 7 FLY") is None
    assert parse_packet("tok ab12 x UP") is None
    with pytest.raises(ValueError):
        format_packet("tok", "ab12", 1, "LAUNCH")


def test_udp_transport_requires_token():
    with pytest.raises(ValueError):
        UdpTransport("127.0.0.1", 4210, token="")


def test_dry_run_sequence_increases():
    t = DryRunTransport("tok")
    first, second = parse_packet(t.send("UP")), parse_packet(t.send("STOP"))
    assert second[2] == first[2] + 1 and first[1] == second[1]


# --------------------------------------------------------------------------- firmware twin


def sim_with_sender(token="tok"):
    return MotorSim(token, MotorParams(us_min=1000, us_max=1200, ramp_us_per_s=100,
                                       watchdog_ms=500)), DryRunTransport(token)


def test_up_ramps_and_is_capped():
    sim, tx = sim_with_sender()
    for ms in range(0, 5000, 50):
        sim.receive(tx.send("UP"), ms)
        sim.tick(ms)
    assert sim.throttle_us == 1200


def test_hold_keeps_and_stop_cuts():
    sim, tx = sim_with_sender()
    for ms in range(0, 1050, 50):
        sim.receive(tx.send("UP"), ms)
        sim.tick(ms)
    level = sim.throttle_us
    assert 1090 < level < 1110
    sim.receive(tx.send("HOLD"), 1100)
    assert sim.tick(1100) == pytest.approx(level, abs=6)
    sim.receive(tx.send("STOP"), 1150)
    assert sim.tick(1150) == 1000


def test_watchdog_stops_motors_when_packets_stop():
    sim, tx = sim_with_sender()
    for ms in range(0, 1000, 50):
        sim.receive(tx.send("UP"), ms)
        sim.tick(ms)
    assert sim.throttle_us > 1000
    assert sim.tick(1400) > 1000  # 450 ms of silence: still within the watchdog window
    assert sim.tick(1600) == 1000  # 650 ms: tripped
    assert sim.watchdog_tripped


def test_wrong_token_is_ignored():
    sim, _ = sim_with_sender("secret")
    attacker = DryRunTransport("guess")
    assert not sim.receive(attacker.send("UP"), 0)
    assert sim.command == "STOP"


def test_replayed_packet_is_ignored_but_restarted_sender_is_accepted():
    sim, tx = sim_with_sender()
    old = tx.send("UP")
    assert sim.receive(old, 0)
    assert sim.receive(tx.send("STOP"), 50)
    assert not sim.receive(old, 100)  # replay of an older UP

    restarted = DryRunTransport("tok")  # new nonce, sequence starts again at 1
    assert sim.receive(restarted.send("UP"), 150)
