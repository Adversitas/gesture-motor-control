"""Python twin of firmware/gesture_motor/gesture_motor.ino.

Same packet checks, same ramp, same watchdog, same constants. It lets the unit tests check
the safety logic, and lets `gesturectl run --dry-run` show what the motors would do, with no
hardware. Keep the two in sync when changing either.
"""

from dataclasses import dataclass

from .transport import parse_packet


@dataclass
class MotorParams:
    us_min: int = 1000  # ESC armed / idle pulse width
    us_max: int = 1200  # bench-safety cap; raise deliberately, props off first
    ramp_us_per_s: float = 100.0  # how fast UP / DOWN move the throttle
    watchdog_ms: int = 500  # no valid packet for this long -> motors to idle


class MotorSim:
    def __init__(self, token: str, params: MotorParams | None = None):
        self.p = params or MotorParams()
        self.token = token
        self.throttle_us = float(self.p.us_min)
        self.command = "STOP"
        self.last_packet_ms: float | None = None
        self.last_tick_ms: float | None = None
        self.nonce = ""
        self.last_seq = 0
        self.watchdog_tripped = False

    def receive(self, packet: str, now_ms: float) -> bool:
        parsed = parse_packet(packet)
        if parsed is None:
            return False
        token, nonce, seq, command = parsed
        if token != self.token:
            return False
        if nonce == self.nonce and seq <= self.last_seq:
            return False  # stale or replayed
        self.nonce, self.last_seq = nonce, seq
        self.command = command
        self.last_packet_ms = now_ms
        self.watchdog_tripped = False
        return True

    def tick(self, now_ms: float) -> float:
        dt = 0.0 if self.last_tick_ms is None else (now_ms - self.last_tick_ms) / 1000
        self.last_tick_ms = now_ms

        if self.last_packet_ms is None or now_ms - self.last_packet_ms > self.p.watchdog_ms:
            self.watchdog_tripped = self.last_packet_ms is not None
            self.command = "STOP"

        if self.command == "UP":
            self.throttle_us += self.p.ramp_us_per_s * dt
        elif self.command == "DOWN":
            self.throttle_us -= self.p.ramp_us_per_s * dt
        elif self.command == "STOP":
            self.throttle_us = self.p.us_min
        self.throttle_us = min(max(self.throttle_us, self.p.us_min), self.p.us_max)
        return self.throttle_us

    @property
    def level(self) -> float:
        """Throttle as a fraction of the allowed range, for display."""
        return (self.throttle_us - self.p.us_min) / (self.p.us_max - self.p.us_min)
