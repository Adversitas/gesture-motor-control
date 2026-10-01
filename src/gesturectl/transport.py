"""PC -> Arduino link.

UDP, one small text packet per command tick: "<token> <nonce> <seq> <CMD>\\n"
- token: shared secret, so other devices on the network can't drive the motors;
- nonce: random per run, so the Arduino can tell a restarted sender from a replay;
- seq: increasing per packet; stale or duplicated packets are ignored.
UDP rather than HTTP because commands are a stream where only the newest one matters:
no connection setup, no retransmission of outdated commands. Losing a packet is harmless;
losing *all* of them trips the firmware watchdog, which stops the motors.
"""

import secrets
import socket
from typing import Protocol

VALID_COMMANDS = {"UP", "DOWN", "STOP", "HOLD"}


def format_packet(token: str, nonce: str, seq: int, command: str) -> str:
    if command not in VALID_COMMANDS:
        raise ValueError(f"Unknown command {command!r}")
    return f"{token} {nonce} {seq} {command}\n"


def parse_packet(text: str) -> tuple[str, str, int, str] | None:
    parts = text.strip().split()
    if len(parts) != 4 or not parts[2].isdigit() or parts[3] not in VALID_COMMANDS:
        return None
    return parts[0], parts[1], int(parts[2]), parts[3]


class Transport(Protocol):
    last_packet: str

    def send(self, command: str) -> str: ...

    def close(self) -> None: ...


class UdpTransport:
    def __init__(self, host: str, port: int, token: str):
        if not token:
            raise ValueError("A shared token is required (set GESTURECTL_TOKEN)")
        self.addr = (host, port)
        self.token = token
        self.nonce = secrets.token_hex(4)
        self.seq = 0
        self.last_packet = ""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: str) -> str:
        self.seq += 1
        self.last_packet = format_packet(self.token, self.nonce, self.seq, command)
        self.sock.sendto(self.last_packet.encode("ascii"), self.addr)
        return self.last_packet

    def close(self) -> None:
        self.sock.close()


class DryRunTransport:
    """Builds the same packets without sending them, for testing with no hardware."""

    def __init__(self, token: str = "dry-run"):
        self.token = token
        self.nonce = secrets.token_hex(4)
        self.seq = 0
        self.last_packet = ""

    def send(self, command: str) -> str:
        self.seq += 1
        self.last_packet = format_packet(self.token, self.nonce, self.seq, command)
        return self.last_packet

    def close(self) -> None:
        pass
