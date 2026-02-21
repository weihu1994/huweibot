from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from typing import Any

try:
    import serial
except Exception:  # pragma: no cover
    serial = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class IRDistanceReader(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def read_mm(self) -> float | None: ...

    def read_distance_mm(self) -> float | None: ...

    def read_samples(self, n: int = 3, interval_ms: int = 20) -> list[float]: ...

    def filtered_mm(self, n: int = 3, interval_ms: int = 20) -> float | None: ...

    def recover_contact(self, retries: int = 1, hold_ms: int = 80) -> list[dict[str, Any]]: ...


class IRReaderBase:
    def read_mm(self) -> float | None:
        raise NotImplementedError

    def read_distance_mm(self) -> float | None:
        return self.read_mm()

    def read_samples(self, n: int = 3, interval_ms: int = 20) -> list[float]:
        samples: list[float] = []
        count = max(1, int(n))
        delay = max(0.0, float(interval_ms) / 1000.0)
        for i in range(count):
            value = self.read_mm()
            if value is not None:
                samples.append(float(value))
            if i < count - 1 and delay > 0:
                time.sleep(delay)
        return samples

    def filtered_mm(self, n: int = 3, interval_ms: int = 20) -> float | None:
        values = self.read_samples(n=n, interval_ms=interval_ms)
        if not values:
            return None
        values = sorted(values)
        return float(values[len(values) // 2])

    def recover_contact(self, retries: int = 1, hold_ms: int = 80) -> list[dict[str, Any]]:
        from huweibot.core.actions import action_from_json, action_to_json

        attempts = max(1, min(2, int(retries)))
        press = max(20, int(hold_ms))
        seq: list[dict[str, Any]] = []
        for _ in range(attempts):
            seq.append(action_to_json(action_from_json({"type": "WAIT", "duration_ms": 80}), as_dict=True))
            seq.append(
                action_to_json(
                    action_from_json(
                        {
                            "type": "TOUCH_PRESS",
                            "coord": {"coord_type": "grid", "x": 0, "y": 0},
                            "duration_ms": press,
                        }
                    ),
                    as_dict=True,
                )
            )
        return seq


@dataclass
class MockIRDistanceReader(IRReaderBase):
    fixed_mm: float = 8.0
    noise_mm: float = 0.0

    def open(self) -> None:
        return

    def close(self) -> None:
        return

    def __enter__(self) -> "MockIRDistanceReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def read_mm(self) -> float | None:
        if self.noise_mm <= 0:
            return float(self.fixed_mm)
        try:
            import random

            return float(self.fixed_mm) + random.uniform(-self.noise_mm, self.noise_mm)
        except Exception:
            return float(self.fixed_mm)


class SerialIRDistanceReader(IRReaderBase):
    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.5):
        self.port = str(port)
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self._ser = None

    def open(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial not available for SerialIRDistanceReader")
        if self._ser is not None:
            return
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)

    def close(self) -> None:
        if self._ser is None:
            return
        try:
            self._ser.close()
        finally:
            self._ser = None

    def __enter__(self) -> "SerialIRDistanceReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @staticmethod
    def _parse_distance_line(line: str) -> float | None:
        m = re.search(r"[-+]?\d*\.?\d+", line)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    def read_mm(self) -> float | None:
        if self._ser is None:
            self.open()
        if self._ser is None:
            return None
        try:
            raw = self._ser.readline()
            if not raw:
                return None
            line = raw.decode("utf-8", errors="ignore").strip()
            value = self._parse_distance_line(line)
            if value is None:
                LOGGER.warning("IR parse failed: %r", line)
            return value
        except Exception as exc:
            LOGGER.warning("IR serial read failed: %s", exc)
            return None


def save_ir_calibration(path: str, contact_threshold_mm: float) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"contact_threshold_mm": float(contact_threshold_mm), "saved_at": time.time()}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ir_calibration(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("contact_threshold_mm")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
