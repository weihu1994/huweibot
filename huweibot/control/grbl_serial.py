from __future__ import annotations

import time
from typing import Any

try:
    import serial
except Exception:  # pragma: no cover
    serial = None  # type: ignore


class GRBLSerial:
    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 2.0):
        self.port = str(port)
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self._ser: Any | None = None

    def __enter__(self) -> "GRBLSerial":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    def open(self) -> None:
        if self.is_open:
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed; cannot open GRBL serial")
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as exc:
            raise RuntimeError(f"Failed to open serial port: port={self.port}, baud={self.baud}, error={exc}") from exc

        time.sleep(2.0)
        self._write("\r\n\r\n")
        self.flush_input()
        self.send_gcode("G91", wait_ok=True)

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def flush_input(self) -> None:
        if not self.is_open:
            return
        assert self._ser is not None
        try:
            self._ser.reset_input_buffer()
        except Exception:
            try:
                self._ser.flushInput()
            except Exception:
                pass

    def _write(self, text: str) -> None:
        if not self.is_open:
            raise RuntimeError("GRBL serial not open")
        assert self._ser is not None
        payload = text.encode("utf-8", errors="ignore")
        self._ser.write(payload)
        self._ser.flush()

    def send_gcode(self, cmd: str, wait_ok: bool = True) -> list[str]:
        command = str(cmd).strip()
        if not command:
            raise ValueError("empty gcode command")
        self._write(command + "\n")
        if not wait_ok:
            return []

        if not self.is_open:
            raise RuntimeError("GRBL serial not open")
        assert self._ser is not None

        deadline = time.time() + self.timeout_s
        lines: list[str] = []
        while time.time() < deadline:
            raw = self._ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            lines.append(line)
            low = line.lower()
            if low == "ok":
                return lines
            if low.startswith("error:"):
                raise RuntimeError(f"GRBL error for '{command}': {line}")
        raise TimeoutError(f"Timeout waiting GRBL response for '{command}'")

    # Compatibility shim used by click-mode same_serial in hardware controller.
    def send(self, command: str) -> str:
        lines = self.send_gcode(command, wait_ok=True)
        return "\n".join(lines)
