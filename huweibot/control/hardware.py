from __future__ import annotations

"""Hardware runs on machine A and physically manipulates machine B mouse baseplate (never OS/API injection)."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import serial
except Exception:  # pragma: no cover
    serial = None  # type: ignore

from huweibot.core.coords import grid_to_screen_px

LOGGER = logging.getLogger(__name__)

ClickMode = Literal["stub", "same_serial", "separate_serial"]


@dataclass
class HardwareStatus:
    ready: bool = True
    note: str = "ok"
    detail: str | None = None


class HardwareController:
    def __init__(
        self,
        *,
        grbl: object | None = None,
        travel_range_mm: tuple[float, float, float, float] = (0.0, 220.0, 0.0, 140.0),
        feed_rate: float = 3000.0,
        homed_flag_path: str = "artifacts/homed.flag",
        enforce_homed: bool = False,
        click_mode: ClickMode = "stub",
        click_serial: object | None = None,
        click_port: str | None = None,
        click_baud: int = 115200,
        screen_w: int = 1920,
        screen_h: int = 1080,
        grid_size: int = 1000,
    ):
        self.grbl = grbl
        self.travel_range_mm = (
            float(travel_range_mm[0]),
            float(travel_range_mm[1]),
            float(travel_range_mm[2]),
            float(travel_range_mm[3]),
        )
        self.feed_rate = float(feed_rate)
        self.homed_flag_path = str(homed_flag_path)
        self.enforce_homed = bool(enforce_homed)

        self.click_mode = click_mode
        self.click_serial = click_serial
        self.click_port = click_port
        self.click_baud = int(click_baud)
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.grid_size = int(grid_size)
        self._clicker = None

        self.pos_mm = (0.0, 0.0)
        self.is_homed = self._read_homed_flag()

    def __enter__(self) -> "HardwareController":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._clicker is not None:
            try:
                self._clicker.close()
            finally:
                self._clicker = None

    def _read_homed_flag(self) -> bool:
        return Path(self.homed_flag_path).exists()

    def mark_homed(self, x_mm: float = 0.0, y_mm: float = 0.0) -> None:
        self.pos_mm = (float(x_mm), float(y_mm))
        self.is_homed = True
        p = Path(self.homed_flag_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{time.time():.6f}\n{self.pos_mm[0]:.4f},{self.pos_mm[1]:.4f}\n", encoding="utf-8")

    def clear_homed_flag(self) -> None:
        self.is_homed = False
        p = Path(self.homed_flag_path)
        if p.exists():
            p.unlink()

    def _check_soft_limit(self, target_x: float, target_y: float) -> None:
        min_x, max_x, min_y, max_y = self.travel_range_mm
        if not (min_x <= target_x <= max_x and min_y <= target_y <= max_y):
            raise ValueError(
                "soft_limit: target out of range "
                f"(target=({target_x:.3f},{target_y:.3f}), "
                f"range=({min_x:.3f},{max_x:.3f},{min_y:.3f},{max_y:.3f}))"
            )

    def move_mm(self, dx_mm: float, dy_mm: float, feed: float | None = None) -> HardwareStatus:
        if self.enforce_homed and not self.is_homed:
            raise RuntimeError("platform is not homed; run home-zero SOP before movement")

        dx = float(dx_mm)
        dy = float(dy_mm)
        cur_x, cur_y = self.pos_mm
        target_x = cur_x + dx
        target_y = cur_y + dy
        self._check_soft_limit(target_x, target_y)

        if self.grbl is None:
            raise RuntimeError("GRBL controller is not attached to HardwareController")

        use_feed = float(feed) if feed is not None else self.feed_rate
        cmd = f"G0 X{dx:.4f} Y{dy:.4f} F{use_feed:.1f}"
        try:
            if hasattr(self.grbl, "send_gcode"):
                self.grbl.send_gcode("G91", wait_ok=True)
                self.grbl.send_gcode(cmd, wait_ok=True)
            elif hasattr(self.grbl, "send"):
                self.grbl.send("G91")
                self.grbl.send(cmd)
            else:
                raise RuntimeError("grbl object does not provide send_gcode/send")
        except Exception as exc:
            raise RuntimeError(f"failed to send movement gcode: {exc}") from exc

        self.pos_mm = (target_x, target_y)
        return HardwareStatus(
            ready=True,
            note="move_mm",
            detail=f"dx={dx:.4f},dy={dy:.4f},feed={use_feed:.1f},pos=({target_x:.4f},{target_y:.4f})",
        )

    def _validate_click_params(self, times: int, interval_ms: int, press_ms: int) -> None:
        if int(times) < 1 or int(times) > 5:
            raise ValueError("times must be in [1,5]")
        if int(interval_ms) < 50 or int(interval_ms) > 1000:
            raise ValueError("interval_ms must be in [50,1000]")
        if int(press_ms) < 10 or int(press_ms) > 200:
            raise ValueError("press_ms must be in [10,200]")

    def _ensure_clicker(self):
        if self.click_mode != "separate_serial":
            return self.click_serial

        if serial is None:
            raise RuntimeError("pyserial not available; cannot use click_mode='separate_serial'")
        if not self.click_port:
            raise RuntimeError("click_port is required when click_mode='separate_serial'")

        if self._clicker is None:
            try:
                self._clicker = serial.Serial(self.click_port, self.click_baud, timeout=1.0)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to open click serial: port={self.click_port}, baud={self.click_baud}, err={exc}"
                ) from exc
        return self._clicker

    def _send_click_command(self, cmd: str) -> None:
        command = cmd.strip().upper()
        if self.click_mode == "stub":
            LOGGER.info("[stub-click] cmd=%s", command)
            return

        client = self._ensure_clicker()
        if client is None:
            raise RuntimeError("click serial client missing; set click_mode=stub or provide click serial")

        if self.click_mode == "same_serial":
            if hasattr(client, "send_gcode"):
                resp = client.send_gcode(command, wait_ok=True)
            elif hasattr(client, "send"):
                resp = client.send(command)
            else:
                raise RuntimeError("click_serial does not support send_gcode/send")
            text = "" if resp is None else str(resp)
            if "error" in text.lower():
                raise RuntimeError(f"click command failed: {text}")
            return

        deadline = time.time() + 2.0
        try:
            client.reset_input_buffer()
        except Exception:
            pass
        client.write((command + "\n").encode("utf-8"))
        client.flush()
        while time.time() < deadline:
            raw = client.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip().lower()
            if not line:
                continue
            if line.startswith("ok"):
                return
            if line.startswith("error"):
                raise RuntimeError(f"click command failed: {line}")
        raise TimeoutError(f"click command timeout waiting for ack: {command}")

    def click_left(self, times: int = 1, interval_ms: int = 120, press_ms: int = 40) -> HardwareStatus:
        self._validate_click_params(times, interval_ms, press_ms)
        for i in range(int(times)):
            self._send_click_command("CL")
            if i < int(times) - 1:
                time.sleep(float(interval_ms) / 1000.0)
        return HardwareStatus(ready=True, note="click_left", detail=f"times={times},press_ms={press_ms}")

    def click_right(self, times: int = 1, interval_ms: int = 120, press_ms: int = 40) -> HardwareStatus:
        self._validate_click_params(times, interval_ms, press_ms)
        for i in range(int(times)):
            self._send_click_command("CR")
            if i < int(times) - 1:
                time.sleep(float(interval_ms) / 1000.0)
        return HardwareStatus(ready=True, note="click_right", detail=f"times={times},press_ms={press_ms}")

    def click_at(
        self,
        x: int,
        y: int,
        coord_type: Literal["screen_px", "grid"] = "screen_px",
        button: Literal["left", "right"] = "left",
        times: int = 1,
        interval_ms: int = 120,
        press_ms: int = 40,
    ) -> HardwareStatus:
        self._validate_click_params(times, interval_ms, press_ms)
        if coord_type == "grid":
            sx, sy = grid_to_screen_px(int(x), int(y), self.screen_w, self.screen_h, self.grid_size)
        else:
            sx, sy = int(round(x)), int(round(y))
        if button == "left":
            status = self.click_left(times=times, interval_ms=interval_ms, press_ms=press_ms)
        elif button == "right":
            status = self.click_right(times=times, interval_ms=interval_ms, press_ms=press_ms)
        else:
            raise ValueError("button must be 'left' or 'right'")
        status.detail = f"coord_type={coord_type},x={sx},y={sy},button={button}"
        return status

    def click(self, button: str = "left") -> HardwareStatus:
        if button == "left":
            return self.click_left()
        if button == "right":
            return self.click_right()
        raise ValueError("button must be left/right")


class RobotMousePlatform(HardwareController):
    """Backward-compatible alias."""

    pass
