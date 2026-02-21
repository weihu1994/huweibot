from __future__ import annotations

"""PHONE mode touch controller (A-side XYZ + touch pen, physically operating the phone screen)."""

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class TouchStatus:
    ok: bool
    reason: str
    detail: str | None = None


class XYZTouchController:
    def __init__(
        self,
        *,
        xy_controller: object | None = None,
        z_axis_enabled: bool = False,
        z_controller: object | None = None,
        ir_enabled: bool = False,
        ir_reader: object | None = None,
        touch_distance_threshold_mm: float = 2.5,
        ir_contact_threshold_mm: float | None = None,
        dry_run: bool = False,
    ):
        self.xy_controller = xy_controller
        self.z_axis_enabled = bool(z_axis_enabled)
        self.z_controller = z_controller
        self.ir_enabled = bool(ir_enabled)
        self.ir_reader = ir_reader
        self.touch_distance_threshold_mm = float(
            touch_distance_threshold_mm if ir_contact_threshold_mm is None else ir_contact_threshold_mm
        )
        self.dry_run = bool(dry_run)
        self.xy_pos_mm: tuple[float, float] = (0.0, 0.0)

    def __enter__(self) -> "XYZTouchController":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        for device in (self.z_controller, self.ir_reader):
            if device is None:
                continue
            try:
                close = getattr(device, "close", None)
                if callable(close):
                    close()
            except Exception:
                LOGGER.warning("failed to close phone-mode device", exc_info=True)

    def move_xy_mm(self, dx: float, dy: float) -> TouchStatus:
        dx_mm = float(dx)
        dy_mm = float(dy)
        if self.xy_controller is None:
            self.xy_pos_mm = (self.xy_pos_mm[0] + dx_mm, self.xy_pos_mm[1] + dy_mm)
            reason = "xy_dry_run" if self.dry_run else "xy_stub"
            return TouchStatus(ok=True, reason=reason, detail=f"dx={dx_mm:.3f},dy={dy_mm:.3f}")

        cmd = f"G0 X{dx_mm:.4f} Y{dy_mm:.4f}"
        try:
            if hasattr(self.xy_controller, "send_gcode"):
                self.xy_controller.send_gcode("G91", wait_ok=True)
                self.xy_controller.send_gcode(cmd, wait_ok=True)
            elif hasattr(self.xy_controller, "move_mm"):
                self.xy_controller.move_mm(dx_mm, dy_mm)
            else:
                raise RuntimeError("xy_controller missing send_gcode/move_mm")
            self.xy_pos_mm = (self.xy_pos_mm[0] + dx_mm, self.xy_pos_mm[1] + dy_mm)
            return TouchStatus(ok=True, reason="ok", detail=cmd)
        except Exception as exc:
            raise RuntimeError(f"phone xy move failed: {exc}") from exc

    def move_xy_to(self, target_mm_x: float, target_mm_y: float) -> TouchStatus:
        tx = float(target_mm_x)
        ty = float(target_mm_y)
        cur_x, cur_y = self.xy_pos_mm
        return self.move_xy_mm(tx - cur_x, ty - cur_y)

    def tap(self, z_press_ms: int = 40) -> TouchStatus:
        press_ms = int(z_press_ms)
        if press_ms < 10 or press_ms > 500:
            raise ValueError("z_press_ms must be in [10,500]")
        distance = self.read_distance_mm()
        if distance is not None and distance > float(self.touch_distance_threshold_mm):
            raise RuntimeError(
                f"distance guard blocked tap: distance_mm={distance:.3f} > threshold={self.touch_distance_threshold_mm:.3f}"
            )
        if self.dry_run:
            return TouchStatus(ok=True, reason="tap_dry_run", detail=f"press_ms={press_ms}")
        if not self.z_axis_enabled:
            raise NotImplementedError("phone tap unavailable: z_axis_enabled=False")

        if self.z_controller is None:
            raise RuntimeError("phone tap unavailable: z-axis controller is not attached")

        cmd = f"TZ {press_ms}"
        try:
            if hasattr(self.z_controller, "send_gcode"):
                self.z_controller.send_gcode(cmd, wait_ok=True)
            elif hasattr(self.z_controller, "write"):
                self.z_controller.write((cmd + "\n").encode("utf-8"))
            else:
                raise RuntimeError("z_controller missing send_gcode/write")
            time.sleep(max(0.0, press_ms / 1000.0))
            return TouchStatus(ok=True, reason="ok", detail=cmd)
        except Exception as exc:
            raise RuntimeError(f"phone tap failed: {exc}") from exc

    def read_distance_mm(self) -> Optional[float]:
        if not self.ir_enabled:
            return None
        if self.ir_reader is None:
            LOGGER.warning("IR enabled but no reader attached")
            return None
        try:
            if hasattr(self.ir_reader, "read_distance_mm"):
                value = self.ir_reader.read_distance_mm()
            elif hasattr(self.ir_reader, "read"):
                value = self.ir_reader.read()
            else:
                return None
            if value is None:
                return None
            return float(value)
        except Exception:
            LOGGER.warning("IR read failed", exc_info=True)
            return None

    def calibrate_contact_threshold(self, samples: list[float] | None = None, margin_mm: float = 0.5) -> float | None:
        values = list(samples or [])
        if not values and self.ir_reader is not None and hasattr(self.ir_reader, "read_samples"):
            try:
                values = list(self.ir_reader.read_samples(n=5, interval_ms=20))
            except Exception:
                values = []
        values = [float(v) for v in values if v is not None]
        if not values:
            return None
        baseline = statistics.median(values)
        return float(max(0.0, baseline + float(margin_mm)))

    def recover_tap_contact(self, retries: int = 1, hold_ms: int = 80) -> list[dict[str, Any]]:
        from huweibot.core.actions import action_from_json, action_to_json

        if self.ir_reader is not None and hasattr(self.ir_reader, "recover_contact"):
            try:
                return list(self.ir_reader.recover_contact(retries=retries, hold_ms=hold_ms))
            except Exception:
                pass
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
