from __future__ import annotations

"""Unified pointer abstraction for PC mouse platform and PHONE touch pen platform."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class PointerResult:
    ok: bool
    reason: str
    detail: str | None = None


class PointerController:
    mode: Literal["pc", "phone"] = "pc"

    def move_to_screen_px(self, x: float, y: float) -> PointerResult:
        raise NotImplementedError

    def click(self, button: str = "left") -> PointerResult:
        raise NotImplementedError

    def tap_screen_px(self, x: float, y: float) -> PointerResult:
        raise NotImplementedError

    @property
    def can_hover_check(self) -> bool:
        return self.mode == "pc"


class PCPointerController(PointerController):
    mode: Literal["pc", "phone"] = "pc"

    def __init__(self, *, loop: Any, hardware: Any):
        self.loop = loop
        self.hardware = hardware

    def move_to_screen_px(self, x: float, y: float) -> PointerResult:
        ret = self.loop.move_to(float(x), float(y))
        return PointerResult(ok=bool(ret.get("ok", False)), reason=str(ret.get("reason", "move_failed")), detail=str(ret))

    def click(self, button: str = "left") -> PointerResult:
        try:
            if button == "right":
                status = self.hardware.click_right()
            else:
                status = self.hardware.click_left()
            return PointerResult(ok=True, reason="ok", detail=str(getattr(status, "detail", "")))
        except Exception as exc:
            return PointerResult(ok=False, reason="click_error", detail=str(exc))

    def tap_screen_px(self, x: float, y: float) -> PointerResult:
        moved = self.move_to_screen_px(x, y)
        if not moved.ok:
            return moved
        return self.click("left")


class PhonePointerController(PointerController):
    mode: Literal["pc", "phone"] = "phone"

    def __init__(
        self,
        *,
        touch_controller: Any,
        screen_bbox: tuple[float, float, float, float] | None = None,
    ):
        self.touch_controller = touch_controller
        self.screen_bbox = screen_bbox

    def set_screen_bbox(self, bbox: tuple[float, float, float, float] | None) -> None:
        self.screen_bbox = bbox

    def move_to_screen_px(self, x: float, y: float) -> PointerResult:
        try:
            # Placeholder mapping for Step integration: use screen px as virtual XY units.
            status = self.touch_controller.move_xy_to(float(x), float(y))
            return PointerResult(ok=bool(getattr(status, "ok", True)), reason=str(getattr(status, "reason", "ok")), detail=str(getattr(status, "detail", "")))
        except Exception as exc:
            return PointerResult(ok=False, reason="move_error", detail=str(exc))

    def click(self, button: str = "left") -> PointerResult:
        try:
            status = self.touch_controller.tap()
            return PointerResult(ok=bool(getattr(status, "ok", True)), reason=str(getattr(status, "reason", "ok")), detail=str(getattr(status, "detail", "")))
        except NotImplementedError as exc:
            return PointerResult(ok=False, reason="not_implemented", detail=str(exc))
        except Exception as exc:
            return PointerResult(ok=False, reason="tap_error", detail=str(exc))

    def tap_screen_px(self, x: float, y: float) -> PointerResult:
        moved = self.move_to_screen_px(x, y)
        if not moved.ok:
            return moved
        return self.click("left")
