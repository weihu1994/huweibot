from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

def _model_dump_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def _model_validate_json(model_cls: type[BaseModel], payload: str) -> BaseModel:
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(payload)
    return model_cls.parse_raw(payload)


class UIElement(BaseModel):
    stable_id: str | None = None
    raw_id: int
    role: Literal["text", "button", "input", "icon", "key", "toggle", "unknown"] = "unknown"
    text: str | None = None
    label: str | None = None
    bbox: tuple[float, float, float, float]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["ocr", "heuristic", "vlm"] = "heuristic"
    clickability_hint: Literal["high", "low"] | None = None

    def to_json(self) -> str:
        return _model_dump_json(self)

    @classmethod
    def from_json(cls, payload: str) -> "UIElement":
        return _model_validate_json(cls, payload)


class Observation(BaseModel):
    timestamp: float = Field(default_factory=lambda: time.time())
    screen_w: int
    screen_h: int
    cursor_xy: tuple[float, float] | None = None
    cursor_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    cursor_type: str = "unknown"
    elements: list[UIElement] = Field(default_factory=list)
    app_hint: str | None = None
    keyboard_mode: bool = False
    keyboard_roi: tuple[float, float, float, float] | None = None
    screen_hash: str | None = None
    ui_change_score: float = 0.0
    phone_screen_bbox: tuple[float, float, float, float] | None = None
    phone_grid_w: int | None = None
    phone_grid_h: int | None = None
    touch_pen_xy: tuple[float, float] | None = None
    touch_pen_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    distance_mm: float | None = None
    device_mode: Literal["pc", "phone"] | None = None

    def to_json(self) -> str:
        return _model_dump_json(self)

    @classmethod
    def from_json(cls, payload: str) -> "Observation":
        return _model_validate_json(cls, payload)
