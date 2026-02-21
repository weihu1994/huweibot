from __future__ import annotations

import json
from typing import Any, Literal, Union

from typing_extensions import Annotated

from huweibot.shared.schemas import ElementRef, VerifyRule
from huweibot._pydantic_compat import BaseModel, Field, TypeAdapter, field_validator, is_pydantic_v2, model_validator, model_to_dict, model_to_json, parse_obj


CONTINUE_UNTIL = Literal["verify_ok", "ui_change", "cursor_lost", "timeout"]
COORD_TYPE = Literal["screen_px", "grid"]

_MAX_MOVE_PX = 3000.0
_MAX_MOVE_GRID = 200.0
_MAX_SCROLL_DELTA = 2400


class Point(BaseModel):
    x: float
    y: float


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_box(self):
            x1 = float(self.x1)
            y1 = float(self.y1)
            x2 = float(self.x2)
            y2 = float(self.y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must satisfy x2>x1 and y2>y1")
            return self
    else:
        @model_validator(mode="after")
        def _validate_box(cls, values: dict[str, Any]) -> dict[str, Any]:
            x1 = float(values.get("x1", 0.0))
            y1 = float(values.get("y1", 0.0))
            x2 = float(values.get("x2", 0.0))
            y2 = float(values.get("y2", 0.0))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must satisfy x2>x1 and y2>y1")
            return values


class Coord(BaseModel):
    coord_type: COORD_TYPE = "screen_px"
    x: float
    y: float


class ActionBase(BaseModel):
    repeat: int = Field(default=1, ge=1, le=6)
    action_ttl_ms: int | None = Field(default=0, ge=0, le=15000)
    continue_until: CONTINUE_UNTIL | None = None

    @property
    def kind(self) -> str:
        return str(getattr(self, "type", "unknown"))

    @property
    def payload(self) -> dict[str, Any]:
        data = model_to_dict(self, by_alias=True, exclude_none=True)
        common = {"type", "repeat", "action_ttl_ms", "continue_until"}
        return {k: v for k, v in data.items() if k not in common}


class MoveToAction(ActionBase):
    type: Literal["MOVE_TO"] = "MOVE_TO"
    target: Coord


class MoveRelAction(ActionBase):
    type: Literal["MOVE_REL"] = "MOVE_REL"
    delta: Coord

    @field_validator("delta")
    def _validate_delta(cls, value: Coord) -> Coord:
        if value.coord_type == "screen_px":
            if abs(value.x) > _MAX_MOVE_PX or abs(value.y) > _MAX_MOVE_PX:
                raise ValueError(f"MOVE_REL screen_px delta exceeds {_MAX_MOVE_PX}")
        else:
            if abs(value.x) > _MAX_MOVE_GRID or abs(value.y) > _MAX_MOVE_GRID:
                raise ValueError(f"MOVE_REL grid delta exceeds {_MAX_MOVE_GRID}")
        return value


class ClickAtAction(ActionBase):
    type: Literal["CLICK_AT"] = "CLICK_AT"
    coord: Coord
    button: Literal["left", "right"] = "left"
    times: int = Field(default=1, ge=1, le=5)
    interval_ms: int = Field(default=120, ge=50, le=1000)
    press_ms: int | None = Field(default=60, ge=20, le=1500)


class ClickElementAction(ActionBase):
    type: Literal["CLICK_ELEMENT"] = "CLICK_ELEMENT"
    target: ElementRef
    button: Literal["left", "right"] = "left"
    times: int = Field(default=1, ge=1, le=5)
    interval_ms: int = Field(default=120, ge=50, le=1000)
    press_ms: int | None = Field(default=60, ge=20, le=1500)


class ClickAction(ActionBase):
    type: Literal["CLICK"] = "CLICK"
    target: ElementRef
    button: Literal["left", "right"] = "left"
    times: int = Field(default=1, ge=1, le=5)
    interval_ms: int = Field(default=120, ge=50, le=1000)
    press_ms: int | None = Field(default=60, ge=20, le=1500)


class TypeTextAction(ActionBase):
    type: Literal["TYPE_TEXT"] = "TYPE_TEXT"
    text: str
    method: Literal["osk", "direct", "paste"] = "osk"
    implementation_status: Literal["enabled", "disabled"] = "enabled"
    note: str | None = None

    @field_validator("text")
    def _validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("TYPE_TEXT requires non-empty text")
        return value

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_method(self):
            method = self.method
            if method != "osk":
                self.implementation_status = "disabled"
                if not self.note:
                    self.note = "method is not implemented in physical-only mode; osk is required"
            return self
    else:
        @model_validator(mode="after")
        def _validate_method(cls, values: dict[str, Any]) -> dict[str, Any]:
            method = values.get("method", "osk")
            if method != "osk":
                values["implementation_status"] = "disabled"
                if not values.get("note"):
                    values["note"] = "method is not implemented in physical-only mode; osk is required"
            return values


class HotkeyAction(ActionBase):
    type: Literal["HOTKEY"] = "HOTKEY"
    keys: list[str] = Field(default_factory=list)
    disabled: bool = True
    note: str = "hotkey injection is disabled by dual-machine boundary"


class ScrollAction(ActionBase):
    type: Literal["SCROLL"] = "SCROLL"
    delta: int = Field(default=320)
    verify: VerifyRule = Field(default_factory=lambda: VerifyRule(mode="TEXT_CHANGED", text="scroll"))

    @field_validator("delta")
    def _validate_scroll_delta(cls, value: int) -> int:
        if abs(int(value)) > _MAX_SCROLL_DELTA:
            raise ValueError(f"SCROLL delta exceeds {_MAX_SCROLL_DELTA}")
        return int(value)

    @field_validator("verify")
    def _validate_scroll_verify(cls, value: VerifyRule) -> VerifyRule:
        # SCROLL defaults to verifiable behavior; explicit NONE is still allowed.
        if value.mode not in {"NONE", "TEXT_CHANGED", "ELEMENT_CHANGED", "INPUT_VERIFIED"}:
            raise ValueError("invalid verify mode")
        return value


class WaitAction(ActionBase):
    type: Literal["WAIT"] = "WAIT"
    duration_ms: int = Field(default=250, ge=0, le=60000)


class AssertTextAction(ActionBase):
    type: Literal["ASSERT_TEXT"] = "ASSERT_TEXT"
    target: ElementRef | None = None
    expected_text: str
    match_mode: Literal["equals", "contains"] = "contains"

    @field_validator("expected_text")
    def _validate_expected_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ASSERT_TEXT expected_text must be non-empty")
        return value


class DragAction(ActionBase):
    type: Literal["DRAG"] = "DRAG"
    from_coord: Coord = Field(..., alias="from")
    to_coord: Coord = Field(..., alias="to")
    hold_ms: int = Field(default=120, ge=20, le=5000)

    @model_validator(mode="before")
    def _validate_from_to_present(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "from" not in values or "to" not in values:
            raise ValueError("DRAG requires both 'from' and 'to'")
        return values

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_drag_amplitude(self):
            f = self.from_coord
            t = self.to_coord
            if f is None or t is None:
                raise ValueError("DRAG requires both 'from' and 'to'")
            if f.coord_type != t.coord_type:
                raise ValueError("DRAG from/to coord_type must match")

            dx = t.x - f.x
            dy = t.y - f.y
            if f.coord_type == "screen_px":
                if abs(dx) > _MAX_MOVE_PX or abs(dy) > _MAX_MOVE_PX:
                    raise ValueError(f"DRAG screen_px delta exceeds {_MAX_MOVE_PX}")
            else:
                if abs(dx) > _MAX_MOVE_GRID or abs(dy) > _MAX_MOVE_GRID:
                    raise ValueError(f"DRAG grid delta exceeds {_MAX_MOVE_GRID}")
            return self
    else:
        @model_validator(mode="after")
        def _validate_drag_amplitude(cls, values: dict[str, Any]) -> dict[str, Any]:
            f: Coord = values.get("from_coord")
            t: Coord = values.get("to_coord")
            if f is None or t is None:
                return values

            if f.coord_type != t.coord_type:
                raise ValueError("DRAG from/to coord_type must match")

            dx = t.x - f.x
            dy = t.y - f.y
            if f.coord_type == "screen_px":
                if abs(dx) > _MAX_MOVE_PX or abs(dy) > _MAX_MOVE_PX:
                    raise ValueError(f"DRAG screen_px delta exceeds {_MAX_MOVE_PX}")
            else:
                if abs(dx) > _MAX_MOVE_GRID or abs(dy) > _MAX_MOVE_GRID:
                    raise ValueError(f"DRAG grid delta exceeds {_MAX_MOVE_GRID}")
            return values


class TouchTapAction(ActionBase):
    type: Literal["TOUCH_TAP"] = "TOUCH_TAP"
    coord: Coord
    times: int = Field(default=1, ge=1, le=5)
    interval_ms: int = Field(default=120, ge=50, le=1000)
    press_ms: int | None = Field(default=60, ge=20, le=1500)


class TouchSwipeAction(ActionBase):
    type: Literal["TOUCH_SWIPE"] = "TOUCH_SWIPE"
    from_coord: Coord = Field(..., alias="from")
    to_coord: Coord = Field(..., alias="to")
    duration_ms: int = Field(default=250, ge=20, le=60000)

    @model_validator(mode="before")
    def _validate_from_to_present(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "from" not in values or "to" not in values:
            raise ValueError("TOUCH_SWIPE requires both 'from' and 'to'")
        return values

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_touch_swipe(self):
            f = self.from_coord
            t = self.to_coord
            if f.coord_type != t.coord_type:
                raise ValueError("TOUCH_SWIPE from/to coord_type must match")
            return self
    else:
        @model_validator(mode="after")
        def _validate_touch_swipe(cls, values: dict[str, Any]) -> dict[str, Any]:
            f: Coord = values.get("from_coord")
            t: Coord = values.get("to_coord")
            if f is None or t is None:
                return values
            if f.coord_type != t.coord_type:
                raise ValueError("TOUCH_SWIPE from/to coord_type must match")
            return values


class TouchPressAction(ActionBase):
    type: Literal["TOUCH_PRESS"] = "TOUCH_PRESS"
    coord: Coord
    duration_ms: int = Field(default=500, ge=20, le=60000)


class OpenAppAction(ActionBase):
    type: Literal["OPEN_APP"] = "OPEN_APP"
    macro_name: str = "OPEN_APP"


class OpenOSKAction(ActionBase):
    type: Literal["OPEN_OSK"] = "OPEN_OSK"
    macro_name: str = "OPEN_OSK"


class DoneAction(ActionBase):
    type: Literal["DONE"] = "DONE"
    reason: str | None = None


Action = Annotated[
    Union[
        MoveToAction,
        MoveRelAction,
        ClickAction,
        ClickAtAction,
        ClickElementAction,
        TypeTextAction,
        HotkeyAction,
        ScrollAction,
        WaitAction,
        AssertTextAction,
        DragAction,
        TouchTapAction,
        TouchSwipeAction,
        TouchPressAction,
        OpenAppAction,
        OpenOSKAction,
        DoneAction,
    ],
    Field(discriminator="type"),
]


class ActionResult(BaseModel):
    ok: bool = True
    message: str = "placeholder"


def action_to_json(action: Action, *, as_dict: bool = False) -> str | dict[str, Any]:
    payload = model_to_dict(action, by_alias=True, exclude_none=True)
    if as_dict:
        return payload
    return model_to_json(action, by_alias=True, exclude_none=True)


def action_from_json(payload: str | bytes | dict[str, Any]) -> Action:
    data: dict[str, Any]
    if isinstance(payload, dict):
        data = payload
    elif isinstance(payload, bytes):
        data = json.loads(payload.decode("utf-8"))
    else:
        data = json.loads(payload)

    if TypeAdapter is not None:
        return TypeAdapter(Action).validate_python(data)
    return parse_obj(Action, data)


def _action_equal(a: Action, b: Action) -> bool:
    return action_to_json(a, as_dict=True) == action_to_json(b, as_dict=True)


def _self_test() -> None:
    samples: list[Action] = [
        MoveToAction(target=Coord(coord_type="screen_px", x=640, y=360)),
        MoveRelAction(delta=Coord(coord_type="screen_px", x=120, y=-30), repeat=2),
        ClickAction(target=ElementRef(by="query", value="role:button text_contains:ok"), times=1),
        ClickAtAction(coord=Coord(coord_type="screen_px", x=200, y=140), times=2, interval_ms=120),
        ClickElementAction(target=ElementRef(by="id", value="btn_submit"), times=1),
        TypeTextAction(text="hello huweibot"),
        ScrollAction(delta=400),
        DragAction(**{"from": {"coord_type": "screen_px", "x": 100, "y": 100}, "to": {"coord_type": "screen_px", "x": 180, "y": 170}}),
        TouchTapAction(coord=Coord(coord_type="grid", x=100, y=50), press_ms=80),
        TouchSwipeAction(**{"from": {"coord_type": "grid", "x": 20, "y": 20}, "to": {"coord_type": "grid", "x": 120, "y": 70}, "duration_ms": 300}),
        TouchPressAction(coord=Coord(coord_type="screen_px", x=300, y=200), duration_ms=700),
        OpenAppAction(),
        OpenOSKAction(),
        AssertTextAction(target=ElementRef(by="query", value="role:button text_contains:ok"), expected_text="ok"),
        DoneAction(reason="task completed"),
    ]

    for idx, action in enumerate(samples, start=1):
        blob = action_to_json(action)
        parsed = action_from_json(blob)
        if not _action_equal(action, parsed):
            raise AssertionError(f"round-trip mismatch at sample#{idx}")

    # Negative checks
    negative_payloads = [
        {"type": "WAIT", "duration_ms": 70000},
        {"type": "DRAG", "from": {"coord_type": "screen_px", "x": 0, "y": 0}},
        {"type": "CLICK_AT", "coord": {"coord_type": "screen_px", "x": 1, "y": 2}, "times": 7},
    ]
    for bad in negative_payloads:
        try:
            action_from_json(bad)
        except Exception:
            continue
        raise AssertionError(f"invalid payload unexpectedly accepted: {bad}")

    print("OK")


if __name__ == "__main__":
    _self_test()
