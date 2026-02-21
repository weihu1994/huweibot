from __future__ import annotations

from typing import Any, Literal

from huweibot._pydantic_compat import BaseModel, Field, field_validator, is_pydantic_v2, model_validator
from huweibot.shared.schemas import ElementRef, VerifyRule


from huweibot.core.actions import (
    Action as NextAction,
    ActionBase as NextActionBase,
    AssertTextAction,
    ClickAction,
    ClickAtAction,
    ClickElementAction,
    Coord as ActionCoord,
    DoneAction,
    DragAction,
    HotkeyAction,
    MoveRelAction,
    MoveToAction,
    OpenAppAction,
    OpenOSKAction,
    ScrollAction,
    TouchPressAction,
    TouchSwipeAction,
    TouchTapAction,
    TypeTextAction,
    WaitAction,
    action_from_json,
    action_to_json,
)


def _coerce_next_action(value: Any) -> NextAction:
    data = value
    if data is None:
        raise ValueError("action is required")
    if isinstance(data, dict) and "payload" in data and isinstance(data.get("payload"), dict):
        merged = dict(data.get("payload") or {})
        merged["type"] = data.get("type")
        for k in ("repeat", "action_ttl_ms", "continue_until"):
            if k in data and k not in merged:
                merged[k] = data.get(k)
        data = merged
    elif hasattr(data, "model_dump"):
        data = data.model_dump(by_alias=True, exclude_none=True)
    elif hasattr(data, "dict"):
        data = data.dict(by_alias=True, exclude_none=True)
    return action_from_json(data)


class NextStep(BaseModel):
    action: NextAction
    verify: VerifyRule = Field(default_factory=VerifyRule)
    repeat: int | None = Field(default=None, ge=1, le=6)
    continue_until: Literal["verify_ok", "ui_change", "cursor_lost", "timeout"] | None = None
    action_ttl_ms: int | None = Field(default=None, ge=0, le=15000)
    obs_mode: Literal["full", "delta"] = "delta"
    reasoning_hint: str | None = None

    @field_validator("reasoning_hint")
    def _limit_reasoning_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if len(value) > 120:
            return value[:120]
        return value

    @model_validator(mode="before")
    def _coerce_action_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values
        if "action" in values:
            values = dict(values)
            values["action"] = _coerce_next_action(values.get("action"))
        return values

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _sync_action_common_fields(self):
            action = self.action
            if action is None:
                return self
            if self.repeat is not None:
                setattr(action, "repeat", int(self.repeat))
            if self.action_ttl_ms is not None:
                setattr(action, "action_ttl_ms", int(self.action_ttl_ms))
            if self.continue_until is not None:
                setattr(action, "continue_until", self.continue_until)
            return self
    else:
        @model_validator(mode="after")
        def _sync_action_common_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
            action = values.get("action")
            if action is not None and not hasattr(action, "type"):
                action = _coerce_next_action(action)
                values["action"] = action
            if action is None:
                return values
            if values.get("repeat") is not None:
                setattr(action, "repeat", int(values["repeat"]))
            if values.get("action_ttl_ms") is not None:
                setattr(action, "action_ttl_ms", int(values["action_ttl_ms"]))
            if values.get("continue_until") is not None:
                setattr(action, "continue_until", values["continue_until"])
            return values

    @classmethod
    def done(cls, reason: str = "done") -> "NextStep":
        return cls(
            action=action_from_json({"type": "DONE", "reason": reason}),
            verify=VerifyRule(type="NONE"),
            obs_mode="delta",
        )


class PlanRequest(BaseModel):
    task: str
    observation_summary: str = ""


VerifyRule.update_forward_refs()


def _schema_touch_smoke() -> None:
    samples = [
        action_from_json({"type": "MOVE_TO", "target": {"coord_type": "screen_px", "x": 640, "y": 360}}),
        action_from_json({"type": "TOUCH_TAP", "coord": {"coord_type": "grid", "x": 10, "y": 12}, "press_ms": 80}),
    ]
    for item in samples:
        payload = action_to_json(action_from_json(action_to_json(item)), as_dict=True)
        action_from_json(payload)
    print("OK")


if __name__ == "__main__":
    _schema_touch_smoke()
