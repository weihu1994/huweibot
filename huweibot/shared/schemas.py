from __future__ import annotations

from typing import Any, Literal

from huweibot._pydantic_compat import BaseModel, Field, is_pydantic_v2, model_validator


def _normalize_verify_values(values: dict[str, Any]) -> dict[str, Any]:
    if values.get("target") is None and values.get("target_ref") is not None:
        values["target"] = values["target_ref"]
    if values.get("target_ref") is None and values.get("target") is not None:
        values["target_ref"] = values["target"]

    rule_type = values.get("type", "NONE")
    if rule_type == "NONE":
        values["text"] = None
        values["target"] = None
        values["target_ref"] = None
        values["rules"] = []
        return values

    if rule_type in {"TEXT_PRESENT", "TEXT_CHANGED"}:
        if not (values.get("text") or "").strip():
            raise ValueError(f"{rule_type} requires text")
        return values

    if rule_type == "ELEMENT_CHANGED":
        if values.get("target") is None and values.get("target_ref") is None:
            raise ValueError("ELEMENT_CHANGED requires target ElementRef")
        return values

    if rule_type == "INPUT_VERIFIED":
        return values

    if rule_type == "ANY_OF":
        rules = values.get("rules", [])
        if not isinstance(rules, list) or len(rules) == 0:
            raise ValueError("ANY_OF requires non-empty rules")
        return values

    raise ValueError(f"unsupported VerifyRule type: {rule_type}")


class ElementRef(BaseModel):
    by: Literal["id", "query"] = "id"
    id: str | None = None
    query: str | None = None
    match: Literal["best", "all"] = "best"

    @model_validator(mode="before")
    def _compat_value_field(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values
        if "value" in values and ("id" not in values and "query" not in values):
            if values.get("by", "id") == "id":
                values["id"] = values["value"]
            else:
                values["query"] = values["value"]
        return values

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_by_target(self):
            by = self.by
            ref_id = (self.id or "").strip() if self.id is not None else ""
            query = (self.query or "").strip() if self.query is not None else ""
            if by == "id":
                if not ref_id:
                    raise ValueError("ElementRef.by='id' requires non-empty id")
                self.id = ref_id
                self.query = None
            elif by == "query":
                if not query:
                    raise ValueError("ElementRef.by='query' requires non-empty query")
                self.query = query
                self.id = None
            else:
                raise ValueError("ElementRef.by must be id|query")
            return self
    else:
        @model_validator(mode="after")
        def _validate_by_target(cls, values: dict[str, Any]) -> dict[str, Any]:
            by = values.get("by")
            ref_id = (values.get("id") or "").strip() if values.get("id") is not None else ""
            query = (values.get("query") or "").strip() if values.get("query") is not None else ""
            if by == "id":
                if not ref_id:
                    raise ValueError("ElementRef.by='id' requires non-empty id")
                values["id"] = ref_id
                values["query"] = None
            elif by == "query":
                if not query:
                    raise ValueError("ElementRef.by='query' requires non-empty query")
                values["query"] = query
                values["id"] = None
            else:
                raise ValueError("ElementRef.by must be id|query")
            return values

    @property
    def value(self) -> str:
        return self.id if self.by == "id" else (self.query or "")


class VerifyRule(BaseModel):
    type: Literal["TEXT_PRESENT", "TEXT_CHANGED", "ELEMENT_CHANGED", "ANY_OF", "NONE", "INPUT_VERIFIED"] = "NONE"
    text: str | None = None
    roi: tuple[int, int, int, int] | None = None
    min_match_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    min_delta: int = Field(default=6, ge=0, le=100)
    target: ElementRef | None = None
    target_ref: ElementRef | None = None
    rules: list["VerifyRule"] = Field(default_factory=list)
    timeout_ms: int = Field(default=1500, ge=0, le=15000)
    allow_vlm: bool = False

    @model_validator(mode="before")
    def _compat_mode_field(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values
        mode = values.get("mode")
        if mode and not values.get("type"):
            values["type"] = mode
        if values.get("target") is None and values.get("target_ref") is not None:
            values["target"] = values.get("target_ref")
        if values.get("target_ref") is None and values.get("target") is not None:
            values["target_ref"] = values.get("target")
        return values

    if is_pydantic_v2():
        @model_validator(mode="after")
        def _validate_by_type(self):
            values = _normalize_verify_values(
                {
                    "type": self.type,
                    "text": self.text,
                    "roi": self.roi,
                    "min_match_ratio": self.min_match_ratio,
                    "min_delta": self.min_delta,
                    "target": self.target,
                    "target_ref": self.target_ref,
                    "rules": self.rules,
                    "timeout_ms": self.timeout_ms,
                    "allow_vlm": self.allow_vlm,
                }
            )
            self.type = values["type"]
            self.text = values.get("text")
            self.target = values.get("target")
            self.target_ref = values.get("target_ref")
            self.rules = values.get("rules", [])
            return self
    else:
        @model_validator(mode="after")
        def _validate_by_type(cls, values: dict[str, Any]) -> dict[str, Any]:
            return _normalize_verify_values(values)

    @property
    def mode(self) -> str:
        return self.type


VerifyRule.update_forward_refs()

