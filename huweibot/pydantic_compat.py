from __future__ import annotations

from typing import Any, Callable

from huweibot._pydantic_compat import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    is_pydantic_v2,
    model_to_dict,
    model_to_json,
    model_validator,
    parse_obj,
)


def compat_root_validator(*, pre: bool = False) -> Callable[[Callable[..., Any]], Any]:
    mode = "before" if pre else "after"
    return model_validator(mode=mode)


__all__ = [
    "BaseModel",
    "Field",
    "ValidationError",
    "ConfigDict",
    "TypeAdapter",
    "field_validator",
    "model_validator",
    "is_pydantic_v2",
    "model_to_dict",
    "model_to_json",
    "parse_obj",
    "compat_root_validator",
]
