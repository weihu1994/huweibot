from __future__ import annotations

import inspect
from typing import Any, Callable

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        TypeAdapter,
        ValidationError,
        field_validator as _field_validator,
        model_validator as _model_validator,
    )

    _IS_V2 = True
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, ValidationError, root_validator, validator

    ConfigDict = dict  # type: ignore[assignment]
    TypeAdapter = None  # type: ignore[assignment]
    _IS_V2 = False


def is_pydantic_v2() -> bool:
    return _IS_V2


def field_validator(*fields: str, mode: str = "after", **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    if _IS_V2:
        return _field_validator(*fields, mode=mode, **kwargs)
    pre = mode == "before"
    return validator(*fields, pre=pre, **kwargs)  # type: ignore[name-defined]


def model_validator(*, mode: str = "after", **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    if _IS_V2:
        def _decorator(func: Callable[..., Any]) -> Any:
            if mode == "before":
                @_model_validator(mode="before", **kwargs)
                @classmethod
                def _before(cls, data: Any) -> Any:
                    sig = inspect.signature(func)
                    argc = len(sig.parameters)
                    if argc <= 1:
                        return func(data)
                    return func(cls, data)

                return _before

            @_model_validator(mode="after", **kwargs)
            def _after(self: BaseModel) -> BaseModel:
                sig = inspect.signature(func)
                argc = len(sig.parameters)
                if argc <= 1:
                    updated = func(self)
                else:
                    values = model_to_dict(self)
                    updated = func(self.__class__, values)
                if updated is None:
                    return self
                if isinstance(updated, self.__class__):
                    return updated
                if isinstance(updated, dict):
                    return self.model_copy(update=updated)  # type: ignore[attr-defined]
                return self

            return _after

        return _decorator

    pre = mode == "before"
    if pre:
        return root_validator(pre=True, **kwargs)  # type: ignore[name-defined]
    return root_validator(pre=False, skip_on_failure=True, **kwargs)  # type: ignore[name-defined]


def model_to_dict(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def model_to_json(model: Any, **kwargs: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)


def parse_obj(model_cls: Any, data: Any) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)
    if TypeAdapter is not None:
        return TypeAdapter(model_cls).validate_python(data)
    from pydantic.tools import parse_obj_as

    return parse_obj_as(model_cls, data)


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
]
