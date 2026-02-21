from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

def _model_validate(model_cls: type[BaseModel], data: Any) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


class MacroStep(BaseModel):
    action: str
    selector: str | None = None
    button: str | None = None
    keys: str | None = None
    wait_ms: int = 0
    note: str | None = None


class MacroPath(BaseModel):
    name: str
    requires_hotkey: bool = False
    steps: list[MacroStep] = Field(default_factory=list)


class MacroDefinition(BaseModel):
    intent: str
    description: str = ""
    path_order: list[str] = Field(default_factory=list)
    paths: dict[str, MacroPath] = Field(default_factory=dict)


class MacroRegistry(BaseModel):
    version: int = 1
    defaults: dict[str, Any] = Field(default_factory=dict)
    macros: dict[str, MacroDefinition] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MacroRegistry":
        raw_path = Path(path)
        payload = yaml.safe_load(raw_path.read_text(encoding="utf-8")) or {}

        defaults = payload.get("defaults", {})
        macro_defs: dict[str, MacroDefinition] = {}

        for intent, body in payload.get("macros", {}).items():
            raw_paths = body.get("paths", {})
            paths: dict[str, MacroPath] = {}

            for path_name, path_value in raw_paths.items():
                requires_hotkey = False
                raw_steps: list[dict[str, Any]]

                if isinstance(path_value, dict):
                    requires_hotkey = bool(path_value.get("requires_hotkey", False))
                    raw_steps = list(path_value.get("steps", []))
                else:
                    raw_steps = list(path_value or [])

                steps = [_model_validate(MacroStep, step) for step in raw_steps]
                paths[path_name] = MacroPath(
                    name=path_name,
                    requires_hotkey=requires_hotkey,
                    steps=steps,
                )

            macro_defs[intent] = MacroDefinition(
                intent=intent,
                description=body.get("description", ""),
                path_order=list(body.get("path_order", [])),
                paths=paths,
            )

        return cls(
            version=int(payload.get("version", 1)),
            defaults=defaults,
            macros=macro_defs,
        )

    def expand_intent(
        self,
        intent: str,
        *,
        allow_hotkey: bool = False,
        preferred_path: str | None = None,
    ) -> list[MacroStep]:
        if intent not in self.macros:
            raise KeyError(f"macro intent not found: {intent}")

        macro = self.macros[intent]

        candidate_paths: list[str]
        if preferred_path:
            candidate_paths = [preferred_path]
        else:
            candidate_paths = macro.path_order or list(macro.paths.keys())

        for path_name in candidate_paths:
            path = macro.paths.get(path_name)
            if path is None:
                continue
            if path.requires_hotkey and not allow_hotkey:
                continue
            return path.steps

        raise ValueError(f"no usable macro path for intent={intent}, allow_hotkey={allow_hotkey}")

    def list_macro_names(self) -> list[str]:
        names: list[str] = []
        for intent, macro in self.macros.items():
            prefix = intent.lower()
            for path_name in macro.paths:
                names.append(f"{prefix}.{path_name}")
        return sorted(names)

    def expand_name(self, name: str, *, allow_hotkey: bool = False) -> list[MacroStep]:
        value = str(name or "").strip()
        if not value:
            raise ValueError("macro name is empty")
        if "." in value:
            prefix, preferred_path = value.split(".", 1)
        else:
            prefix, preferred_path = value, None
        return self.expand_intent(prefix.upper(), allow_hotkey=allow_hotkey, preferred_path=preferred_path)


def load_macros(path: str = "config/macros.yaml") -> MacroRegistry:
    return MacroRegistry.from_yaml(path)
