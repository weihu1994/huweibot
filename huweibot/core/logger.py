from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


class RunLogger:
    def __init__(self, artifacts_dir: str = "artifacts", enabled: bool = True):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(enabled)
        self._records: dict[int, dict[str, Any]] = {}

    def _prefix(self, step: int) -> str:
        return f"step_{int(step):04d}"

    def _path(self, step: int, suffix: str) -> Path:
        return self.artifacts_dir / f"{self._prefix(step)}_{suffix}"

    def _record_path(self, step: int) -> Path:
        return self.artifacts_dir / f"{self._prefix(step)}.json"

    def _save_record(self, step: int) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        self._record_path(step).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    def begin_step(self, step: int, *, task: str, obs_mode: str) -> None:
        if not self.enabled:
            return
        rec = {
            "step": int(step),
            "time": _iso_now(),
            "phase": "observe",
            "obs": {
                "app_hint": None,
                "screen_hash": None,
                "ui_change_score": 0.0,
                "cursor": {"x": None, "y": None, "confidence": 0.0, "type": "unknown"},
                "elements_count": 0,
                "keyboard_mode": False,
                "keyboard_roi": None,
            },
            "planner": {
                "task": task,
                "model": None,
                "provider": None,
                "capabilities": {},
                "output": None,
                "repair_used": False,
                "error": None,
                "obs_mode": obs_mode,
            },
            "vlm": {
                "used": False,
                "trigger": None,
                "roi": None,
                "max_side": None,
                "jpeg_quality": None,
                "cache_hit": False,
                "error": None,
            },
            "resolve": {
                "target": None,
                "status": "skipped",
                "chosen": None,
                "candidates": [],
            },
            "execution": {"action_executed": None, "success": False, "reason": ""},
            "osk": {
                "text": None,
                "profile": None,
                "shift_state": None,
                "compiled_keys": None,
                "failed_char": None,
            },
            "macro": {"name": None, "expanded_steps": None, "failed_step": None},
            "verify": {"ok": None, "score": None, "method": None, "details": None},
            "drift": {"updated": False, "mm_per_px_before": None, "mm_per_px_after": None},
            "cursor_recovery": {"attempts": 0, "ok": True},
            "notes": "",
        }
        self._records[int(step)] = rec
        self.write_json(step, "resolve.json", rec["resolve"])
        self._save_record(step)

    def set_phase(self, step: int, phase: str) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        rec["phase"] = phase
        self._save_record(step)

    def append_note(self, step: int, note: str) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        prev = rec.get("notes") or ""
        rec["notes"] = note if not prev else f"{prev}; {note}"
        self._save_record(step)

    def write_text(self, step: int, suffix: str, content: str) -> None:
        if not self.enabled:
            return
        self._path(step, suffix).write_text(content, encoding="utf-8")

    def write_json(self, step: int, suffix: str, payload: Any) -> None:
        if not self.enabled:
            return
        self._path(step, suffix).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_screen(self, step: int, frame_bgr: Any | None) -> None:
        if not self.enabled:
            return
        if frame_bgr is None:
            self.append_note(step, "screen_not_written:no_frame")
            return
        cv2.imwrite(str(self._path(step, "screen.png")), frame_bgr)

    def write_elements(self, step: int, elements: list[Any]) -> None:
        if not self.enabled:
            return
        serializable = []
        for elem in elements:
            item = _model_dump(elem)
            if isinstance(item, dict):
                serializable.append(
                    {
                        "stable_id": item.get("stable_id"),
                        "raw_id": item.get("raw_id"),
                        "role": item.get("role"),
                        "text": item.get("text"),
                        "label": item.get("label"),
                        "bbox": item.get("bbox"),
                        "confidence": item.get("confidence"),
                        "source": item.get("source"),
                        "clickability_hint": item.get("clickability_hint"),
                    }
                )
        self.write_json(step, "elements.json", serializable)

    def log_observation(self, step: int, obs: Any, *, screen_bgr: Any | None = None) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        cursor_xy = getattr(obs, "cursor_xy", None)
        rec["obs"] = {
            "app_hint": getattr(obs, "app_hint", None),
            "screen_hash": getattr(obs, "screen_hash", None),
            "ui_change_score": float(getattr(obs, "ui_change_score", 0.0)),
            "cursor": {
                "x": None if cursor_xy is None else int(round(float(cursor_xy[0]))),
                "y": None if cursor_xy is None else int(round(float(cursor_xy[1]))),
                "confidence": float(getattr(obs, "cursor_conf", 0.0)),
                "type": getattr(obs, "cursor_type", "unknown"),
            },
            "elements_count": len(getattr(obs, "elements", [])),
            "keyboard_mode": bool(getattr(obs, "keyboard_mode", False)),
            "keyboard_roi": getattr(obs, "keyboard_roi", None),
        }
        self.write_screen(step, screen_bgr)
        self.write_elements(step, list(getattr(obs, "elements", [])))
        self._save_record(step)

    def log_planner(self, step: int, planner_meta: dict[str, Any] | None, *, task: str, obs_mode: str) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        meta = planner_meta or {}
        rec["planner"] = {
            "task": task,
            "model": meta.get("model"),
            "provider": meta.get("provider"),
            "capabilities": meta.get("capabilities") or {},
            "output": meta.get("output"),
            "repair_used": bool(meta.get("repair_used", False)),
            "error": meta.get("error"),
            "obs_mode": obs_mode,
        }
        self._save_record(step)

    def log_vlm(self, step: int, vlm_meta: dict[str, Any] | None) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        meta = vlm_meta or {}
        rec["vlm"] = {
            "used": bool(meta.get("used", False)),
            "trigger": meta.get("trigger"),
            "roi": meta.get("roi"),
            "max_side": meta.get("max_side"),
            "jpeg_quality": meta.get("jpeg_quality"),
            "cache_hit": bool(meta.get("cache_hit", False)),
            "error": meta.get("error"),
        }
        self._save_record(step)

    def log_resolve(
        self,
        step: int,
        *,
        target: dict[str, Any] | None,
        status: str,
        chosen: dict[str, Any] | None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        payload = {
            "target": target,
            "status": status,
            "chosen": chosen,
            "candidates": candidates or [],
        }
        rec["resolve"] = payload
        self.write_json(step, "resolve.json", payload)
        self._save_record(step)

    def log_macro(self, step: int, *, name: str | None, expanded_steps: list[Any] | None, failed_step: int | None) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        payload = {"name": name, "expanded_steps": expanded_steps, "failed_step": failed_step}
        rec["macro"] = payload
        self.write_json(step, "macro_expand.json", payload)
        self._save_record(step)

    def log_osk(
        self,
        step: int,
        *,
        text: str | None,
        profile: str | None,
        shift_state: str | None,
        compiled_keys: list[Any] | None,
        failed_char: str | None,
    ) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        payload = {
            "text": text,
            "profile": profile,
            "shift_state": shift_state,
            "compiled_keys": compiled_keys,
            "failed_char": failed_char,
        }
        rec["osk"] = payload
        self.write_json(step, "osk_compile.json", payload)
        self._save_record(step)

    def log_drift(self, step: int, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        rec["drift"] = {
            "updated": bool(payload.get("updated", False)),
            "mm_per_px_before": payload.get("mm_per_px_before"),
            "mm_per_px_after": payload.get("mm_per_px_after"),
        }
        self.write_json(step, "drift.json", payload)
        self._save_record(step)

    def log_cursor_recovery(self, step: int, *, attempts: int, ok: bool) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        rec["cursor_recovery"] = {"attempts": int(attempts), "ok": bool(ok)}
        self._save_record(step)

    def write_cursor_lost(self, step: int, frame_bgr: Any | None) -> None:
        if not self.enabled:
            return
        if frame_bgr is None:
            self.append_note(step, "cursor_lost:no_frame")
            return
        cv2.imwrite(str(self._path(step, "cursor_lost.png")), frame_bgr)

    def log_execution(self, step: int, *, action_executed: Any, success: bool, reason: str) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        rec["execution"] = {
            "action_executed": _model_dump(action_executed),
            "success": bool(success),
            "reason": reason,
        }
        self._save_record(step)

    def log_verify(self, step: int, verify: Any | None) -> None:
        if not self.enabled:
            return
        rec = self._records.get(int(step))
        if rec is None:
            return
        if verify is None:
            rec["verify"] = {"ok": None, "score": None, "method": None, "details": None}
        else:
            data = _model_dump(verify)
            if isinstance(data, dict):
                rec["verify"] = {
                    "ok": data.get("ok"),
                    "score": data.get("score"),
                    "method": data.get("method"),
                    "details": data.get("details"),
                }
            else:
                rec["verify"] = {"ok": None, "score": None, "method": None, "details": {"raw": str(data)}}
        self._save_record(step)
