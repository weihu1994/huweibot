from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from huweibot.agent.tasks import TaskStore


_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts"
_TASKS_DB = _ROOT / "artifacts" / "tasks.json"
_NOTIFICATION_LOG = _ROOT / "artifacts" / "notifications.log"


class RunRequest(BaseModel):
    action: str = Field(default="run_task")
    mode: str = Field(default="pc")
    task: str = Field(default="")
    dry_run: bool | None = Field(default=None)
    confirm_real: bool = Field(default=False)
    extra_args: list[str] = Field(default_factory=list)


class ModeRequest(BaseModel):
    mode: Literal["pc", "phone"]
    stop_current: bool = True


class TaskRunRequest(BaseModel):
    task_id: str | None = None
    instruction: str = ""
    mode: Literal["pc", "phone"] = "pc"
    dry_run: bool | None = None
    max_steps: int = 30
    retry: int = 0
    confirm_real: bool = False


class TaskCreateRequest(BaseModel):
    name: str
    mode: Literal["pc", "phone"] = "pc"
    instruction: str
    every_seconds: int | None = None
    at_ts: float | None = None
    enabled: bool = True


class SchedulerAddRequest(BaseModel):
    task_id: str | None = None
    name: str | None = None
    mode: Literal["pc", "phone"] = "pc"
    instruction: str | None = None
    every_seconds: int | None = None
    at_ts: float | None = None


class ProviderPingRequest(BaseModel):
    provider: Literal["dummy", "openai", "anthropic", "gemini", "openai_compat"] = "dummy"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_style: Literal["auto", "responses", "chat_completions"] | None = None
    profile: Literal["generic", "azure", "openrouter", "together", "groq"] | None = None
    headers_json: str | None = None
    query_json: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str | None = None
    or_referer: str | None = None
    or_title: str | None = None
    or_app_id: str | None = None
    dry_run: bool = True


class PCActionRequest(BaseModel):
    action: str
    x: int | None = None
    y: int | None = None
    query: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PhoneActionRequest(BaseModel):
    action: Literal["tap", "swipe", "calibrate", "overlay"]
    gx: int | None = None
    gy: int | None = None
    gx2: int | None = None
    gy2: int | None = None
    dry_run: bool | None = None
    confirm_real: bool = False


class SettingsPatchRequest(BaseModel):
    dry_run: bool | None = None
    allow_real_execution: bool | None = None
    allow_fallback_to_dummy: bool | None = None
    default_provider: str | None = None


@dataclass
class RunnerState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=3000))
    running: bool = False
    mode: str = "pc"
    dry_run: bool = True
    allow_real_execution: bool = False
    allow_fallback_to_dummy: bool = False
    default_provider: str = "dummy"
    last_exit_code: int | None = None
    last_error: str | None = None
    current_cmd: list[str] = field(default_factory=list)
    proc: subprocess.Popen[str] | None = None
    thread: threading.Thread | None = None
    active_task_id: str | None = None
    last_snapshot: str | None = None

    def append(self, text: str) -> None:
        line = text.rstrip("\n")
        if not line:
            return
        with self.lock:
            self.logs.append(line)
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with (_ARTIFACTS / "web_console.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@dataclass
class SpawnContext:
    mode: str
    task_id: str | None = None


def _notify(message: str) -> None:
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with _NOTIFICATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def _redact_text(text: str, secrets: list[str] | None = None) -> str:
    out = str(text or "")
    for secret in secrets or []:
        if secret:
            out = out.replace(secret, "***")
    out = out.replace("Bearer ", "Bearer ***")
    return out


def _status_components(state: RunnerState) -> dict[str, dict[str, Any]]:
    try:
        import cv2  # noqa: F401

        camera_status = {"status": "ready", "detail": "opencv available"}
    except Exception:
        camera_status = {"status": "warn", "detail": "opencv unavailable"}

    calib_path = _ROOT / "config" / "calibration.json"
    phone_calib = _ROOT / "config" / "phone_screen.json"
    calib_status = {
        "status": "ready" if calib_path.exists() else "warn",
        "detail": str(calib_path if calib_path.exists() else "missing calibration.json"),
        "phone": str(phone_calib if phone_calib.exists() else "missing phone_screen.json"),
    }

    provider_status = {
        "status": "ready",
        "detail": state.default_provider,
    }

    controller_status = {
        "status": "idle",
        "detail": "hardware bridge not opened by web console",
    }

    task_status = {
        "status": "running" if state.running else "idle",
        "detail": state.active_task_id or "none",
    }

    return {
        "camera": camera_status,
        "calib": calib_status,
        "controller": controller_status,
        "provider": provider_status,
        "task_runner": task_status,
    }


def _safety_status() -> dict[str, Any]:
    drift_file = _ARTIFACTS / "last_drift_check.json"
    drift = None
    if drift_file.exists():
        try:
            drift = json.loads(drift_file.read_text(encoding="utf-8"))
        except Exception:
            drift = {"status": "invalid"}
    return {
        "cursor_visible": None,
        "drift_last": drift,
        "bounds_protection": True,
        "emergency_stop": True,
    }


def _build_command(req: RunRequest, *, effective_dry_run: bool) -> list[str]:
    mode = "phone" if str(req.mode).lower() == "phone" else "computer"
    action = str(req.action or "run_task").strip().lower()
    if action == "run_task":
        cmd = [sys.executable, "-m", "huweibot.main", "run-task", "--mode", mode]
        if effective_dry_run:
            cmd.append("--dry-run")
        if req.task:
            cmd.extend(["--task", str(req.task)])
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    if action == "phone_dry_run":
        cmd = [sys.executable, str(_ROOT / "scripts" / "phone_dry_run.py")]
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    if action == "inspect_elements":
        cmd = [sys.executable, str(_ROOT / "scripts" / "inspect_elements.py")]
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    if action == "preview_camera":
        cmd = [sys.executable, str(_ROOT / "scripts" / "preview_camera.py")]
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    if action == "calibrate_screen":
        cmd = [sys.executable, str(_ROOT / "scripts" / "calibrate_screen.py")]
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    if action == "calibrate_phone":
        cmd = [sys.executable, str(_ROOT / "scripts" / "calibrate_phone_screen.py")]
        cmd.extend([str(x) for x in req.extra_args])
        return cmd
    raise ValueError("unsupported action")


def _set_task_status(task_id: str, *, running: bool, ok: bool | None = None) -> None:
    store = TaskStore(str(_TASKS_DB))
    task = store.get(task_id)
    if task is None:
        return
    if running:
        store.mark_running(task)
        return
    if ok is None:
        task.status = "pending"
        store.replace(task)
        return
    store.mark_result(task, ok=ok)


def _spawn_cmd(state: RunnerState, cmd: list[str], ctx: SpawnContext) -> None:
    state.append(f"$ {' '.join(cmd)}")
    _notify(f"TRIGGER mode={ctx.mode} cmd={' '.join(cmd)}")
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with state.lock:
            state.proc = proc
            state.current_cmd = cmd
            state.mode = ctx.mode
            state.active_task_id = ctx.task_id
        if ctx.task_id:
            _set_task_status(ctx.task_id, running=True)
        if proc.stdout is not None:
            for line in proc.stdout:
                state.append(line)
        rc = proc.wait()
        with state.lock:
            state.last_exit_code = int(rc)
            state.last_error = None if rc == 0 else f"exit_code={rc}"
        if ctx.task_id:
            _set_task_status(ctx.task_id, running=False, ok=(rc == 0))
        _notify(f"FINISH mode={ctx.mode} rc={rc}")
    except Exception as exc:
        with state.lock:
            state.last_error = str(exc)
            state.last_exit_code = 1
        if ctx.task_id:
            _set_task_status(ctx.task_id, running=False, ok=False)
        state.append(f"[error] {exc}")
        _notify(f"FAIL mode={ctx.mode} err={exc}")
    finally:
        with state.lock:
            state.running = False
            state.proc = None
            state.thread = None
            state.active_task_id = None
            if proc is not None and proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass


def _start_async(state: RunnerState, *, cmd: list[str], mode: str, task_id: str | None = None) -> dict[str, Any]:
    with state.lock:
        if state.running:
            raise HTTPException(status_code=409, detail="task already running")
        state.running = True
        state.last_error = None
        state.last_exit_code = None
    ctx = SpawnContext(mode=mode, task_id=task_id)
    t = threading.Thread(target=_spawn_cmd, args=(state, cmd, ctx), daemon=True)
    with state.lock:
        state.thread = t
    t.start()
    return {"ok": True, "running": True, "mode": mode, "cmd": cmd}


def _terminate_process(state: RunnerState) -> bool:
    with state.lock:
        proc = state.proc
    if proc is None:
        return False
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True
    except Exception:
        return False


def _run_provider_check(req: ProviderPingRequest) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "huweibot.tools.provider_check", "--provider", req.provider]
    if req.model:
        cmd.extend(["--model", req.model])
    if req.base_url:
        cmd.extend(["--base-url", req.base_url])
    if req.api_key is not None:
        cmd.extend(["--api-key", req.api_key])
    if req.api_style:
        cmd.extend(["--api-style", req.api_style])
    if req.profile:
        cmd.extend(["--profile", req.profile])
    if req.headers_json:
        cmd.extend(["--headers-json", req.headers_json])
    if req.query_json:
        cmd.extend(["--query-json", req.query_json])
    if req.azure_deployment:
        cmd.extend(["--azure-deployment", req.azure_deployment])
    if req.azure_api_version:
        cmd.extend(["--azure-api-version", req.azure_api_version])
    if req.or_referer:
        cmd.extend(["--or-referer", req.or_referer])
    if req.or_title:
        cmd.extend(["--or-title", req.or_title])
    if req.or_app_id:
        cmd.extend(["--or-app-id", req.or_app_id])
    cmd.append("--dry-run" if req.dry_run else "--ping")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), timeout=60)
    out = (proc.stdout or "") + (proc.stderr or "")
    return int(proc.returncode), _redact_text(out, secrets=[req.api_key or ""])


def _load_tasks() -> list[dict[str, Any]]:
    store = TaskStore(str(_TASKS_DB))
    return [t.to_dict() for t in store.list()]


def _find_task(task_id: str) -> dict[str, Any] | None:
    store = TaskStore(str(_TASKS_DB))
    t = store.get(task_id)
    return t.to_dict() if t else None


def _parse_jsonish(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    return parsed


def create_app() -> FastAPI:
    app = FastAPI(title="huweibot web console", version="0.2.0")
    state = RunnerState()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "name": "huweibot", "ts": time.time()}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        with state.lock:
            payload = {
                "ok": True,
                "name": "huweibot",
                "running": bool(state.running),
                "mode": state.mode,
                "dry_run": bool(state.dry_run),
                "allow_real_execution": bool(state.allow_real_execution),
                "allow_fallback_to_dummy": bool(state.allow_fallback_to_dummy),
                "default_provider": state.default_provider,
                "last_exit_code": state.last_exit_code,
                "last_error": state.last_error,
                "current_cmd": list(state.current_cmd),
                "components": _status_components(state),
                "safety": _safety_status(),
            }
        return payload

    @app.post("/api/mode")
    def set_mode(req: ModeRequest) -> dict[str, Any]:
        target = "phone" if req.mode == "phone" else "pc"
        with state.lock:
            was_running = bool(state.running)
            current = state.mode
        if was_running and current != target and not req.stop_current:
            raise HTTPException(status_code=409, detail="mode switch requires stopping current task")
        if was_running and current != target and req.stop_current:
            _terminate_process(state)
            time.sleep(0.1)
        with state.lock:
            state.mode = target
        return {"ok": True, "mode": target, "stopped": bool(was_running and current != target)}

    @app.post("/api/settings")
    def patch_settings(req: SettingsPatchRequest) -> dict[str, Any]:
        with state.lock:
            if req.dry_run is not None:
                state.dry_run = bool(req.dry_run)
            if req.allow_real_execution is not None:
                state.allow_real_execution = bool(req.allow_real_execution)
            if req.allow_fallback_to_dummy is not None:
                state.allow_fallback_to_dummy = bool(req.allow_fallback_to_dummy)
            if req.default_provider:
                state.default_provider = str(req.default_provider)
            return {
                "ok": True,
                "dry_run": state.dry_run,
                "allow_real_execution": state.allow_real_execution,
                "allow_fallback_to_dummy": state.allow_fallback_to_dummy,
                "default_provider": state.default_provider,
            }

    @app.get("/api/logs")
    def logs_compat(tail: int = Query(default=200, ge=1, le=3000)) -> JSONResponse:
        with state.lock:
            lines = list(state.logs)[-int(tail) :]
        return JSONResponse({"lines": lines, "text": "\n".join(lines)})

    @app.get("/api/logs/tail")
    def logs_tail(lines: int = Query(default=200, ge=1, le=3000)) -> JSONResponse:
        with state.lock:
            chunk = list(state.logs)[-int(lines) :]
        return JSONResponse({"ok": True, "lines": chunk, "text": "\n".join(chunk)})

    @app.get("/api/logs/download")
    def logs_download(lines: int = Query(default=500, ge=1, le=5000)) -> PlainTextResponse:
        with state.lock:
            chunk = list(state.logs)[-int(lines) :]
        return PlainTextResponse("\n".join(chunk), media_type="text/plain")

    @app.post("/api/run")
    def run(req: RunRequest) -> dict[str, Any]:
        effective_dry = state.dry_run if req.dry_run is None else bool(req.dry_run)
        if not effective_dry:
            if not state.allow_real_execution:
                raise HTTPException(status_code=403, detail="real execution disabled in settings")
            if not req.confirm_real:
                raise HTTPException(status_code=400, detail="confirm_real required for non-dry-run actions")
        try:
            cmd = _build_command(req, effective_dry_run=effective_dry)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _start_async(state, cmd=cmd, mode=("phone" if req.mode == "phone" else "pc"))

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        stopped = _terminate_process(state)
        return {"ok": True, "stopped": bool(stopped)}

    @app.post("/api/emergency-stop")
    def emergency_stop() -> dict[str, Any]:
        stopped = _terminate_process(state)
        state.append("[warn] emergency stop requested")
        return {"ok": True, "stopped": bool(stopped), "reason": "emergency_stop"}

    @app.post("/api/provider/ping")
    def provider_ping(req: ProviderPingRequest) -> dict[str, Any]:
        rc, out = _run_provider_check(req)
        parsed: Any = None
        try:
            parsed = json.loads(out)
        except Exception:
            parsed = None
        with state.lock:
            state.default_provider = req.provider
        return {
            "ok": rc == 0,
            "exit_code": rc,
            "provider": req.provider,
            "dry_run": req.dry_run,
            "result": parsed,
            "text": None if parsed is not None else out.strip(),
        }

    @app.post("/api/pc/preview")
    def pc_preview() -> dict[str, Any]:
        req = RunRequest(action="preview_camera", mode="pc", dry_run=True)
        cmd = _build_command(req, effective_dry_run=True)
        return _start_async(state, cmd=cmd, mode="pc")

    @app.post("/api/pc/snapshot")
    def pc_snapshot() -> dict[str, Any]:
        try:
            from huweibot.vision.camera import Camera
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"camera module unavailable: {exc}") from exc
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out_path = _ARTIFACTS / "web_snapshot.png"
        try:
            import cv2

            with Camera(0, 1280, 720, 30) as cam:
                frame_pack = cam.read_latest(2)
                frame = frame_pack.bgr
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
            h, w = frame.shape[:2]
            with state.lock:
                state.last_snapshot = str(out_path)
            return {"ok": True, "path": str(out_path), "resolution": f"{w}x{h}", "timestamp": frame_pack.timestamp}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"snapshot failed: {exc}") from exc

    @app.post("/api/pc/inspect_elements")
    def pc_inspect_elements() -> dict[str, Any]:
        req = RunRequest(action="inspect_elements", mode="pc", dry_run=True)
        cmd = _build_command(req, effective_dry_run=True)
        return _start_async(state, cmd=cmd, mode="pc")

    @app.post("/api/pc/action")
    def pc_action(req: PCActionRequest) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "Not implemented",
            "status": "Coming soon",
            "action": req.action,
        }

    @app.post("/api/phone/calibrate")
    def phone_calibrate() -> dict[str, Any]:
        req = RunRequest(action="calibrate_phone", mode="phone", dry_run=True)
        cmd = _build_command(req, effective_dry_run=True)
        return _start_async(state, cmd=cmd, mode="phone")

    @app.post("/api/phone/tap")
    def phone_tap(req: PhoneActionRequest) -> dict[str, Any]:
        effective_dry = state.dry_run if req.dry_run is None else bool(req.dry_run)
        if not effective_dry:
            if not state.allow_real_execution:
                raise HTTPException(status_code=403, detail="real execution disabled in settings")
            if not req.confirm_real:
                raise HTTPException(status_code=400, detail="confirm_real required for non-dry-run actions")
            return {"ok": False, "error": "Not implemented", "status": "Coming soon"}
        if req.gx is None or req.gy is None:
            raise HTTPException(status_code=400, detail="gx/gy required")
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "phone_dry_run.py"),
            "--tap",
            str(req.gx),
            str(req.gy),
            "--out",
            str(_ARTIFACTS / "phone_dry_run.jsonl"),
        ]
        return _start_async(state, cmd=cmd, mode="phone")

    @app.post("/api/phone/swipe")
    def phone_swipe(req: PhoneActionRequest) -> dict[str, Any]:
        effective_dry = state.dry_run if req.dry_run is None else bool(req.dry_run)
        if not effective_dry:
            if not state.allow_real_execution:
                raise HTTPException(status_code=403, detail="real execution disabled in settings")
            if not req.confirm_real:
                raise HTTPException(status_code=400, detail="confirm_real required for non-dry-run actions")
            return {"ok": False, "error": "Not implemented", "status": "Coming soon"}
        if req.gx is None or req.gy is None or req.gx2 is None or req.gy2 is None:
            raise HTTPException(status_code=400, detail="gx/gy/gx2/gy2 required")
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "phone_dry_run.py"),
            "--swipe",
            str(req.gx),
            str(req.gy),
            str(req.gx2),
            str(req.gy2),
            "--out",
            str(_ARTIFACTS / "phone_dry_run.jsonl"),
        ]
        return _start_async(state, cmd=cmd, mode="phone")

    @app.get("/api/tasks")
    def tasks() -> dict[str, Any]:
        return {"ok": True, "tasks": _load_tasks()}

    @app.post("/api/task/add")
    def task_add(req: TaskCreateRequest) -> dict[str, Any]:
        store = TaskStore(str(_TASKS_DB))
        task = store.add(
            name=req.name,
            mode=("phone" if req.mode == "phone" else "computer"),
            goal=req.instruction,
            every_seconds=req.every_seconds,
            at_ts=req.at_ts,
            enabled=req.enabled,
        )
        return {"ok": True, "task": task.to_dict()}

    @app.post("/api/task/run")
    def task_run(req: TaskRunRequest) -> dict[str, Any]:
        effective_dry = state.dry_run if req.dry_run is None else bool(req.dry_run)
        if not effective_dry:
            if not state.allow_real_execution:
                raise HTTPException(status_code=403, detail="real execution disabled in settings")
            if not req.confirm_real:
                raise HTTPException(status_code=400, detail="confirm_real required for non-dry-run actions")
        mode = "phone" if req.mode == "phone" else "pc"
        instruction = req.instruction
        task_id = req.task_id
        if task_id:
            t = _find_task(task_id)
            if t is None:
                raise HTTPException(status_code=404, detail="task not found")
            instruction = str(t.get("goal") or "")
            mode = "phone" if str(t.get("mode", "computer")) == "phone" else "pc"
        if not str(instruction or "").strip():
            raise HTTPException(status_code=400, detail="instruction is required")
        cmd = [sys.executable, "-m", "huweibot.main", "run-task", "--mode", ("phone" if mode == "phone" else "computer"), "--task", instruction, "--max-steps", str(max(1, int(req.max_steps)))]
        if effective_dry:
            cmd.append("--dry-run")
        return _start_async(state, cmd=cmd, mode=mode, task_id=task_id)

    @app.post("/api/task/stop")
    def task_stop() -> dict[str, Any]:
        stopped = _terminate_process(state)
        return {"ok": True, "stopped": bool(stopped)}

    @app.post("/api/scheduler/add")
    def scheduler_add(req: SchedulerAddRequest) -> dict[str, Any]:
        store = TaskStore(str(_TASKS_DB))
        if req.task_id:
            task = store.get(req.task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            if req.every_seconds is not None:
                task.schedule.kind = "interval"
                task.schedule.every_seconds = max(1, int(req.every_seconds))
                task.next_run_at = time.time()
            elif req.at_ts is not None:
                task.schedule.kind = "once"
                task.schedule.at_ts = float(req.at_ts)
                task.next_run_at = float(req.at_ts)
            else:
                raise HTTPException(status_code=400, detail="every_seconds or at_ts is required")
            store.replace(task)
            return {"ok": True, "task": task.to_dict()}

        if not req.name or not req.instruction:
            raise HTTPException(status_code=400, detail="name and instruction required when task_id not provided")
        task = store.add(
            name=req.name,
            mode=("phone" if req.mode == "phone" else "computer"),
            goal=req.instruction,
            every_seconds=req.every_seconds,
            at_ts=req.at_ts,
            enabled=True,
        )
        return {"ok": True, "task": task.to_dict()}

    @app.get("/api/scheduler")
    def scheduler_list() -> dict[str, Any]:
        tasks = _load_tasks()
        scheduled = [t for t in tasks if (t.get("schedule") or {}).get("kind") in {"once", "interval"}]
        return {"ok": True, "items": scheduled}

    @app.get("/api/version")
    def version() -> dict[str, Any]:
        git_commit = None
        try:
            proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(_ROOT), timeout=3)
            if proc.returncode == 0:
                git_commit = (proc.stdout or "").strip() or None
        except Exception:
            git_commit = None
        return {
            "ok": True,
            "name": "huweibot",
            "version": "0.2.0",
            "git_commit": git_commit,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(Path(__file__).resolve().parent / "static" / "index.html"))

    return app
