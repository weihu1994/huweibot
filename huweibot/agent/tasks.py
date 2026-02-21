from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


TaskMode = Literal["computer", "phone"]
TaskStatus = Literal["pending", "running", "done", "failed"]
ScheduleKind = Literal["once", "interval"]


def _now_ts() -> float:
    return time.time()


def _normalize_mode(mode: str) -> TaskMode:
    value = str(mode or "computer").strip().lower()
    if value in {"computer", "pc"}:
        return "computer"
    if value == "phone":
        return "phone"
    raise ValueError("mode must be computer|phone")


@dataclass
class Schedule:
    kind: ScheduleKind
    at_ts: float | None = None
    every_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Schedule":
        kind = str(payload.get("kind", "once"))
        at_ts = payload.get("at_ts")
        every_seconds = payload.get("every_seconds")
        return cls(
            kind="interval" if kind == "interval" else "once",
            at_ts=None if at_ts is None else float(at_ts),
            every_seconds=None if every_seconds is None else max(1, int(every_seconds)),
        )


@dataclass
class Task:
    id: str
    name: str
    mode: TaskMode
    goal: str
    payload: dict[str, Any]
    created_at: float
    enabled: bool
    schedule: Schedule
    status: TaskStatus = "pending"
    last_run_at: float | None = None
    next_run_at: float | None = None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schedule"] = self.schedule.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        schedule_payload = payload.get("schedule", {})
        schedule = schedule_payload if isinstance(schedule_payload, dict) else {}
        mode = _normalize_mode(str(payload.get("mode", "computer")))
        task = cls(
            id=str(payload.get("id") or uuid.uuid4().hex[:12]),
            name=str(payload.get("name") or payload.get("title") or "task"),
            mode=mode,
            goal=str(payload.get("goal") or payload.get("instruction") or ""),
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            created_at=float(payload.get("created_at", _now_ts())),
            enabled=bool(payload.get("enabled", True)),
            schedule=Schedule.from_dict(schedule),
            status=str(payload.get("status", "pending")),
            last_run_at=None if payload.get("last_run_at") is None else float(payload.get("last_run_at")),
            next_run_at=None if payload.get("next_run_at") is None else float(payload.get("next_run_at")),
            retries=int(payload.get("retries", 0)),
        )
        if task.next_run_at is None:
            if task.schedule.kind == "interval" and task.schedule.every_seconds is not None:
                task.next_run_at = _now_ts()
            elif task.schedule.kind == "once" and task.schedule.at_ts is not None:
                task.next_run_at = float(task.schedule.at_ts)
            else:
                task.next_run_at = _now_ts()
        return task


class TaskStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"tasks": [], "updated_at": _now_ts()}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"tasks": [], "updated_at": _now_ts()}
        if not isinstance(payload, dict):
            return {"tasks": [], "updated_at": _now_ts()}
        if not isinstance(payload.get("tasks"), list):
            payload["tasks"] = []
        return payload

    def load(self) -> list[Task]:
        payload = self._read_payload()
        out: list[Task] = []
        for item in payload.get("tasks", []):
            if isinstance(item, dict):
                out.append(Task.from_dict(item))
        return out

    def save(self, tasks: list[Task]) -> None:
        self._atomic_write({"tasks": [t.to_dict() for t in tasks], "updated_at": _now_ts()})

    def list(self) -> list[Task]:
        return self.load()

    def get(self, task_id: str) -> Task | None:
        for task in self.load():
            if task.id == task_id:
                return task
        return None

    def replace(self, new_task: Task) -> None:
        tasks = self.load()
        replaced = False
        for idx, task in enumerate(tasks):
            if task.id == new_task.id:
                tasks[idx] = new_task
                replaced = True
                break
        if not replaced:
            tasks.append(new_task)
        self.save(tasks)

    def add(
        self,
        *,
        name: str,
        mode: str,
        goal: str,
        payload: dict[str, Any] | None = None,
        at_ts: float | None = None,
        every_seconds: int | None = None,
        enabled: bool = True,
    ) -> Task:
        mode_norm = _normalize_mode(mode)
        name_text = str(name or "").strip()
        goal_text = str(goal or "").strip()
        if not name_text:
            raise ValueError("task name is required")
        if not goal_text:
            raise ValueError("task goal is required")

        if every_seconds is not None:
            schedule = Schedule(kind="interval", at_ts=None, every_seconds=max(1, int(every_seconds)))
            next_run_at = _now_ts()
        else:
            schedule = Schedule(kind="once", at_ts=(None if at_ts is None else float(at_ts)), every_seconds=None)
            next_run_at = _now_ts() if at_ts is None else float(at_ts)

        task = Task(
            id=uuid.uuid4().hex[:12],
            name=name_text,
            mode=mode_norm,
            goal=goal_text,
            payload=payload or {},
            created_at=_now_ts(),
            enabled=bool(enabled),
            schedule=schedule,
            status="pending",
            last_run_at=None,
            next_run_at=next_run_at,
            retries=0,
        )
        tasks = self.load()
        tasks.append(task)
        self.save(tasks)
        return task

    def remove(self, task_id: str) -> bool:
        tasks = self.load()
        new_tasks = [t for t in tasks if t.id != task_id]
        if len(new_tasks) == len(tasks):
            return False
        self.save(new_tasks)
        return True

    def set_enabled(self, task_id: str, enabled: bool) -> Task | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.enabled = bool(enabled)
        if not task.enabled:
            task.status = "pending"
        elif task.next_run_at is None:
            if task.schedule.kind == "interval" and task.schedule.every_seconds is not None:
                task.next_run_at = _now_ts()
            elif task.schedule.kind == "once" and task.schedule.at_ts is not None:
                task.next_run_at = float(task.schedule.at_ts)
            else:
                task.next_run_at = _now_ts()
        self.replace(task)
        return task

    def due(self, now_ts: float | None = None) -> list[Task]:
        now = _now_ts() if now_ts is None else float(now_ts)
        due_tasks: list[Task] = []
        for task in self.load():
            if not task.enabled:
                continue
            if task.next_run_at is None:
                continue
            if task.status == "running":
                continue
            if float(task.next_run_at) <= now:
                due_tasks.append(task)
        return due_tasks

    def mark_running(self, task: Task) -> Task:
        task.status = "running"
        task.last_run_at = _now_ts()
        self.replace(task)
        return task

    def mark_result(self, task: Task, *, ok: bool) -> Task:
        now = _now_ts()
        task.last_run_at = now
        if task.schedule.kind == "interval" and task.schedule.every_seconds is not None:
            task.status = "pending"
            task.next_run_at = now + int(task.schedule.every_seconds)
        else:
            task.status = "done" if ok else "failed"
            task.next_run_at = None
        if not ok:
            task.retries += 1
        self.replace(task)
        return task


# Backward-compatible function wrappers used by existing CLI glue.
def load_tasks(path: str) -> list[Task]:
    return TaskStore(path).load()


def save_tasks(path: str, tasks: list[Task]) -> None:
    TaskStore(path).save(tasks)


def add_task(
    path: str,
    *,
    name: str | None = None,
    title: str | None = None,
    goal: str | None = None,
    instruction: str | None = None,
    mode: str,
    every_seconds: int | None = None,
    at_ts: float | None = None,
    payload: dict[str, Any] | None = None,
) -> Task:
    store = TaskStore(path)
    return store.add(
        name=(name or title or "task"),
        mode=mode,
        goal=(goal or instruction or ""),
        payload=payload,
        at_ts=at_ts,
        every_seconds=every_seconds,
        enabled=True,
    )


def get_task(path: str, task_id: str) -> Task | None:
    return TaskStore(path).get(task_id)


def replace_task(path: str, new_task: Task) -> None:
    TaskStore(path).replace(new_task)


def list_due_tasks(path: str, now_ts: float | None = None) -> list[Task]:
    return TaskStore(path).due(now_ts)


def mark_task_running(path: str, task: Task) -> Task:
    return TaskStore(path).mark_running(task)


def mark_task_result(path: str, task: Task, *, ok: bool) -> Task:
    return TaskStore(path).mark_result(task, ok=ok)


def remove_task(path: str, task_id: str) -> bool:
    return TaskStore(path).remove(task_id)


def set_task_enabled(path: str, task_id: str, enabled: bool) -> Task | None:
    return TaskStore(path).set_enabled(task_id, enabled)
