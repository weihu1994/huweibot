from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huweibot.agent.macros import MacroRegistry, load_macros
from huweibot.agent.schemas import ElementRef, NextStep, VerifyRule, action_from_json
from huweibot.agent.verifier import VerifyResult
from huweibot.core.logger import RunLogger
from huweibot.core.observation import Observation, UIElement
from huweibot.core.semantic_guard import SemanticGuard


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _element_ref_from_any(payload: Any) -> ElementRef:
    if isinstance(payload, ElementRef):
        return payload
    if isinstance(payload, dict):
        return ElementRef.parse_obj(payload)
    raise ValueError(f"invalid ElementRef payload: {payload!r}")


@dataclass
class ExecutorState:
    step: int = 0
    done: bool = False
    done_reason: str | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    last_obs: Observation | None = None
    failures: int = 0


@dataclass
class StepResult:
    step: int
    ok: bool
    done: bool
    reason: str
    action: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None
    guard: dict[str, Any] | None = None
    planner: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ok": self.ok,
            "done": self.done,
            "reason": self.reason,
            "action": self.action,
            "verify": self.verify,
            "guard": self.guard,
            "planner": self.planner,
        }


class Executor:
    def __init__(
        self,
        *,
        loop: Any,
        planner: Any,
        config: Any,
        router: Any | None = None,
        guard: SemanticGuard | None = None,
        run_logger: RunLogger | None = None,
    ):
        self.loop = loop
        self.planner = planner
        self.config = config
        self.router = router
        self.state = ExecutorState(memory={"debug_reasoning": bool(getattr(config, "debug_reasoning", False))})
        self.artifacts_dir = str(getattr(config, "artifacts_dir", "artifacts"))
        self.logger = run_logger or RunLogger(self.artifacts_dir, enabled=True)
        self.environment_state: dict[str, Any] = {"environment_valid": True, "validate_screen_ok": True}
        self.guard = guard or SemanticGuard(
            config,
            getattr(loop, "selector", None),
            hardware=getattr(loop, "hardware", None),
            kinematics=getattr(loop, "kinematics", None),
            environment_state=self.environment_state,
        )
        self._macros: MacroRegistry | None = None

    def _macro_registry(self) -> MacroRegistry:
        if self._macros is None:
            self._macros = load_macros(str(getattr(self.config, "macros_path", "config/macros.yaml")))
        return self._macros

    def _step_path(self, step: int, suffix: str) -> Path:
        root = Path(self.artifacts_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f"step_{step:04d}_{suffix}"

    def _write_if_missing(self, step: int, suffix: str, payload: Any) -> None:
        path = self._step_path(step, suffix)
        if path.exists():
            return
        if suffix.endswith(".txt"):
            path.write_text(str(payload), encoding="utf-8")
            return
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _capture_screen_frame(self) -> Any | None:
        try:
            if hasattr(self.loop, "_read_one_frame"):
                frame, _ = self.loop._read_one_frame()
                return frame
            packet = self.loop.camera.read()
            frame = packet.bgr if hasattr(packet, "bgr") else packet
            if hasattr(self.loop, "_apply_rectify"):
                return self.loop._apply_rectify(frame)
            return frame
        except Exception:
            return None

    def _ensure_planner_files(
        self,
        step: int,
        *,
        obs_mode: str,
        planner_meta: dict[str, Any] | None,
        next_step: NextStep,
        task: str,
    ) -> None:
        meta = planner_meta or {}
        fallback_payload = {"task": task if obs_mode == "full" else "", "obs_mode": obs_mode, "input": "rule_fallback_or_local"}
        self._write_if_missing(step, "planner_llm_raw.txt", meta.get("raw", ""))
        self._write_if_missing(step, "planner_out.json", _model_dump(next_step))
        self._write_if_missing(
            step,
            f"planner_in_{obs_mode}.json",
            fallback_payload,
        )
        self._write_if_missing(step, "planner_obs_packed.json", fallback_payload)
        self._write_if_missing(
            step,
            "obs_digest.json",
            {
                "mode": obs_mode,
                "last_sent_obs_digest": None,
                "new_obs_digest": None,
                "elements_topk": 0 if obs_mode == "delta" else len(getattr(self.state.last_obs, "elements", []) if self.state.last_obs else []),
                "elements_delta": 0,
                "sampling_rule": "executor_minimal",
                "clipped": False,
                "clip_reason": "",
                "prompt_chars": 0,
                "packed_elements": 0,
            },
        )

    def _ensure_cursor(self, obs: Observation, step: int) -> bool:
        if obs.cursor_xy is not None:
            self.logger.log_cursor_recovery(step, attempts=0, ok=True)
            return True
        frame = self._capture_screen_frame()
        self.logger.write_cursor_lost(step, frame)
        self.logger.log_cursor_recovery(step, attempts=1, ok=False)
        return False

    def _default_scroll_verify(self, obs: Observation) -> VerifyRule:
        roi = (
            int(obs.screen_w * 0.2),
            int(obs.screen_h * 0.2),
            int(obs.screen_w * 0.8),
            int(obs.screen_h * 0.8),
        )
        return VerifyRule(type="TEXT_CHANGED", text="ui_change", roi=roi, min_delta=6)

    def _run_verify(self, before: Observation, after: Observation, rule: VerifyRule, fallback_ref: ElementRef | None = None) -> VerifyResult:
        return self.loop._run_verify_rule(before, after, rule, fallback_ref=fallback_ref)

    def _execute_scroll(self, step: int, delta: int, verify_rule: VerifyRule | None) -> dict[str, Any]:
        before = self.loop.observe(force_ui=True)
        active_rule = verify_rule if verify_rule is not None and verify_rule.type != "NONE" else self._default_scroll_verify(before)

        success = False
        last_verify: VerifyResult | None = None
        reason = "scroll_stuck"

        for idx, delta_once in enumerate([int(delta), int(delta), -int(delta)]):
            if hasattr(self.loop.hardware, "scroll"):
                self.loop.hardware.scroll(delta_once)
            time.sleep(0.12)
            after = self.loop.observe(force_ui=True)
            if not self._ensure_cursor(after, step):
                return {"ok": False, "reason": "cursor_lost", "verify": None}
            verify_result = self._run_verify(before, after, active_rule, fallback_ref=None)
            last_verify = verify_result
            if verify_result.ok:
                success = True
                reason = "ok"
                break
            if idx >= 1:
                time.sleep(0.08)

        return {
            "ok": success,
            "reason": reason,
            "verify": _model_dump(last_verify),
            "after_obs": _model_dump(after) if "after" in locals() else None,
        }

    def _execute_macro_intent(self, step: int, intent: str, preferred_path: str | None = None) -> dict[str, Any]:
        try:
            steps = self._macro_registry().expand_intent(
                intent,
                allow_hotkey=bool(getattr(self.config, "allow_hotkey", False)),
                preferred_path=preferred_path,
            )
        except Exception as exc:
            return {"ok": False, "reason": f"macro_expand_failed:{exc}", "failed_step": 0}

        self.logger.log_macro(
            step,
            name=intent,
            expanded_steps=[_model_dump(s) for s in steps],
            failed_step=None,
        )

        for idx, macro_step in enumerate(steps, start=1):
            obs_i = self.loop.observe(force_ui=True)
            if not self._ensure_cursor(obs_i, step):
                self.logger.log_macro(
                    step,
                    name=intent,
                    expanded_steps=[_model_dump(s) for s in steps],
                    failed_step=idx,
                )
                return {"ok": False, "reason": "cursor_lost", "failed_step": idx}

            action_name = str(getattr(macro_step, "action", "")).lower()
            if action_name == "wait":
                wait_ms = int(getattr(macro_step, "wait_ms", 200))
                time.sleep(max(0.0, wait_ms / 1000.0))
                continue
            if action_name in {"click", "right_click"}:
                selector = getattr(macro_step, "selector", None)
                if not selector:
                    self.logger.log_macro(
                        step,
                        name=intent,
                        expanded_steps=[_model_dump(s) for s in steps],
                        failed_step=idx,
                    )
                    return {"ok": False, "reason": "macro_step_missing_selector", "failed_step": idx}
                ref = ElementRef(by="query", query=str(selector))
                guard_step = NextStep(
                    action=action_from_json({"type": "CLICK", "target": _model_dump(ref), "button": "right" if action_name == "right_click" else "left"}),
                    verify=VerifyRule(type="NONE"),
                )
                guard_result = self.guard.check(guard_step, obs_i)
                if not guard_result.allowed:
                    self.logger.log_macro(
                        step,
                        name=intent,
                        expanded_steps=[_model_dump(s) for s in steps],
                        failed_step=idx,
                    )
                    return {
                        "ok": False,
                        "reason": guard_result.reason,
                        "failed_step": idx,
                        "policy_hit": guard_result.policy_hit,
                    }

                if action_name == "click":
                    ret = self.loop.click_and_verify(ref, verify_rule=None, retry=1)
                    if not bool(ret.get("ok", False)):
                        self.logger.log_macro(
                            step,
                            name=intent,
                            expanded_steps=[_model_dump(s) for s in steps],
                            failed_step=idx,
                        )
                        return {"ok": False, "reason": "macro_click_failed", "failed_step": idx, "details": ret}
                else:
                    ret = self.loop.click_target(ref, button="right")
                    if not bool(ret.get("ok", False)):
                        self.logger.log_macro(
                            step,
                            name=intent,
                            expanded_steps=[_model_dump(s) for s in steps],
                            failed_step=idx,
                        )
                        return {"ok": False, "reason": "macro_right_click_failed", "failed_step": idx, "details": ret}
                continue

            self.logger.log_macro(
                step,
                name=intent,
                expanded_steps=[_model_dump(s) for s in steps],
                failed_step=idx,
            )
            return {"ok": False, "reason": f"unsupported_macro_action:{action_name}", "failed_step": idx}

        self.logger.log_macro(
            step,
            name=intent,
            expanded_steps=[_model_dump(s) for s in steps],
            failed_step=None,
        )
        return {"ok": True, "reason": "ok", "failed_step": None}

    def _execute_action_once(self, step: int, next_step: NextStep, before_obs: Observation) -> dict[str, Any]:
        action_type = str(next_step.action.type).upper()
        payload = dict(next_step.action.payload or {})
        verify_rule = next_step.verify

        if action_type == "CLICK":
            ref = _element_ref_from_any(payload.get("target"))
            chosen: UIElement | None = None
            candidates: list[dict[str, Any]] = []
            resolve_status = "ok"
            try:
                chosen = self.loop._resolve_element(ref, before_obs)
                candidates = [
                    {
                        "stable_id": chosen.stable_id,
                        "role": chosen.role,
                        "text": chosen.text or chosen.label,
                        "confidence": chosen.confidence,
                        "bbox": chosen.bbox,
                    }
                ]
            except Exception:
                resolve_status = "no_match"
            self.logger.log_resolve(
                step,
                target=_model_dump(ref),
                status=resolve_status,
                chosen=None if chosen is None else candidates[0],
                candidates=candidates,
            )
            if chosen is None:
                return {"ok": False, "reason": "no_match", "verify": None}

            active_verify = verify_rule
            if active_verify.type == "NONE":
                active_verify = self.loop._default_verify_rule(before_obs, ref)

            button = str(payload.get("button", "left")).lower()
            if button == "right":
                click_ret = self.loop.click_target(ref, button="right")
                if not bool(click_ret.get("ok", False)):
                    return {"ok": False, "reason": "click_failed", "verify": None}
                after = self.loop.observe(force_ui=True)
                verify_ret = self._run_verify(before_obs, after, active_verify, fallback_ref=ref)
                return {
                    "ok": bool(verify_ret.ok),
                    "reason": "ok" if verify_ret.ok else "verify_failed",
                    "verify": _model_dump(verify_ret),
                    "after_obs": after,
                }

            click_ret = self.loop.click_and_verify(ref, verify_rule=active_verify, retry=1)
            return {
                "ok": bool(click_ret.get("ok", False)),
                "reason": str(click_ret.get("reason", "click_failed")),
                "verify": click_ret.get("verify"),
                "after_obs": None,
            }

        if action_type == "TYPE_TEXT":
            method = str(payload.get("method", "osk")).lower()
            if method != "osk":
                return {"ok": False, "reason": "keyboard_inject_disabled", "verify": None}
            text = str(payload.get("text", ""))
            target_input = payload.get("target_input")
            target_ref = _element_ref_from_any(target_input) if target_input else None
            profile = payload.get("keyboard_profile") or payload.get("profile")
            ret = self.loop.type_text_osk(text, target_input=target_ref, keyboard_profile=profile)
            self.logger.log_osk(
                step,
                text=text,
                profile=str(profile) if profile else None,
                shift_state="auto",
                compiled_keys=ret.get("details", {}).get("compiled", None) if isinstance(ret.get("details"), dict) else None,
                failed_char=ret.get("failed_char"),
            )
            return {
                "ok": bool(ret.get("ok", False)),
                "reason": str(ret.get("reason", "type_failed")),
                "verify": ret.get("verify"),
                "after_obs": None,
            }

        if action_type in {"OPEN_APP", "OPEN_OSK"}:
            intent = action_type
            macro_ret = self._execute_macro_intent(step, intent)
            return {
                "ok": bool(macro_ret.get("ok", False)),
                "reason": str(macro_ret.get("reason", "macro_failed")),
                "verify": None,
                "after_obs": None,
            }

        if action_type == "SCROLL":
            delta = int(payload.get("delta", 320))
            return self._execute_scroll(step, delta, verify_rule if verify_rule.type != "NONE" else None)

        if action_type == "WAIT":
            duration_ms = int(payload.get("duration_ms", 500))
            time.sleep(max(0.0, float(duration_ms) / 1000.0))
            after = self.loop.observe(force_ui=True)
            return {"ok": True, "reason": "ok", "verify": _model_dump(VerifyResult(ok=True, score=1.0, method="NONE", details={})), "after_obs": after}

        if action_type == "TOUCH_TAP":
            ret = self.loop.tap_target(
                x=payload.get("x"),
                y=payload.get("y"),
                gx=payload.get("gx"),
                gy=payload.get("gy"),
                coord_type=str(payload.get("coord_type", "phone_grid")),
            )
            after = self.loop.observe(force_ui=True)
            verify_ret = self._run_verify(before_obs, after, verify_rule if verify_rule.type != "NONE" else self._default_scroll_verify(before_obs))
            return {
                "ok": bool(ret.get("ok", False) and verify_ret.ok),
                "reason": "ok" if bool(ret.get("ok", False) and verify_ret.ok) else str(ret.get("reason", "tap_failed")),
                "verify": _model_dump(verify_ret),
                "after_obs": after,
            }

        if action_type == "DONE":
            return {"ok": True, "reason": payload.get("reason") or "done", "verify": None, "after_obs": None, "done": True}

        return {"ok": False, "reason": f"unsupported_action:{action_type}", "verify": None}

    def _execute_with_controls(self, step: int, next_step: NextStep, before_obs: Observation) -> dict[str, Any]:
        repeats = int(next_step.repeat or next_step.action.repeat or 1)
        ttl_ms = int(next_step.action_ttl_ms or next_step.action.action_ttl_ms or 0)
        continue_until = next_step.continue_until or next_step.action.continue_until
        deadline = time.time() + (ttl_ms / 1000.0) if ttl_ms > 0 else None

        last: dict[str, Any] = {"ok": False, "reason": "not_executed", "verify": None}
        for _ in range(max(1, repeats)):
            if deadline is not None and time.time() > deadline:
                last = {"ok": False, "reason": "timeout", "verify": last.get("verify")}
                break
            last = self._execute_action_once(step, next_step, before_obs)
            if last.get("done"):
                break
            after_obs = last.get("after_obs")
            if isinstance(after_obs, Observation) and not self._ensure_cursor(after_obs, step):
                return {"ok": False, "reason": "cursor_lost", "verify": last.get("verify")}

            if continue_until == "verify_ok":
                verify_obj = last.get("verify") or {}
                verify_ok = bool(verify_obj.get("ok")) if isinstance(verify_obj, dict) else False
                if verify_ok:
                    break
            elif continue_until == "ui_change":
                if isinstance(after_obs, Observation) and float(getattr(after_obs, "ui_change_score", 0.0)) >= 0.05:
                    break
            elif continue_until == "cursor_lost":
                if isinstance(after_obs, Observation) and after_obs.cursor_xy is None:
                    break
            elif continue_until == "timeout" and deadline is not None and time.time() > deadline:
                break

            if not bool(last.get("ok", False)):
                break

        return last

    def run_step(self, task: str) -> StepResult:
        if self.state.done:
            return StepResult(
                step=self.state.step,
                ok=True,
                done=True,
                reason=self.state.done_reason or "done",
                action=None,
                verify=None,
                guard=None,
                planner=None,
            )

        self.state.step += 1
        step = self.state.step
        obs_mode = "full" if step == 1 else "delta"
        self.logger.begin_step(step, task=task, obs_mode=obs_mode)

        try:
            self.logger.set_phase(step, "observe")
            before_obs = self.loop.observe(force_ui=True)
            self.state.last_obs = before_obs
            screen_bgr = self._capture_screen_frame()
            self.logger.log_observation(step, before_obs, screen_bgr=screen_bgr)
            if not self._ensure_cursor(before_obs, step):
                self.state.done = True
                self.state.done_reason = "cursor_lost"
                self.logger.log_execution(step, action_executed=None, success=False, reason="cursor_lost")
                self.logger.log_verify(step, None)
                return StepResult(step=step, ok=False, done=True, reason="cursor_lost")

            self.logger.set_phase(step, "plan")
            next_step = self.planner.plan(
                task,
                before_obs,
                self.state.memory,
                step=step,
                obs_mode=obs_mode,
                artifacts_dir=self.artifacts_dir,
            )
            planner_meta = getattr(self.planner, "last_meta", {}) or {}
            self.logger.log_planner(step, planner_meta, task=task, obs_mode=obs_mode)
            self._ensure_planner_files(step, obs_mode=obs_mode, planner_meta=planner_meta, next_step=next_step, task=task)

            if str(next_step.action.type).upper() == "DONE":
                reason = str(next_step.action.payload.get("reason", "done"))
                self.state.done = True
                self.state.done_reason = reason
                self.logger.log_execution(step, action_executed=next_step.action, success=True, reason=reason)
                self.logger.log_verify(step, None)
                return StepResult(
                    step=step,
                    ok=True,
                    done=True,
                    reason=reason,
                    action=_model_dump(next_step.action),
                    verify=None,
                    guard={"allowed": True, "reason": "skipped_done"},
                    planner=planner_meta,
                )

            self.logger.set_phase(step, "guard")
            guard_result = self.guard.check(next_step, before_obs)
            if not guard_result.allowed:
                self.state.done = True
                self.state.done_reason = guard_result.reason
                self.logger.log_execution(step, action_executed=next_step.action, success=False, reason=guard_result.reason)
                self.logger.log_verify(step, None)
                self.logger.append_note(step, f"policy_hit={','.join(guard_result.policy_hit)}")
                return StepResult(
                    step=step,
                    ok=False,
                    done=True,
                    reason=guard_result.reason,
                    action=_model_dump(next_step.action),
                    verify=None,
                    guard=guard_result.to_dict(),
                    planner=planner_meta,
                )

            self.logger.set_phase(step, "act")
            exec_ret = self._execute_with_controls(step, next_step, before_obs)
            if str(exec_ret.get("reason", "")) == "cursor_lost":
                self.logger.write_cursor_lost(step, self._capture_screen_frame())
            self.logger.log_execution(
                step,
                action_executed=next_step.action,
                success=bool(exec_ret.get("ok", False)),
                reason=str(exec_ret.get("reason", "")),
            )

            self.logger.set_phase(step, "verify")
            self.logger.log_verify(step, exec_ret.get("verify"))

            self.state.memory["last_action"] = _model_dump(next_step.action)
            self.state.memory["last_verify"] = exec_ret.get("verify")
            if str(next_step.action.type).upper() in {"OPEN_APP", "OPEN_OSK"}:
                self.state.memory["macro_state"] = {"last_macro": str(next_step.action.type), "status": exec_ret.get("reason")}

            action_type = str(next_step.action.type).upper()
            done = False
            reason = str(exec_ret.get("reason", ""))
            if action_type == "DONE":
                done = True
            if reason in {"planner_failed", "click_failed", "scroll_stuck", "macro_failed", "cursor_lost", "soft_limit"}:
                done = True
            if reason.startswith("macro_"):
                done = True

            if not bool(exec_ret.get("ok", False)):
                self.state.failures += 1
            else:
                self.state.failures = 0

            if self.state.failures >= 3 and not done:
                done = True
                reason = "failure_limit"

            if done:
                self.state.done = True
                self.state.done_reason = reason or "done"

            return StepResult(
                step=step,
                ok=bool(exec_ret.get("ok", False)),
                done=done,
                reason=reason,
                action=_model_dump(next_step.action),
                verify=exec_ret.get("verify"),
                guard=guard_result.to_dict(),
                planner=planner_meta,
            )

        except KeyboardInterrupt:
            self.state.done = True
            self.state.done_reason = "interrupted"
            self.logger.append_note(step, "interrupted")
            self.logger.log_execution(step, action_executed=None, success=False, reason="interrupted")
            return StepResult(step=step, ok=False, done=True, reason="interrupted")
        except Exception as exc:
            self.state.done = True
            self.state.done_reason = "executor_exception"
            self.logger.append_note(step, f"exception={exc}")
            self.logger.log_execution(step, action_executed=None, success=False, reason=str(exc))
            return StepResult(
                step=step,
                ok=False,
                done=True,
                reason=f"executor_exception:{exc}",
                planner=getattr(self.planner, "last_meta", None),
            )

    def run_task(self, task: str, max_steps: int = 30) -> dict[str, Any]:
        task_text = str(task or "").strip()
        if not task_text:
            return {"ok": False, "reason": "empty_task", "steps": []}

        results: list[dict[str, Any]] = []
        for _ in range(max(1, int(max_steps))):
            result = self.run_step(task_text)
            results.append(result.to_dict())
            if result.done:
                break

        return {
            "ok": bool(results and results[-1]["ok"]),
            "done": bool(results and results[-1]["done"]),
            "reason": results[-1]["reason"] if results else "no_steps",
            "steps": results,
        }
