from __future__ import annotations

from typing import Any

from huweibot.agent.router import Router
from huweibot.agent.schemas import NextStep, VerifyRule, action_from_json
from huweibot.core.observation import Observation


class BasePlanner:
    def __init__(self) -> None:
        self.last_meta: dict[str, Any] = {}

    def plan(
        self,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        *,
        step: int,
        obs_mode: str,
        artifacts_dir: str,
    ) -> NextStep:
        raise NotImplementedError


class RulePlanner(BasePlanner):
    def plan(
        self,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        *,
        step: int,
        obs_mode: str,
        artifacts_dir: str,
    ) -> NextStep:
        del memory, step, artifacts_dir
        text_task = (task or "").strip()
        if not text_task:
            next_step = NextStep(
                action=action_from_json({"type": "DONE", "reason": "empty_task"}),
                verify=VerifyRule(type="NONE"),
                obs_mode=obs_mode,
            )
            self.last_meta = {
                "provider": "rule",
                "model": "rule",
                "capabilities": {},
                "repair_used": False,
                "error": None,
                "obs_mode": obs_mode,
                "output": next_step.dict(),
            }
            return next_step

        # Conservative local fallback: click obvious positive buttons if present.
        candidates = []
        for element in obs.elements:
            text = (element.text or element.label or "").lower()
            if element.role in {"button", "toggle"} and any(k in text for k in ("ok", "yes", "next", "continue", "确认", "下一步")):
                candidates.append(element)
        if candidates:
            chosen = sorted(candidates, key=lambda e: float(e.confidence), reverse=True)[0]
            if chosen.stable_id:
                target = {"by": "id", "id": chosen.stable_id}
            else:
                seed = (chosen.text or chosen.label or "ok").strip()
                target = {"by": "query", "query": f"role:button text_contains:{seed}"}
            next_step = NextStep(
                action=action_from_json({"type": "CLICK", "target": target, "button": "left"}),
                verify=VerifyRule(type="ELEMENT_CHANGED", target=target),
                obs_mode=obs_mode,
            )
        else:
            next_step = NextStep(
                action=action_from_json({"type": "WAIT", "duration_ms": 500}),
                verify=VerifyRule(type="NONE"),
                obs_mode=obs_mode,
            )

        self.last_meta = {
            "provider": "rule",
            "model": "rule",
            "capabilities": {},
            "repair_used": False,
            "error": None,
            "obs_mode": obs_mode,
            "output": next_step.dict(),
        }
        return next_step


class LLMPlanner(BasePlanner):
    def __init__(self, router: Router):
        super().__init__()
        self.router = router
        self.rule_fallback = RulePlanner()

    def plan(
        self,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        *,
        step: int,
        obs_mode: str,
        artifacts_dir: str,
    ) -> NextStep:
        try:
            next_step, meta = self.router.request_plan(
                task=task,
                obs=obs,
                memory=memory,
                step=step,
                obs_mode=obs_mode,
                debug_reasoning=bool(memory.get("debug_reasoning", False)),
                artifacts_dir=artifacts_dir,
            )
            self.last_meta = meta
            return next_step
        except Exception as exc:
            self.last_meta = {
                "provider": "router",
                "model": "unknown",
                "capabilities": {},
                "repair_used": False,
                "error": str(exc),
                "obs_mode": obs_mode,
                "output": None,
            }
            # Hard fallback when planner output is invalid.
            return NextStep.done(reason="planner_failed")


def build_planner(router: Router | None) -> BasePlanner:
    if router is None:
        return RulePlanner()
    return LLMPlanner(router)
