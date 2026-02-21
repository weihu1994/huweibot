from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huweibot.agent.schemas import ElementRef, NextStep
from huweibot.core.observation import Observation, UIElement
from huweibot.core.selector import NoMatch, select_elements


def _to_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except Exception:
            return None
    return None


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _point_in_bbox(x: float, y: float, b: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _extract_action(next_step: NextStep) -> tuple[str, dict[str, Any]]:
    action = next_step.action
    payload = dict(action.payload or {})
    return str(action.type).upper(), payload


@dataclass
class GuardResult:
    allowed: bool
    reason: str = "ok"
    policy_hit: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "policy_hit": list(self.policy_hit),
            "needs_confirmation": bool(self.needs_confirmation),
        }


class SemanticGuard:
    def __init__(
        self,
        config: Any,
        selector: Any | None = None,
        *,
        hardware: Any | None = None,
        kinematics: Any | None = None,
        environment_state: dict[str, Any] | None = None,
    ):
        self.config = config
        self.selector = selector
        self.hardware = hardware
        self.kinematics = kinematics
        self.environment_state = environment_state if environment_state is not None else {}

    def _resolve(self, ref: ElementRef, obs: Observation) -> UIElement | None:
        if ref.by == "id":
            for elem in obs.elements:
                if elem.stable_id == ref.id:
                    return elem
            return None

        query = ref.query or ""
        if not query:
            return None

        try:
            if self.selector is not None and hasattr(self.selector, "resolve"):
                return self.selector.resolve(ref, obs.elements)  # type: ignore[no-any-return]
            if self.selector is not None and callable(self.selector):
                return self.selector(query, obs.elements, observation=obs)  # type: ignore[no-any-return]
            return select_elements(obs.elements, query, observation=obs, config=self.config, match="best")  # type: ignore[return-value]
        except NoMatch:
            return None
        except Exception:
            return None

    def _environment_check(self) -> GuardResult:
        if self.environment_state.get("environment_valid", True) is False:
            return GuardResult(allowed=False, reason="environment_invalid", policy_hit=["environment_invalid"])
        if self.environment_state.get("validate_screen_ok", True) is False:
            return GuardResult(allowed=False, reason="environment_invalid", policy_hit=["validate_screen_failed"])

        require_homed = bool(getattr(self.config, "guard_require_homed", True))
        if require_homed:
            homed_ok = False
            if self.hardware is not None:
                homed_ok = bool(getattr(self.hardware, "is_homed", False))
            if not homed_ok:
                flag_path = Path(str(getattr(self.config, "homed_flag_path", "artifacts/homed.flag")))
                homed_ok = flag_path.exists()
            if not homed_ok:
                return GuardResult(allowed=False, reason="not_homed", policy_hit=["not_homed"])
        return GuardResult(allowed=True)

    def _permission_prompt_check(self, obs: Observation) -> GuardResult | None:
        if not bool(getattr(self.config, "block_permission_prompt", True)):
            return None
        keywords = [
            "uac",
            "administrator",
            "admin",
            "allow this app",
            "allow changes",
            "install",
            "uninstall",
            "privacy",
            "permission",
            "权限",
            "管理员",
            "安装",
            "卸载",
            "允许更改",
            "隐私",
        ]
        haystacks: list[str] = []
        if obs.app_hint:
            haystacks.append(obs.app_hint.lower())
        for elem in obs.elements:
            for text in (elem.text, elem.label):
                if text:
                    haystacks.append(text.lower())
        joined = "\n".join(haystacks)
        if any(k in joined for k in keywords):
            return GuardResult(
                allowed=False,
                reason="permission_prompt",
                policy_hit=["permission_prompt"],
                needs_confirmation=False,
            )
        return None

    def _type_text_check(self, payload: dict[str, Any], obs: Observation) -> GuardResult:
        method = str(payload.get("method", "osk")).lower()
        if method != "osk":
            return GuardResult(
                allowed=False,
                reason="keyboard_inject_disabled",
                policy_hit=["keyboard_inject_disabled"],
            )
        text = str(payload.get("text", ""))
        max_len = int(getattr(self.config, "max_type_text_len", 200))
        if len(text) > max_len:
            return GuardResult(allowed=False, reason="text_too_long", policy_hit=["text_too_long"])
        patterns = getattr(
            self.config,
            "block_sensitive_text_patterns",
            ["password", "passwd", "otp", "验证码", "cvv", "card", "银行卡", "信用卡"],
        )
        lowered = text.lower()
        if any(str(p).lower() in lowered for p in patterns):
            return GuardResult(allowed=False, reason="sensitive_text", policy_hit=["sensitive_text"])
        if not obs.keyboard_mode:
            return GuardResult(allowed=True, reason="ok", policy_hit=["will_open_osk"])
        return GuardResult(allowed=True)

    def _screen_bounds_check(self, point: tuple[float, float] | None, obs: Observation) -> GuardResult | None:
        if point is None:
            return None
        x, y = point
        if x < 0 or y < 0 or x > float(obs.screen_w - 1) or y > float(obs.screen_h - 1):
            return GuardResult(allowed=False, reason="outside_screen", policy_hit=["outside_screen"])
        return None

    def _region_check(
        self,
        *,
        target_bbox: tuple[float, float, float, float] | None,
        target_point: tuple[float, float] | None,
    ) -> GuardResult | None:
        blocked_regions_raw = getattr(self.config, "blocked_regions", []) or []
        blocked_regions = [_to_bbox(v) for v in blocked_regions_raw]
        blocked_regions = [b for b in blocked_regions if b is not None]
        allowed_region = _to_bbox(getattr(self.config, "allowed_region", None))

        if target_bbox is not None:
            for blocked in blocked_regions:
                if _bbox_intersects(target_bbox, blocked):
                    return GuardResult(allowed=False, reason="blocked_region", policy_hit=["blocked_region"])
            if allowed_region is not None and not _bbox_intersects(target_bbox, allowed_region):
                return GuardResult(
                    allowed=False,
                    reason="outside_allowed_region",
                    policy_hit=["outside_allowed_region"],
                )

        if target_point is not None:
            x, y = target_point
            for blocked in blocked_regions:
                if _point_in_bbox(x, y, blocked):
                    return GuardResult(allowed=False, reason="blocked_region", policy_hit=["blocked_region"])
            if allowed_region is not None and not _point_in_bbox(x, y, allowed_region):
                return GuardResult(
                    allowed=False,
                    reason="outside_allowed_region",
                    policy_hit=["outside_allowed_region"],
                )
        return None

    def _soft_limit_check(
        self,
        *,
        target_point: tuple[float, float] | None,
        obs: Observation,
    ) -> GuardResult | None:
        if target_point is None:
            return None
        if obs.cursor_xy is None:
            return None
        if self.kinematics is None or self.hardware is None:
            return None
        try:
            dx_px = float(target_point[0]) - float(obs.cursor_xy[0])
            dy_px = float(target_point[1]) - float(obs.cursor_xy[1])
            cur_pos = getattr(self.hardware, "pos_mm", None)
            if hasattr(self.kinematics, "px_to_mm"):
                try:
                    self.kinematics.px_to_mm(dx_px, dy_px, current_pos_mm=cur_pos)
                except TypeError:
                    self.kinematics.px_to_mm(dx_px, dy_px)
            if hasattr(self.hardware, "_check_soft_limit") and cur_pos is not None:
                dx_mm, dy_mm = self.kinematics.px_to_mm(dx_px, dy_px, current_pos_mm=cur_pos)
                self.hardware._check_soft_limit(float(cur_pos[0]) + dx_mm, float(cur_pos[1]) + dy_mm)
        except Exception:
            return GuardResult(allowed=False, reason="soft_limit", policy_hit=["soft_limit"])
        return None

    def check(self, next_step: NextStep, obs: Observation) -> GuardResult:
        env = self._environment_check()
        if not env.allowed:
            return env

        permission = self._permission_prompt_check(obs)
        if permission is not None and not permission.allowed:
            return permission

        action_type, payload = _extract_action(next_step)
        target_bbox: tuple[float, float, float, float] | None = None
        target_point: tuple[float, float] | None = None
        policy_hit: list[str] = []

        if action_type in {"CLICK", "DRAG", "SCROLL", "CLICK_AT"}:
            target_ref_raw = payload.get("target")
            target_ref = None
            if target_ref_raw is not None:
                target_ref = target_ref_raw if isinstance(target_ref_raw, ElementRef) else ElementRef.parse_obj(target_ref_raw)
                resolved = self._resolve(target_ref, obs)
                if resolved is None:
                    return GuardResult(allowed=False, reason="cannot_resolve_target", policy_hit=["cannot_resolve_target"])
                target_bbox = resolved.bbox
                target_point = _bbox_center(resolved.bbox)

        if action_type == "CLICK_AT":
            point = payload.get("coord") or payload
            try:
                target_point = (float(point["x"]), float(point["y"]))
            except Exception:
                return GuardResult(allowed=False, reason="invalid_click_point", policy_hit=["invalid_click_point"])

        if action_type == "DRAG":
            from_coord = payload.get("from") or payload.get("from_coord")
            to_coord = payload.get("to") or payload.get("to_coord")
            try:
                from_point = (float(from_coord["x"]), float(from_coord["y"]))
                to_point = (float(to_coord["x"]), float(to_coord["y"]))
            except Exception:
                return GuardResult(allowed=False, reason="invalid_drag_points", policy_hit=["invalid_drag_points"])
            target_point = to_point
            target_bbox = (from_point[0], from_point[1], to_point[0], to_point[1])

        if action_type == "TYPE_TEXT":
            type_check = self._type_text_check(payload, obs)
            if not type_check.allowed:
                return type_check
            policy_hit.extend(type_check.policy_hit)

        bounds = self._screen_bounds_check(target_point, obs)
        if bounds is not None and not bounds.allowed:
            return bounds

        region = self._region_check(target_bbox=target_bbox, target_point=target_point)
        if region is not None and not region.allowed:
            return region

        soft = self._soft_limit_check(target_point=target_point, obs=obs)
        if soft is not None and not soft.allowed:
            return soft

        return GuardResult(allowed=True, reason="ok", policy_hit=policy_hit)
