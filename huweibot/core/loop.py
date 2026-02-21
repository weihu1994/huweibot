from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from huweibot.agent.macros import load_macros
from huweibot.agent.osk import OSKExecutor, OSKPlanner
from huweibot.agent.schemas import ElementRef, VerifyRule
from huweibot.agent.verifier import (
    VerifyResult,
    verify_element_changed_ref,
    verify_input_effect,
    verify_text_changed,
    verify_text_present,
)
from huweibot.core.observation import Observation, UIElement
from huweibot.core.coords import grid_to_screen_px_phone
from huweibot.core.selector import AmbiguousMatch, NoMatch, select_elements
from huweibot.control.pointer import PCPointerController, PhonePointerController
from huweibot.control.touch import XYZTouchController
from huweibot.vision.cursor_detect import detect_cursor, recover_cursor
from huweibot.vision.phone_screen import auto_detect_phone_screen, load_phone_screen
from huweibot.vision.ui_elements import (
    compute_screen_hash,
    compute_ui_change_score,
    detect_keyboard_roi,
    extract_ui_elements,
    infer_app_hint,
)

LOGGER = logging.getLogger(__name__)


class LoopState(BaseModel):
    step_index: int = 0
    done: bool = False
    last_error: str | None = None


@dataclass
class LoopConfig:
    threshold_px: float = 3.0
    max_iters: int = 20
    k_p: float = 0.55
    max_step_px: float = 180.0
    settle_ms: int = 120
    best_of_n: int = 3
    min_confidence: float = 0.55
    no_progress_limit: int = 4
    ui_every_n_iters: int = 3
    allow_vlm: bool = False
    hover_delta_px: int = 8


class XBotLoop:
    def __init__(
        self,
        *,
        camera: Any,
        hardware: Any,
        kinematics: Any,
        rectifier: Any | None = None,
        ui_extractor: Any | None = None,
        cursor_detector: Any | None = None,
        tracker: Any | None = None,
        selector: Any | None = None,
        config: LoopConfig | None = None,
        runtime_config: Any | None = None,
        pointer_controller: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self.camera = camera
        self.hardware = hardware
        self.kinematics = kinematics
        self.rectifier = rectifier
        self.ui_extractor = ui_extractor or extract_ui_elements
        self.cursor_detector = cursor_detector or detect_cursor
        self.tracker = tracker
        self.selector = selector
        self.config = config or LoopConfig()
        self.runtime_config = runtime_config
        self.logger = logger or LOGGER
        self.mode = str(getattr(runtime_config, "mode", "pc")).lower()
        self.phone_screen_detection = str(getattr(runtime_config, "phone_screen_detection", "manual")).lower()
        self.phone_screen_calibration_path = str(
            getattr(runtime_config, "phone_screen_calibration_path", "config/phone_screen.json")
        )
        self.phone_grid_w = int(getattr(runtime_config, "phone_grid_w", 200))
        self.phone_grid_h = int(getattr(runtime_config, "phone_grid_h", 100))
        self.phone_allowed_margin_px = int(getattr(runtime_config, "phone_allowed_margin_px", 6))
        self._phone_screen_bbox_cache: tuple[float, float, float, float] | None = None

        self._iteration = 0
        self._last_obs: Observation | None = None
        self._macros = None
        self._vlm_state: dict[str, Any] = {
            "last_vlm_ts": 0.0,
            "vlm_calls": 0,
            "vlm_cooldown_s": float(getattr(runtime_config, "vlm_cooldown_s", 10.0)),
            "vlm_max_per_task": int(getattr(runtime_config, "vlm_max_per_task", 8)),
            "vlm_force_allow_if_elements0_streak": int(
                getattr(runtime_config, "vlm_force_allow_if_elements0_streak", 3)
            ),
            "elements_min": int(getattr(getattr(runtime_config, "vlm_trigger_thresholds", None), "elements_min", 3)),
            "ambiguous_trigger": int(getattr(getattr(runtime_config, "vlm_trigger_thresholds", None), "ambiguous_streak", 2)),
            "macro_fail_trigger": int(getattr(getattr(runtime_config, "vlm_trigger_thresholds", None), "macro_fail_streak", 2)),
            "ui_change_threshold": float(
                getattr(getattr(runtime_config, "vlm_trigger_thresholds", None), "ui_change_threshold", 0.15)
            ),
        }
        if pointer_controller is not None:
            self.pointer_controller = pointer_controller
        elif self.mode == "phone":
            touch_controller = hardware if isinstance(hardware, XYZTouchController) else XYZTouchController(
                xy_controller=getattr(hardware, "grbl", None) if hardware is not None else None,
                z_axis_enabled=bool(getattr(runtime_config, "z_axis_enabled", False)),
                ir_enabled=bool(getattr(runtime_config, "ir_enabled", False)),
            )
            self.pointer_controller = PhonePointerController(
                touch_controller=touch_controller,
                screen_bbox=None,
            )
        else:
            self.pointer_controller = PCPointerController(loop=self, hardware=hardware)
        keyword_path = getattr(runtime_config, "keyboard_keyword_list_path", "assets/ui/keyboard_words.txt")
        self.osk_planner = OSKPlanner(keyword_list_path=keyword_path)
        self.osk_executor = OSKExecutor()

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        if kwargs:
            self.logger.log(level, "%s | %s", message, kwargs)
        else:
            self.logger.log(level, message)

    def _apply_rectify(self, frame_bgr: Any) -> Any:
        if self.rectifier is None:
            return frame_bgr
        if callable(self.rectifier):
            return self.rectifier(frame_bgr)
        if hasattr(self.rectifier, "rectify"):
            return self.rectifier.rectify(frame_bgr)
        return frame_bgr

    def _read_one_frame(self) -> tuple[Any, float]:
        packet = self.camera.read()
        if hasattr(packet, "bgr"):
            frame = packet.bgr
            ts = float(getattr(packet, "timestamp", time.time()))
        else:
            frame = packet
            ts = time.time()
        return self._apply_rectify(frame), ts

    def _detect_cursor_on_frame(self, frame_bgr: Any) -> Any:
        detector = self.cursor_detector
        if callable(detector):
            try:
                return detector(frame_bgr)
            except TypeError:
                return detector.detect(frame_bgr)
        if hasattr(detector, "detect"):
            return detector.detect(frame_bgr)
        return detect_cursor(frame_bgr)

    def _extract_elements(self, frame_bgr: Any, allow_vlm: bool) -> list[UIElement]:
        ui_mode = getattr(self.runtime_config, "ui_mode", "local")
        max_elements = int(getattr(self.runtime_config, "max_elements", 128))
        min_text_conf = float(getattr(self.runtime_config, "min_text_conf", 0.4))
        min_elem_conf = float(getattr(self.runtime_config, "min_elem_conf", 0.3))
        element_merge_iou = float(getattr(self.runtime_config, "element_merge_iou", 0.5))
        txt_trunc_policy = str(getattr(self.runtime_config, "txt_trunc_policy", "head12~tail6"))

        kwargs = {
            "ui_mode": ui_mode,
            "max_elements": max_elements,
            "min_text_conf": min_text_conf,
            "min_elem_conf": min_elem_conf,
            "element_merge_iou": element_merge_iou,
            "txt_trunc_policy": txt_trunc_policy,
            "allow_vlm": bool(allow_vlm),
            "vlm_gate_passed": None,
            "vlm_state": self._vlm_state,
            "ui_change_score": float(self._last_obs.ui_change_score) if self._last_obs is not None else 0.0,
            "artifacts_dir": str(getattr(self.runtime_config, "artifacts_dir", "artifacts")),
            "vlm_image_max_side": int(getattr(self.runtime_config, "vlm_image_max_side", 1280)),
            "vlm_jpeg_quality": int(getattr(self.runtime_config, "vlm_jpeg_quality", 80)),
        }
        try:
            result = self.ui_extractor(frame_bgr, **kwargs)
        except TypeError:
            kwargs.pop("vlm_state", None)
            kwargs.pop("vlm_gate_passed", None)
            kwargs.pop("ui_change_score", None)
            kwargs.pop("artifacts_dir", None)
            kwargs.pop("vlm_image_max_side", None)
            kwargs.pop("vlm_jpeg_quality", None)
            result = self.ui_extractor(frame_bgr, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def _detect_phone_screen_bbox(self, frame_bgr: Any) -> tuple[float, float, float, float] | None:
        if self.phone_screen_detection == "manual":
            try:
                bbox = load_phone_screen(self.phone_screen_calibration_path)
                self._phone_screen_bbox_cache = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except Exception as exc:
                self._log(logging.WARNING, "phone screen calibration unavailable", error=str(exc))
        elif self.phone_screen_detection == "auto":
            auto_bbox = auto_detect_phone_screen(frame_bgr)
            if auto_bbox is not None:
                self._phone_screen_bbox_cache = (
                    float(auto_bbox[0]),
                    float(auto_bbox[1]),
                    float(auto_bbox[2]),
                    float(auto_bbox[3]),
                )
        return self._phone_screen_bbox_cache

    def _bbox_with_margin(
        self,
        bbox: tuple[float, float, float, float],
        margin_px: int,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        margin = max(0, int(margin_px))
        nx1 = x1 + margin
        ny1 = y1 + margin
        nx2 = x2 - margin
        ny2 = y2 - margin
        if nx2 <= nx1 or ny2 <= ny1:
            return bbox
        return (nx1, ny1, nx2, ny2)

    def _point_in_bbox(self, x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = bbox
        return float(x1) <= float(x) <= float(x2) and float(y1) <= float(y) <= float(y2)

    def _resolve_phone_screen_bbox(self) -> tuple[float, float, float, float] | None:
        if self._last_obs is not None and self._last_obs.phone_screen_bbox is not None:
            return self._last_obs.phone_screen_bbox
        try:
            frame, _ = self._read_one_frame()
        except Exception:
            return self._phone_screen_bbox_cache
        return self._detect_phone_screen_bbox(frame)

    def _track_elements(self, prev: Observation | None, cur_elements: list[UIElement]) -> list[UIElement]:
        if not bool(getattr(self.runtime_config, "enable_tracking", True)):
            return cur_elements
        if self.tracker is None:
            return cur_elements

        prev_elements = prev.elements if prev is not None else []
        if hasattr(self.tracker, "track"):
            return self.tracker.track(prev_elements, cur_elements)
        if callable(self.tracker):
            return self.tracker(prev_elements, cur_elements)
        return cur_elements

    def observe(self, force_ui: bool = False) -> Observation:
        best_detection = None
        best_frame = None
        best_ts = time.time()

        for _ in range(max(1, int(self.config.best_of_n))):
            frame, ts = self._read_one_frame()
            detection = self._detect_cursor_on_frame(frame)
            if best_detection is None or float(getattr(detection, "cursor_conf", 0.0)) >= float(
                getattr(best_detection, "cursor_conf", 0.0)
            ):
                best_detection = detection
                best_frame = frame
                best_ts = ts

        if best_detection is None or best_frame is None:
            raise RuntimeError("observe failed: no frame available")

        cursor_conf = float(getattr(best_detection, "cursor_conf", 0.0))
        cursor_xy = getattr(best_detection, "cursor_xy", None)
        cursor_type = str(getattr(best_detection, "cursor_type", "unknown") or "unknown")

        if cursor_xy is None or cursor_conf < float(self.config.min_confidence):
            recovered = recover_cursor(
                self.camera,
                min_confidence=float(self.config.min_confidence),
                frame_transform=self._apply_rectify,
            )
            if recovered.cursor_xy is not None and recovered.cursor_conf >= float(self.config.min_confidence):
                cursor_xy = recovered.cursor_xy
                cursor_conf = float(recovered.cursor_conf)
                cursor_type = recovered.cursor_type
                if recovered.frame_bgr is not None:
                    best_frame = recovered.frame_bgr
                if recovered.timestamp is not None:
                    best_ts = float(recovered.timestamp)
            else:
                cursor_xy = None

        do_ui = force_ui or (self._iteration % max(1, int(self.config.ui_every_n_iters)) == 0)
        elements: list[UIElement] = []
        if do_ui:
            elements = self._extract_elements(best_frame, allow_vlm=bool(self.config.allow_vlm))

        screen_h, screen_w = best_frame.shape[:2]
        phone_screen_bbox: tuple[float, float, float, float] | None = None
        distance_mm: float | None = None
        if self.mode == "phone":
            phone_screen_bbox = self._detect_phone_screen_bbox(best_frame)
            if hasattr(self.pointer_controller, "set_screen_bbox"):
                self.pointer_controller.set_screen_bbox(phone_screen_bbox)
            touch = getattr(self.pointer_controller, "touch_controller", None)
            if touch is not None and hasattr(touch, "read_distance_mm"):
                try:
                    distance_mm = touch.read_distance_mm()
                except Exception:
                    distance_mm = None
        keyboard_roi = detect_keyboard_roi(
            elements,
            screen_w,
            screen_h,
            keyword_list_path=getattr(self.runtime_config, "keyboard_keyword_list_path", "assets/ui/keyboard_words.txt"),
            density_threshold=float(getattr(self.runtime_config, "keyboard_density_threshold", 0.18)),
        )
        keyboard_mode = keyboard_roi is not None

        tracked = self._track_elements(self._last_obs, elements)
        screen_hash = compute_screen_hash(best_frame)
        ui_change = compute_ui_change_score(
            self._last_obs.screen_hash if self._last_obs else None,
            screen_hash,
            self._last_obs.elements if self._last_obs else [],
            tracked,
        )

        obs = Observation(
            timestamp=best_ts,
            screen_w=int(screen_w),
            screen_h=int(screen_h),
            cursor_xy=cursor_xy,
            cursor_conf=cursor_conf,
            cursor_type=cursor_type,
            elements=tracked,
            app_hint=infer_app_hint(tracked),
            keyboard_mode=keyboard_mode,
            keyboard_roi=keyboard_roi,
            screen_hash=screen_hash,
            ui_change_score=ui_change,
            phone_screen_bbox=phone_screen_bbox,
            phone_grid_w=self.phone_grid_w if self.mode == "phone" else None,
            phone_grid_h=self.phone_grid_h if self.mode == "phone" else None,
            touch_pen_xy=None,
            touch_pen_conf=0.0,
            distance_mm=distance_mm,
            device_mode=self.mode,
        )

        self._last_obs = obs
        self._iteration += 1
        return obs

    def _resolve_element(self, ref: ElementRef, obs: Observation) -> UIElement:
        if ref.by == "id":
            for element in obs.elements:
                if element.stable_id == ref.value:
                    return element
            raise NoMatch(f"stable_id not found: {ref.value}")
        if ref.by == "query":
            return select_elements(
                obs.elements,
                ref.value,
                observation=obs,
                config=self.runtime_config,
                match="best",
            )
        raise ValueError(f"unsupported ElementRef.by: {ref.by}")

    def move_to(self, x: float, y: float) -> dict[str, Any]:
        if self.mode == "phone":
            bbox = self._resolve_phone_screen_bbox()
            if bbox is None:
                return {"ok": False, "reason": "phone_screen_not_calibrated", "iters": 0}
            safe_bbox = self._bbox_with_margin(bbox, self.phone_allowed_margin_px)
            if not self._point_in_bbox(float(x), float(y), safe_bbox):
                return {"ok": False, "reason": "out_of_phone_bounds", "iters": 0}
            moved = self.pointer_controller.move_to_screen_px(float(x), float(y))
            return {
                "ok": bool(moved.ok),
                "reason": str(moved.reason),
                "iters": 1,
                "target": {"xy": (float(x), float(y)), "bbox": safe_bbox},
            }

        obs = self.observe(force_ui=False)
        if obs.cursor_xy is None:
            return {"ok": False, "reason": "cursor_lost", "iters": 0}

        target_x = float(x)
        target_y = float(y)
        no_progress = 0
        prev_err = None

        for it in range(int(self.config.max_iters)):
            cx, cy = obs.cursor_xy
            err_x = target_x - float(cx)
            err_y = target_y - float(cy)
            if abs(err_x) <= float(self.config.threshold_px) and abs(err_y) <= float(self.config.threshold_px):
                return {
                    "ok": True,
                    "reason": "ok",
                    "iters": it,
                    "final_error": {"dx": err_x, "dy": err_y},
                }

            step_x = max(-self.config.max_step_px, min(self.config.max_step_px, self.config.k_p * err_x))
            step_y = max(-self.config.max_step_px, min(self.config.max_step_px, self.config.k_p * err_y))
            if abs(step_x) < 1.0 and abs(err_x) > float(self.config.threshold_px):
                step_x = 1.0 if err_x > 0 else -1.0
            if abs(step_y) < 1.0 and abs(err_y) > float(self.config.threshold_px):
                step_y = 1.0 if err_y > 0 else -1.0

            try:
                cur_pos = getattr(self.hardware, "pos_mm", None)
                try:
                    dx_mm, dy_mm = self.kinematics.px_to_mm(step_x, step_y, current_pos_mm=cur_pos)
                except TypeError:
                    dx_mm, dy_mm = self.kinematics.px_to_mm(step_x, step_y)
                self.hardware.move_mm(dx_mm, dy_mm)
            except ValueError:
                return {"ok": False, "reason": "soft_limit", "iters": it}
            except Exception as exc:  # pragma: no cover
                return {"ok": False, "reason": f"move_error:{exc}", "iters": it}

            time.sleep(max(0.0, float(self.config.settle_ms) / 1000.0))
            obs = self.observe(force_ui=False)
            if obs.cursor_xy is None:
                return {"ok": False, "reason": "cursor_lost", "iters": it + 1}

            cur_err = abs(target_x - obs.cursor_xy[0]) + abs(target_y - obs.cursor_xy[1])
            if prev_err is not None and (prev_err - cur_err) < 2.0:
                no_progress += 1
            else:
                no_progress = 0
            prev_err = cur_err
            if no_progress >= int(self.config.no_progress_limit):
                return {"ok": False, "reason": "no_progress", "iters": it + 1}

        cx, cy = obs.cursor_xy
        return {
            "ok": False,
            "reason": "max_iters",
            "iters": int(self.config.max_iters),
            "final_error": {"dx": target_x - cx, "dy": target_y - cy},
        }

    def tap_target(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        gx: int | None = None,
        gy: int | None = None,
        coord_type: str = "screen_px",
    ) -> dict[str, Any]:
        if self.mode != "phone":
            return {"ok": False, "reason": "tap_target_requires_phone_mode"}

        bbox = self._resolve_phone_screen_bbox()
        if bbox is None:
            return {"ok": False, "reason": "phone_screen_not_calibrated"}
        safe_bbox = self._bbox_with_margin(bbox, self.phone_allowed_margin_px)

        target_x: float
        target_y: float
        if coord_type == "phone_grid":
            if gx is None or gy is None:
                return {"ok": False, "reason": "missing_phone_grid"}
            px, py = grid_to_screen_px_phone(
                int(gx),
                int(gy),
                safe_bbox,
                grid_w=self.phone_grid_w,
                grid_h=self.phone_grid_h,
            )
            target_x, target_y = float(px), float(py)
        else:
            if x is None or y is None:
                return {"ok": False, "reason": "missing_screen_px"}
            target_x, target_y = float(x), float(y)

        if not self._point_in_bbox(target_x, target_y, safe_bbox):
            return {"ok": False, "reason": "out_of_phone_bounds", "target": {"x": target_x, "y": target_y}}

        tapped = self.pointer_controller.tap_screen_px(target_x, target_y)
        return {
            "ok": bool(tapped.ok),
            "reason": str(tapped.reason),
            "target": {"x": target_x, "y": target_y, "coord_type": coord_type},
            "phone_screen_bbox": safe_bbox,
            "detail": tapped.detail,
        }

    def hover_check(self, element: UIElement, obs: Observation) -> str:
        if self.mode == "phone" or not bool(getattr(self.pointer_controller, "can_hover_check", True)):
            return "unknown"
        if obs.cursor_xy is None:
            element.clickability_hint = None
            return "unknown"

        x1, y1, x2, y2 = element.bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        first = self.move_to(center_x, center_y)
        if not bool(first.get("ok", False)):
            return "unknown"

        offset_x = min(float(obs.screen_w - 1), center_x + float(self.config.hover_delta_px))
        second = self.move_to(offset_x, center_y)
        if not bool(second.get("ok", False)):
            return "unknown"

        after = self.observe(force_ui=False)
        if not after.cursor_type or after.cursor_type == "unknown":
            return "unknown"

        before_type = (obs.cursor_type or "unknown").lower()
        after_type = after.cursor_type.lower()
        if (before_type == "arrow" and after_type in {"hand", "ibeam"}) or after_type in {"hand", "ibeam"}:
            element.clickability_hint = "high"
            return "high"
        element.clickability_hint = "low"
        return "low"

    def _expand_bbox_roi(
        self,
        bbox: tuple[float, float, float, float],
        obs: Observation,
        ratio: float = 0.25,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        pad_x = w * float(ratio)
        pad_y = h * float(ratio)
        return (
            max(0, int(round(x1 - pad_x))),
            max(0, int(round(y1 - pad_y))),
            min(obs.screen_w - 1, int(round(x2 + pad_x))),
            min(obs.screen_h - 1, int(round(y2 + pad_y))),
        )

    def _default_verify_rule(self, before: Observation, ref: ElementRef) -> VerifyRule:
        try:
            element = self._resolve_element(ref, before)
            return VerifyRule(
                mode="ELEMENT_CHANGED",
                target_ref=ref,
                roi=self._expand_bbox_roi(element.bbox, before, ratio=0.30),
                min_delta=6,
            )
        except Exception:
            return VerifyRule(
                mode="TEXT_CHANGED",
                text="ui_change",
                roi=(
                    int(before.screen_w * 0.2),
                    int(before.screen_h * 0.2),
                    int(before.screen_w * 0.8),
                    int(before.screen_h * 0.8),
                ),
                min_delta=6,
            )

    def _run_verify_rule(
        self,
        before: Observation,
        after: Observation,
        verify_rule: VerifyRule,
        fallback_ref: ElementRef | None = None,
    ) -> VerifyResult:
        if verify_rule.mode == "NONE":
            return VerifyResult(ok=True, score=1.0, method="NONE", details={"verify_skipped": True})

        if verify_rule.mode == "TEXT_PRESENT":
            return verify_text_present(
                after,
                verify_rule.text or "",
                roi=verify_rule.roi,
                min_match_ratio=float(verify_rule.min_match_ratio),
            )

        if verify_rule.mode == "TEXT_CHANGED":
            return verify_text_changed(
                before,
                after,
                roi=verify_rule.roi,
                min_delta=int(verify_rule.min_delta),
            )

        if verify_rule.mode == "ELEMENT_CHANGED":
            target_ref = verify_rule.target_ref or fallback_ref
            if target_ref is None:
                return VerifyResult(ok=False, score=0.0, method="ELEMENT_CHANGED", details={"reason": "missing_target_ref"})
            primary = verify_element_changed_ref(
                before,
                after,
                target_ref=target_ref,
                selector=self.selector,
            )
            if (
                not primary.ok
                and primary.details.get("reason") == "cannot_resolve"
                and verify_rule.roi is not None
            ):
                fallback = verify_text_changed(
                    before,
                    after,
                    roi=verify_rule.roi,
                    min_delta=int(verify_rule.min_delta),
                )
                fallback.details["fallback_from"] = "ELEMENT_CHANGED"
                return fallback
            return primary

        if verify_rule.mode == "INPUT_VERIFIED":
            return verify_input_effect(
                before,
                after,
                target_input=verify_rule.target_ref or fallback_ref,
                expected_text=verify_rule.text,
                selector=self.selector,
                allow_vlm=bool(verify_rule.allow_vlm),
            )

        return VerifyResult(ok=False, score=0.0, method="UNKNOWN", details={"reason": "unsupported_verify_mode"})

    def click_and_verify(
        self,
        ref: ElementRef,
        verify_rule: VerifyRule | None = None,
        timeout_ms: int = 2000,
        poll_ms: int = 200,
        retry: int = 1,
    ) -> dict[str, Any]:
        timeout_ms = max(0, int(timeout_ms))
        poll_ms = max(50, int(poll_ms))
        retry = max(0, int(retry))

        last_click: dict[str, Any] | None = None
        last_verify: VerifyResult | None = None

        for attempt in range(retry + 1):
            before = self.observe(force_ui=True)
            active_rule = verify_rule or self._default_verify_rule(before, ref)

            click_ret = self.click_target(ref, button="left")
            last_click = click_ret
            if not bool(click_ret.get("ok", False)):
                if attempt < retry:
                    continue
                return {
                    "ok": False,
                    "reason": "click_failed",
                    "attempt": attempt,
                    "click": click_ret,
                }

            if active_rule.mode == "NONE":
                return {
                    "ok": True,
                    "reason": "ok",
                    "attempt": attempt,
                    "verify_skipped": True,
                    "verify": {"ok": True, "method": "NONE", "details": {"verify_skipped": True}},
                    "click": click_ret,
                }

            time.sleep(0.15)
            deadline = time.time() + (float(timeout_ms) / 1000.0)
            while time.time() <= deadline:
                after = self.observe(force_ui=True)
                verify_ret = self._run_verify_rule(before, after, active_rule, fallback_ref=ref)
                last_verify = verify_ret
                if verify_ret.ok:
                    return {
                        "ok": True,
                        "reason": "ok",
                        "attempt": attempt,
                        "click": click_ret,
                        "verify": verify_ret.model_dump() if hasattr(verify_ret, "model_dump") else verify_ret.dict(),
                    }
                time.sleep(float(poll_ms) / 1000.0)

        return {
            "ok": False,
            "reason": "verify_failed",
            "click": last_click,
            "verify": None
            if last_verify is None
            else (last_verify.model_dump() if hasattr(last_verify, "model_dump") else last_verify.dict()),
        }

    def move_to_target(self, ref: ElementRef, anchor: str = "center") -> dict[str, Any]:
        obs = self.observe(force_ui=True)
        try:
            element = self._resolve_element(ref, obs)
        except AmbiguousMatch:
            return {"ok": False, "reason": "ambiguous"}
        except NoMatch:
            return {"ok": False, "reason": "no_match"}

        x1, y1, x2, y2 = element.bbox
        anchor_key = (anchor or "center").lower()
        if anchor_key in {"center", "c"}:
            tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        elif anchor_key in {"top_left", "tl"}:
            tx, ty = x1, y1
        elif anchor_key in {"top_right", "tr"}:
            tx, ty = x2, y1
        elif anchor_key in {"bottom_left", "bl"}:
            tx, ty = x1, y2
        elif anchor_key in {"bottom_right", "br"}:
            tx, ty = x2, y2
        else:
            return {"ok": False, "reason": f"unsupported_anchor:{anchor}"}

        moved = self.move_to(tx, ty)
        moved["target"] = {
            "stable_id": element.stable_id,
            "bbox": element.bbox,
            "anchor": anchor_key,
            "xy": (tx, ty),
        }
        return moved

    def click_target(self, ref: ElementRef, button: str = "left") -> dict[str, Any]:
        obs = self.observe(force_ui=True)
        if not obs.elements:
            return {"ok": False, "reason": "no_elements"}

        try:
            element = self._resolve_element(ref, obs)
        except AmbiguousMatch:
            return {"ok": False, "reason": "ambiguous"}
        except NoMatch:
            return {"ok": False, "reason": "no_match"}

        text_len = len((element.text or element.label or "").strip())
        if element.source == "heuristic" and text_len <= 1:
            self.hover_check(element, obs)

        moved = self.move_to_target(ref, anchor="center")
        if not bool(moved.get("ok", False)):
            return {"ok": False, "reason": moved.get("reason", "move_failed"), "move": moved}

        try:
            if self.mode == "phone":
                click_ret = self.pointer_controller.click("left")
                if not bool(getattr(click_ret, "ok", False)):
                    return {"ok": False, "reason": f"tap_error:{getattr(click_ret, 'reason', 'unknown')}"}
            elif button == "right":
                click_ret = self.hardware.click_right(
                    times=1,
                    interval_ms=120,
                    press_ms=int(getattr(self.runtime_config, "click_press_ms", 40)),
                )
            else:
                click_ret = self.hardware.click_left(
                    times=1,
                    interval_ms=120,
                    press_ms=int(getattr(self.runtime_config, "click_press_ms", 40)),
                )
        except Exception as exc:
            return {"ok": False, "reason": f"click_error:{exc}"}

        return {
            "ok": True,
            "reason": "ok",
            "target": {"stable_id": element.stable_id, "bbox": element.bbox, "button": button},
            "move": moved,
            "click": {"note": getattr(click_ret, "note", "click"), "detail": getattr(click_ret, "detail", "")},
        }

    def _macro_registry(self):
        if self._macros is None:
            macros_path = getattr(self.runtime_config, "macros_path", "config/macros.yaml")
            self._macros = load_macros(macros_path)
        return self._macros

    def _run_open_osk_macro(self) -> dict[str, Any]:
        try:
            registry = self._macro_registry()
            steps = registry.expand_intent(
                "OPEN_OSK",
                allow_hotkey=bool(getattr(self.runtime_config, "allow_hotkey", False)),
            )
        except Exception as exc:
            return {"ok": False, "reason": f"open_osk_macro_error:{exc}"}

        for step in steps:
            action = (step.action or "").lower()
            if action == "wait":
                wait_ms = max(0, int(step.wait_ms))
                time.sleep(wait_ms / 1000.0)
                continue
            if action in {"click", "right_click"}:
                if not step.selector:
                    return {"ok": False, "reason": "macro_step_missing_selector"}
                click_ret = self.click_target(
                    ElementRef(by="query", value=step.selector),
                    button="right" if action == "right_click" else "left",
                )
                if not bool(click_ret.get("ok", False)):
                    return {"ok": False, "reason": "macro_click_failed", "details": click_ret}
                continue
            return {"ok": False, "reason": f"unsupported_macro_action:{step.action}"}
        return {"ok": True, "reason": "ok"}

    def type_text_osk(
        self,
        text: str,
        target_input: ElementRef | None = None,
        keyboard_profile: str | None = None,
    ) -> dict[str, Any]:
        profile = keyboard_profile or str(getattr(self.runtime_config, "keyboard_profile_default", "EN_US"))
        verify_rule = VerifyRule(
            mode="INPUT_VERIFIED",
            target_ref=target_input,
            text=text,
            allow_vlm=False,
        )

        def _run_once() -> tuple[bool, dict[str, Any], Observation | None, VerifyResult | None]:
            before_obs = self.observe(force_ui=True)

            if target_input is not None:
                focus = self.click_target(target_input, button="left")
                if not bool(focus.get("ok", False)):
                    return False, {"ok": False, "reason": "focus_failed", "details": focus}, None, None
                before_obs = self.observe(force_ui=True)

            if not before_obs.keyboard_mode or before_obs.keyboard_roi is None:
                macro_ret = self._run_open_osk_macro()
                if not bool(macro_ret.get("ok", False)):
                    return False, {"ok": False, "reason": "open_osk_failed", "details": macro_ret}, None, None
                after_open = self.observe(force_ui=True)
                if not after_open.keyboard_mode or after_open.keyboard_roi is None:
                    after_open = self.observe(force_ui=True)
                if not after_open.keyboard_mode or after_open.keyboard_roi is None:
                    return False, {"ok": False, "reason": "keyboard_not_detected"}, None, None
                before_obs = after_open

            compiled = self.osk_planner.compile(text, before_obs, keyboard_profile=profile, shift_state="auto")
            if not compiled.ok:
                return (
                    False,
                    {
                        "ok": False,
                        "reason": "compile_failed",
                        "failed_char": compiled.failed_char,
                        "details": compiled.to_dict(),
                    },
                    before_obs,
                    None,
                )

            exec_ret = self.osk_executor.execute(compiled, loop=self, target_input=None)
            if not bool(exec_ret.get("ok", False)):
                return (
                    False,
                    {
                        "ok": False,
                        "reason": "type_failed",
                        "failed_char": exec_ret.get("failed_char") or compiled.failed_char,
                        "details": exec_ret,
                    },
                    before_obs,
                    None,
                )

            after_obs = self.observe(force_ui=True)
            verify_ret = self._run_verify_rule(before_obs, after_obs, verify_rule, fallback_ref=target_input)
            if not verify_ret.ok:
                return (
                    False,
                    {
                        "ok": False,
                        "reason": "input_verify_failed",
                        "failed_char": exec_ret.get("failed_char") or compiled.failed_char,
                        "details": exec_ret,
                    },
                    before_obs,
                    verify_ret,
                )

            return True, {"ok": True, "reason": "ok", "details": exec_ret}, after_obs, verify_ret

        ok, payload, _obs, verify_ret = _run_once()
        if ok:
            return {
                "ok": True,
                "reason": "ok",
                "attempt": 0,
                "details": payload.get("details", {}),
                "verify": None if verify_ret is None else (verify_ret.model_dump() if hasattr(verify_ret, "model_dump") else verify_ret.dict()),
            }

        # One retry for focus / keyboard / verify failure.
        ok2, payload2, _obs2, verify_ret2 = _run_once()
        if ok2:
            return {
                "ok": True,
                "reason": "ok",
                "attempt": 1,
                "details": payload2.get("details", {}),
                "verify": None if verify_ret2 is None else (verify_ret2.model_dump() if hasattr(verify_ret2, "model_dump") else verify_ret2.dict()),
            }

        return {
            "ok": False,
            "reason": "type_failed",
            "failed_char": payload2.get("failed_char") or payload.get("failed_char"),
            "details": payload2,
            "verify": None if verify_ret2 is None else (verify_ret2.model_dump() if hasattr(verify_ret2, "model_dump") else verify_ret2.dict()),
        }


def run_step_placeholder() -> LoopState:
    return LoopState(step_index=1, done=False, last_error=None)
