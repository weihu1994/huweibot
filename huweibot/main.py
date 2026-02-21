from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from huweibot import __version__
from huweibot._pydantic_compat import model_to_dict
from huweibot.agent.macros import load_macros
from huweibot.agent.planner import RulePlanner, build_planner
from huweibot.agent.router import Router
from huweibot.agent.schemas import ElementRef, NextStep, VerifyRule
from huweibot.agent.tasks import (
    TaskStore,
    add_task,
    get_task,
    list_due_tasks,
    load_tasks,
    mark_task_result,
    mark_task_running,
    remove_task,
    set_task_enabled,
)
from huweibot.agent.verifier import verify_text_present
from huweibot.config import XBotConfig, load_config
from huweibot.control.grbl_serial import GRBLSerial
from huweibot.control.hardware import HardwareController
from huweibot.control.kinematics import PxMmMapping, drift_check_update
from huweibot.control.touch import XYZTouchController
from huweibot.core.actions import action_from_json, action_to_json
from huweibot.core.executor import Executor
from huweibot.core.logger import RunLogger
from huweibot.core.loop import LoopConfig, XBotLoop
from huweibot.core.observation import Observation, UIElement
from huweibot.core.tracker import StableTracker
from huweibot.vision.camera import Camera
from huweibot.vision.cursor_detect import detect_cursor_best_of_n
from huweibot.vision.screen_rectify import load_calibration, rectify, validate_calibration


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _apply_mode_override(cfg: XBotConfig, args: argparse.Namespace) -> None:
    mode = getattr(args, "mode", None)
    if mode in {"pc", "computer"}:
        cfg.mode = "pc"
    elif mode == "phone":
        cfg.mode = "phone"


def _load_cfg(args: argparse.Namespace | None = None) -> XBotConfig:
    cfg = load_config()
    if args is not None:
        _apply_mode_override(cfg, args)
    return cfg


def _parse_at_datetime(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError("invalid --at format, expected 'YYYY-mm-dd HH:MM:SS'") from exc
    return dt.timestamp()


def _add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rectify", action="store_true", help="Use homography-rectified B-screen view")
    parser.add_argument("--validate-screen", action="store_true", help="Validate calibration/environment before run")
    parser.add_argument("--calib", default="config/calibration.json")


def _add_hardware_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="", help="GRBL serial port on machine A")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mapping", default="config/mapping.json")
    parser.add_argument("--click-mode", choices=["stub", "same_serial", "separate_serial"], default=None)


def _validate_screen_if_needed(args: argparse.Namespace, camera: Camera, calib: Any) -> None:
    if not bool(getattr(args, "validate_screen", False)):
        return
    if calib is None:
        raise RuntimeError("validate-screen requires --calib")
    frame = camera.read_latest(2).bgr
    ok, err, debug = validate_calibration(frame, calib)
    if not ok:
        reason = debug.get("reason") or "unknown"
        raise RuntimeError(f"validate_screen failed: {reason}; reprojection_error_px={err:.2f}")


@contextmanager
def _runtime(
    cfg: XBotConfig,
    args: argparse.Namespace,
    *,
    require_camera: bool,
    require_hardware: bool,
    use_rule_planner: bool = False,
) -> Iterator[dict[str, Any]]:
    with ExitStack() as stack:
        runtime_mode = str(getattr(cfg, "mode", "pc")).lower()
        calib = None
        rectifier = None
        camera = None
        mapping = None
        grbl = None
        hardware = None

        if require_hardware and runtime_mode == "pc":
            port = str(getattr(args, "port", "")).strip()
            if not port:
                raise RuntimeError("需要 --port（GRBL on A）")
            mapping_path = str(getattr(args, "mapping", cfg.mapping_path))
            try:
                mapping = PxMmMapping.load_json(mapping_path)
            except Exception as exc:
                raise RuntimeError(f"failed to load mapping ({mapping_path}); Step7 required: {exc}") from exc

        if require_camera:
            camera = stack.enter_context(Camera(args.camera_id, args.width, args.height, args.fps))
            if bool(getattr(args, "rectify", False)) or bool(getattr(args, "validate_screen", False)):
                calib_path = getattr(args, "calib", cfg.calibration_path)
                try:
                    calib = load_calibration(calib_path)
                except Exception as exc:
                    raise RuntimeError(f"failed to load calibration ({calib_path}): {exc}") from exc
                if bool(getattr(args, "rectify", False)):
                    rectifier = lambda frame_bgr: rectify(frame_bgr, calib)  # noqa: E731
            _validate_screen_if_needed(args, camera, calib)

        if require_hardware:
            port = str(getattr(args, "port", "")).strip()
            if runtime_mode == "pc" or port:
                if not port:
                    raise RuntimeError("需要 --port（GRBL on A）")
                grbl = stack.enter_context(
                    GRBLSerial(
                        port,
                        baud=int(getattr(args, "baud", cfg.serial_baud)),
                        timeout_s=float(cfg.serial_timeout_s),
                    )
                )

            if runtime_mode == "phone":
                hardware = XYZTouchController(
                    xy_controller=grbl,
                    z_axis_enabled=bool(cfg.z_axis_enabled),
                    ir_enabled=bool(cfg.ir_enabled),
                    touch_distance_threshold_mm=float(cfg.touch_distance_threshold_mm),
                    dry_run=bool(getattr(args, "dry_run", False)),
                )
                setattr(hardware, "is_homed", True)
                setattr(hardware, "pos_mm", (0.0, 0.0))
            else:
                click_mode = getattr(args, "click_mode", None) or cfg.click_mode
                screen_w = int(getattr(calib, "screen_w", args.width if require_camera else cfg.capture_width))
                screen_h = int(getattr(calib, "screen_h", args.height if require_camera else cfg.capture_height))
                hardware = HardwareController(
                    grbl=grbl,
                    travel_range_mm=cfg.travel_range_mm,
                    feed_rate=float(cfg.feed_rate_mm_min),
                    homed_flag_path=cfg.homed_flag_path,
                    enforce_homed=bool(cfg.enforce_homed),
                    click_mode=click_mode,
                    click_port=cfg.click_port,
                    click_baud=int(cfg.click_baud),
                    screen_w=screen_w,
                    screen_h=screen_h,
                )

        if require_camera:
            tracker = StableTracker()
            loop_cfg = LoopConfig(
                allow_vlm=bool(getattr(args, "allow_vlm", False)),
            )
            loop = XBotLoop(
                camera=camera,
                hardware=hardware,
                kinematics=mapping,
                rectifier=rectifier,
                tracker=tracker,
                config=loop_cfg,
                runtime_config=cfg,
            )
        else:
            loop = None

        router = Router(cfg)
        planner = RulePlanner() if use_rule_planner else build_planner(router)
        run_logger = RunLogger(cfg.artifacts_dir, enabled=bool(cfg.logging_enabled))
        executor = None
        if loop is not None:
            executor = Executor(loop=loop, planner=planner, config=cfg, router=router, run_logger=run_logger)

        yield {
            "cfg": cfg,
            "camera": camera,
            "calib": calib,
            "mapping": mapping,
            "grbl": grbl,
            "hardware": hardware,
            "loop": loop,
            "router": router,
            "planner": planner,
            "executor": executor,
            "logger": run_logger,
        }


def _placeholder_handler(args: argparse.Namespace) -> int:
    _json_print(
        {
            "command": args.command,
            "status": "placeholder",
            "message": "Command registered but implementation is outside current step scope.",
        }
    )
    return 0


def _trace_handler(_: argparse.Namespace) -> int:
    cfg = _load_cfg()
    router = Router(cfg)
    _json_print(
        {
            "planner_provider": router.get_planner_provider().name,
            "vlm_provider": router.get_vlm_provider().name if router.get_vlm_provider() else None,
            "capabilities": router.capabilities_report(),
        }
    )
    return 0


def _web_handler(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception:
        print("[error] huweibot web console requires fastapi and uvicorn. install with: pip install -e '.[dev]'")
        return 2
    try:
        from huweibot.web import create_app
    except Exception as exc:
        print(f"[error] failed to load web app: {exc}")
        return 2
    uvicorn.run(create_app(), host=str(args.host), port=int(args.port), log_level=str(args.log_level))
    return 0


def _resolve_goal_text(args: argparse.Namespace) -> str:
    task_text = str(getattr(args, "task", "") or "").strip()
    if task_text:
        return task_text
    goal_text = str(getattr(args, "goal", "") or "").strip()
    return goal_text


def _dry_run_task(task_text: str, mode: str, cfg: XBotConfig, max_steps: int = 1) -> dict[str, Any]:
    run_logger = RunLogger(cfg.artifacts_dir, enabled=bool(cfg.logging_enabled))
    router = Router(cfg)
    planner = build_planner(router)
    step = 1
    obs = Observation(
        screen_w=1920,
        screen_h=1080,
        app_hint="dry_run",
        screen_hash="dry_run_hash",
        ui_change_score=0.0,
        device_mode=("phone" if mode == "phone" else "pc"),
    )
    if mode == "phone":
        try:
            from huweibot.vision.phone_screen import load_phone_screen

            bbox = load_phone_screen(cfg.phone_screen_calibration_path)
            obs.phone_screen_bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            obs.phone_grid_w = int(cfg.phone_grid_w)
            obs.phone_grid_h = int(cfg.phone_grid_h)
        except Exception:
            obs.phone_screen_bbox = None
            obs.phone_grid_w = int(cfg.phone_grid_w)
            obs.phone_grid_h = int(cfg.phone_grid_h)

    run_logger.begin_step(step, task=task_text, obs_mode="full")
    run_logger.log_observation(step, obs, screen_bgr=None)
    next_step = planner.plan(
        task_text,
        obs,
        {},
        step=step,
        obs_mode="full",
        artifacts_dir=cfg.artifacts_dir,
    )
    planner_meta = getattr(planner, "last_meta", {}) or {}
    run_logger.log_planner(step, planner_meta, task=task_text, obs_mode="full")
    run_logger.log_execution(step, action_executed=next_step.action, success=True, reason="dry_run_stub")
    run_logger.log_verify(step, {"ok": True, "score": 1.0, "method": "NONE", "details": {"dry_run": True}})
    return {
        "ok": True,
        "done": True,
        "reason": "dry_run_stub" if not (mode == "phone" and obs.phone_screen_bbox is None) else "dry_run_no_phone_calibration",
        "steps": [
            {
                "step": step,
                "ok": True,
                "done": True,
                "reason": "dry_run_stub",
                "action": _model(next_step.action),
                "verify": {"ok": True, "method": "NONE", "details": {"dry_run": True}},
                "planner": planner_meta,
            }
        ],
        "max_steps": int(max_steps),
    }


def _run_step_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    task_text = _resolve_goal_text(args)
    if not task_text:
        print("[error] run-step requires --task or --goal")
        return 2
    if str(getattr(cfg, "mode", "pc")) == "phone":
        cfg.guard_require_homed = False
    if bool(getattr(args, "dry_run", False)):
        ret = _dry_run_task(task_text, str(getattr(args, "mode", "computer")), cfg, max_steps=1)
        _json_print(ret)
        return 0 if bool(ret.get("ok", False)) else 1
    if str(getattr(cfg, "mode", "pc")) == "phone" and str(getattr(cfg, "phone_screen_detection", "manual")) == "manual":
        if not Path(str(cfg.phone_screen_calibration_path)).exists():
            print("[error] phone mode requires ROI calibration first (run scripts/calibrate_phone_screen.py)")
            return 2
    try:
        with _runtime(
            cfg,
            args,
            require_camera=True,
            require_hardware=True,
            use_rule_planner=bool(args.rule_planner),
        ) as rt:
            executor: Executor = rt["executor"]
            result = executor.run_step(task_text)
            _json_print(result.to_dict())
            return 0 if result.ok else 1
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] run-step failed: {exc}")
        return 1


def _run_task_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    task_text = _resolve_goal_text(args)
    if not task_text:
        print("[error] run-task requires --task or --goal")
        return 2
    mode = str(getattr(args, "mode", "computer"))
    if str(getattr(cfg, "mode", "pc")) == "phone":
        cfg.guard_require_homed = False
    if bool(getattr(args, "dry_run", False)):
        ret = _dry_run_task(task_text, mode, cfg, max_steps=int(getattr(args, "max_steps", 30)))
        _json_print(ret)
        return 0 if bool(ret.get("ok", False)) else 1
    if str(getattr(cfg, "mode", "pc")) == "phone" and str(getattr(cfg, "phone_screen_detection", "manual")) == "manual":
        if not Path(str(cfg.phone_screen_calibration_path)).exists():
            print("[error] phone mode requires ROI calibration first (run scripts/calibrate_phone_screen.py)")
            return 2
    try:
        with _runtime(
            cfg,
            args,
            require_camera=True,
            require_hardware=True,
            use_rule_planner=bool(args.rule_planner),
        ) as rt:
            executor: Executor = rt["executor"]
            result = executor.run_task(task_text, max_steps=args.max_steps)
            _json_print(result)
            return 0 if bool(result.get("ok", False)) else 1
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] run-task failed: {exc}")
        return 1


def _list_macros_handler(_: argparse.Namespace) -> int:
    cfg = _load_cfg()
    reg = load_macros(cfg.macros_path)
    names: list[str] = []
    for intent, macro in reg.macros.items():
        prefix = intent.lower()
        for path_name in macro.paths:
            names.append(f"{prefix}.{path_name}")
    _json_print({"macros": sorted(names)})
    return 0


def _parse_macro_name(name: str) -> tuple[str, str | None]:
    value = str(name or "").strip()
    if not value:
        raise ValueError("macro name is empty")
    if "." in value:
        head, tail = value.split(".", 1)
        intent = head.upper()
        return intent, tail
    return value.upper(), None


def _run_macro_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        intent, preferred_path = _parse_macro_name(args.name)
        reg = load_macros(cfg.macros_path)
        try:
            steps = reg.expand_intent(intent, allow_hotkey=bool(cfg.allow_hotkey), preferred_path=preferred_path)
        except Exception:
            steps = reg.expand_intent(intent, allow_hotkey=bool(cfg.allow_hotkey), preferred_path=None)
            preferred_path = None
    except Exception as exc:
        print(f"[error] run-macro parse/expand failed: {exc}")
        return 2

    if args.dry_run:
        _json_print({"intent": intent, "preferred_path": preferred_path, "dry_run": True, "steps": [_model(step) for step in steps]})
        return 0

    try:
        with _runtime(cfg, args, require_camera=True, require_hardware=True) as rt:
            executor: Executor = rt["executor"]
            executor.state.step += 1
            step = executor.state.step
            executor.logger.begin_step(step, task=f"run-macro:{intent}", obs_mode="full")
            obs = executor.loop.observe(force_ui=True)
            executor.logger.log_observation(step, obs, screen_bgr=executor._capture_screen_frame())
            ret = executor._execute_macro_intent(step, intent, preferred_path=preferred_path)
            executor.logger.log_execution(step, action_executed={"type": intent}, success=bool(ret.get("ok", False)), reason=str(ret.get("reason")))
            _json_print(ret)
            return 0 if bool(ret.get("ok", False)) else 1
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] run-macro failed: {exc}")
        return 1


def _home_check_handler(_: argparse.Namespace) -> int:
    cfg = _load_cfg()
    path = Path(cfg.homed_flag_path)
    if path.exists():
        print(f"[ok] homed flag detected: {path}")
        return 0
    print(f"[warn] not homed: missing {path}. 请先执行回零 SOP。")
    return 1


def _drift_check_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        with _runtime(cfg, args, require_camera=True, require_hardware=True) as rt:
            camera: Camera = rt["camera"]
            mapping: PxMmMapping = rt["mapping"]
            hardware: HardwareController = rt["hardware"]
            calib = rt["calib"]
            run_logger: RunLogger = rt["logger"]

            def _transform(frame_bgr):
                if bool(args.rectify) and calib is not None:
                    return rectify(frame_bgr, calib)
                return frame_bgr

            before = detect_cursor_best_of_n(camera, n=3, frame_transform=_transform)
            if before.cursor_xy is None:
                print("[error] drift-check failed: cursor not found before move")
                return 1

            step_mm = float(args.step_mm)
            hardware.move_mm(step_mm, 0.0)
            time.sleep(max(0.0, float(args.settle_ms) / 1000.0))
            after = detect_cursor_best_of_n(camera, n=3, frame_transform=_transform)
            hardware.move_mm(-step_mm, 0.0)

            if after.cursor_xy is None:
                print("[error] drift-check failed: cursor not found after move")
                return 1

            observed_dx_px = float(after.cursor_xy[0] - before.cursor_xy[0])
            updated = drift_check_update(
                mapping,
                observed_dx_px=observed_dx_px,
                dx_mm=step_mm,
                alpha=float(cfg.drift_alpha),
                min_px=float(cfg.drift_min_px),
            )
            changed = abs(updated.mm_per_px_x - mapping.mm_per_px_x) > 1e-9
            payload = {
                "updated": changed,
                "observed_dx_px": observed_dx_px,
                "mm_per_px_before": {"x": mapping.mm_per_px_x, "y": mapping.mm_per_px_y},
                "mm_per_px_after": {"x": updated.mm_per_px_x, "y": updated.mm_per_px_y},
            }
            if args.write:
                updated.save_json(args.mapping)
                payload["mapping_saved"] = args.mapping
            run_logger.begin_step(1, task="drift-check", obs_mode="delta")
            run_logger.log_drift(1, payload)
            _json_print(payload)
            return 0
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] drift-check failed: {exc}")
        return 1


def _osk_demo_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        with _runtime(cfg, args, require_camera=True, require_hardware=True) as rt:
            loop: XBotLoop = rt["loop"]
            obs = loop.observe(force_ui=True)
            if args.input_target_query:
                focus = loop.click_target(ElementRef(by="query", query=args.input_target_query), button="left")
                if not bool(focus.get("ok", False)):
                    _json_print({"ok": False, "reason": "focus_failed", "details": focus})
                    return 1
                obs = loop.observe(force_ui=True)
            if not hasattr(loop, "type_text_osk"):
                print("Step10 required: OSK compile/execute not implemented")
                return 2
            ret = loop.type_text_osk(
                args.text,
                target_input=None,
                keyboard_profile=args.keyboard_profile,
            )
            ret["keyboard_mode_before"] = bool(obs.keyboard_mode)
            _json_print(ret)
            return 0 if bool(ret.get("ok", False)) else 1
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] osk-demo failed: {exc}")
        return 1


def _rule_script_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        with _runtime(cfg, args, require_camera=True, require_hardware=True, use_rule_planner=True) as rt:
            loop: XBotLoop = rt["loop"]
            out: dict[str, Any] = {"ok": True, "steps": []}
            if args.iq:
                ret = loop.click_and_verify(ElementRef(by="query", query=args.iq), verify_rule=None, retry=1)
                out["steps"].append({"op": "focus_input", "result": ret})
                if not bool(ret.get("ok", False)):
                    out["ok"] = False
                    _json_print(out)
                    return 1
            if args.t:
                ret = loop.type_text_osk(args.t, target_input=None, keyboard_profile=args.keyboard_profile)
                out["steps"].append({"op": "type_text", "result": ret})
                if not bool(ret.get("ok", False)):
                    out["ok"] = False
                    _json_print(out)
                    return 1
            if args.sq:
                ret = loop.click_and_verify(ElementRef(by="query", query=args.sq), verify_rule=None, retry=1)
                out["steps"].append({"op": "submit", "result": ret})
                if not bool(ret.get("ok", False)):
                    out["ok"] = False
                    _json_print(out)
                    return 1
            if args.vq:
                after = loop.observe(force_ui=True)
                verify = verify_text_present(after, args.vq)
                out["steps"].append({"op": "verify_text_present", "result": _model(verify)})
                out["ok"] = bool(verify.ok)
                if not verify.ok:
                    _json_print(out)
                    return 1
            _json_print(out)
            return 0
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] rule-script failed: {exc}")
        return 1


def _phone_tap_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    cfg.mode = "phone"
    if bool(getattr(args, "dry_run", False)):
        if args.grid and (args.gx is None or args.gy is None):
            print("[error] phone-tap dry-run with --grid requires --gx and --gy")
            return 2
        if (not args.grid) and (args.x is None or args.y is None):
            print("[error] phone-tap dry-run requires --x and --y (or --grid --gx --gy)")
            return 2
        coord = (
            {"coord_type": "grid", "x": int(args.gx), "y": int(args.gy)}
            if args.grid
            else {"coord_type": "screen_px", "x": float(args.x), "y": float(args.y)}
        )
        action = action_to_json(
            action_from_json({"type": "CLICK_AT", "coord": coord, "times": 1}),
            as_dict=True,
        )
        verify = model_to_dict(VerifyRule(type="NONE"), exclude_none=True)
        _json_print({"ok": True, "reason": "dry_run_stub", "mode": "phone", "action": action, "verify": verify})
        return 0
    try:
        with _runtime(cfg, args, require_camera=True, require_hardware=False, use_rule_planner=True) as rt:
            loop: XBotLoop = rt["loop"]
            _ = loop.observe(force_ui=True)
            if args.grid:
                result = loop.tap_target(
                    gx=int(args.gx) if args.gx is not None else None,
                    gy=int(args.gy) if args.gy is not None else None,
                    coord_type="phone_grid",
                )
            else:
                result = loop.tap_target(x=args.x, y=args.y, coord_type="screen_px")
            _json_print(result)
            return 0 if bool(result.get("ok", False)) else 1
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:
        print(f"[error] phone-tap failed: {exc}")
        return 1


def _task_to_dict(task: Any) -> dict[str, Any]:
    if hasattr(task, "to_dict"):
        return task.to_dict()
    return _model(task)


def _task_add_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    try:
        task = add_task(
            cfg.tasks_db_path,
            title=args.title,
            instruction=args.instruction,
            mode=args.task_mode,
            every_seconds=args.every_seconds,
        )
        _json_print({"ok": True, "task": _task_to_dict(task), "db": cfg.tasks_db_path})
        return 0
    except Exception as exc:
        print(f"[error] task-add failed: {exc}")
        return 1


def _task_list_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    tasks = [_task_to_dict(t) for t in load_tasks(cfg.tasks_db_path)]
    _json_print({"ok": True, "count": len(tasks), "tasks": tasks, "db": cfg.tasks_db_path})
    return 0


def _build_task_run_namespace(args: argparse.Namespace, task: Any) -> argparse.Namespace:
    return argparse.Namespace(
        task=str(getattr(task, "goal", "")),
        goal=str(getattr(task, "goal", "")),
        max_steps=int(getattr(args, "max_steps", 30)),
        rule_planner=bool(getattr(args, "rule_planner", False)),
        allow_vlm=bool(getattr(args, "allow_vlm", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        camera_id=int(getattr(args, "camera_id", 0)),
        width=int(getattr(args, "width", 1920)),
        height=int(getattr(args, "height", 1080)),
        fps=int(getattr(args, "fps", 30)),
        rectify=bool(getattr(args, "rectify", False)),
        validate_screen=bool(getattr(args, "validate_screen", False)),
        calib=str(getattr(args, "calib", "config/calibration.json")),
        port=str(getattr(args, "port", "")),
        baud=int(getattr(args, "baud", 115200)),
        mapping=str(getattr(args, "mapping", "config/mapping.json")),
        click_mode=getattr(args, "click_mode", None),
        mode=str(getattr(task, "mode", "computer")),
    )


def _task_run_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    task = get_task(cfg.tasks_db_path, args.id)
    if task is None:
        print(f"[error] task not found: {args.id}")
        return 2

    print(f"[info] task-run id={task.id} mode={task.mode} name={task.name}")
    print(f"[info] goal={task.goal}")
    _notify_event(cfg, f"TASK_TRIGGER id={task.id} mode={task.mode} name={task.name}")
    mark_task_running(cfg.tasks_db_path, task)

    run_args = _build_task_run_namespace(args, task)
    rc = _run_task_handler(run_args)
    mark_task_result(cfg.tasks_db_path, task, ok=(rc == 0))
    _notify_event(cfg, f"TASK_DONE id={task.id} rc={rc}")
    return rc


def _emit_reminder(cfg: XBotConfig, message: str) -> None:
    if not bool(cfg.reminder_enabled):
        return
    if cfg.reminder_channel == "logfile":
        log_path = Path(cfg.artifacts_dir) / "reminders.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    else:
        print(message)


def _notify_event(cfg: XBotConfig, message: str) -> None:
    print(message)
    path = Path(cfg.artifacts_dir) / "notifications.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _scheduler_run_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    tick = float(args.tick if args.tick is not None else cfg.scheduler_tick_s)
    tick = max(0.1, tick)
    print(f"[info] scheduler-run started, tick={tick:.2f}s, db={cfg.tasks_db_path}")
    try:
        while True:
            due = list_due_tasks(cfg.tasks_db_path)
            for task in due:
                _emit_reminder(
                    cfg,
                    f"REMINDER task_due id={task.id} mode={task.mode} title={task.name}",
                )
                run_ns = argparse.Namespace(**{**vars(args), "id": task.id})
                rc = _task_run_handler(run_ns)
                if rc != 0:
                    _emit_reminder(cfg, f"REMINDER task_failed id={task.id} rc={rc}")
            if bool(getattr(args, "once", False)):
                return 0
            time.sleep(tick)
    except KeyboardInterrupt:
        print("[warn] scheduler interrupted")
        return 130


def _task_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    store = TaskStore(cfg.tasks_db_path)
    cmd = str(getattr(args, "task_command", "") or "")
    if cmd == "add":
        try:
            at_ts = _parse_at_datetime(getattr(args, "at", None))
            task = store.add(
                name=args.name,
                mode=args.mode,
                goal=args.goal,
                payload={},
                at_ts=at_ts,
                every_seconds=args.every,
                enabled=True,
            )
            _json_print({"ok": True, "task": _task_to_dict(task), "db": cfg.tasks_db_path})
            return 0
        except Exception as exc:
            print(f"[error] task add failed: {exc}")
            return 1
    if cmd == "list":
        tasks = [_task_to_dict(t) for t in store.list()]
        _json_print({"ok": True, "count": len(tasks), "tasks": tasks, "db": cfg.tasks_db_path})
        return 0
    if cmd == "remove":
        ok = store.remove(args.id)
        _json_print({"ok": bool(ok), "id": args.id})
        return 0 if ok else 1
    if cmd == "enable":
        task = store.set_enabled(args.id, True)
        _json_print({"ok": task is not None, "task": None if task is None else _task_to_dict(task)})
        return 0 if task is not None else 1
    if cmd == "disable":
        task = store.set_enabled(args.id, False)
        _json_print({"ok": task is not None, "task": None if task is None else _task_to_dict(task)})
        return 0 if task is not None else 1
    if cmd == "run":
        run_ns = argparse.Namespace(
            id=args.id,
            camera_id=getattr(args, "camera_id", 0),
            width=getattr(args, "width", 1920),
            height=getattr(args, "height", 1080),
            fps=getattr(args, "fps", 30),
            rectify=getattr(args, "rectify", False),
            validate_screen=getattr(args, "validate_screen", False),
            calib=getattr(args, "calib", "config/calibration.json"),
            port=getattr(args, "port", ""),
            baud=getattr(args, "baud", 115200),
            mapping=getattr(args, "mapping", "config/mapping.json"),
            click_mode=getattr(args, "click_mode", None),
            max_steps=getattr(args, "max_steps", 30),
            rule_planner=getattr(args, "rule_planner", False),
            allow_vlm=getattr(args, "allow_vlm", False),
            dry_run=getattr(args, "dry_run", False),
            mode=getattr(args, "mode", "computer"),
        )
        return _task_run_handler(run_ns)
    if cmd == "daemon":
        tick = max(0.1, float(getattr(args, "tick", None) or cfg.scheduler_tick_s))
        print(f"[info] task daemon started, tick={tick:.2f}s, db={cfg.tasks_db_path}")
        try:
            while True:
                due_tasks = store.due()
                for task in due_tasks:
                    run_ns = argparse.Namespace(
                        id=task.id,
                        camera_id=getattr(args, "camera_id", 0),
                        width=getattr(args, "width", 1920),
                        height=getattr(args, "height", 1080),
                        fps=getattr(args, "fps", 30),
                        rectify=getattr(args, "rectify", False),
                        validate_screen=getattr(args, "validate_screen", False),
                        calib=getattr(args, "calib", "config/calibration.json"),
                        port=getattr(args, "port", ""),
                        baud=getattr(args, "baud", 115200),
                        mapping=getattr(args, "mapping", "config/mapping.json"),
                        click_mode=getattr(args, "click_mode", None),
                        max_steps=getattr(args, "max_steps", 30),
                        rule_planner=getattr(args, "rule_planner", False),
                        allow_vlm=getattr(args, "allow_vlm", False),
                        dry_run=getattr(args, "dry_run", False),
                        mode=getattr(args, "mode", task.mode),
                    )
                    rc = _task_run_handler(run_ns)
                if bool(getattr(args, "once", False)):
                    return 0
                time.sleep(tick)
        except KeyboardInterrupt:
            print("[warn] task daemon interrupted")
            return 130
    print("[error] unknown task command")
    return 2


def _doctor_fake_observation() -> Observation:
    return Observation(
        timestamp=time.time(),
        screen_w=1920,
        screen_h=1080,
        cursor_xy=(480, 320),
        cursor_conf=0.9,
        cursor_type="arrow",
        elements=[
            UIElement(
                stable_id="btn_ok_for_doctor_check",
                raw_id=1,
                role="button",
                text="  OK  ",
                label="OK",
                bbox=(120.0, 220.0, 280.0, 288.0),
                confidence=0.93,
                source="ocr",
            ),
            UIElement(
                stable_id="input_name_for_doctor_check",
                raw_id=2,
                role="input",
                text="Name Field",
                label="Name",
                bbox=(320.0, 220.0, 860.0, 300.0),
                confidence=0.88,
                source="heuristic",
            ),
            UIElement(
                stable_id="title_for_doctor_check",
                raw_id=3,
                role="text",
                text="VeryLongTitle_ABCDEFGHIJKLMN",
                label="VeryLongTitle_ABCDEFGHIJKLMN",
                bbox=(80.0, 30.0, 780.0, 110.0),
                confidence=0.84,
                source="ocr",
            ),
        ],
        app_hint="Settings App",
        keyboard_mode=False,
        keyboard_roi=None,
        screen_hash="doctor_hash",
        ui_change_score=0.2,
    )


def _find_forbidden_media_paths(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    media_tokens = ("image", "frame", "base64", "jpeg", "png")
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{path}.{k}" if path else str(k)
            key_lower = str(k).lower()
            if any(token in key_lower for token in media_tokens):
                found.append(child)
            found.extend(_find_forbidden_media_paths(v, child))
        return found
    if isinstance(value, list):
        for idx, item in enumerate(value):
            found.extend(_find_forbidden_media_paths(item, f"{path}[{idx}]"))
        return found
    if isinstance(value, str) and value.strip().lower().startswith("data:image"):
        found.append(path or "<root>")
    return found


def _doctor_handler(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    if args.artifacts_dir:
        cfg.artifacts_dir = args.artifacts_dir
    cfg.obs_model_encoding = "packed"
    cfg.obs_key_minify = True
    artifacts = Path(cfg.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    router = Router(cfg)
    obs = _doctor_fake_observation()
    memory: dict[str, Any] = {
        "last_action": {"type": "CLICK", "payload": {"target": {"by": "id", "id": "btn_ok_for_doctor_check"}}},
        "last_verify": {"ok": False, "score": 0.0},
        "macro_state": {"long_text": "x" * 900},
    }

    checks: list[tuple[str, bool, str]] = []

    try:
        payload_full, _packed_full, _digest_full, _, _ = router.build_planner_payload(
            task="doctor_full",
            obs=obs,
            memory=memory,
            obs_mode="full",
        )
        forbidden = _find_forbidden_media_paths(payload_full)
        checks.append(("packed/minified_no_image", len(forbidden) == 0, "ok" if not forbidden else ",".join(forbidden)))
    except Exception as exc:
        checks.append(("packed/minified_no_image", False, str(exc)))

    step_full = int(args.step_base)
    step_delta = int(args.step_base) + 1
    try:
        router.request_plan(
            task="doctor_full",
            obs=obs,
            memory=memory,
            step=step_full,
            obs_mode="full",
            artifacts_dir=str(artifacts),
        )
        router.request_plan(
            task="doctor_delta",
            obs=obs,
            memory=memory,
            step=step_delta,
            obs_mode="delta",
            artifacts_dir=str(artifacts),
        )
        delta_path = artifacts / f"step_{step_delta:04d}_planner_in_delta.json"
        digest_path = artifacts / f"step_{step_delta:04d}_obs_digest.json"
        packed_path = artifacts / f"step_{step_delta:04d}_planner_obs_packed.json"
        full_path = artifacts / f"step_{step_full:04d}_planner_in_full.json"
        delta_payload = json.loads(delta_path.read_text(encoding="utf-8"))
        digest = json.loads(digest_path.read_text(encoding="utf-8"))
        allowed_delta_keys = {"last_action", "last_verify", "cursor", "ui_change_score", "elements_delta", "macro_state"}
        delta_ok = set(delta_payload.keys()).issubset(allowed_delta_keys)
        prune_logged = bool(isinstance(digest.get("delta_prune"), dict))
        checks.append(("delta_prune_lock", delta_ok and prune_logged, "ok" if (delta_ok and prune_logged) else "delta_keys_or_digest_invalid"))
        logs_ok = full_path.exists() and delta_path.exists() and packed_path.exists()
        checks.append(("planner_logs_present", logs_ok, "ok" if logs_ok else "missing_planner_log_files"))
    except Exception as exc:
        checks.append(("delta_prune_lock", False, str(exc)))
        checks.append(("planner_logs_present", False, str(exc)))

    try:
        now = time.time()
        cooldown_state = {
            "elements0_streak": 0,
            "ambiguous_streak": 0,
            "no_match_streak": 0,
            "macro_fail_streak": 0,
            "vlm_calls": 0,
            "last_vlm_time": now - 1.0,
            "local_boost_done": False,
            "local_boost_executed": False,
        }
        c_ok, c_reason = router.should_call_vlm(
            state=dict(cooldown_state),
            triggers=["elements_below_min"],
            now_ts=now,
            elements_count=1,
        )
        b_state = dict(cooldown_state)
        b_state["last_vlm_time"] = now - 30.0
        b_state["vlm_calls"] = int(cfg.vlm_max_per_task)
        b_ok, b_reason = router.should_call_vlm(
            state=b_state,
            triggers=["selector_streak"],
            now_ts=now,
            elements_count=1,
        )
        f_state = dict(cooldown_state)
        f_state["elements0_streak"] = int(cfg.vlm_force_allow_if_elements0_streak)
        f_state["local_boost_done"] = True
        f_ok, f_reason = router.should_call_vlm(
            state=f_state,
            triggers=["elements_below_min"],
            now_ts=now,
            elements_count=0,
        )
        gate_ok = (not c_ok and c_reason == "cooldown") and (not b_ok and b_reason == "vlm_budget_exceeded") and (f_ok and f_reason == "force_break_cooldown")
        checks.append(("vlm_gate_cooldown_budget", gate_ok, "ok" if gate_ok else f"cooldown={c_reason},budget={b_reason},force={f_reason}"))
    except Exception as exc:
        checks.append(("vlm_gate_cooldown_budget", False, str(exc)))

    all_ok = True
    for name, ok, reason in checks:
        print(f"{'OK' if ok else 'FAIL'} {name}: {reason}")
        all_ok = all_ok and ok
    print("OK doctor" if all_ok else "FAIL doctor")
    return 0 if all_ok else 1


def _model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _build_subcommand_handlers() -> dict[str, Callable[[argparse.Namespace], int]]:
    return {
        "preview": _placeholder_handler,
        "calibrate-screen": _placeholder_handler,
        "test-cursor": _placeholder_handler,
        "inspect-elements": _placeholder_handler,
        "manual-move": _placeholder_handler,
        "move-to": _placeholder_handler,
        "run-step": _run_step_handler,
        "run-task": _run_task_handler,
        "osk-demo": _osk_demo_handler,
        "list-macros": _list_macros_handler,
        "run-macro": _run_macro_handler,
        "home-check": _home_check_handler,
        "drift-check": _drift_check_handler,
        "rule-script": _rule_script_handler,
        "phone-tap": _phone_tap_handler,
        "task-add": _task_add_handler,
        "task-list": _task_list_handler,
        "task-run": _task_run_handler,
        "scheduler-run": _scheduler_run_handler,
        "task": _task_handler,
        "doctor": _doctor_handler,
        "trace": _trace_handler,
        "web": _web_handler,
        "stop": _placeholder_handler,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huweibot",
        description=(
            "huweibot CLI. Dual-machine boundary: A controls B via camera observation + physical mouse actuation only."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--mode", choices=["pc", "phone"], default=None, help="Runtime mode override (default: config.mode, usually pc)")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name in ["preview", "calibrate-screen", "test-cursor", "inspect-elements", "manual-move", "move-to", "stop"]:
        sub = subparsers.add_parser(name, help=f"{name} command (placeholder)")
        sub.set_defaults(command=name)

    run_step = subparsers.add_parser("run-step", help="Observe->Plan->Act->Verify one step")
    run_step.add_argument("--task", default="")
    run_step.add_argument("--goal", default="")
    run_step.add_argument("--mode", choices=["computer", "phone"], default="computer")
    run_step.add_argument("--dry-run", action="store_true", default=False)
    run_step.add_argument("--rule-planner", action="store_true", help="Use local rule planner without provider")
    run_step.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(run_step)
    _add_hardware_args(run_step)
    run_step.set_defaults(command="run-step")

    run_task = subparsers.add_parser("run-task", help="Run multi-step loop until DONE/failure")
    run_task.add_argument("--task", default="")
    run_task.add_argument("--goal", default="")
    run_task.add_argument("--mode", choices=["computer", "phone"], default="computer")
    run_task.add_argument("--dry-run", action="store_true", default=False)
    run_task.add_argument("--max-steps", type=int, default=30)
    run_task.add_argument("--rule-planner", action="store_true", help="Use local rule planner without provider")
    run_task.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(run_task)
    _add_hardware_args(run_task)
    run_task.set_defaults(command="run-task")

    osk_demo = subparsers.add_parser("osk-demo", help="TYPE_TEXT(method=osk) demo")
    osk_demo.add_argument("--text", required=True)
    osk_demo.add_argument("--input-target-query", default="")
    osk_demo.add_argument("--keyboard-profile", choices=["EN_US", "ZH_CN", "NUMERIC"], default="EN_US")
    osk_demo.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(osk_demo)
    _add_hardware_args(osk_demo)
    osk_demo.set_defaults(command="osk-demo")

    list_macros = subparsers.add_parser("list-macros", help="List macro names from config/macros.yaml")
    list_macros.set_defaults(command="list-macros")

    run_macro = subparsers.add_parser("run-macro", help="Run macro intent by name")
    run_macro.add_argument("--name", required=True, help="e.g. open_app.path_a or open_osk.path_b")
    run_macro.add_argument("--dry-run", action="store_true")
    _add_camera_args(run_macro)
    _add_hardware_args(run_macro)
    run_macro.set_defaults(command="run-macro")

    home_check = subparsers.add_parser("home-check", help="Check whether platform homed flag exists")
    home_check.set_defaults(command="home-check")

    drift_check = subparsers.add_parser("drift-check", help="Manual mapping drift maintenance entry")
    drift_check.add_argument("--step-mm", type=float, default=1.0)
    drift_check.add_argument("--settle-ms", type=int, default=350)
    drift_check.add_argument("--write", action="store_true", help="Write updated mapping back to --mapping")
    _add_camera_args(drift_check)
    _add_hardware_args(drift_check)
    drift_check.set_defaults(command="drift-check")

    rule_script = subparsers.add_parser("rule-script", help="Token-light rule template execution entry")
    rule_script.add_argument("--iq", default="", help="Input-focus selector query")
    rule_script.add_argument("--t", default="", help="Text for TYPE_TEXT(osk)")
    rule_script.add_argument("--sq", default="", help="Submit selector query")
    rule_script.add_argument("--vq", default="", help="Verify text present after script")
    rule_script.add_argument("--keyboard-profile", choices=["EN_US", "ZH_CN", "NUMERIC"], default="EN_US")
    _add_camera_args(rule_script)
    _add_hardware_args(rule_script)
    rule_script.set_defaults(command="rule-script")

    phone_tap = subparsers.add_parser("phone-tap", help="PHONE mode tap by screen_px or phone grid")
    phone_tap.add_argument("--mode", choices=["computer", "phone"], default="phone")
    phone_tap.add_argument("--dry-run", action="store_true", default=False)
    phone_tap.add_argument("--x", type=float, default=None)
    phone_tap.add_argument("--y", type=float, default=None)
    phone_tap.add_argument("--grid", action="store_true", help="Use phone grid coordinates")
    phone_tap.add_argument("--gx", type=int, default=None)
    phone_tap.add_argument("--gy", type=int, default=None)
    _add_camera_args(phone_tap)
    _add_hardware_args(phone_tap)
    phone_tap.set_defaults(command="phone-tap")

    task_add = subparsers.add_parser("task-add", help="Add immediate or interval task")
    task_add.add_argument("--title", required=True)
    task_add.add_argument("--instruction", required=True)
    task_add.add_argument("--task-mode", choices=["pc", "phone"], required=True)
    task_add.add_argument("--every-seconds", type=int, default=None)
    task_add.set_defaults(command="task-add")

    task_list = subparsers.add_parser("task-list", help="List tasks from tasks DB")
    task_list.set_defaults(command="task-list")

    task_run = subparsers.add_parser("task-run", help="Run one task by id")
    task_run.add_argument("--id", required=True)
    task_run.add_argument("--max-steps", type=int, default=30)
    task_run.add_argument("--rule-planner", action="store_true")
    task_run.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(task_run)
    _add_hardware_args(task_run)
    task_run.set_defaults(command="task-run")

    scheduler_run = subparsers.add_parser("scheduler-run", help="Foreground scheduler loop for due tasks")
    scheduler_run.add_argument("--tick", type=float, default=None, help="Override scheduler tick seconds")
    scheduler_run.add_argument("--once", action="store_true", help="Run one scan then exit")
    scheduler_run.add_argument("--max-steps", type=int, default=30)
    scheduler_run.add_argument("--rule-planner", action="store_true")
    scheduler_run.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(scheduler_run)
    _add_hardware_args(scheduler_run)
    scheduler_run.set_defaults(command="scheduler-run")

    task = subparsers.add_parser("task", help="Task management commands")
    task_sub = task.add_subparsers(dest="task_command", metavar="TASK_CMD")

    task_add2 = task_sub.add_parser("add", help="Add a task")
    task_add2.add_argument("--name", required=True)
    task_add2.add_argument("--mode", choices=["computer", "phone"], required=True)
    task_add2.add_argument("--goal", required=True)
    task_add2.add_argument("--at", default=None, help="one-shot time: YYYY-mm-dd HH:MM:SS")
    task_add2.add_argument("--every", type=int, default=None, help="interval seconds")
    task_add2.set_defaults(command="task")

    task_list2 = task_sub.add_parser("list", help="List tasks")
    task_list2.set_defaults(command="task")

    task_remove2 = task_sub.add_parser("remove", help="Remove a task")
    task_remove2.add_argument("--id", required=True)
    task_remove2.set_defaults(command="task")

    task_enable2 = task_sub.add_parser("enable", help="Enable a task")
    task_enable2.add_argument("--id", required=True)
    task_enable2.set_defaults(command="task")

    task_disable2 = task_sub.add_parser("disable", help="Disable a task")
    task_disable2.add_argument("--id", required=True)
    task_disable2.set_defaults(command="task")

    task_run2 = task_sub.add_parser("run", help="Run a task once by id")
    task_run2.add_argument("--id", required=True)
    task_run2.add_argument("--dry-run", action="store_true", default=False)
    task_run2.add_argument("--max-steps", type=int, default=30)
    task_run2.add_argument("--rule-planner", action="store_true")
    task_run2.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(task_run2)
    _add_hardware_args(task_run2)
    task_run2.set_defaults(command="task")

    task_daemon2 = task_sub.add_parser("daemon", help="Run scheduler loop for pending tasks")
    task_daemon2.add_argument("--tick", type=float, default=None)
    task_daemon2.add_argument("--once", action="store_true", default=False)
    task_daemon2.add_argument("--dry-run", action="store_true", default=False)
    task_daemon2.add_argument("--max-steps", type=int, default=30)
    task_daemon2.add_argument("--rule-planner", action="store_true")
    task_daemon2.add_argument("--allow-vlm", action="store_true", default=False)
    _add_camera_args(task_daemon2)
    _add_hardware_args(task_daemon2)
    task_daemon2.set_defaults(command="task")

    doctor = subparsers.add_parser("doctor", help="Run interface consistency self-check (no hardware required)")
    doctor.add_argument("--artifacts-dir", default="artifacts")
    doctor.add_argument("--step-base", type=int, default=9000)
    doctor.set_defaults(command="doctor")

    trace = subparsers.add_parser("trace", help="Print router/provider capability report")
    trace.set_defaults(command="trace")

    web = subparsers.add_parser("web", help="Start huweibot Web Console MVP+")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--log-level", default="info")
    web.set_defaults(command="web")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handlers = _build_subcommand_handlers()
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 2
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("[warn] interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
