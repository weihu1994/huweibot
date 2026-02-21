#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.agent.schemas import ElementRef
from huweibot.config import load_config
from huweibot.control.grbl_serial import GRBLSerial
from huweibot.control.hardware import HardwareController
from huweibot.control.kinematics import PxMmMapping
from huweibot.core.coords import grid_to_screen_px
from huweibot.core.loop import LoopConfig, XBotLoop
from huweibot.core.tracker import StableTracker
from huweibot.vision.camera import Camera
from huweibot.vision.screen_rectify import load_calibration, rectify, validate_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop move_to for machine B screen target.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--validate-screen", action="store_true")
    parser.add_argument("--calib", default="config/calibration.json")

    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mapping", default="config/mapping.json")

    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--gx", type=int, default=None)
    parser.add_argument("--gy", type=int, default=None)
    parser.add_argument("--target-id", type=str, default="")
    parser.add_argument("--target-query", type=str, default="")
    return parser


def _target_mode(args: argparse.Namespace) -> str:
    modes = []
    if args.target_id:
        modes.append("target_id")
    if args.target_query:
        modes.append("target_query")
    if args.grid:
        modes.append("grid")
    if args.x is not None or args.y is not None:
        modes.append("xy")
    if len(modes) != 1:
        raise ValueError("choose exactly one target mode: (--x --y) | (--grid --gx --gy) | --target-id | --target-query")
    mode = modes[0]
    if mode == "xy" and (args.x is None or args.y is None):
        raise ValueError("both --x and --y are required for xy mode")
    if mode == "grid" and (args.gx is None or args.gy is None):
        raise ValueError("--grid requires --gx and --gy")
    return mode


def _print_candidates(obs) -> None:
    print("[info] resolve candidates (top 12):")
    for idx, e in enumerate(obs.elements[:12], start=1):
        text = (e.text or e.label or "").strip()
        sid = e.stable_id or "-"
        print(f"  {idx:02d}. sid={sid} role={e.role} conf={e.confidence:.2f} text='{text[:32]}' bbox={e.bbox}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()

    try:
        mode = _target_mode(args)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    calib = None
    if args.rectify or args.validate_screen:
        try:
            calib = load_calibration(args.calib)
        except Exception as exc:
            print(f"[error] failed to load calibration: {exc}")
            return 1

    try:
        mapping = PxMmMapping.load_json(args.mapping)
    except Exception as exc:
        print(f"[error] failed to load mapping: {exc}")
        return 1

    rectifier = None
    if args.rectify and calib is not None:
        rectifier = lambda frame_bgr: rectify(frame_bgr, calib)  # noqa: E731

    loop_cfg = LoopConfig()
    tracker = StableTracker()

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam, GRBLSerial(
            args.port,
            baud=args.baud,
            timeout_s=float(cfg.serial_timeout_s),
        ) as grbl:
            if args.validate_screen and calib is not None:
                frame = cam.read_latest(2).bgr
                ok, err, debug = validate_calibration(frame, calib)
                if not ok:
                    print(f"[error] validate_screen failed: {debug.get('reason')} reprojection_error_px={err:.2f}")
                    return 3

            hw = HardwareController(
                grbl=grbl,
                travel_range_mm=cfg.travel_range_mm,
                feed_rate=float(cfg.feed_rate_mm_min),
                homed_flag_path=cfg.homed_flag_path,
                enforce_homed=bool(cfg.enforce_homed),
                click_mode=cfg.click_mode,
                click_port=cfg.click_port,
                click_baud=cfg.click_baud,
                screen_w=(calib.screen_w if calib is not None else args.width),
                screen_h=(calib.screen_h if calib is not None else args.height),
            )

            loop = XBotLoop(
                camera=cam,
                hardware=hw,
                kinematics=mapping,
                rectifier=rectifier,
                tracker=tracker,
                config=loop_cfg,
                runtime_config=cfg,
            )

            obs = loop.observe(force_ui=True)
            print(f"[info] cursor={obs.cursor_xy} conf={obs.cursor_conf:.2f} type={obs.cursor_type}")

            if mode == "xy":
                target_x = float(args.x)
                target_y = float(args.y)
                result = loop.move_to(target_x, target_y)
                print(f"[info] target=({target_x:.2f},{target_y:.2f})")
            elif mode == "grid":
                target_x, target_y = grid_to_screen_px(
                    int(args.gx),
                    int(args.gy),
                    obs.screen_w,
                    obs.screen_h,
                    1000,
                )
                result = loop.move_to(float(target_x), float(target_y))
                print(f"[info] target_grid=({args.gx},{args.gy}) -> target_px=({target_x},{target_y})")
            elif mode == "target_id":
                ref = ElementRef(by="id", value=args.target_id)
                # Pre-resolve; failure must not move.
                pre = loop.observe(force_ui=True)
                try:
                    loop._resolve_element(ref, pre)
                except Exception as exc:
                    print(f"[error] resolve failed: {exc}")
                    _print_candidates(pre)
                    return 4
                result = loop.move_to_target(ref)
            else:
                ref = ElementRef(by="query", value=args.target_query)
                pre = loop.observe(force_ui=True)
                try:
                    loop._resolve_element(ref, pre)
                except Exception as exc:
                    print(f"[error] resolve failed: {exc}")
                    _print_candidates(pre)
                    return 4
                result = loop.move_to_target(ref)

            print(f"[result] ok={bool(result.get('ok', False))} reason={result.get('reason')}")
            print(f"[result] iters={result.get('iters')} final_error={result.get('final_error')}")
            return 0 if bool(result.get("ok", False)) else 5
    except Exception as exc:
        print(f"[error] move_to failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
