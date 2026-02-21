#!/usr/bin/env python3
from __future__ import annotations

"""
Dual-machine boundary:
- cursor_xy refers to machine B rectified screen coordinates.
- platform movement acts on machine B physical mouse baseplate from machine A.
"""

import argparse
import time
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.config import load_config
from huweibot.control.grbl_serial import GRBLSerial
from huweibot.control.hardware import HardwareController
from huweibot.control.kinematics import PxMmMapping
from huweibot.vision.camera import Camera
from huweibot.vision.cursor_detect import CursorDetection, detect_cursor_best_of_n
from huweibot.vision.screen_rectify import load_calibration, rectify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate px->mm mapping using cursor displacement on machine B.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--calib", default="config/calibration.json")
    parser.add_argument("--templates", default="assets/cursor_templates")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--settle-ms", type=int, default=350)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--out", default="config/mapping.json")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser


def _observe_cursor_best(
    cam: Camera,
    *,
    template_dir: str,
    threshold: float,
    frame_transform,
    retries: int,
) -> CursorDetection | None:
    for _ in range(max(1, retries)):
        detection = detect_cursor_best_of_n(
            cam,
            n=3,
            template_dir=template_dir,
            frame_transform=frame_transform,
        )
        if detection.cursor_xy is not None and detection.cursor_conf >= float(threshold):
            return detection
    return None


def _save_artifact(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty values")
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) * 0.5)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()

    if args.samples < 1:
        print("[error] --samples must be >= 1")
        return 2
    if args.step_mm <= 0:
        print("[error] --step-mm must be > 0")
        return 2

    try:
        calib = load_calibration(args.calib)
    except Exception as exc:
        print(f"[error] failed to load calibration: {exc}")
        return 1

    def _rectify(frame_bgr):
        return rectify(frame_bgr, calib)

    artifacts = Path(cfg.artifacts_dir)
    failures = 0

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam, GRBLSerial(
            args.port, baud=args.baud, timeout_s=float(cfg.serial_timeout_s)
        ) as grbl:
            hw = HardwareController(
                grbl=grbl,
                travel_range_mm=cfg.travel_range_mm,
                feed_rate=float(cfg.feed_rate_mm_min),
                homed_flag_path=cfg.homed_flag_path,
                enforce_homed=False,
            )

            def observe_with_retry(local_retries: int = 2) -> CursorDetection | None:
                nonlocal failures
                det = _observe_cursor_best(
                    cam,
                    template_dir=args.templates,
                    threshold=args.threshold,
                    frame_transform=_rectify,
                    retries=local_retries,
                )
                if det is None:
                    failures += 1
                return det

            def measure_axis(axis: str) -> tuple[float, int]:
                nonlocal failures
                values: list[float] = []
                signs: list[int] = []

                while len(values) < args.samples:
                    if failures > args.max_retries:
                        raise RuntimeError(
                            f"cursor detect failed over max retries ({args.max_retries}); "
                            "check lighting/template/mouse visibility"
                        )

                    p0 = observe_with_retry(3)
                    if p0 is None or p0.cursor_xy is None:
                        continue

                    dx = args.step_mm if axis == "x" else 0.0
                    dy = args.step_mm if axis == "y" else 0.0
                    hw.move_mm(dx, dy)
                    time.sleep(max(0.0, args.settle_ms / 1000.0))
                    p1 = observe_with_retry(3)
                    if p1 is None or p1.cursor_xy is None:
                        # Try to move back even if observation failed.
                        try:
                            hw.move_mm(-dx, -dy)
                            time.sleep(max(0.0, args.settle_ms / 1000.0))
                        except Exception:
                            pass
                        continue

                    delta_px = (p1.cursor_xy[0] - p0.cursor_xy[0]) if axis == "x" else (p1.cursor_xy[1] - p0.cursor_xy[1])
                    if abs(delta_px) < 1e-6:
                        failures += 1
                        try:
                            hw.move_mm(-dx, -dy)
                            time.sleep(max(0.0, args.settle_ms / 1000.0))
                        except Exception:
                            pass
                        continue

                    values.append(float(args.step_mm) / abs(float(delta_px)))
                    signs.append(1 if (float(args.step_mm) / float(delta_px)) > 0 else -1)

                    # Move back to reduce drift accumulation.
                    hw.move_mm(-dx, -dy)
                    time.sleep(max(0.0, args.settle_ms / 1000.0))

                sign = 1 if signs.count(1) >= signs.count(-1) else -1
                return _median(values), sign

            # Capture one frame for artifact debug.
            if args.save_artifacts:
                shot = cam.read_latest(2).bgr
                _save_artifact(artifacts / "calib_px_to_mm_raw.png", shot)
                _save_artifact(artifacts / "calib_px_to_mm_screen.png", _rectify(shot))

            mm_per_px_x, sign_x = measure_axis("x")
            mm_per_px_y, sign_y = measure_axis("y")

            mapping = PxMmMapping(
                mm_per_px_x=mm_per_px_x,
                mm_per_px_y=mm_per_px_y,
                sign_x=sign_x,
                sign_y=sign_y,
                max_move_mm=float(cfg.max_move_mm),
                travel_range_mm=cfg.travel_range_mm,
                last_calib_time=time.time(),
            )
            mapping.save_json(args.out)

            # 100px sanity move (magnitude check only).
            check_before = observe_with_retry(3)
            check_dx_mm, _ = mapping.px_to_mm(100.0, 0.0, current_pos_mm=hw.pos_mm)
            hw.move_mm(check_dx_mm, 0.0)
            time.sleep(max(0.0, args.settle_ms / 1000.0))
            check_after = observe_with_retry(3)
            observed = None
            if check_before and check_before.cursor_xy and check_after and check_after.cursor_xy:
                observed = check_after.cursor_xy[0] - check_before.cursor_xy[0]

            # Move back after sanity check.
            hw.move_mm(-check_dx_mm, 0.0)
            time.sleep(max(0.0, args.settle_ms / 1000.0))

            print("[ok] mapping calibrated")
            print(
                f"[ok] mm_per_px_x={mapping.mm_per_px_x:.6f} sign_x={mapping.sign_x} "
                f"mm_per_px_y={mapping.mm_per_px_y:.6f} sign_y={mapping.sign_y}"
            )
            if observed is not None:
                print(f"[ok] 100px sanity observed_dx_px={observed:.2f}")
            else:
                print("[warn] 100px sanity check observation unavailable")
            print(f"[ok] mapping saved: {args.out}")
            return 0
    except Exception as exc:
        print(f"[error] calibration failed: {exc}")
        print("[hint] Check B-screen pointer visibility, lighting, template quality, and disable mouse acceleration on B.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
