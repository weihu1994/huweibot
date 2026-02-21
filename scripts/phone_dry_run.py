#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot._pydantic_compat import model_to_dict
from huweibot.agent.schemas import VerifyRule
from huweibot.config import load_config
from huweibot.control.ir_reader import MockIRDistanceReader, save_ir_calibration
from huweibot.core.actions import action_from_json, action_to_json
from huweibot.core.coords import phone_grid_to_screen_px
from huweibot.vision.camera import Camera
from huweibot.vision.phone_screen import (
    PhoneScreenROI,
    auto_detect_phone_screen_roi,
    draw_phone_screen_overlay,
    load_phone_screen_roi,
)


class MockXYZHardware:
    def __init__(self):
        self.actions: list[dict[str, Any]] = []
        self._last_px: tuple[int, int] = (0, 0)

    @staticmethod
    def _normalize_action(payload: dict[str, Any]) -> dict[str, Any]:
        return action_to_json(action_from_json(payload), as_dict=True)

    def move_to_px(self, x: int, y: int) -> None:
        self._last_px = (int(x), int(y))
        self.actions.append(
            self._normalize_action(
                {
                    "type": "MOVE_TO",
                    "target": {"coord_type": "screen_px", "x": int(x), "y": int(y)},
                }
            )
        )

    def tap(self, press_ms: int = 60) -> dict[str, Any]:
        x, y = self._last_px
        action = self._normalize_action(
            {
                "type": "TOUCH_PRESS",
                "coord": {"coord_type": "screen_px", "x": int(x), "y": int(y)},
                "duration_ms": max(20, min(60000, int(press_ms))),
            }
        )
        self.actions.append(action)
        return action


def _parse_tap(value: list[str]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _parse_swipe(value: list[str]) -> tuple[int, int, int, int]:
    return int(value[0]), int(value[1]), int(value[2]), int(value[3])


def _clamp_grid(gx: int, gy: int, grid_w: int, grid_h: int) -> tuple[int, int, bool]:
    cgx = max(0, min(int(grid_w - 1), int(gx)))
    cgy = max(0, min(int(grid_h - 1), int(gy)))
    return cgx, cgy, (cgx != gx or cgy != gy)


def _write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phone mode dry-run pipeline (no hardware required).")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--phone-screen", default="config/phone_screen.json")
    parser.add_argument("--auto-screen", action="store_true", default=False)
    parser.add_argument("--allow-vlm", action="store_true", default=False)

    parser.add_argument("--grid-w", type=int, default=200)
    parser.add_argument("--grid-h", type=int, default=100)

    parser.add_argument("--tap", nargs=2, action="append", metavar=("GX", "GY"), default=[])
    parser.add_argument("--swipe", nargs=4, action="append", metavar=("GX1", "GY1", "GX2", "GY2"), default=[])
    parser.add_argument("--steps", type=int, default=6)

    parser.add_argument("--mock-ir-mm", type=float, default=8.0)
    parser.add_argument("--ir-contact-threshold-mm", type=float, default=2.0)
    parser.add_argument("--tap-press-ms", type=int, default=60)
    parser.add_argument("--calibrate-ir-threshold-mm", type=float, default=None)
    parser.add_argument("--ir-calib-out", default="config/ir_calibration.json")

    parser.add_argument("--out", default="artifacts/phone_dry_run.jsonl")
    parser.add_argument("--save-artifacts", action="store_true", default=False)
    parser.add_argument("--preview", action="store_true", default=False)
    return parser


def _load_or_detect_roi(frame_bgr: Any, args: argparse.Namespace) -> tuple[PhoneScreenROI | None, str]:
    if bool(args.auto_screen):
        router = None
        if bool(args.allow_vlm):
            try:
                from huweibot.agent.router import Router

                router = Router(load_config())
            except Exception:
                router = None
        roi = auto_detect_phone_screen_roi(
            frame_bgr,
            allow_vlm=bool(args.allow_vlm),
            router=router,
            vlm_image_max_side=768,
            vlm_jpeg_quality=70,
        )
        if roi is not None:
            return roi, "auto"
    try:
        return load_phone_screen_roi(args.phone_screen), "file"
    except Exception:
        return None, "none"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.calibrate_ir_threshold_mm is not None:
        try:
            save_ir_calibration(args.ir_calib_out, float(args.calibrate_ir_threshold_mm))
            print(f"[ok] IR calibration saved: {args.ir_calib_out} threshold_mm={float(args.calibrate_ir_threshold_mm):.3f}")
        except Exception as exc:
            print(f"[error] failed to save IR calibration: {exc}")
            return 2

    taps = [_parse_tap(v) for v in args.tap]
    swipes = [_parse_swipe(v) for v in args.swipe]

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam:
            frame = cam.read_latest(3).bgr
    except Exception as exc:
        print(f"[error] camera unavailable: {exc}")
        return 1

    roi, roi_source = _load_or_detect_roi(frame, args)
    if roi is None:
        print("[error] phone ROI unavailable: run scripts/calibrate_phone_screen.py or pass --auto-screen")
        return 2

    ir = MockIRDistanceReader(fixed_mm=float(args.mock_ir_mm))
    hw = MockXYZHardware()
    records: list[dict[str, Any]] = []
    verify_none = model_to_dict(VerifyRule(type="NONE"), exclude_none=True)
    press_ms = max(20, min(1500, int(args.tap_press_ms)))

    def _action(payload: dict[str, Any]) -> dict[str, Any]:
        return action_to_json(action_from_json(payload), as_dict=True)

    def log(action: dict[str, Any]) -> None:
        rec = {"ts": time.time(), **action}
        records.append(rec)

    clamp_count = 0
    oob_detected = False

    with ir:
        for gx, gy in taps:
            cgx, cgy, clamped = _clamp_grid(gx, gy, int(args.grid_w), int(args.grid_h))
            clamp_count += int(clamped)
            px, py = phone_grid_to_screen_px(cgx, cgy, roi.bbox, int(args.grid_w), int(args.grid_h))
            in_bbox = roi.bbox[0] <= px <= roi.bbox[2] and roi.bbox[1] <= py <= roi.bbox[3]
            oob_detected = oob_detected or (not in_bbox)
            hw.move_to_px(px, py)
            action_click = _action(
                {
                    "type": "TOUCH_TAP",
                    "coord": {"coord_type": "grid", "x": cgx, "y": cgy},
                    "times": 1,
                    "press_ms": press_ms,
                }
            )
            log(
                {
                    "kind": "ACTION",
                    "op": "tap",
                    "action": action_click,
                    "verify": verify_none,
                    "grid": [gx, gy],
                    "grid_clamped": [cgx, cgy],
                    "clamped": clamped,
                    "to_px": [px, py],
                    "in_roi": in_bbox,
                }
            )
            dist = ir.filtered_mm(n=3, interval_ms=15)
            log({"kind": "IR_READ", "distance_mm": dist})
            if dist is not None and dist > float(args.ir_contact_threshold_mm):
                recovery = []
                for item in ir.recover_contact(retries=1, hold_ms=press_ms):
                    try:
                        recovery.append(_action(item))
                    except Exception:
                        continue
                log(
                    {
                        "kind": "RECOVER_CONTACT",
                        "ok": False,
                        "reason": "distance_too_far",
                        "distance_mm": dist,
                        "recovery_actions": recovery,
                    }
                )
            else:
                exec_action = hw.tap(press_ms=press_ms)
                log({"kind": "EXEC", "action": exec_action})

        for gx1, gy1, gx2, gy2 in swipes:
            c1x, c1y, cl1 = _clamp_grid(gx1, gy1, int(args.grid_w), int(args.grid_h))
            c2x, c2y, cl2 = _clamp_grid(gx2, gy2, int(args.grid_w), int(args.grid_h))
            clamp_count += int(cl1) + int(cl2)
            action_drag = _action(
                {
                    "type": "TOUCH_SWIPE",
                    "from": {"coord_type": "grid", "x": c1x, "y": c1y},
                    "to": {"coord_type": "grid", "x": c2x, "y": c2y},
                    "duration_ms": max(20, min(60000, int(press_ms) * max(2, int(args.steps)))),
                }
            )
            swipe_path_px: list[list[int]] = []
            for i in range(max(2, int(args.steps))):
                t = i / float(max(1, int(args.steps) - 1))
                igx = int(round((1.0 - t) * c1x + t * c2x))
                igy = int(round((1.0 - t) * c1y + t * c2y))
                px, py = phone_grid_to_screen_px(igx, igy, roi.bbox, int(args.grid_w), int(args.grid_h))
                in_bbox = roi.bbox[0] <= px <= roi.bbox[2] and roi.bbox[1] <= py <= roi.bbox[3]
                oob_detected = oob_detected or (not in_bbox)
                hw.move_to_px(px, py)
                swipe_path_px.append([int(px), int(py)])
            log(
                {
                    "kind": "ACTION",
                    "op": "swipe",
                    "action": action_drag,
                    "verify": verify_none,
                    "from_grid": [gx1, gy1],
                    "to_grid": [gx2, gy2],
                    "from_grid_clamped": [c1x, c1y],
                    "to_grid_clamped": [c2x, c2y],
                    "path_px": swipe_path_px,
                }
            )

    summary = {
        "roi_detected": True,
        "roi_source": roi_source,
        "roi": roi.to_dict(),
        "actions": len(records),
        "clamp_count": clamp_count,
        "out_of_bounds": oob_detected,
        "ir_threshold_mm": float(args.ir_contact_threshold_mm),
    }
    log({"kind": "SUMMARY", **summary})
    _write_jsonl(args.out, records)

    if bool(args.save_artifacts) or bool(args.preview):
        overlay = draw_phone_screen_overlay(frame, roi)
        for rec in records:
            if rec.get("kind") == "ACTION" and rec.get("op") == "tap" and isinstance(rec.get("to_px"), list):
                px, py = rec["to_px"]
                cv2.circle(overlay, (int(px), int(py)), 3, (255, 220, 0), -1)
            if rec.get("kind") == "ACTION" and rec.get("op") == "swipe":
                for pt in rec.get("path_px", []):
                    if isinstance(pt, list) and len(pt) == 2:
                        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (0, 200, 255), -1)
        if bool(args.save_artifacts):
            artifacts = Path("artifacts")
            artifacts.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(artifacts / "phone_dry_run_overlay.png"), overlay)
        if bool(args.preview):
            cv2.imshow("phone_dry_run", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(
        f"[ok] roi_detected={summary['roi_detected']} source={summary['roi_source']} "
        f"actions={summary['actions']} clamp_count={summary['clamp_count']} out_of_bounds={summary['out_of_bounds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
