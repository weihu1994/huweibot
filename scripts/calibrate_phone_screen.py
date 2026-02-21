#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.vision.camera import Camera
from huweibot.vision.phone_screen import (
    PhoneScreenROI,
    auto_detect_phone_screen_roi,
    draw_phone_screen_overlay,
    save_phone_screen,
)

WINDOW_NAME = "calibrate_phone_screen"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual phone screen bbox calibration from camera frame.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="config/phone_screen.json")
    parser.add_argument("--auto", action="store_true", help="Use local CV heuristic auto detection first")
    parser.add_argument("--allow-vlm", action="store_true", default=False, help="Allow one VLM fallback when auto CV fails")
    parser.add_argument("--preview", action="store_true", default=False, help="Show auto-detect overlay preview")
    parser.add_argument("--save-artifacts", action="store_true", default=False)
    return parser


def _order_points(pts: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = arr[np.argmin(s)]
    ordered[2] = arr[np.argmax(s)]
    ordered[1] = arr[np.argmin(d)]
    ordered[3] = arr[np.argmax(d)]
    return ordered


def _mouse_handler(event: int, x: int, y: int, _flags: int, state: dict[str, object]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    points = state["points"]
    assert isinstance(points, list)
    if len(points) < 4:
        points.append((x, y))


def _draw(frame: np.ndarray, points: list[tuple[int, int]], confirmed: bool, auto_roi: PhoneScreenROI | None = None) -> np.ndarray:
    canvas = frame.copy()
    if auto_roi is not None:
        canvas = draw_phone_screen_overlay(canvas, auto_roi)
    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(canvas, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if len(points) == 4:
        poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [poly], True, (0, 220, 0) if confirmed else (0, 220, 255), 2)
    tips = [
        "Click phone corners: TL -> TR -> BR -> BL",
        "c/Enter: confirm  r: reset  y: save  q/Esc: quit",
    ]
    y = 28
    for tip in tips:
        cv2.putText(canvas, tip, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    cv2.putText(
        canvas,
        f"points={len(points)}/4 confirmed={confirmed}",
        (12, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam:
            raw = cam.read_latest(3).bgr.copy()
    except Exception as exc:
        print(f"[error] failed to open/read camera: {exc}")
        return 1

    state: dict[str, Any] = {"points": []}
    confirmed = False
    bbox: tuple[int, int, int, int] | None = None
    auto_roi: PhoneScreenROI | None = None

    if args.auto:
        router = None
        if args.allow_vlm:
            try:
                from huweibot.agent.router import Router
                from huweibot.config import load_config

                router = Router(load_config())
            except Exception:
                router = None
        auto_roi = auto_detect_phone_screen_roi(
            raw,
            allow_vlm=bool(args.allow_vlm),
            router=router,
            vlm_image_max_side=768,
            vlm_jpeg_quality=70,
        )
        if auto_roi is not None:
            bbox = auto_roi.bbox
            confirmed = True
            print(f"[info] auto detect ({auto_roi.method}) bbox={bbox} confidence={auto_roi.confidence:.2f}")
        else:
            print("[warn] auto detect failed; switch to manual point selection")

    if args.save_artifacts:
        artifacts = Path("artifacts")
        artifacts.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(artifacts / "phone_calib_raw.png"), raw)
        if auto_roi is not None:
            overlay = draw_phone_screen_overlay(raw, auto_roi)
            cv2.imwrite(str(artifacts / "phone_calib_auto.png"), overlay)

    if args.auto and confirmed and not args.preview:
        h, w = raw.shape[:2]
        save_phone_screen(
            args.out,
            bbox,  # type: ignore[arg-type]
            source_w=w,
            source_h=h,
            corners=auto_roi.corners if auto_roi is not None else None,
            confidence=(auto_roi.confidence if auto_roi is not None else None),
            method=(auto_roi.method if auto_roi is not None else "manual"),
        )
        print(f"[ok] phone screen calibration saved: {args.out} at {time.time():.3f}")
        return 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_handler, state)

    try:
        while True:
            points = state["points"]
            assert isinstance(points, list)
            cv2.imshow(WINDOW_NAME, _draw(raw, points, confirmed, auto_roi=auto_roi if args.preview else None))
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                print("[info] phone screen calibration canceled")
                return 1
            if key == ord("r"):
                state["points"] = []
                confirmed = False
                bbox = None
                auto_roi = None
                continue
            if key in (13, ord("c")):
                if len(points) != 4:
                    print("[warn] need exactly 4 points before confirm")
                    continue
                ordered = _order_points(points)
                x1 = int(round(float(np.min(ordered[:, 0]))))
                y1 = int(round(float(np.min(ordered[:, 1]))))
                x2 = int(round(float(np.max(ordered[:, 0]))))
                y2 = int(round(float(np.max(ordered[:, 1]))))
                if x2 <= x1 or y2 <= y1:
                    print("[warn] invalid bbox from selected points, reselect")
                    continue
                bbox = (x1, y1, x2, y2)
                confirmed = True
                auto_roi = PhoneScreenROI(
                    bbox=bbox,
                    corners=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                    confidence=1.0,
                    method="manual",
                )
                print(f"[info] bbox confirmed: {bbox}; press y to save")
                continue
            if key == ord("y"):
                if not confirmed or bbox is None:
                    print("[warn] phone screen is not confirmed yet")
                    continue
                h, w = raw.shape[:2]
                save_phone_screen(
                    args.out,
                    bbox,
                    source_w=w,
                    source_h=h,
                    corners=(auto_roi.corners if auto_roi is not None else None),
                    confidence=(auto_roi.confidence if auto_roi is not None else None),
                    method=(auto_roi.method if auto_roi is not None else "manual"),
                )
                print(f"[ok] phone screen calibration saved: {args.out} at {time.time():.3f}")
                return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
