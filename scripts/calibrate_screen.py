#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.vision.camera import Camera
from huweibot.vision.screen_rectify import build_baseline_signature


WINDOW_NAME = "calibrate_screen"
PREVIEW_WINDOW = "calibrate_screen_rectified_preview"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual 4-corner homography calibration for machine B screen.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--screen-w", type=int, default=1920, help="Machine B screen width")
    parser.add_argument("--screen-h", type=int, default=1080, help="Machine B screen height")
    parser.add_argument("--out", default="config/calibration.json")
    return parser


def _order_points(pts: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = arr[np.argmin(s)]  # TL
    ordered[2] = arr[np.argmax(s)]  # BR
    ordered[1] = arr[np.argmin(d)]  # TR
    ordered[3] = arr[np.argmax(d)]  # BL
    return ordered


def _mouse_handler(event: int, x: int, y: int, _flags: int, state: dict[str, object]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    points = state["points"]
    assert isinstance(points, list)

    if len(points) >= 4:
        return

    points.append((x, y))


def _draw_overlay(base: np.ndarray, points: list[tuple[int, int]], confirmed: bool) -> np.ndarray:
    frame = base.copy()

    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            str(idx),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if len(points) == 4:
        poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 200, 0) if confirmed else (0, 200, 255)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

    instructions = [
        "Click corners in order: TL -> TR -> BR -> BL",
        "c/Enter: confirm    r: reset    y: save+exit    q/Esc: quit",
    ]

    y = 28
    for text in instructions:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    status = f"points={len(points)}/4 confirmed={confirmed}"
    cv2.putText(frame, status, (12, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _compute_homography(
    points: list[tuple[int, int]],
    screen_w: int,
    screen_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = _order_points(points)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(screen_w - 1), 0.0],
            [float(screen_w - 1), float(screen_h - 1)],
            [0.0, float(screen_h - 1)],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H, src, dst


def _save_calibration(
    out_path: Path,
    screen_w: int,
    screen_h: int,
    H: np.ndarray,
    src: np.ndarray,
    baseline: dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "screen_w": int(screen_w),
        "screen_h": int(screen_h),
        "H": H.astype(float).tolist(),
        "src_points": src.astype(float).tolist(),
        "created_at": time.time(),
        "baseline": baseline,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam:
            captured = cam.read_latest(3)
            raw = captured.bgr.copy()
    except Exception as exc:
        print(f"[error] failed to open/read camera: {exc}")
        return 1

    state: dict[str, object] = {"points": []}
    confirmed = False
    H: np.ndarray | None = None
    src: np.ndarray | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_handler, state)

    try:
        while True:
            points = state["points"]
            assert isinstance(points, list)

            canvas = _draw_overlay(raw, points, confirmed)
            cv2.imshow(WINDOW_NAME, canvas)

            if confirmed and H is not None:
                rectified_preview = cv2.warpPerspective(raw, H, (args.screen_w, args.screen_h))
                cv2.imshow(PREVIEW_WINDOW, rectified_preview)

            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                print("[info] calibration canceled")
                return 1

            if key == ord("r"):
                state["points"] = []
                confirmed = False
                H = None
                src = None
                continue

            if key in (13, ord("c")):
                if len(points) != 4:
                    print("[warn] need exactly 4 points before confirm")
                    continue

                H, src, _ = _compute_homography(points, args.screen_w, args.screen_h)
                confirmed = True
                print("[info] homography confirmed; press y to save or r to reselect")
                continue

            if key == ord("y"):
                if not confirmed or H is None or src is None:
                    print("[warn] calibration not confirmed yet (press c/Enter first)")
                    continue

                rectified = cv2.warpPerspective(raw, H, (args.screen_w, args.screen_h))
                baseline = build_baseline_signature(rectified)
                out_path = Path(args.out)
                _save_calibration(out_path, args.screen_w, args.screen_h, H, src, baseline)
                print(f"[ok] calibration saved: {out_path}")
                return 0

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
