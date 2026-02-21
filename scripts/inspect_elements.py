#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.config import load_config
from huweibot.core.tracker import StableTracker
from huweibot.vision.camera import Camera
from huweibot.vision.cursor_detect import CursorDetection, detect_cursor_best_of_n, recover_cursor
from huweibot.vision.screen_rectify import load_calibration, rectify, validate_calibration
from huweibot.vision.ui_elements import (
    compute_screen_hash,
    compute_ui_change_score,
    detect_keyboard_roi,
    extract_ui_elements,
    infer_app_hint,
    maybe_extract_vlm_elements,
    merge_ui_elements,
)


def _model_dump(model: object) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect rectified screen, cursor, and UI elements.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)

    parser.set_defaults(rectify=True, validate_screen=True)
    parser.add_argument("--rectify", dest="rectify", action="store_true", help="Enable screen rectification")
    parser.add_argument("--no-rectify", dest="rectify", action="store_false", help="Disable screen rectification")
    parser.add_argument("--validate-screen", dest="validate_screen", action="store_true", help="Enable calibration validation")
    parser.add_argument("--no-validate-screen", dest="validate_screen", action="store_false", help="Disable calibration validation")

    parser.add_argument("--calib", default="config/calibration.json")
    parser.add_argument("--template-dir", default="assets/cursor_templates")
    parser.add_argument("--cursor-min-confidence", type=float, default=0.55)
    parser.add_argument("--cursor-max-failures", type=int, default=8)

    parser.add_argument("--allow-vlm", action="store_true", default=False)
    parser.add_argument("--json-out", default="", help="Optional path to dump elements JSON")
    return parser


def _draw_elements(frame, elements, cursor: CursorDetection | None, keyboard_roi, app_hint: str | None):
    canvas = frame.copy()

    for elem in elements:
        x1, y1, x2, y2 = map(int, elem.bbox)
        role = elem.role

        if role == "key":
            color = (0, 180, 255)
        elif role in {"button", "toggle"}:
            color = (0, 255, 0)
        elif role == "input":
            color = (255, 180, 0)
        else:
            color = (200, 200, 200)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = elem.text or elem.label or ""
        sid = elem.stable_id or "-"
        label = f"{sid}:{role}"
        if text:
            label += f":{text[:20]}"

        cv2.putText(
            canvas,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if cursor and cursor.cursor_xy is not None:
        cx, cy = int(cursor.cursor_xy[0]), int(cursor.cursor_xy[1])
        cv2.circle(canvas, (cx, cy), 7, (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"cursor={cursor.cursor_type} conf={cursor.cursor_conf:.2f}",
            (max(0, cx + 10), max(15, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if keyboard_roi is not None:
        kx1, ky1, kx2, ky2 = map(int, keyboard_roi)
        cv2.rectangle(canvas, (kx1, ky1), (kx2, ky2), (255, 0, 255), 2)
        cv2.putText(canvas, "keyboard_roi", (kx1, max(14, ky1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if app_hint:
        cv2.putText(canvas, f"app_hint={app_hint}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return canvas


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_cursor_lost_artifacts(screen_frame, elements, artifacts_dir: str) -> None:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(root / "cursor_lost.png"), screen_frame)

    payload = {
        "status": "cursor_lost",
        "elements": [_model_dump(e) for e in elements],
    }
    (root / "elements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()

    calib = None
    if args.rectify or args.validate_screen:
        try:
            calib = load_calibration(args.calib)
        except Exception as exc:
            print(f"[error] failed to load calibration ({args.calib}): {exc}")
            return 1

    tracker = StableTracker()
    prev_tracked = []
    prev_hash: str | None = None

    vlm_state: dict[str, Any] = {
        "elements_min": int(cfg.vlm_trigger_thresholds.elements_min),
        "ambiguous_trigger": int(cfg.vlm_trigger_thresholds.ambiguous_streak),
        "macro_fail_trigger": int(cfg.vlm_trigger_thresholds.macro_fail_streak),
        "ui_change_threshold": float(cfg.vlm_trigger_thresholds.ui_change_threshold),
        "vlm_cooldown_s": float(cfg.vlm_cooldown_s),
        "vlm_max_per_task": int(cfg.vlm_max_per_task),
        "vlm_force_allow_if_elements0_streak": int(cfg.vlm_force_allow_if_elements0_streak),
        "vlm_calls": 0,
        "last_vlm_ts": 0.0,
        "elements0_streak": 0,
        "local_boost_done": False,
        "ambiguous_streak": 0,
        "macro_fail_streak": 0,
    }

    exit_code = 0

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam:
            while True:
                frame_pack = cam.read_latest(2)
                raw_frame = frame_pack.bgr

                if args.validate_screen:
                    assert calib is not None
                    ok, err, debug = validate_calibration(raw_frame, calib)
                    if not ok:
                        dump = debug.get("suggested_dump") or "artifacts/calib_pose_drift.png"
                        Path(dump).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dump), raw_frame)
                        print(f"[error] validate_screen failed: {debug.get('reason')} reprojection_error_px={err:.3f}")
                        print("[hint] please re-run scripts/calibrate_screen.py")
                        exit_code = 3
                        break

                def _transform(frame_bgr):
                    if args.rectify:
                        assert calib is not None
                        return rectify(frame_bgr, calib)
                    return frame_bgr

                cursor = detect_cursor_best_of_n(
                    cam,
                    n=3,
                    template_dir=args.template_dir,
                    frame_transform=_transform,
                )

                screen_frame = cursor.frame_bgr if cursor.frame_bgr is not None else _transform(raw_frame)

                if cursor.cursor_xy is None or cursor.cursor_conf < args.cursor_min_confidence:
                    recovered = recover_cursor(
                        cam,
                        min_confidence=args.cursor_min_confidence,
                        max_failures=args.cursor_max_failures,
                        template_dir=args.template_dir,
                        frame_transform=_transform,
                    )
                    if recovered.cursor_xy is None or recovered.cursor_conf < args.cursor_min_confidence:
                        _save_cursor_lost_artifacts(screen_frame, [], cfg.artifacts_dir)
                        print(
                            f"[error] cursor lost after {args.cursor_max_failures} recovery attempts. "
                            f"Artifacts: {cfg.artifacts_dir}/cursor_lost.png, {cfg.artifacts_dir}/elements.json"
                        )
                        exit_code = 4
                        break

                    cursor = recovered
                    if recovered.frame_bgr is not None:
                        screen_frame = recovered.frame_bgr

                local_elements, local_meta = extract_ui_elements(
                    screen_frame,
                    ocr_backend=None,
                    ui_mode=cfg.ui_mode,
                    max_elements=cfg.max_elements,
                    min_text_conf=cfg.min_text_conf,
                    min_elem_conf=cfg.min_elem_conf,
                    element_merge_iou=cfg.element_merge_iou,
                    keyboard_mode=False,
                    txt_trunc_policy=cfg.txt_trunc_policy,
                    return_meta=True,
                )

                app_hint = infer_app_hint(local_elements)
                keyboard_roi = detect_keyboard_roi(
                    local_elements,
                    screen_w=screen_frame.shape[1],
                    screen_h=screen_frame.shape[0],
                    keyword_list_path=cfg.keyboard_keyword_list_path,
                    density_threshold=cfg.keyboard_density_threshold,
                )
                keyboard_mode = keyboard_roi is not None
                vlm_state["local_boost_done"] = False

                if keyboard_mode:
                    key_elements, _ = extract_ui_elements(
                        screen_frame,
                        ocr_backend=None,
                        ui_mode="heuristic",
                        max_elements=max(32, cfg.max_elements // 2),
                        min_text_conf=cfg.min_text_conf,
                        min_elem_conf=min(cfg.min_elem_conf, 0.20),
                        element_merge_iou=cfg.element_merge_iou,
                        keyboard_mode=True,
                        txt_trunc_policy=cfg.txt_trunc_policy,
                        return_meta=True,
                    )
                    local_elements = merge_ui_elements(
                        local_elements + key_elements,
                        iou_threshold=cfg.element_merge_iou,
                        max_elements=cfg.max_elements,
                    )
                    vlm_state["local_boost_done"] = True

                screen_hash = compute_screen_hash(screen_frame)
                ui_change_score = compute_ui_change_score(prev_hash, screen_hash, prev_tracked, local_elements)

                vlm_elements = []
                vlm_meta = {"called": False, "reason": "disabled"}
                if args.allow_vlm:
                    vlm_elements, vlm_meta = maybe_extract_vlm_elements(
                        screen_frame,
                        local_elements,
                        allow_vlm=True,
                        ui_change_score=ui_change_score,
                        state=vlm_state,
                        roi=keyboard_roi,
                        artifacts_dir=cfg.artifacts_dir,
                        vlm_image_max_side=cfg.vlm_image_max_side,
                        vlm_jpeg_quality=cfg.vlm_jpeg_quality,
                    )

                all_elements = merge_ui_elements(
                    local_elements + vlm_elements,
                    iou_threshold=cfg.element_merge_iou,
                    max_elements=cfg.max_elements,
                )

                tracked = tracker.track(prev_tracked, all_elements)
                prev_tracked = tracked
                prev_hash = screen_hash

                canvas = _draw_elements(screen_frame, tracked, cursor, keyboard_roi, app_hint)
                cv2.imshow("inspect_elements", canvas)

                payload = {
                    "timestamp": frame_pack.timestamp,
                    "cursor": {
                        "xy": cursor.cursor_xy,
                        "conf": cursor.cursor_conf,
                        "type": cursor.cursor_type,
                    },
                    "app_hint": app_hint,
                    "keyboard_mode": keyboard_mode,
                    "keyboard_roi": keyboard_roi,
                    "screen_hash": screen_hash,
                    "ui_change_score": ui_change_score,
                    "elements": [_model_dump(e) for e in tracked],
                    "meta": {
                        "local": local_meta,
                        "vlm": vlm_meta,
                    },
                }

                if args.json_out:
                    _save_json(args.json_out, payload)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    target_img = Path(cfg.artifacts_dir) / "screen_elements.png"
                    target_img.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target_img), canvas)
                    print(f"[saved] {target_img}")

                    if args.json_out:
                        _save_json(args.json_out, payload)
                        print(f"[saved] {args.json_out}")

    except RuntimeError as exc:
        print(f"[error] {exc}")
        exit_code = 1
    except cv2.error as exc:
        print(f"[error] OpenCV failure: {exc}")
        exit_code = 1
    finally:
        cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
