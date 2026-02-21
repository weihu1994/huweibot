#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.vision.camera import Camera
from huweibot.vision.screen_rectify import load_calibration, rectify, validate_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview machine A camera stream for observing machine B screen.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index on machine A")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="artifacts/raw.png", help="Output path when pressing s")
    parser.add_argument("--rectify", action="store_true", help="Rectify to machine B screen view (requires Step 3 calibration)")
    parser.add_argument(
        "--validate-screen",
        action="store_true",
        help="Validate calibration each loop; on failure stop and request recalibration",
    )
    parser.add_argument("--calib", default="config/calibration.json", help="Calibration JSON path")
    return parser


def _save_frame(path: str | Path, frame, timestamp: float) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), frame):
        raise RuntimeError(f"Failed to save image: {p}")

    h, w = frame.shape[:2]
    print(f"[saved] path={p} resolution={w}x{h} timestamp={timestamp:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    calib = None
    if args.rectify or args.validate_screen:
        try:
            calib = load_calibration(args.calib)
        except Exception as exc:
            print(f"[error] failed to load calibration ({args.calib}): {exc}")
            print("[hint] Step3 required: run scripts/calibrate_screen.py first")
            return 1

    save_out = args.out
    if args.rectify and save_out == "artifacts/raw.png":
        save_out = "artifacts/screen.png"

    exit_code = 0

    try:
        with Camera(args.camera_id, args.width, args.height, args.fps) as cam:
            while True:
                frame_pack = cam.read_latest(2)
                raw_frame = frame_pack.bgr
                shown_frame = raw_frame

                if args.rectify:
                    assert calib is not None
                    shown_frame = rectify(raw_frame, calib)

                if args.validate_screen:
                    assert calib is not None
                    ok, reproj_error_px, debug = validate_calibration(raw_frame, calib)
                    if not ok:
                        suggested = debug.get("suggested_dump") or "artifacts/calib_pose_drift.png"
                        dump_target = raw_frame if "pose" in str(suggested) else shown_frame
                        try:
                            _save_frame(suggested, dump_target, frame_pack.timestamp)
                        except Exception as save_exc:
                            print(f"[warn] failed to save validation debug frame: {save_exc}")

                        print("[error] calibration validation failed")
                        print(f"[error] reason={debug.get('reason', 'unknown')} reprojection_error_px={reproj_error_px:.3f}")
                        print("[hint] Re-run scripts/calibrate_screen.py and keep camera/screen pose fixed")
                        exit_code = 2
                        break

                cv2.imshow("preview", shown_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("s"):
                    _save_frame(save_out, shown_frame, frame_pack.timestamp)

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
