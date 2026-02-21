#!/usr/bin/env python3
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 11 click+verify demo CLI (flow stub; use click_mode=stub when hardware is unavailable)."
    )

    # Target selector.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-query", type=str, help="Selector DSL query (ElementRef.by=query)")
    group.add_argument("--target-id", type=str, help="Stable id (ElementRef.by=id)")

    # Verify mode shortcuts.
    parser.add_argument("--verify-text-present", type=str, default="", help="Verify that text appears after click")
    parser.add_argument(
        "--verify-text-changed",
        action="store_true",
        help="Verify text/UI changed after click",
    )
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--retry", type=int, default=1)

    # Camera / rectify passthrough.
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rectify", action="store_true")
    parser.add_argument("--validate-screen", action="store_true")
    parser.add_argument("--calib", type=str, default="config/calibration.json")

    # Hardware passthrough.
    parser.add_argument("--port", type=str, default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mapping", type=str, default="config/mapping.json")
    parser.add_argument("--click-mode", choices=["stub", "same_serial", "separate_serial"], default="stub")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.click_mode != "stub" and not args.port:
        print("error: --port is required unless --click-mode=stub")
        return 2

    verify_mode = "TEXT_CHANGED" if args.verify_text_changed else "TEXT_PRESENT" if args.verify_text_present else "AUTO"
    target = f"id={args.target_id}" if args.target_id else f"query={args.target_query}"

    print("click_verify_demo Step11 flow stub")
    print(f"target: {target}")
    print(f"verify_mode: {verify_mode}")
    print(f"camera: id={args.camera_id} {args.width}x{args.height}@{args.fps} rectify={args.rectify} validate={args.validate_screen}")
    print(f"hardware: click_mode={args.click_mode} port={args.port or '<none>'} baud={args.baud} mapping={args.mapping}")
    print("note: use --click-mode stub when hardware is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
