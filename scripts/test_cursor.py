#!/usr/bin/env python3
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test cursor detection (Step 1 placeholder).")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--template-dir", default="assets/cursor_templates")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"test_cursor placeholder: camera_id={args.camera_id}, template_dir={args.template_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
