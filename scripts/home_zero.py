#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.control.hardware import HardwareController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record homed/zero flag after manual physical homing.")
    parser.add_argument("--flag", default="artifacts/homed.flag", help="Homed flag output path")
    parser.add_argument("--x-mm", type=float, default=0.0)
    parser.add_argument("--y-mm", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Move platform to physical zero/reference manually, then press Enter to confirm.")
    input()

    hw = HardwareController(homed_flag_path=args.flag)
    hw.mark_homed(args.x_mm, args.y_mm)
    print(f"[ok] homed flag written: {Path(args.flag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
