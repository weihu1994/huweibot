#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.config import load_config
from huweibot.control.grbl_serial import GRBLSerial
from huweibot.control.hardware import HardwareController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual relative move via GRBL (machine A hardware only).")
    parser.add_argument("--port", required=True, help="GRBL serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dx-mm", type=float, default=0.0)
    parser.add_argument("--dy-mm", type=float, default=0.0)
    parser.add_argument("--feed", type=float, default=None, help="Override feed rate (mm/min)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between repeats in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repeat < 1:
        print("[error] --repeat must be >= 1")
        return 2

    cfg = load_config()
    try:
        with GRBLSerial(args.port, baud=args.baud, timeout_s=float(cfg.serial_timeout_s)) as grbl:
            hw = HardwareController(
                grbl=grbl,
                travel_range_mm=cfg.travel_range_mm,
                feed_rate=float(cfg.feed_rate_mm_min),
                homed_flag_path=cfg.homed_flag_path,
                enforce_homed=False,
            )
            for i in range(args.repeat):
                status = hw.move_mm(args.dx_mm, args.dy_mm, feed=args.feed)
                print(f"[ok] step={i + 1}/{args.repeat} {status.detail}")
                if i < args.repeat - 1:
                    time.sleep(max(0.0, float(args.delay)))
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1
    except KeyboardInterrupt:
        print("[warn] interrupted by user")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
