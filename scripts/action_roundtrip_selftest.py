#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.core.actions import action_from_json, action_to_json


def _samples() -> list[dict]:
    return [
        {"type": "MOVE_TO", "target": {"coord_type": "screen_px", "x": 640, "y": 360}},
        {"type": "MOVE_REL", "delta": {"coord_type": "screen_px", "x": 50, "y": -20}},
        {"type": "CLICK_AT", "coord": {"coord_type": "screen_px", "x": 200, "y": 140}, "times": 1, "interval_ms": 120},
        {"type": "SCROLL", "delta": 320, "verify": {"type": "NONE"}},
        {"type": "TYPE_TEXT", "text": "hello", "method": "osk"},
        {"type": "TOUCH_TAP", "coord": {"coord_type": "grid", "x": 20, "y": 10}, "press_ms": 80},
        {"type": "TOUCH_SWIPE", "from": {"coord_type": "grid", "x": 10, "y": 10}, "to": {"coord_type": "grid", "x": 90, "y": 40}, "duration_ms": 400},
        {"type": "TOUCH_PRESS", "coord": {"coord_type": "grid", "x": 33, "y": 22}, "duration_ms": 700},
    ]


def run_once() -> int:
    passed = 0
    for payload in _samples():
        a1 = action_from_json(payload)
        j1 = action_to_json(a1, as_dict=True)
        a2 = action_from_json(j1)
        j2 = action_to_json(a2, as_dict=True)
        if j1 != j2:
            raise ValueError(f"round-trip mismatch: {json.dumps(payload, ensure_ascii=False)}")
        passed += 1
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Action schema round-trip self-test (no hardware required).")
    parser.add_argument("--n", type=int, default=1, help="Repeat count for stability check")
    args = parser.parse_args(argv)

    loops = max(1, int(args.n))
    passed = 0
    try:
        for _ in range(loops):
            passed = run_once()
    except Exception as exc:
        print(f"FAIL action_roundtrip_error={exc}")
        return 1
    print(f"OK action_roundtrip={passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
