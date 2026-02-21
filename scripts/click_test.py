#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huweibot.agent.schemas import ElementRef, VerifyRule
from huweibot._pydantic_compat import model_to_dict, model_to_json, parse_obj
from huweibot.core.actions import (
    Action,
    ClickAtAction,
    ClickElementAction,
    Coord,
    DragAction,
    ScrollAction,
    TypeTextAction,
    action_from_json,
    action_to_json,
)
from huweibot.agent.schemas import NextAction, NextStep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step5 action schema self-test.")
    parser.add_argument("--schema-self-test", action="store_true", help="Run Action JSON round-trip test")
    parser.add_argument("--schema-smoke", action="store_true", help="Run NextStep/NextAction/VerifyRule schema smoke test")
    return parser


def _roundtrip(action: Action) -> bool:
    blob = action_to_json(action)
    again = action_from_json(blob)
    return action_to_json(action, as_dict=True) == action_to_json(again, as_dict=True)


def run_schema_self_test() -> int:
    cases: list[Action] = [
        ClickAtAction(coord=Coord(coord_type="screen_px", x=100, y=120), times=1),
        ClickElementAction(target=ElementRef(by="id", value="stable_btn_ok"), times=2),
        ScrollAction(delta=600),
        ScrollAction(delta=-320, verify=VerifyRule(mode="NONE")),
        TypeTextAction(text="hello", method="osk"),
        DragAction(**{"from": {"coord_type": "screen_px", "x": 30, "y": 50}, "to": {"coord_type": "screen_px", "x": 80, "y": 100}}),
    ]

    for idx, action in enumerate(cases, start=1):
        if not _roundtrip(action):
            print(f"FAIL: round-trip mismatch at case#{idx}")
            return 1

    invalid = [
        {"type": "WAIT", "duration_ms": 999999},
        {"type": "DRAG", "from": {"coord_type": "screen_px", "x": 1, "y": 1}},
    ]
    for payload in invalid:
        try:
            action_from_json(payload)
        except Exception:
            continue
        print(f"FAIL: invalid payload accepted: {json.dumps(payload, ensure_ascii=False)}")
        return 1

    print("OK")
    return 0


def run_schema_smoke_test() -> int:
    payloads = [
        NextStep(
            action=NextAction(type="WAIT", payload={"duration_ms": 300}),
            verify=VerifyRule(type="NONE"),
        ),
        NextStep(
            action=NextAction(type="CLICK", target=ElementRef(by="query", query="role:button text:ok")),
            verify=VerifyRule(type="TEXT_PRESENT", text="ok"),
            repeat=2,
        ),
        NextStep(
            action=NextAction(type="SCROLL", payload={"delta": 320}),
            verify=VerifyRule(type="TEXT_CHANGED", text="scroll"),
            action_ttl_ms=1200,
        ),
    ]

    for idx, obj in enumerate(payloads, start=1):
        as_dict = model_to_dict(obj)
        as_json = model_to_json(obj)
        obj_from_dict = parse_obj(NextStep, as_dict)
        obj_from_json = parse_obj(NextStep, json.loads(as_json))
        if model_to_dict(obj_from_dict) != model_to_dict(obj_from_json):
            print(f"FAIL: schema round-trip mismatch at case#{idx}")
            return 1

    print("OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.schema_self_test:
        return run_schema_self_test()
    if args.schema_smoke:
        return run_schema_smoke_test()

    print("click_test placeholder: use --schema-self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
