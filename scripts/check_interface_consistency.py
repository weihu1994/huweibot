#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huweibot.agent.router import Router
from huweibot.config import XBotConfig
from huweibot.core.observation import Observation, UIElement


def _fake_observation() -> Observation:
    elements = [
        UIElement(
            stable_id="btn_ok",
            raw_id=1,
            role="button",
            text="  OK  ",
            label="OK",
            bbox=(100.0, 200.0, 220.0, 260.0),
            confidence=0.92,
            source="ocr",
        ),
        UIElement(
            stable_id="input_name",
            raw_id=2,
            role="input",
            text="Name Field",
            label="Name",
            bbox=(260.0, 200.0, 720.0, 280.0),
            confidence=0.87,
            source="heuristic",
        ),
        UIElement(
            stable_id="txt_title",
            raw_id=3,
            role="text",
            text="VeryLongTitle_ABCDEFGHIJKLMN",
            label="VeryLongTitle_ABCDEFGHIJKLMN",
            bbox=(80.0, 40.0, 760.0, 120.0),
            confidence=0.85,
            source="ocr",
        ),
        UIElement(
            stable_id="icon_settings",
            raw_id=4,
            role="icon",
            text=None,
            label="Settings",
            bbox=(820.0, 30.0, 880.0, 90.0),
            confidence=0.72,
            source="heuristic",
        ),
    ]
    return Observation(
        timestamp=time.time(),
        screen_w=1920,
        screen_h=1080,
        cursor_xy=(360, 240),
        cursor_conf=0.91,
        cursor_type="arrow",
        elements=elements,
        app_hint="Settings App",
        keyboard_mode=False,
        keyboard_roi=None,
        screen_hash="fakehash",
        ui_change_score=0.18,
    )


def check_packed_plain_consistency() -> tuple[bool, str]:
    obs = _fake_observation()
    cfg_plain = XBotConfig(obs_model_encoding="plain", obs_key_minify=False, elements_delta_max=3)
    cfg_packed = XBotConfig(obs_model_encoding="packed", obs_key_minify=True, elements_delta_max=3)
    router_plain = Router(cfg_plain)
    router_packed = Router(cfg_packed)

    topk_plain = router_plain._elements_topk(obs, 3)
    topk_packed = router_packed._elements_topk(obs, 3)
    if [e.stable_id for e in topk_plain] != [e.stable_id for e in topk_packed]:
        return False, "TopK mismatch between plain and packed"

    plain_obs = router_plain.build_planner_observation(obs, mode="full", elements=topk_plain)
    packed_obs = router_packed.build_planner_observation(obs, mode="full", elements=topk_packed)
    plain_elements = plain_obs.get("elements_topk", [])
    packed_elements = packed_obs.get("e", [])
    if len(plain_elements) != len(packed_elements):
        return False, "Element count mismatch between plain and packed"
    for i, pe in enumerate(plain_elements):
        pt = pe.get("text", "")
        tt = packed_elements[i][2] if i < len(packed_elements) else ""
        if pt != tt:
            return False, f"text normalize/truncate mismatch at index {i}"
    return True, "ok"


def check_delta_prune(artifacts_dir: str) -> tuple[bool, str]:
    cfg = XBotConfig(obs_model_encoding="packed", obs_key_minify=True)
    router = Router(cfg)
    obs = _fake_observation()
    payload_built, _packed, _digest, _clipped, _reason = router.build_planner_payload(
        task="delta_check",
        obs=obs,
        memory={"last_action": {"type": "WAIT"}, "last_verify": {"ok": True}, "macro_state": {"x": 1}},
        obs_mode="delta",
    )
    if "c" not in payload_built or "u" not in payload_built:
        return False, "delta payload missing required short keys c/u"
    if not payload_built:
        return False, "delta payload unexpectedly empty"
    if "task" in payload_built or "constraints" in payload_built:
        return False, "delta payload leaked task/constraints"

    payload = {
        "task": "should_be_removed",
        "constraints": ["should_be_removed"],
        "history": "x" * 600,
        "la": {"type": "CLICK"},
        "lv": {"ok": False},
        "c": [10, 20],
        "u": 0.2,
        "ed": [],
        "ms": {"long_text": "y" * 800},
    }
    pruned, removed_keys, removed_sizes = router.prune_delta_payload(payload)
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "interface_delta_prune_check.json").write_text(
        json.dumps(
            {
                "removed_keys": removed_keys,
                "removed_sizes": removed_sizes,
                "pruned_keys": sorted(pruned.keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not removed_keys:
        return False, "delta prune did not remove any banned keys"
    for forbidden in ("task", "constraints"):
        if forbidden in pruned:
            return False, f"delta prune leaked forbidden key {forbidden}"
    return True, "ok"


def check_vlm_gate() -> tuple[bool, str]:
    cfg = XBotConfig(vlm_cooldown_s=10, vlm_max_per_task=2, vlm_force_allow_if_elements0_streak=3)
    router = Router(cfg)
    now = time.time()
    base_state = {
        "elements0_streak": 0,
        "ambiguous_streak": 0,
        "no_match_streak": 0,
        "macro_fail_streak": 0,
        "vlm_calls": 0,
        "last_vlm_time": now - 1.0,
        "local_boost_done": False,
        "local_boost_executed": False,
    }

    ok1, reason1 = router.should_call_vlm(
        state=dict(base_state),
        triggers=["elements_below_min"],
        now_ts=now,
        elements_count=1,
    )
    if ok1 or reason1 != "cooldown":
        return False, f"expected cooldown deny, got allow={ok1}, reason={reason1}"

    budget_state = dict(base_state)
    budget_state["last_vlm_time"] = now - 30.0
    budget_state["vlm_calls"] = int(cfg.vlm_max_per_task)
    ok2, reason2 = router.should_call_vlm(
        state=budget_state,
        triggers=["selector_streak"],
        now_ts=now,
        elements_count=1,
    )
    if ok2 or reason2 != "vlm_budget_exceeded":
        return False, f"expected budget deny, got allow={ok2}, reason={reason2}"

    force_state = dict(base_state)
    force_state["elements0_streak"] = int(cfg.vlm_force_allow_if_elements0_streak)
    force_state["local_boost_done"] = True
    ok3, reason3 = router.should_call_vlm(
        state=force_state,
        triggers=["elements_below_min"],
        now_ts=now,
        elements_count=0,
    )
    if not ok3 or reason3 != "force_break_cooldown":
        return False, f"expected force break allow, got allow={ok3}, reason={reason3}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="huweibot interface consistency self-check (no hardware, no VLM call).")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to write check artifacts.")
    args = parser.parse_args()

    checks = [
        ("packed/minified", check_packed_plain_consistency),
        ("delta prune lock", lambda: check_delta_prune(args.artifacts_dir)),
        ("vlm cooldown/budget", check_vlm_gate),
    ]
    ok_all = True
    for name, fn in checks:
        ok, reason = fn()
        print(f"{'PASS' if ok else 'FAIL'} {name}: {reason}")
        ok_all = ok_all and ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
