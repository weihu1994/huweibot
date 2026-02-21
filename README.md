[![CI](https://github.com/weihu1994/huweibot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/weihu1994/huweibot/actions/workflows/ci.yml)

# huweibot

huweibot is a tool-style agent framework for physical GUI automation.

## Hard System Constraints (Must Hold)
- Two-machine topology: machine A controls machine B.
- A can only "see + act": camera observation of B screen + mechanical mouse movement/click on B.
- A must not inject system APIs into B, must not use remote-control protocols, and must not use keyboard input injection on B.
- Text input must be done via clicking B's on-screen keyboard (OSK) keys.

## Step 3 MVP Environment Assumptions
- B uses a single display.
- B resolution is fixed (e.g. 1920x1080) and display scale is 100%.
- Camera on A and B screen pose are physically fixed; moving either requires recalibration.
- If screen validation fails, runtime must stop and require recalibration.

## Current Scope (Step 1-15)
- Step 1: installable package, CLI skeleton, router dummy provider, selector/tracker/macros baseline.
- Step 2: camera acquisition + preview (`q` exit, `s` save).
- Step 3: manual 4-point homography calibration + rectification + runtime validation.
- Step 4: cursor detection/recovery + heuristic/optional OCR UI element extraction + keyboard ROI + stable ID tracking + inspect overlay.
- Step 5: structured action schema (JSON round-trip, strict validation, discriminator union).
- Step 6: GRBL serial communication + manual relative move + soft limits + homing flag.
- Step 7: px->mm mapping calibration and minimal drift update skeleton.
- Step 8: closed-loop `move_to` and target-based move safety gating.
- Step 9: fixed 1000x1000 grid coordinate fallback (`screen_px <-> grid`) with clamp-safe conversion.
- Step 10: click actuator modes (`stub|same_serial|separate_serial`), loop-level `click_target`, OSK compile/execute (`type_text_osk`), and heuristic-target hover recheck.
- Step 11: verifier primitives (`TEXT_PRESENT/TEXT_CHANGED/ELEMENT_CHANGED/INPUT_VERIFIED`) and loop-level `click_and_verify` / input verify retry.
- Step 12: planner/executor/router integration (`run-step` / `run-task`), strict JSON planning contract, low-token full/delta observation protocol.
- Step 13: runtime validation commands (`osk-demo`, `list-macros`, `run-macro`, `home-check`, `drift-check`, `rule-script`).
- Step 14: per-step artifact logging (`step_XXXX_*`) for observe/plan/resolve/act/verify replay.
- Step 15: semantic safety guard (region/environment/homing/soft-limit/type-text permission rails) with hard blocking.

## Quick Start
```bash
pip install -e .
python3 -m huweibot.main --help
python3 scripts/preview_camera.py --help
python3 scripts/calibrate_screen.py --help
python3 scripts/inspect_elements.py --help
python3 -m huweibot.main doctor --help
```

Smoke command note:
- Prefer `.venv_smoke/bin/python` when running smoke checks; if unavailable, use `python3`.
- For fresh repositories without commits, use `git rev-parse --short HEAD 2>/dev/null || echo "N/A (no commits yet)"`.


## Migration (xbot -> huweibot)
- New package and command: `huweibot`.
- Backward compatibility: `xbot` remains available as a lightweight shim for one transition cycle.
- Deprecated entry still works:
```bash
python3 -m xbot.main --help
```
- Please migrate your imports and command usage to:
```bash
python3 -m huweibot.main --help
```

Pydantic compatibility: the codebase supports both `pydantic v1` and `pydantic v2`.
Recommended Python runtime is 3.12/3.13; Python 3.15 is not supported due to `pydantic-core`/PyO3 build constraints.

## Step 2 Preview
```bash
python3 scripts/preview_camera.py --camera-id 0 --width 1280 --height 720 --fps 30
```
- `q`: quit
- `s`: save image (default `artifacts/raw.png`)

## Step 3 Calibration + Rectified Preview
```bash
python3 scripts/calibrate_screen.py --camera-id 0 --screen-w 1920 --screen-h 1080 --out config/calibration.json
python3 scripts/preview_camera.py --rectify --validate-screen --calib config/calibration.json
```

## Step 4 Inspect Elements
```bash
python3 scripts/inspect_elements.py --calib config/calibration.json --allow-vlm
```
- `q`: quit
- `s`: save `artifacts/screen_elements.png`
- `--json-out artifacts/elements.json` to save structured output

## Step 5 Action Schema
```bash
python3 huweibot/core/actions.py
python3 scripts/click_test.py --schema-self-test
```
- `CLICK_ELEMENT` supports stable_id via `ElementRef(by='id', value='...')`
- `SCROLL` defaults to verifiable rule (`TEXT_CHANGED`) unless explicitly set to `NONE`
- Invalid actions (e.g. DRAG missing `from`/`to`, WAIT out of range, times out of range) raise readable validation errors

## Step 6 GRBL + Manual Move
```bash
python3 scripts/home_zero.py --flag artifacts/homed.flag
python3 scripts/manual_move.py --port /dev/ttyUSB0 --dx-mm 1.0 --dy-mm 0.0 --repeat 3 --delay 0.2
```
- Every session should start from known zero/reference; otherwise mapping is unsafe.
- Soft limit is enforced in `HardwareController.move_mm`; out-of-range move is rejected before sending G-code.
- Dual-machine boundary remains unchanged: hardware is on A and only physically drives B mouse.

## Step 7 px->mm Calibration
Before calibration on machine B:
- Disable mouse acceleration.
- Keep pointer speed fixed.

```bash
python3 scripts/calibrate_px_to_mm.py \
  --port /dev/ttyUSB0 \
  --calib config/calibration.json \
  --out config/mapping.json \
  --step-mm 1.0 --samples 3 --max-retries 8
```
- Script observes B cursor on rectified screen, estimates `mm_per_px_x/y` and `sign_x/sign_y`, and writes `config/mapping.json`.
- A 100px sanity move is included for scale check (rough magnitude, not full closed-loop precision).

## Step 8 Closed-Loop Move
```bash
python3 scripts/move_to.py --port /dev/ttyUSB0 --mapping config/mapping.json --x 960 --y 540 --rectify --calib config/calibration.json
python3 scripts/move_to.py --port /dev/ttyUSB0 --mapping config/mapping.json --target-query "role:button text_contains:ok" --rectify --calib config/calibration.json
```
- `cursor_lost` => move is aborted (no blind movement).
- resolve failure for `--target-id/--target-query` => script exits with candidate summary and does not move.
- Soft limit rejection is surfaced as `reason=soft_limit`.

## Step 9 Grid Coordinates (Fallback Only)
```bash
python3 huweibot/core/coords.py
```
- Elements (`ElementRef` / detected `elements`) are always preferred.
- Elements are preferred; grid is fallback-only: use grid coordinates only when elements are unavailable or in emergency fallback scenarios, and Planner/Executor should prioritize Selector/ElementRef resolution.
- Grid coordinates are emergency fallback only when reliable elements are unavailable.
- Mapping rules are fixed and clamp-safe: `screen_px -> grid(0..1000) -> screen_px` with round-trip error within `0..1px`.

## Step 10 Pure-Mouse Input (OSK)
- OSK must be on machine B and should stay docked in a fixed location (recommended bottom area); do not float/resize during tasks.
- During task execution, do not switch input method or keyboard layout on B.
- `TYPE_TEXT` must be executed by clicking B-screen OSK keys one-by-one; A never injects keyboard APIs into B.
- `click_mode` options:
  - `stub`: no hardware click signal is sent; logs only.
  - `same_serial`: click commands use the same serial channel as platform controller.
  - `separate_serial`: click commands use an independent serial port.

Minimal loop-level usage:
```python
from huweibot.core.loop import XBotLoop
from huweibot.agent.schemas import ElementRef

# loop.type_text_osk("Hello", target_input=ElementRef(by="query", value="role:input"))
```

Step 10 acceptance targets:
- `click_left/right/click_at` validates `times/interval_ms/press_ms` and can run in `stub` mode without hardware.
- `type_text_osk` tries `OPEN_OSK` macro when keyboard is not detected; if keyboard ROI is still missing it fails with clear reason.
- Heuristic targets with no reliable text trigger `hover_check`, producing `clickability_hint` before click.

## Step 11 Verify-First Execution
- Default policy: CLICK / SCROLL / TYPE_TEXT must run with verify unless explicitly `VerifyRule(mode='NONE')`.
- `Verifier` functions are pure observation checks and do not read camera/send serial/sleep.
- `click_and_verify` is enabled in loop layer with retry and timeout polling.
- If caller does not provide a `VerifyRule`, loop auto-fills a conservative default:
  - first `ELEMENT_CHANGED(target=ref)`;
  - fallback `TEXT_CHANGED` over a safe ROI.
- `VerifyRule(mode=NONE)` is allowed but reported as `verify_skipped=True`.
- `type_text_osk` now verifies with `INPUT_VERIFIED`; on failure it retries once (refocus + reopen OSK if needed).

Step 11 acceptance targets:
- `TEXT_PRESENT` and `TEXT_CHANGED` are available as structured verifier methods.
- `ELEMENT_CHANGED` supports `ElementRef(by='query'|'id')`.
- CLICK path supports verify loop with retry.
- TYPE_TEXT(OSK) returns explicit verify details and failure reason for non-echo/password-like scenarios.

## Step 12 Agent Runtime (Planner + Executor + Router)
- Planner receives text-only observation summaries. Planner never receives full-screen images.
- Router/provider contract includes capabilities (`supports_strict_json`, `supports_image`, `max_context_tokens`, `cost_tier`).
- Default planner provider is a local dummy provider; strict JSON parse + repair is applied before schema validation.
- Cloud API providers are optional and disabled by default. Keep `XBOT_PROVIDER=dummy` (default) unless explicitly configured.
- Supported provider env switches:
  - OpenAI: `XBOT_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_MODEL`, `OPENAI_BASE_URL`
  - Anthropic: `XBOT_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`
  - Gemini: `XBOT_PROVIDER=gemini`, `GEMINI_API_KEY`, optional `GEMINI_MODEL`, `GEMINI_BASE_URL`
- If a cloud provider is selected but missing/invalid key, Router raises a clear error. It will only fallback to dummy when `XBOT_ALLOW_FALLBACK_TO_DUMMY=true` (or config `allow_fallback_to_dummy=true`).
- Manual cloud-provider ping/dry-run commands are documented in `docs/PROVIDERS.md`.
- Low-token protocol:
  - `full`: task + constraints + TopK elements.
  - `delta`: only `last_action/last_verify/cursor/ui_change/elements_delta/macro_state`; task/constraints/history are clipped.
- VLM is disabled by default (`allow_vlm=false`); image-model route is placeholder only.
- VLM gate is Router-only: cooldown/budget/force-break checks are centralized in Router, and non-Router extraction paths keep `allow_vlm` blocked unless Router explicitly grants the gate.

Interface consistency quick check:
```bash
python3 scripts/check_interface_consistency.py
python3 -m huweibot.main doctor
```

Run:
```bash
python3 -m huweibot.main run-step --task "..." --port /dev/ttyUSB0 --mapping config/mapping.json --rectify --calib config/calibration.json
python3 -m huweibot.main run-task --task "..." --max-steps 30 --port /dev/ttyUSB0 --mapping config/mapping.json --rectify --calib config/calibration.json
```

## Step 13 Console Entry Points
```bash
python3 -m huweibot.main osk-demo --help
python3 -m huweibot.main list-macros
python3 -m huweibot.main run-macro --name open_osk.path_a --dry-run
python3 -m huweibot.main home-check
python3 -m huweibot.main drift-check --help
python3 -m huweibot.main rule-script --help
```

Behavior constraints:
- Commands requiring GRBL must provide `--port` (GRBL on machine A).
- Commands requiring camera must open camera successfully; otherwise exit with clear error.
- `run-macro --dry-run` does not open camera or trigger hardware.
- `allow_vlm` defaults to false.

## Step 14 Replay Artifacts
Each step writes stable artifacts under `artifacts/`:
- `step_XXXX.json`
- `step_XXXX_screen.png`
- `step_XXXX_elements.json`
- `step_XXXX_resolve.json`
- `step_XXXX_planner_llm_raw.txt`
- `step_XXXX_planner_out.json`
- `step_XXXX_planner_in_full.json` or `step_XXXX_planner_in_delta.json`
- `step_XXXX_planner_obs_packed.json`
- `step_XXXX_obs_digest.json`
- `step_XXXX_macro_expand.json` (macro only)
- `step_XXXX_osk_compile.json` (osk only)
- `step_XXXX_cursor_lost.png` (cursor-lost case)
- `step_XXXX_drift.json` (drift-check case)

These files are intentionally stable for later replay/parsing scripts.

## Step 15 Safety Guard (Hard Block)
Guard checks are enforced before action execution (including macro-expanded sub-steps):
- Resolve-first safety: target `ElementRef` must resolve before region checks.
- Region rails: `blocked_regions` hard block; `allowed_region` allowlist boundary.
- Environment rails: invalid screen validation / not-homed / soft-limit risk block execution.
- TYPE_TEXT rails:
  - only `method='osk'` allowed,
  - max length limit (`max_type_text_len`),
  - sensitive text patterns blocked.
- Permission/UAC rails: admin/install/uninstall/privacy/permission prompts are blocked and logged with `policy_hit`.
- KeyboardInterrupt (`Ctrl+C`) stops run safely and records interruption note.

## PC/PHONE Mode Switch + Task Scheduler (Minimal Framework)
- Runtime mode can be selected via config `mode` (`pc|phone`, default `pc`) or CLI override `--mode`.
- PC mode behavior remains unchanged.
- PHONE mode is physical touch-pen control only (camera observe + mechanical actuation), no API injection/remote-control/keyboard injection.
- PHONE fallback grid is fixed `200x100` within calibrated phone screen bbox.
- PHONE touch actions are hard-bounded to calibrated ROI (with margin shrink); out-of-bounds targets are rejected.

Phone calibration + tap entry:
```bash
python3 scripts/calibrate_phone_screen.py --help
python3 scripts/calibrate_phone_screen.py --camera-id 0 --out config/phone_screen.json
python3 scripts/calibrate_phone_screen.py --camera-id 0 --auto --allow-vlm --preview --save-artifacts
python3 scripts/phone_dry_run.py --help
python3 scripts/phone_dry_run.py --auto-screen --tap 10 10 --tap 210 120 --mock-ir-mm 1.5 --save-artifacts
python3 -m huweibot.main --mode phone phone-tap --x 320 --y 640
python3 -m huweibot.main --mode phone phone-tap --grid --gx 100 --gy 50
```
- If `config/phone_screen.json` is missing, PHONE tap/move is rejected (`phone_screen_not_calibrated`).
- Any PHONE target outside calibrated bbox (with margin shrink) is rejected (`out_of_phone_bounds`) and no hardware action is sent.
- IR distance guard can block tap when distance is above threshold (`touch_distance_threshold_mm`).

Task/interval/reminder minimal commands:
```bash
python3 -m huweibot.main task add --name "Demo" --mode computer --goal "open settings"
python3 -m huweibot.main task add --name "Ping" --mode computer --goal "check status" --every 30
python3 -m huweibot.main task list
python3 -m huweibot.main task disable --id <task_id>
python3 -m huweibot.main task enable --id <task_id>
python3 -m huweibot.main task remove --id <task_id>
python3 -m huweibot.main task run --id <task_id> --dry-run
python3 -m huweibot.main task daemon --dry-run
```
- `scheduler-run` scans due tasks every tick and emits `REMINDER` to stdout/logfile based on config.
- `task daemon` emits trigger/finish reminders to terminal and appends events into `artifacts/notifications.log`.
- `run-task` supports mode/goal flags; phone dry-run can be validated without hardware:
```bash
python3 -m huweibot.main run-task --mode phone --dry-run --goal "Open an app and tap a target area"
```

## Notes
- OCR backend is optional. Without OCR backend, extraction runs in heuristic-only mode and warns.
- Verifier defaults to local observation only and does not trigger VLM in Step 11 (`allow_vlm` is passthrough-only placeholder).
- Planner/Verifier/Executor default to local-only operation; VLM escalation remains gated/off by default.

## huweibot Web Console (MVP+)
```bash
python3 -m huweibot.main web --host 0.0.0.0 --port 8000
```
- If missing dependencies: `pip install -e ".[dev]"`.
- This is an entry-layer console only; it triggers existing commands and keeps core execution behavior unchanged.
