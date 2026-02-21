from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

def _model_dump(model: Any, **kwargs: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(**kwargs)
    return model.dict(**kwargs)


def _model_dump_json(model: Any, **kwargs: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)


def _model_fields(model_cls: Any) -> dict[str, Any]:
    if hasattr(model_cls, "model_fields"):
        return model_cls.model_fields
    return model_cls.__fields__


def _model_validate(model_cls: Any, data: Any) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


class VLMTriggerThresholds(BaseModel):
    elements_min: int = 3
    ambiguous_streak: int = 2
    macro_fail_streak: int = 2
    ui_change_threshold: float = 0.15


class ProviderSpec(BaseModel):
    name: str
    type: str = "dummy"
    enabled: bool = True
    model: str = "dummy-model"
    base_url: str | None = None
    api_key_env: str | None = None


class XBotConfig(BaseModel):
    # Runtime mode selection.
    mode: Literal["pc", "phone"] = "pc"

    # Camera on machine A observing machine B.
    camera_id: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30

    # Calibration of B display in A's camera frame.
    calibration_path: str = "config/calibration.json"

    # Phone screen detection/calibration in A's camera frame.
    phone_screen_detection: Literal["manual", "auto"] = "manual"
    phone_screen_calibration_path: str = "config/phone_screen.json"
    phone_grid_w: int = 200
    phone_grid_h: int = 100
    phone_allowed_margin_px: int = 6
    phone_travel_range_mm: tuple[float, float, float, float] = (0.0, 120.0, 0.0, 220.0)

    # UI element extraction.
    ui_mode: str = "local"
    max_elements: int = 128
    min_text_conf: float = 0.4
    min_elem_conf: float = 0.3
    element_merge_iou: float = 0.5
    app_hint_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.12)

    # On-screen keyboard on machine B.
    osk_enabled: bool = False
    keyboard_density_threshold: float = 0.18
    keyboard_roi_hint: tuple[float, float, float, float] = (0.0, 0.55, 1.0, 1.0)
    keyboard_grid_prior: str = "qwerty"
    keyboard_keyword_list_path: str = "assets/ui/keyboard_words.txt"
    keyboard_profile_default: Literal["EN_US", "ZH_CN", "NUMERIC"] = "EN_US"

    # Tracking.
    enable_tracking: bool = True
    track_iou: float = 0.3
    track_text_sim: float = 0.55
    track_center_dist_px: float = 120.0
    track_max_age: int = 8
    stable_id_fallback: bool = True

    # Selector.
    selector_max_candidates: int = 20
    selector_min_conf: float = 0.2
    selector_ambiguous_gap: float = 0.05
    clickability_weight: float = 1.0

    # Macros.
    macros_path: str = "config/macros.yaml"
    allow_hotkey: bool = False

    # Click actuator.
    click_mode: Literal["stub", "same_serial", "separate_serial"] = "stub"
    click_port: str | None = None
    click_baud: int = 115200
    click_press_ms: int = 40

    # Z-axis touch pen and IR distance sensor for phone mode (A-side hardware).
    z_axis_enabled: bool = False
    z_axis_port: str | None = None
    z_axis_baud: int = 115200
    safe_z_mm: float = 3.0
    tap_z_mm: float = 0.5
    tap_press_ms: int = 40
    ir_enabled: bool = False
    ir_port: str | None = None
    ir_baud: int = 115200
    ir_timeout_s: float = 0.5
    ir_i2c_addr: str | None = None
    ir_contact_threshold_mm: float = 2.0
    touch_distance_threshold_mm: float = 2.5

    # Motion / GRBL / homing.
    serial_port: str = "/dev/ttyUSB0"
    serial_baud: int = 115200
    serial_timeout_s: float = 2.0
    travel_range_mm: tuple[float, float, float, float] = (0.0, 220.0, 0.0, 140.0)
    feed_rate_mm_min: float = 3000.0
    homed_flag_path: str = "artifacts/homed.flag"
    enforce_homed: bool = False

    # px->mm mapping.
    mapping_path: str = "config/mapping.json"
    max_move_mm: float = 12.0
    drift_alpha: float = 0.1
    drift_min_px: float = 3.0

    # Providers / router.
    providers: list[ProviderSpec] = Field(
        default_factory=lambda: [
            ProviderSpec(name="dummy_planner", type="dummy", enabled=True, model="dummy-plan"),
            ProviderSpec(name="dummy_vlm", type="dummy", enabled=False, model="dummy-vlm"),
        ]
    )
    default_planner_provider: str = "dummy_planner"
    default_vlm_provider: str = "dummy_vlm"
    provider_capabilities_cache: dict[str, dict[str, Any]] = Field(default_factory=dict)
    router_enable_fallback: bool = True
    router_enable_upgrade: bool = False
    router_enable_downgrade: bool = True
    allow_fallback_to_dummy: bool = False
    provider_timeout_s: float = 30.0
    provider_max_retries: int = 3

    # Token and VLM policy.
    vlm_enabled_default: bool = False
    vlm_trigger_thresholds: VLMTriggerThresholds = Field(default_factory=VLMTriggerThresholds)
    vlm_image_max_side: int = 1280
    vlm_jpeg_quality: int = 80
    vlm_cache_ttl_s: int = 120
    vlm_cooldown_s: int = 10
    vlm_max_per_task: int = 8
    vlm_force_allow_if_elements0_streak: int = 3
    vlm_force_allow_requires_local_boost: bool = True
    debug_reasoning: bool = False
    obs_model_encoding: Literal["plain", "packed"] = "packed"
    obs_key_minify: bool = True
    elements_delta_max: int = 10
    txt_norm_policy: str = "whitelist+normalize"
    txt_trunc_policy: str = "head12~tail6"

    # Output.
    artifacts_dir: str = "artifacts"
    logging_enabled: bool = True

    # Task/scheduler/reminder.
    tasks_db_path: str = "artifacts/tasks.json"
    scheduler_tick_s: float = 1.0
    reminder_enabled: bool = True
    reminder_channel: Literal["stdout", "logfile"] = "stdout"

    # Semantic guard and safety rail.
    allowed_region: tuple[float, float, float, float] | None = None
    blocked_regions: list[tuple[float, float, float, float]] = Field(default_factory=list)
    guard_require_homed: bool = True
    max_type_text_len: int = 200
    block_sensitive_text_patterns: list[str] = Field(
        default_factory=lambda: [
            "password",
            "passwd",
            "otp",
            "验证码",
            "cvv",
            "card",
            "银行卡",
            "信用卡",
        ]
    )
    block_permission_prompt: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _model_dump(self)

    def to_json(self) -> str:
        return _model_dump_json(self, indent=2)


def _parse_env_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return value

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_config(env: dict[str, str] | None = None) -> XBotConfig:
    source = env if env is not None else dict(os.environ)
    overrides: dict[str, Any] = {}

    for field_name in _model_fields(XBotConfig):
        env_key = f"XBOT_{field_name.upper()}"
        if env_key in source:
            overrides[field_name] = _parse_env_value(source[env_key])

    return _model_validate(XBotConfig, overrides)
