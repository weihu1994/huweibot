from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from huweibot.core.geometry import iou
from huweibot.core.observation import UIElement

LOGGER = logging.getLogger(__name__)
_OCR_WARNING_EMITTED = False


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.strip().split())
    return normalized


def _truncate_text(value: str, policy: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""

    # Expected policy format similar to "head12~tail6".
    if "head" in policy and "tail" in policy and "~" in policy:
        try:
            left_part, right_part = policy.split("~", 1)
            head_n = int(left_part.replace("head", ""))
            tail_n = int(right_part.replace("tail", ""))
        except ValueError:
            return text

        if len(text) <= head_n + tail_n:
            return text
        return f"{text[:head_n]}~{text[-tail_n:]}"

    return text


def _safe_conf(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _role_from_geometry(x: int, y: int, w: int, h: int, screen_w: int, screen_h: int, keyboard_mode: bool) -> str:
    if keyboard_mode:
        return "key"

    area_ratio = float(w * h) / float(max(screen_w * screen_h, 1))
    aspect = float(w) / float(max(h, 1))

    if h > 0.07 * screen_h and w > 0.40 * screen_w:
        return "input"
    if area_ratio < 0.001 and 0.7 <= aspect <= 1.3:
        return "icon"
    if 1.4 <= aspect <= 8.0 and h < 0.12 * screen_h:
        return "button"
    if h < 0.05 * screen_h and w > 0.15 * screen_w:
        return "text"
    return "unknown"


def _clickability_from_role(role: str) -> str | None:
    if role in {"button", "input", "key", "toggle", "icon"}:
        return "high"
    if role == "unknown":
        return None
    return "low"


def _bbox_from_rect(x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
    return (float(x), float(y), float(x + w), float(y + h))


def _heuristic_rectangles(frame_bgr: np.ndarray, keyboard_mode: bool) -> list[tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    min_area = max(150.0, 0.00018 * float(w * h))
    max_area = 0.90 * float(w * h)

    rects: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, rw, rh = cv2.boundingRect(contour)
        area = float(rw * rh)

        if area < min_area or area > max_area:
            continue
        if rw < 14 or rh < 10:
            continue

        if keyboard_mode and y < int(0.50 * h):
            continue

        # Confidence heuristic: larger / more rectangular boxes score higher.
        contour_area = max(cv2.contourArea(contour), 1.0)
        fill_ratio = min(contour_area / area, 1.0)
        conf = _safe_conf(0.25 + 0.45 * fill_ratio + 0.30 * min(area / (0.05 * w * h), 1.0))
        rects.append((x, y, rw, rh, conf))

    rects.sort(key=lambda item: item[4], reverse=True)
    return rects


def _extract_ocr_candidates(frame_bgr: np.ndarray, ocr_backend: object | None) -> list[dict[str, Any]]:
    if ocr_backend is None:
        return []

    try:
        if hasattr(ocr_backend, "extract"):
            data = ocr_backend.extract(frame_bgr)
        elif hasattr(ocr_backend, "readtext"):
            data = ocr_backend.readtext(frame_bgr)
        elif callable(ocr_backend):
            data = ocr_backend(frame_bgr)
        else:
            LOGGER.warning("Unsupported OCR backend interface: %s", type(ocr_backend))
            return []
    except Exception as exc:
        LOGGER.warning("OCR backend failed: %s", exc)
        return []

    if not isinstance(data, list):
        return []

    parsed: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            bbox = item.get("bbox")
            text = item.get("text")
            conf = float(item.get("conf", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Common easyocr-like format: [bbox, text, conf]
            bbox = item[0]
            text = item[1]
            conf = float(item[2]) if len(item) >= 3 else 0.0
        else:
            continue

        if bbox is None:
            continue

        bbox_arr = np.array(bbox, dtype=np.float32).reshape(-1, 2)
        if bbox_arr.shape[0] < 4:
            continue

        x1 = float(np.min(bbox_arr[:, 0]))
        y1 = float(np.min(bbox_arr[:, 1]))
        x2 = float(np.max(bbox_arr[:, 0]))
        y2 = float(np.max(bbox_arr[:, 1]))
        if x2 <= x1 or y2 <= y1:
            continue

        parsed.append(
            {
                "bbox": (x1, y1, x2, y2),
                "text": _normalize_text(str(text) if text is not None else ""),
                "conf": _safe_conf(conf),
            }
        )

    return parsed


def merge_ui_elements(
    elements: list[UIElement],
    *,
    iou_threshold: float = 0.5,
    max_elements: int = 128,
) -> list[UIElement]:
    if not elements:
        return []

    sorted_elems = sorted(elements, key=lambda e: float(e.confidence), reverse=True)
    merged: list[UIElement] = []

    for elem in sorted_elems:
        duplicate = False
        for kept in merged:
            if iou(kept.bbox, elem.bbox) < iou_threshold:
                continue

            kept_text = _normalize_text(kept.text or kept.label)
            elem_text = _normalize_text(elem.text or elem.label)

            # Merge only if text is same or either side has no text.
            if kept_text and elem_text and kept_text != elem_text:
                continue

            duplicate = True
            if float(elem.confidence) > float(kept.confidence):
                kept.raw_id = elem.raw_id
                kept.role = elem.role
                kept.text = elem.text
                kept.label = elem.label
                kept.bbox = elem.bbox
                kept.confidence = elem.confidence
                kept.source = elem.source
                kept.clickability_hint = elem.clickability_hint
            break

        if not duplicate:
            merged.append(elem)

        if len(merged) >= max_elements:
            break

    return merged


def _read_keyword_list(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    words = []
    for line in p.read_text(encoding="utf-8").splitlines():
        text = _normalize_text(line)
        if text:
            words.append(text.lower())
    return words


def infer_app_hint(elements: list[UIElement], top_ratio: float = 0.12) -> str | None:
    if not elements:
        return None

    max_y = max(e.bbox[3] for e in elements)
    top_limit = top_ratio * max_y if max_y > 0 else float("inf")

    candidates = [
        e
        for e in elements
        if (e.text or e.label) and e.bbox[1] <= top_limit
    ]
    if not candidates:
        candidates = [e for e in elements if (e.text or e.label)]
    if not candidates:
        return None

    best = max(candidates, key=lambda e: float(e.confidence))
    text = _normalize_text(best.text or best.label)
    return text if text else None


def detect_keyboard_roi(
    elements: list[UIElement],
    screen_w: int,
    screen_h: int,
    *,
    keyword_list_path: str = "assets/ui/keyboard_words.txt",
    density_threshold: float = 0.18,
) -> tuple[float, float, float, float] | None:
    if screen_w <= 0 or screen_h <= 0:
        return None

    keywords = _read_keyword_list(keyword_list_path)

    key_like: list[UIElement] = []
    for elem in elements:
        text = _normalize_text((elem.text or elem.label or "").lower())
        if elem.role == "key":
            key_like.append(elem)
            continue
        if any(word in text for word in keywords):
            key_like.append(elem)

    if key_like:
        x1 = min(e.bbox[0] for e in key_like)
        y1 = min(e.bbox[1] for e in key_like)
        x2 = max(e.bbox[2] for e in key_like)
        y2 = max(e.bbox[3] for e in key_like)

        pad_x = 0.02 * screen_w
        pad_y = 0.03 * screen_h
        return (
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(screen_w), x2 + pad_x),
            min(float(screen_h), y2 + pad_y),
        )

    bottom_limit = 0.55 * float(screen_h)
    small_bottom = [
        e
        for e in elements
        if e.bbox[1] >= bottom_limit and (e.bbox[2] - e.bbox[0]) <= 0.18 * screen_w and (e.bbox[3] - e.bbox[1]) <= 0.14 * screen_h
    ]

    if len(small_bottom) < 12:
        return None

    x1 = min(e.bbox[0] for e in small_bottom)
    y1 = min(e.bbox[1] for e in small_bottom)
    x2 = max(e.bbox[2] for e in small_bottom)
    y2 = max(e.bbox[3] for e in small_bottom)

    density = min(1.0, len(small_bottom) / max(1.0, ((x2 - x1) * (y2 - y1)) / (screen_w * screen_h * 0.1)))
    if density < density_threshold:
        return None

    return (x1, y1, x2, y2)


def compute_screen_hash(frame: object) -> str:
    if frame is None:
        return ""

    if isinstance(frame, np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        resized = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        payload = resized.tobytes()
    elif hasattr(frame, "tobytes"):
        payload = frame.tobytes()
    elif isinstance(frame, bytes):
        payload = frame
    else:
        payload = repr(frame).encode("utf-8")

    return hashlib.sha1(payload).hexdigest()


def _hash_distance_fraction(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 1.0
    if len(a) != len(b):
        return 1.0

    try:
        left = bytes.fromhex(a)
        right = bytes.fromhex(b)
    except ValueError:
        return 1.0

    bits = len(left) * 8
    if bits == 0:
        return 0.0

    diff = 0
    for x, y in zip(left, right):
        diff += (x ^ y).bit_count()
    return float(diff) / float(bits)


def compute_ui_change_score(
    prev_screen_hash: str | None,
    cur_screen_hash: str | None,
    prev_elements: list[UIElement] | None = None,
    cur_elements: list[UIElement] | None = None,
) -> float:
    if not prev_screen_hash and not cur_screen_hash:
        return 0.0

    prev_elems = prev_elements or []
    cur_elems = cur_elements or []

    hash_delta = _hash_distance_fraction(prev_screen_hash, cur_screen_hash)

    prev_count = len(prev_elems)
    cur_count = len(cur_elems)
    count_delta = abs(cur_count - prev_count) / float(max(prev_count, cur_count, 1))

    prev_texts = {_normalize_text(e.text or e.label).lower() for e in prev_elems if (e.text or e.label)}
    cur_texts = {_normalize_text(e.text or e.label).lower() for e in cur_elems if (e.text or e.label)}
    union = len(prev_texts | cur_texts)
    text_delta = 0.0 if union == 0 else 1.0 - (len(prev_texts & cur_texts) / float(union))

    score = 0.55 * hash_delta + 0.25 * count_delta + 0.20 * text_delta
    return float(max(0.0, min(1.0, score)))


@dataclass
class DummyVLMProvider:
    name: str = "dummy_vlm"

    def extract(self, _image_bgr: np.ndarray, _reason: str) -> list[UIElement]:
        # Step 4 skeleton: keep network-free. Real VLM provider wiring is a later step.
        return []


def _vlm_should_trigger(
    local_elements: list[UIElement],
    ui_change_score: float,
    state: dict[str, Any],
) -> tuple[bool, str]:
    now = time.time()

    elements_min = int(state.get("elements_min", 3))
    ambiguous_streak = int(state.get("ambiguous_streak", 0))
    ambiguous_trigger = int(state.get("ambiguous_trigger", 2))
    macro_fail_streak = int(state.get("macro_fail_streak", 0))
    macro_fail_trigger = int(state.get("macro_fail_trigger", 2))
    ui_change_trigger = float(state.get("ui_change_threshold", 0.15))

    if len(local_elements) == 0:
        state["elements0_streak"] = int(state.get("elements0_streak", 0)) + 1
    else:
        state["elements0_streak"] = 0

    reasons: list[str] = []
    if len(local_elements) < elements_min:
        reasons.append("elements_below_min")
    if ambiguous_streak >= ambiguous_trigger:
        reasons.append("ambiguous_streak")
    if macro_fail_streak >= macro_fail_trigger:
        reasons.append("macro_fail_streak")
    if ui_change_score >= ui_change_trigger:
        reasons.append("ui_change_high")

    if not reasons:
        return False, "trigger_not_met"

    cooldown_s = float(state.get("vlm_cooldown_s", 10.0))
    last_vlm_ts = float(state.get("last_vlm_ts", 0.0))
    max_per_task = int(state.get("vlm_max_per_task", 8))
    used = int(state.get("vlm_calls", 0))

    if used >= max_per_task:
        return False, "budget_exhausted"

    force_streak_n = int(state.get("vlm_force_allow_if_elements0_streak", 3))
    local_boost_done = bool(state.get("local_boost_done", False))
    force_allow = local_boost_done and int(state.get("elements0_streak", 0)) >= force_streak_n

    if (now - last_vlm_ts) < cooldown_s and not force_allow:
        return False, "cooldown"

    return True, ",".join(reasons)


def _prepare_vlm_roi(frame_bgr: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    if roi is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    x1 = int(max(0, min(w - 1, round(roi[0]))))
    y1 = int(max(0, min(h - 1, round(roi[1]))))
    x2 = int(max(x1 + 1, min(w, round(roi[2]))))
    y2 = int(max(y1 + 1, min(h, round(roi[3]))))
    return frame_bgr[y1:y2, x1:x2].copy()


def _resize_for_vlm(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image_bgr

    scale = float(max_side) / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def maybe_extract_vlm_elements(
    frame_bgr: np.ndarray,
    local_elements: list[UIElement],
    *,
    allow_vlm: bool,
    vlm_gate_passed: bool | None = None,
    vlm_gate_reason: str | None = None,
    ui_change_score: float,
    state: dict[str, Any] | None = None,
    roi: tuple[float, float, float, float] | None = None,
    artifacts_dir: str = "artifacts",
    vlm_image_max_side: int = 1280,
    vlm_jpeg_quality: int = 80,
) -> tuple[list[UIElement], dict[str, Any]]:
    runtime = state if state is not None else {}

    meta: dict[str, Any] = {
        "called": False,
        "reason": "allow_vlm_false",
    }

    if not allow_vlm:
        return [], meta

    if vlm_gate_passed is False:
        meta["reason"] = "router_gate_denied"
        return [], meta

    should_call, auto_reason = _vlm_should_trigger(local_elements, ui_change_score, runtime)
    if vlm_gate_passed is None:
        if not should_call:
            meta["reason"] = auto_reason
            return [], meta
        reason = auto_reason
    else:
        if not should_call and auto_reason in {"budget_exhausted", "cooldown"}:
            meta["reason"] = auto_reason
            return [], meta
        reason = vlm_gate_reason or "router_gate_allowed"

    provider = DummyVLMProvider()

    roi_img = _prepare_vlm_roi(frame_bgr, roi)
    roi_img = _resize_for_vlm(roi_img, max_side=vlm_image_max_side)

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    vlm_raw_path = artifacts / "vlm_raw.txt"
    vlm_json_path = artifacts / "vlm_elements.json"

    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        roi_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(10, min(95, vlm_jpeg_quality)))],
    )
    if encoded_ok:
        (artifacts / "vlm_request.jpg").write_bytes(encoded.tobytes())

    payload = {
        "ts": time.time(),
        "trigger_reason": reason,
        "local_elements": len(local_elements),
        "ui_change_score": ui_change_score,
        "provider": provider.name,
    }
    vlm_raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    vlm_elements = provider.extract(roi_img, reason)

    def _dump_model(e: UIElement) -> dict[str, Any]:
        if hasattr(e, "model_dump"):
            return e.model_dump()
        return e.dict()

    vlm_json_path.write_text(
        json.dumps([_dump_model(e) for e in vlm_elements], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runtime["last_vlm_ts"] = time.time()
    runtime["vlm_calls"] = int(runtime.get("vlm_calls", 0)) + 1

    meta["called"] = True
    meta["reason"] = reason
    meta["artifacts"] = {
        "raw": str(vlm_raw_path),
        "json": str(vlm_json_path),
    }
    return vlm_elements, meta


def extract_ui_elements(
    frame: object,
    *,
    ocr_backend: object | None = None,
    ui_mode: str = "local",
    max_elements: int = 128,
    min_text_conf: float = 0.4,
    min_elem_conf: float = 0.3,
    element_merge_iou: float = 0.5,
    keyboard_mode: bool = False,
    txt_norm_policy: str = "whitelist+normalize",
    txt_trunc_policy: str = "head12~tail6",
    allow_vlm: bool = False,
    vlm_gate_passed: bool | None = None,
    vlm_gate_reason: str | None = None,
    vlm_state: dict[str, Any] | None = None,
    vlm_roi: tuple[float, float, float, float] | None = None,
    ui_change_score: float = 0.0,
    artifacts_dir: str = "artifacts",
    vlm_image_max_side: int = 1280,
    vlm_jpeg_quality: int = 80,
    return_meta: bool = False,
) -> list[UIElement] | tuple[list[UIElement], dict[str, Any]]:
    global _OCR_WARNING_EMITTED

    del txt_norm_policy  # Reserved for richer policies in later steps.

    if frame is None or not isinstance(frame, np.ndarray):
        warnings.warn("Invalid frame input. Returning empty UI elements.", RuntimeWarning, stacklevel=2)
        return ([], {"reason": "invalid_frame"}) if return_meta else []

    mode = (ui_mode or "local").lower()
    if mode in {"ocr", "ocr_only"} and ocr_backend is None:
        warnings.warn("No OCR backend configured for ocr mode. Returning empty UI element list.", RuntimeWarning, stacklevel=2)
        return ([], {"reason": "missing_ocr_backend"}) if return_meta else []

    h, w = frame.shape[:2]
    raw_candidates: list[UIElement] = []
    raw_id = 1

    heuristic_rects = _heuristic_rectangles(frame, keyboard_mode=keyboard_mode)
    for x, y, rw, rh, conf in heuristic_rects:
        if conf < min_elem_conf:
            continue

        role = _role_from_geometry(x, y, rw, rh, w, h, keyboard_mode=keyboard_mode)
        text = None
        label = None

        raw_candidates.append(
            UIElement(
                raw_id=raw_id,
                role=role,  # type: ignore[arg-type]
                text=text,
                label=label,
                bbox=_bbox_from_rect(x, y, rw, rh),
                confidence=_safe_conf(conf),
                source="heuristic",
                clickability_hint=_clickability_from_role(role),  # type: ignore[arg-type]
            )
        )
        raw_id += 1

    if ocr_backend is None and mode in {"local", "heuristic", "local=ocr+heuristic", "default"} and not _OCR_WARNING_EMITTED:
        warnings.warn(
            "No OCR backend configured. Using heuristic-only element extraction.",
            RuntimeWarning,
            stacklevel=2,
        )
        _OCR_WARNING_EMITTED = True

    ocr_candidates = _extract_ocr_candidates(frame, ocr_backend)
    for item in ocr_candidates:
        text = _truncate_text(str(item.get("text", "")), txt_trunc_policy)
        conf = max(float(item.get("conf", 0.0)), min_text_conf)
        if conf < min_text_conf:
            continue

        bbox = item["bbox"]
        x1, y1, x2, y2 = bbox
        rw = int(max(1.0, x2 - x1))
        rh = int(max(1.0, y2 - y1))

        role = "text"
        if keyboard_mode:
            role = "key"
        elif rw > 0.35 * w and rh > 0.08 * h:
            role = "input"
        elif rw > 0.10 * w and rh < 0.10 * h:
            role = "button"

        raw_candidates.append(
            UIElement(
                raw_id=raw_id,
                role=role,  # type: ignore[arg-type]
                text=text,
                label=text,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=_safe_conf(conf),
                source="ocr",
                clickability_hint=_clickability_from_role(role),  # type: ignore[arg-type]
            )
        )
        raw_id += 1

    # Keyboard fallback grid when keyboard mode is inferred but detection is sparse.
    if keyboard_mode and len(raw_candidates) < 16:
        rows, cols = 4, 10
        y_start = int(h * 0.58)
        y_end = int(h * 0.97)
        x_start = int(w * 0.03)
        x_end = int(w * 0.97)
        key_w = max(6, (x_end - x_start) // cols)
        key_h = max(6, (y_end - y_start) // rows)

        for row in range(rows):
            for col in range(cols):
                x1 = x_start + col * key_w
                y1 = y_start + row * key_h
                x2 = min(x_end, x1 + key_w - 2)
                y2 = min(y_end, y1 + key_h - 2)
                raw_candidates.append(
                    UIElement(
                        raw_id=raw_id,
                        role="key",
                        text=None,
                        label=None,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=0.36,
                        source="heuristic",
                        clickability_hint="high",
                    )
                )
                raw_id += 1

    merged = merge_ui_elements(raw_candidates, iou_threshold=element_merge_iou, max_elements=max_elements)

    vlm_meta = {"called": False, "reason": "disabled"}
    if allow_vlm:
        vlm_elems, vlm_meta = maybe_extract_vlm_elements(
            frame,
            merged,
            allow_vlm=True,
            vlm_gate_passed=vlm_gate_passed,
            vlm_gate_reason=vlm_gate_reason,
            ui_change_score=ui_change_score,
            state=vlm_state,
            roi=vlm_roi,
            artifacts_dir=artifacts_dir,
            vlm_image_max_side=vlm_image_max_side,
            vlm_jpeg_quality=vlm_jpeg_quality,
        )
        merged = merge_ui_elements(merged + vlm_elems, iou_threshold=element_merge_iou, max_elements=max_elements)

    if not return_meta:
        return merged

    meta = {
        "mode": mode,
        "heuristic_candidates": len(heuristic_rects),
        "final_elements": len(merged),
        "vlm": vlm_meta,
    }
    return merged, meta
