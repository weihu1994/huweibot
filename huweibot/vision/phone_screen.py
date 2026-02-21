from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PhoneScreenROI:
    bbox: tuple[int, int, int, int]
    corners: list[tuple[int, int]]
    confidence: float
    method: str = "cv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])],
            "corners": [[int(x), int(y)] for x, y in self.corners],
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "method": self.method,
        }


def _normalize_bbox(raw_bbox: Any) -> tuple[int, int, int, int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("phone screen bbox must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid phone screen bbox: {raw_bbox}")
    return x1, y1, x2, y2


def _clamp_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(w - 1), int(x1)))
    y1 = max(0, min(int(h - 1), int(y1)))
    x2 = max(0, min(int(w - 1), int(x2)))
    y2 = max(0, min(int(h - 1), int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_first_float(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_first_json_obj(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                if isinstance(obj, dict):
                    return obj
                return None
    return None


def _heuristic_detect_phone_screen(
    frame_bgr: Any,
    *,
    min_area_ratio: float = 0.18,
    aspect_ratio_min: float = 1.55,
    aspect_ratio_max: float = 2.45,
) -> PhoneScreenROI | None:
    if frame_bgr is None or not hasattr(frame_bgr, "shape"):
        return None
    h, w = frame_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(w * h)
    if img_area <= 1:
        return None

    best: PhoneScreenROI | None = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        area_ratio = area / img_area
        if area_ratio < float(min_area_ratio):
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw <= 0 or rh <= 0:
            continue
        ar = max(float(rw) / float(rh), float(rh) / float(rw))
        if ar < float(aspect_ratio_min) or ar > float(aspect_ratio_max):
            continue

        rectangularity = min(1.0, area / max(1.0, float(rw * rh)))
        poly_bonus = 1.0 if len(approx) == 4 else 0.7 if len(approx) <= 6 else 0.5
        aspect_mid = (float(aspect_ratio_min) + float(aspect_ratio_max)) * 0.5
        aspect_span = max(0.001, (float(aspect_ratio_max) - float(aspect_ratio_min)) * 0.5)
        aspect_score = max(0.0, 1.0 - abs(ar - aspect_mid) / aspect_span)
        score = (0.55 * area_ratio) + (0.25 * rectangularity) + (0.15 * poly_bonus) + (0.05 * aspect_score)

        bbox = (int(rx), int(ry), int(rx + rw), int(ry + rh))
        clamped = _clamp_bbox(bbox, w, h)
        if clamped is None:
            continue
        if len(approx) >= 4:
            corners = [(int(p[0][0]), int(p[0][1])) for p in approx[:4]]
        else:
            x1, y1, x2, y2 = clamped
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        roi = PhoneScreenROI(
            bbox=clamped,
            corners=corners,
            confidence=float(max(0.0, min(1.0, score))),
            method="cv",
        )
        if score > best_score:
            best_score = score
            best = roi
    return best


def _vlm_detect_phone_screen(
    frame_bgr: Any,
    *,
    router: Any | None,
    max_side: int = 768,
    jpeg_quality: int = 70,
) -> PhoneScreenROI | None:
    if router is None:
        return None
    provider = None
    if hasattr(router, "get_vlm_provider"):
        provider = router.get_vlm_provider()
    if provider is None:
        LOGGER.warning("phone auto-detect VLM fallback skipped: no VLM provider")
        return None
    capabilities = getattr(provider, "capabilities", None)
    if capabilities is not None and not bool(getattr(capabilities, "supports_image", False)):
        LOGGER.warning("phone auto-detect VLM fallback skipped: provider does not support image")
        return None

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(float(h), float(w)))
    if scale < 1.0:
        resized = cv2.resize(frame_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
    else:
        resized = frame_bgr
    ok, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None

    img_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    prompt = (
        "Detect phone screen region and output strict JSON only: "
        "{\"bbox\":[x1,y1,x2,y2],\"corners\":[[x,y],[x,y],[x,y],[x,y]],\"confidence\":0.0}."
        f" Input image is JPEG base64 ({resized.shape[1]}x{resized.shape[0]}): {img_b64}"
    )
    try:
        raw = provider.generate(prompt, system="Return strict JSON only.")
    except Exception as exc:
        LOGGER.warning("phone auto-detect VLM fallback failed: %s", exc)
        return None

    obj = _extract_first_json_obj(raw if isinstance(raw, str) else str(raw))
    if not isinstance(obj, dict):
        return None
    bbox_raw = obj.get("bbox")
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        return None

    if scale != 1.0:
        bbox_vals = [float(v) / scale for v in bbox_raw]
    else:
        bbox_vals = [float(v) for v in bbox_raw]
    bbox = _clamp_bbox(
        (int(round(bbox_vals[0])), int(round(bbox_vals[1])), int(round(bbox_vals[2])), int(round(bbox_vals[3]))),
        w,
        h,
    )
    if bbox is None:
        return None

    corners_raw = obj.get("corners")
    corners: list[tuple[int, int]] = []
    if isinstance(corners_raw, list):
        for p in corners_raw[:4]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                px = float(p[0]) / scale if scale != 1.0 else float(p[0])
                py = float(p[1]) / scale if scale != 1.0 else float(p[1])
                corners.append((int(round(px)), int(round(py))))
    if len(corners) != 4:
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    conf = _extract_first_float(str(obj.get("confidence", "0.5")))
    return PhoneScreenROI(
        bbox=bbox,
        corners=corners,
        confidence=float(0.5 if conf is None else max(0.0, min(1.0, conf))),
        method="vlm",
    )


def auto_detect_phone_screen_roi(
    frame_bgr: Any,
    *,
    min_area_ratio: float = 0.18,
    aspect_ratio_min: float = 1.55,
    aspect_ratio_max: float = 2.45,
    min_confidence: float = 0.35,
    allow_vlm: bool = False,
    router: Any | None = None,
    vlm_image_max_side: int = 768,
    vlm_jpeg_quality: int = 70,
) -> PhoneScreenROI | None:
    roi = _heuristic_detect_phone_screen(
        frame_bgr,
        min_area_ratio=min_area_ratio,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
    )
    if roi is not None and roi.confidence >= float(min_confidence):
        return roi
    if not allow_vlm:
        return roi
    vlm_roi = _vlm_detect_phone_screen(
        frame_bgr,
        router=router,
        max_side=int(vlm_image_max_side),
        jpeg_quality=int(vlm_jpeg_quality),
    )
    if vlm_roi is not None:
        return vlm_roi
    return roi


def auto_detect_phone_screen(frame_bgr: Any, **kwargs: Any) -> tuple[int, int, int, int] | None:
    roi = auto_detect_phone_screen_roi(frame_bgr, **kwargs)
    if roi is None:
        return None
    return roi.bbox


def draw_phone_screen_overlay(frame_bgr: Any, roi: PhoneScreenROI | None) -> Any:
    if frame_bgr is None:
        return frame_bgr
    canvas = frame_bgr.copy()
    if roi is None:
        return canvas
    x1, y1, x2, y2 = roi.bbox
    color = (0, 220, 0) if roi.method == "cv" else (0, 180, 255)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    for px, py in roi.corners:
        cv2.circle(canvas, (int(px), int(py)), 4, color, -1)
    label = f"{roi.method} conf={roi.confidence:.2f}"
    cv2.putText(canvas, label, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def save_phone_screen(
    path: str,
    bbox: tuple[int, int, int, int],
    *,
    source_w: int | None = None,
    source_h: int | None = None,
    corners: list[tuple[int, int]] | None = None,
    confidence: float | None = None,
    method: str | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "screen_bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        "created_at": time.time(),
        "source_w": None if source_w is None else int(source_w),
        "source_h": None if source_h is None else int(source_h),
        "corners": None if corners is None else [[int(x), int(y)] for x, y in corners],
        "confidence": None if confidence is None else float(confidence),
        "method": method,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_phone_screen_from_roi(
    path: str,
    roi: PhoneScreenROI,
    *,
    source_w: int | None = None,
    source_h: int | None = None,
) -> None:
    save_phone_screen(
        path,
        roi.bbox,
        source_w=source_w,
        source_h=source_h,
        corners=roi.corners,
        confidence=roi.confidence,
        method=roi.method,
    )


def load_phone_screen_roi(path: str) -> PhoneScreenROI:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"phone screen calibration not found: {path}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    bbox = _normalize_bbox(payload.get("screen_bbox", payload.get("bbox")))
    corners_raw = payload.get("corners")
    corners: list[tuple[int, int]] = []
    if isinstance(corners_raw, list):
        for pnt in corners_raw[:4]:
            if isinstance(pnt, (list, tuple)) and len(pnt) >= 2:
                corners.append((int(round(float(pnt[0]))), int(round(float(pnt[1])))))
    if len(corners) != 4:
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    confidence_raw = payload.get("confidence")
    conf = 0.0
    if confidence_raw is not None:
        try:
            conf = float(confidence_raw)
        except (TypeError, ValueError):
            conf = 0.0
    method = str(payload.get("method") or "manual")
    return PhoneScreenROI(
        bbox=bbox,
        corners=corners,
        confidence=max(0.0, min(1.0, conf)),
        method=method,
    )


def load_phone_screen(path: str) -> tuple[int, int, int, int]:
    return load_phone_screen_roi(path).bbox
