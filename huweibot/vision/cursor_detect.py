from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CursorDetection:
    cursor_xy: tuple[float, float] | None
    cursor_conf: float
    cursor_type: str
    frame_bgr: np.ndarray | None = None
    timestamp: float | None = None
    attempts: int = 0


def _infer_cursor_type(path: Path) -> str:
    name = path.stem.lower()
    if "ibeam" in name or "text" in name:
        return "ibeam"
    if "hand" in name:
        return "hand"
    if "arrow" in name or "pointer" in name:
        return "arrow"
    return "unknown"


def _load_templates(template_dir: str | Path) -> list[tuple[str, np.ndarray]]:
    root = Path(template_dir)
    if not root.exists():
        LOGGER.warning("Cursor template directory missing: %s", root)
        return []

    templates: list[tuple[str, np.ndarray]] = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image.ndim != 2:
            continue

        templates.append((_infer_cursor_type(path), image))

    if not templates:
        LOGGER.warning("No cursor templates found in %s", root)

    return templates


def _detect_on_gray(
    gray: np.ndarray,
    templates: list[tuple[str, np.ndarray]],
    scales: list[float],
) -> CursorDetection:
    best = CursorDetection(cursor_xy=None, cursor_conf=0.0, cursor_type="unknown")

    for cursor_type, template in templates:
        for scale in scales:
            if scale <= 0:
                continue

            scaled_w = max(1, int(round(template.shape[1] * scale)))
            scaled_h = max(1, int(round(template.shape[0] * scale)))
            if scaled_w >= gray.shape[1] or scaled_h >= gray.shape[0]:
                continue

            if scaled_w < 3 or scaled_h < 3:
                continue

            scaled = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            try:
                result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue

            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            conf = float(max(max_val, 0.0))
            if conf <= best.cursor_conf:
                continue

            center_x = float(max_loc[0] + scaled_w / 2.0)
            center_y = float(max_loc[1] + scaled_h / 2.0)
            best = CursorDetection(
                cursor_xy=(center_x, center_y),
                cursor_conf=conf,
                cursor_type=cursor_type,
            )

    return best


def detect_cursor(
    frame_bgr: np.ndarray | None,
    *,
    template_dir: str = "assets/cursor_templates",
    scales: list[float] | None = None,
) -> CursorDetection:
    if frame_bgr is None:
        return CursorDetection(cursor_xy=None, cursor_conf=0.0, cursor_type="unknown")

    templates = _load_templates(template_dir)
    if not templates:
        return CursorDetection(cursor_xy=None, cursor_conf=0.0, cursor_type="unknown")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    candidate = _detect_on_gray(gray, templates, scales or [0.70, 0.85, 1.0, 1.15, 1.30])
    candidate.frame_bgr = frame_bgr
    return candidate


def detect_cursor_best_of_n(
    camera,
    n: int = 3,
    *,
    template_dir: str = "assets/cursor_templates",
    scales: list[float] | None = None,
    frame_transform=None,
) -> CursorDetection:
    if n < 1:
        n = 1

    best = CursorDetection(cursor_xy=None, cursor_conf=0.0, cursor_type="unknown")

    for idx in range(n):
        frame_pack = camera.read()
        frame = frame_pack.bgr
        if frame_transform is not None:
            frame = frame_transform(frame)

        current = detect_cursor(frame, template_dir=template_dir, scales=scales)
        current.timestamp = frame_pack.timestamp
        current.attempts = idx + 1

        if current.cursor_conf >= best.cursor_conf:
            best = current

    return best


def _enhance_frame_for_cursor(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 40, 120)
    merged = cv2.addWeighted(enhanced, 0.8, edges, 0.2, 0)
    return cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)


def recover_cursor(
    camera,
    *,
    min_confidence: float = 0.55,
    max_failures: int = 8,
    template_dir: str = "assets/cursor_templates",
    frame_transform=None,
) -> CursorDetection:
    failures = 0
    last_result = CursorDetection(cursor_xy=None, cursor_conf=0.0, cursor_type="unknown")

    while failures < max_failures:
        result = detect_cursor_best_of_n(
            camera,
            n=3,
            template_dir=template_dir,
            frame_transform=frame_transform,
        )

        if result.cursor_xy is not None and result.cursor_conf >= min_confidence:
            result.attempts = failures + 1
            return result

        last_result = result

        if result.frame_bgr is not None:
            enhanced = _enhance_frame_for_cursor(result.frame_bgr)
            enhanced_result = detect_cursor(enhanced, template_dir=template_dir)
            if enhanced_result.cursor_xy is not None and enhanced_result.cursor_conf >= min_confidence:
                enhanced_result.frame_bgr = result.frame_bgr
                enhanced_result.timestamp = result.timestamp
                enhanced_result.attempts = failures + 1
                return enhanced_result

            # Optional 0.5x retry and map coordinates back.
            down = cv2.resize(result.frame_bgr, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            down_result = detect_cursor(down, template_dir=template_dir)
            if down_result.cursor_xy is not None and down_result.cursor_conf >= min_confidence:
                x, y = down_result.cursor_xy
                down_result.cursor_xy = (x * 2.0, y * 2.0)
                down_result.frame_bgr = result.frame_bgr
                down_result.timestamp = result.timestamp
                down_result.attempts = failures + 1
                return down_result

        failures += 1

    last_result.attempts = max_failures
    return last_result
