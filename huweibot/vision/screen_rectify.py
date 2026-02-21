from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Calibration:
    screen_w: int
    screen_h: int
    H: np.ndarray
    created_at: float | None = None
    src_points: np.ndarray | None = None
    baseline: dict[str, Any] | None = None


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    if pts.shape != (4, 2):
        raise ValueError(f"expected quad points with shape (4,2), got {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _ensure_matrix_3x3(matrix: Any) -> np.ndarray:
    arr = np.array(matrix, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"calibration matrix must be 3x3, got {arr.shape}")
    return arr


def load_calibration(path: str | Path) -> Calibration:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")

    payload = json.loads(p.read_text(encoding="utf-8"))

    if "screen_w" not in payload or "screen_h" not in payload or "H" not in payload:
        raise ValueError("Calibration JSON must include screen_w, screen_h, H")

    matrix = _ensure_matrix_3x3(payload["H"])

    src_points: np.ndarray | None = None
    if "src_points" in payload:
        src_points = np.array(payload["src_points"], dtype=np.float32)
        if src_points.shape != (4, 2):
            raise ValueError(f"src_points must have shape (4,2), got {src_points.shape}")

    baseline = payload.get("baseline")
    if baseline is not None and not isinstance(baseline, dict):
        raise ValueError("baseline must be a JSON object if provided")

    return Calibration(
        screen_w=int(payload["screen_w"]),
        screen_h=int(payload["screen_h"]),
        H=matrix,
        created_at=float(payload.get("created_at", 0.0)) if payload.get("created_at") is not None else None,
        src_points=src_points,
        baseline=baseline,
    )


def rectify(frame_bgr: np.ndarray, calib: Calibration) -> np.ndarray:
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")

    if calib.H.shape != (3, 3):
        raise ValueError(f"Invalid calibration matrix shape: {calib.H.shape}")

    try:
        rectified = cv2.warpPerspective(
            frame_bgr,
            calib.H.astype(np.float32),
            (int(calib.screen_w), int(calib.screen_h)),
            flags=cv2.INTER_LINEAR,
        )
    except cv2.error as exc:
        raise RuntimeError(f"Failed to warp perspective: {exc}") from exc

    if rectified is None:
        raise RuntimeError("Failed to warp perspective: empty result")

    return rectified


def _detect_screen_corners(frame_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        area = abs(cv2.contourArea(quad))
        if area > best_area:
            best_area = area
            best_quad = quad

    if best_quad is None:
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        best_area = abs(float(cv2.contourArea(largest)))
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        best_quad = box

    frame_h, frame_w = frame_bgr.shape[:2]
    min_area = 0.10 * float(frame_h * frame_w)
    if best_area < min_area:
        return None

    return _order_quad_points(best_quad)


def _dhash(gray: np.ndarray, size: int = 8) -> int:
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    bit = 0
    for row in diff:
        for cell in row:
            if cell:
                value |= 1 << bit
            bit += 1
    return value


def _hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


def build_baseline_signature(screen_bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "edge_density": _edge_density(gray),
        "dhash": f"{_dhash(gray):016x}",
        "shape": [int(screen_bgr.shape[1]), int(screen_bgr.shape[0])],
    }


def validate_calibration(
    frame_bgr: np.ndarray,
    calib: Calibration,
    reprojection_threshold_px: float = 12.0,
) -> tuple[bool, float, dict[str, Any]]:
    debug: dict[str, Any] = {
        "size_ok": False,
        "pose_ok": False,
        "baseline_ok": True,
        "reason": "",
        "suggested_dump": "",
    }

    rectified = rectify(frame_bgr, calib)
    expected_shape = (int(calib.screen_h), int(calib.screen_w))
    got_shape = rectified.shape[:2]

    if got_shape != expected_shape:
        debug["reason"] = f"rectified size mismatch: expected={expected_shape}, got={got_shape}"
        debug["suggested_dump"] = "artifacts/screen_mismatch.png"
        return False, float("inf"), debug

    debug["size_ok"] = True

    dst_corners = np.array(
        [
            [0.0, 0.0],
            [float(calib.screen_w - 1), 0.0],
            [float(calib.screen_w - 1), float(calib.screen_h - 1)],
            [0.0, float(calib.screen_h - 1)],
        ],
        dtype=np.float32,
    )

    try:
        inv_h = np.linalg.inv(calib.H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Calibration matrix is not invertible") from exc

    predicted = cv2.perspectiveTransform(dst_corners.reshape(-1, 1, 2), inv_h.astype(np.float32)).reshape(4, 2)
    predicted = _order_quad_points(predicted)
    detected = _detect_screen_corners(frame_bgr)

    if detected is None:
        debug["reason"] = "failed to detect screen corners for pose validation"
        debug["suggested_dump"] = "artifacts/calib_pose_drift.png"
        debug["predicted_corners"] = predicted.tolist()
        return False, float("inf"), debug

    detected = _order_quad_points(detected)
    per_corner = np.linalg.norm(predicted - detected, axis=1)
    reproj_error = float(np.mean(per_corner))

    debug["predicted_corners"] = predicted.tolist()
    debug["detected_corners"] = detected.tolist()
    debug["reprojection_error_px"] = reproj_error

    pose_ok = reproj_error <= float(reprojection_threshold_px)
    debug["pose_ok"] = pose_ok
    if not pose_ok:
        debug["reason"] = (
            f"pose drift detected: reprojection_error={reproj_error:.2f}px > "
            f"threshold={reprojection_threshold_px:.2f}px"
        )
        debug["suggested_dump"] = "artifacts/calib_pose_drift.png"

    baseline_ok = True
    baseline = calib.baseline or {}
    if baseline:
        gray_rectified = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

        baseline_edge = baseline.get("edge_density")
        if baseline_edge is not None:
            cur_edge = _edge_density(gray_rectified)
            edge_delta = abs(float(cur_edge) - float(baseline_edge))
            debug["edge_density"] = cur_edge
            debug["edge_density_delta"] = edge_delta
            if edge_delta > 0.12:
                baseline_ok = False
                debug["reason"] = (
                    f"screen content geometry drift: edge_density_delta={edge_delta:.3f} > 0.120"
                )
                debug["suggested_dump"] = "artifacts/screen_mismatch.png"

        baseline_hash = baseline.get("dhash")
        if baseline_hash is not None:
            try:
                baseline_int = int(str(baseline_hash), 16)
            except ValueError:
                baseline_int = None

            if baseline_int is not None:
                cur_hash_int = _dhash(gray_rectified)
                hash_dist = _hamming_distance(cur_hash_int, baseline_int)
                debug["dhash_distance"] = hash_dist
                if hash_dist > 20:
                    baseline_ok = False
                    debug["reason"] = (
                        f"screen signature mismatch: dhash_distance={hash_dist} > 20"
                    )
                    debug["suggested_dump"] = "artifacts/screen_mismatch.png"

    debug["baseline_ok"] = baseline_ok

    ok = debug["size_ok"] and debug["pose_ok"] and baseline_ok
    if ok and not debug["reason"]:
        debug["reason"] = "ok"

    if not math.isfinite(reproj_error):
        reproj_error = 1e9

    return ok, reproj_error, debug
