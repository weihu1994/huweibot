from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    bgr: np.ndarray
    timestamp: float
    width: int
    height: int


class Camera:
    """OpenCV camera wrapper for machine A camera observing machine B display."""

    def __init__(self, camera_id: int, width: int, height: int, fps: int):
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        if self.is_open:
            return

        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap or not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"Failed to open camera: camera_id={self.camera_id}")

        self._set_capture_property(cv2.CAP_PROP_FRAME_WIDTH, float(self.width), "CAP_PROP_FRAME_WIDTH")
        self._set_capture_property(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height), "CAP_PROP_FRAME_HEIGHT")
        self._set_capture_property(cv2.CAP_PROP_FPS, float(self.fps), "CAP_PROP_FPS")

    def _set_capture_property(self, prop: int, value: float, name: str) -> None:
        if not self._cap:
            return

        ok = self._cap.set(prop, value)
        if not ok:
            LOGGER.warning("Failed to set camera property %s=%s", name, value)

    def read(self) -> CameraFrame:
        if not self.is_open:
            raise RuntimeError("Camera is not open")

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame")

        h, w = frame.shape[:2]
        return CameraFrame(
            bgr=frame,
            timestamp=time.time(),
            width=int(w),
            height=int(h),
        )

    def read_latest(self, n: int = 3) -> CameraFrame:
        if n < 1:
            n = 1

        latest: CameraFrame | None = None
        for _ in range(n):
            latest = self.read()

        assert latest is not None
        return latest

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @staticmethod
    def frame_export(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        h, w = frame_bgr.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return frame_bgr.copy()

        scale = float(max_side) / float(side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
