from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass
class PxMmMapping:
    mm_per_px_x: float
    mm_per_px_y: float
    sign_x: int = 1
    sign_y: int = 1
    max_move_mm: float = 12.0
    travel_range_mm: tuple[float, float, float, float] = (0.0, 220.0, 0.0, 140.0)
    last_calib_time: float | str = 0.0

    def __post_init__(self) -> None:
        self.mm_per_px_x = float(self.mm_per_px_x)
        self.mm_per_px_y = float(self.mm_per_px_y)
        if self.mm_per_px_x <= 0 or self.mm_per_px_y <= 0:
            raise ValueError("mm_per_px_x/mm_per_px_y must be > 0")
        if int(self.sign_x) not in (-1, 1) or int(self.sign_y) not in (-1, 1):
            raise ValueError("sign_x/sign_y must be +1 or -1")
        self.sign_x = int(self.sign_x)
        self.sign_y = int(self.sign_y)
        self.max_move_mm = float(max(0.1, self.max_move_mm))
        if len(self.travel_range_mm) != 4:
            raise ValueError("travel_range_mm must be (min_x,max_x,min_y,max_y)")
        self.travel_range_mm = (
            float(self.travel_range_mm[0]),
            float(self.travel_range_mm[1]),
            float(self.travel_range_mm[2]),
            float(self.travel_range_mm[3]),
        )

    def px_to_mm(
        self,
        dx_px: float,
        dy_px: float,
        current_pos_mm: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        dx_mm = float(dx_px) * self.mm_per_px_x * float(self.sign_x)
        dy_mm = float(dy_px) * self.mm_per_px_y * float(self.sign_y)

        dx_mm = _clamp(dx_mm, -self.max_move_mm, self.max_move_mm)
        dy_mm = _clamp(dy_mm, -self.max_move_mm, self.max_move_mm)

        min_x, max_x, min_y, max_y = self.travel_range_mm
        if current_pos_mm is not None:
            tx = float(current_pos_mm[0]) + dx_mm
            ty = float(current_pos_mm[1]) + dy_mm
            if tx < min_x or tx > max_x or ty < min_y or ty > max_y:
                raise ValueError("soft_limit from mapping: target out of travel range")
        else:
            # Conservative check without current pose: delta must not exceed axis span.
            if abs(dx_mm) > abs(max_x - min_x) or abs(dy_mm) > abs(max_y - min_y):
                raise ValueError("mapping delta exceeds travel span")

        return dx_mm, dy_mm

    def mm_to_px(self, dx_mm: float, dy_mm: float) -> tuple[float, float]:
        dx_px = (float(dx_mm) / self.mm_per_px_x) * float(self.sign_x)
        dy_px = (float(dy_mm) / self.mm_per_px_y) * float(self.sign_y)
        return dx_px, dy_px

    def to_dict(self) -> dict[str, object]:
        return {
            "mm_per_px_x": self.mm_per_px_x,
            "mm_per_px_y": self.mm_per_px_y,
            "sign_x": self.sign_x,
            "sign_y": self.sign_y,
            "max_move_mm": self.max_move_mm,
            "travel_range_mm": list(self.travel_range_mm),
            "last_calib_time": self.last_calib_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PxMmMapping":
        return cls(
            mm_per_px_x=float(data.get("mm_per_px_x", 0.0)),
            mm_per_px_y=float(data.get("mm_per_px_y", 0.0)),
            sign_x=int(data.get("sign_x", 1)),
            sign_y=int(data.get("sign_y", 1)),
            max_move_mm=float(data.get("max_move_mm", 12.0)),
            travel_range_mm=tuple(data.get("travel_range_mm", (0.0, 220.0, 0.0, 140.0))),  # type: ignore[arg-type]
            last_calib_time=data.get("last_calib_time", 0.0),
        )

    @classmethod
    def load_json(cls, path: str) -> "PxMmMapping":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def save_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if not payload.get("last_calib_time"):
            payload["last_calib_time"] = time.time()
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def px_to_mm(
    dx_px: float,
    dy_px: float,
    mm_per_px_x: float,
    mm_per_px_y: float,
    sign_x: int = 1,
    sign_y: int = 1,
    max_move_mm: float = 12.0,
    travel_range_mm: tuple[float, float, float, float] = (0.0, 220.0, 0.0, 140.0),
    current_pos_mm: tuple[float, float] | None = None,
) -> tuple[float, float]:
    mapping = PxMmMapping(
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
        sign_x=sign_x,
        sign_y=sign_y,
        max_move_mm=max_move_mm,
        travel_range_mm=travel_range_mm,
        last_calib_time=0.0,
    )
    return mapping.px_to_mm(dx_px, dy_px, current_pos_mm=current_pos_mm)


def mm_to_px(dx_mm: float, dy_mm: float, mm_per_px_x: float, mm_per_px_y: float, sign_x: int = 1, sign_y: int = 1) -> tuple[float, float]:
    mapping = PxMmMapping(
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
        sign_x=sign_x,
        sign_y=sign_y,
    )
    return mapping.mm_to_px(dx_mm, dy_mm)


def drift_check_update(
    old_mapping: PxMmMapping,
    observed_dx_px: float,
    dx_mm: float,
    alpha: float = 0.1,
    min_px: float = 3.0,
) -> PxMmMapping:
    if abs(float(observed_dx_px)) < float(min_px):
        return old_mapping

    estimate = abs(float(dx_mm)) / abs(float(observed_dx_px))
    a = _clamp(float(alpha), 0.0, 1.0)
    new_x = (1.0 - a) * old_mapping.mm_per_px_x + a * estimate

    return PxMmMapping(
        mm_per_px_x=new_x,
        mm_per_px_y=old_mapping.mm_per_px_y,
        sign_x=old_mapping.sign_x,
        sign_y=old_mapping.sign_y,
        max_move_mm=old_mapping.max_move_mm,
        travel_range_mm=old_mapping.travel_range_mm,
        last_calib_time=time.time(),
    )
