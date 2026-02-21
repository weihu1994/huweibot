from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from huweibot.core.geometry import center_distance, iou
from huweibot.core.observation import UIElement


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(normalized.lower().split())


def _text_similarity(a: UIElement, b: UIElement) -> float:
    left = _normalize_text(a.text) or _normalize_text(a.label)
    right = _normalize_text(b.text) or _normalize_text(b.label)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _fallback_stable_id(element: UIElement) -> str:
    payload = f"{element.role}|{element.bbox}|{_normalize_text(element.text)}|{_normalize_text(element.label)}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"fb_{digest}"


@dataclass
class TrackerConfig:
    track_iou: float = 0.3
    track_text_sim: float = 0.55
    track_center_dist_px: float = 120.0
    track_max_age: int = 8
    stable_id_fallback: bool = True


class StableTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._counter = 1
        self._ages: dict[str, int] = {}

    def _new_stable_id(self, element: UIElement) -> str:
        if self.config.stable_id_fallback:
            return _fallback_stable_id(element)

        value = f"sid_{self._counter:06d}"
        self._counter += 1
        return value

    def _cleanup_aged_tracks(self) -> None:
        expired = [sid for sid, age in self._ages.items() if age > self.config.track_max_age]
        for sid in expired:
            self._ages.pop(sid, None)

    def track(self, prev_elements: list[UIElement], cur_elements: list[UIElement]) -> list[UIElement]:
        assigned_prev: set[int] = set()
        matched_stable_ids: set[str] = set()

        for current in cur_elements:
            best_idx: int | None = None
            best_score = -1.0

            for idx, previous in enumerate(prev_elements):
                if idx in assigned_prev:
                    continue

                geom_iou = iou(previous.bbox, current.bbox)
                text_sim = _text_similarity(previous, current)
                dist = center_distance(previous.bbox, current.bbox)
                dist_score = max(0.0, 1.0 - (dist / max(self.config.track_center_dist_px, 1.0)))

                if (
                    geom_iou < self.config.track_iou
                    and text_sim < self.config.track_text_sim
                    and dist > self.config.track_center_dist_px
                ):
                    continue

                score = 0.55 * geom_iou + 0.25 * text_sim + 0.20 * dist_score
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                previous = prev_elements[best_idx]
                sid = previous.stable_id or _fallback_stable_id(previous)
                current.stable_id = sid
                assigned_prev.add(best_idx)
                matched_stable_ids.add(sid)
                self._ages[sid] = 0
            else:
                sid = self._new_stable_id(current)
                current.stable_id = sid
                matched_stable_ids.add(sid)
                self._ages[sid] = 0

        for sid in list(self._ages):
            if sid not in matched_stable_ids:
                self._ages[sid] += 1

        self._cleanup_aged_tracks()
        return cur_elements


def track(prev_elements: list[UIElement], cur_elements: list[UIElement], config: object | None = None) -> list[UIElement]:
    if config is None:
        tracker = StableTracker()
        return tracker.track(prev_elements, cur_elements)

    tracker_cfg = TrackerConfig(
        track_iou=float(getattr(config, "track_iou", 0.3)),
        track_text_sim=float(getattr(config, "track_text_sim", 0.55)),
        track_center_dist_px=float(getattr(config, "track_center_dist_px", 120.0)),
        track_max_age=int(getattr(config, "track_max_age", 8)),
        stable_id_fallback=bool(getattr(config, "stable_id_fallback", True)),
    )
    tracker = StableTracker(tracker_cfg)
    return tracker.track(prev_elements, cur_elements)
