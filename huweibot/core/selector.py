from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from huweibot.core.geometry import bbox_area, bbox_center, center_distance, intersects, intersection_area
from huweibot.core.observation import Observation, UIElement


class SelectorError(Exception):
    pass


class NoMatch(SelectorError):
    pass


class AmbiguousMatch(SelectorError):
    def __init__(self, message: str, candidates: list[UIElement] | None = None):
        super().__init__(message)
        self.candidates = candidates or []


@dataclass
class SelectorQuery:
    role: str | None = None
    text: str | None = None
    text_contains: str | None = None
    text_contains_any: list[str] | None = None
    label: str | None = None
    label_contains: str | None = None
    label_contains_any: list[str] | None = None
    source: str | None = None
    min_conf: float | None = None
    in_roi: tuple[float, float, float, float] | None = None
    near: tuple[float, float, float] | None = None
    on_screen: bool | None = None
    clickability_hint: Literal["high", "low"] | None = None


_ALLOWED_KEYS = {
    "role",
    "text",
    "text_contains",
    "text_contains_any",
    "label",
    "label_contains",
    "label_contains_any",
    "source",
    "min_conf",
    "in_roi",
    "near",
    "on_screen",
    "clickability_hint",
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    lowered = normalized.lower()
    return "".join(lowered.split())


def parse_selector(query: str) -> SelectorQuery:
    if not query or not query.strip():
        return SelectorQuery()

    parsed = SelectorQuery()
    tokens = query.strip().split()

    for token in tokens:
        if ":" not in token:
            raise SelectorError(f"invalid token: {token}")

        key, raw_value = token.split(":", 1)
        key = key.strip().lower()
        value = raw_value.strip()

        if key not in _ALLOWED_KEYS:
            raise SelectorError(f"unsupported selector key: {key}")

        if key == "role":
            parsed.role = normalize_text(value)
        elif key == "text":
            parsed.text = normalize_text(value)
        elif key == "text_contains":
            parsed.text_contains = normalize_text(value)
        elif key == "text_contains_any":
            parsed.text_contains_any = [normalize_text(v) for v in value.split(",") if v.strip()]
        elif key == "label":
            parsed.label = normalize_text(value)
        elif key == "label_contains":
            parsed.label_contains = normalize_text(value)
        elif key == "label_contains_any":
            parsed.label_contains_any = [normalize_text(v) for v in value.split(",") if v.strip()]
        elif key == "source":
            parsed.source = normalize_text(value)
        elif key == "min_conf":
            parsed.min_conf = float(value)
        elif key == "in_roi":
            parts = [float(v) for v in value.split(",")]
            if len(parts) != 4:
                raise SelectorError("in_roi expects x1,y1,x2,y2")
            parsed.in_roi = (parts[0], parts[1], parts[2], parts[3])
        elif key == "near":
            parts = [float(v) for v in value.split(",")]
            if len(parts) != 3:
                raise SelectorError("near expects x,y,dist")
            parsed.near = (parts[0], parts[1], parts[2])
        elif key == "on_screen":
            lowered = value.lower()
            if lowered not in {"true", "false"}:
                raise SelectorError("on_screen expects true|false")
            parsed.on_screen = lowered == "true"
        elif key == "clickability_hint":
            lowered = value.lower()
            if lowered not in {"high", "low"}:
                raise SelectorError("clickability_hint expects high|low")
            parsed.clickability_hint = lowered

    return parsed


def _text_for_matching(element: UIElement, prefer_text: bool = True) -> str:
    text = normalize_text(element.text)
    label = normalize_text(element.label)
    if prefer_text:
        return text or label
    return label or text


def _query_anchor_text(query: SelectorQuery) -> str:
    # text fields take precedence over label fields.
    if query.text:
        return query.text
    if query.text_contains:
        return query.text_contains
    if query.text_contains_any:
        return query.text_contains_any[0]
    if query.label:
        return query.label
    if query.label_contains:
        return query.label_contains
    if query.label_contains_any:
        return query.label_contains_any[0]
    return ""


def _text_similarity(query: SelectorQuery, element: UIElement) -> float:
    anchor = _query_anchor_text(query)
    if not anchor:
        return 0.5

    candidate = _text_for_matching(element, prefer_text=True)
    if not candidate:
        return 0.0

    return SequenceMatcher(None, anchor, candidate).ratio()


def _area_penalty_score(element: UIElement, obs: Observation | None) -> float:
    elem_area = bbox_area(element.bbox)
    if elem_area <= 0:
        return 0.0

    if obs is None:
        screen_area = 1920.0 * 1080.0
    else:
        screen_area = max(float(obs.screen_w * obs.screen_h), 1.0)

    ratio = min(elem_area / screen_area, 1.0)
    return 1.0 - ratio


def _clickability_bonus(element: UIElement) -> float:
    if element.clickability_hint == "high":
        return 1.0
    if element.clickability_hint == "low":
        return 0.0
    return 0.5


def _bbox_overlap_ratio_with_screen(element: UIElement, obs: Observation | None) -> float:
    if obs is None:
        return 1.0

    screen_bbox = (0.0, 0.0, float(obs.screen_w), float(obs.screen_h))
    inter = intersection_area(element.bbox, screen_bbox)
    area = bbox_area(element.bbox)
    if area <= 0:
        return 0.0
    return inter / area


def _passes_query_filters(
    element: UIElement,
    query: SelectorQuery,
    obs: Observation | None,
    selector_min_conf: float,
) -> bool:
    if element.confidence < max(selector_min_conf, query.min_conf or 0.0):
        return False

    if query.role and normalize_text(element.role) != query.role:
        return False

    if query.source and normalize_text(element.source) != query.source:
        return False

    if query.clickability_hint and element.clickability_hint != query.clickability_hint:
        return False

    elem_text = normalize_text(element.text)
    elem_label = normalize_text(element.label)

    if query.text and elem_text != query.text:
        return False
    if query.text_contains and query.text_contains not in elem_text:
        return False
    if query.text_contains_any and not any(part in elem_text for part in query.text_contains_any):
        return False

    if query.label and elem_label != query.label:
        return False
    if query.label_contains and query.label_contains not in elem_label:
        return False
    if query.label_contains_any and not any(part in elem_label for part in query.label_contains_any):
        return False

    if query.in_roi and not intersects(element.bbox, query.in_roi):
        return False

    if query.near:
        x, y, max_dist = query.near
        cx, cy = bbox_center(element.bbox)
        dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        if dist > max_dist:
            return False

    if query.on_screen is not None:
        overlap = _bbox_overlap_ratio_with_screen(element, obs)
        on_screen_now = overlap >= 0.2
        if query.on_screen != on_screen_now:
            return False

    return True


def _resolve_selector_config(config: object | None) -> tuple[float, int, float, float]:
    if config is None:
        return 0.2, 20, 0.05, 1.0

    selector_min_conf = float(getattr(config, "selector_min_conf", 0.2))
    selector_max_candidates = int(getattr(config, "selector_max_candidates", 20))
    selector_ambiguous_gap = float(getattr(config, "selector_ambiguous_gap", 0.05))
    clickability_weight = float(getattr(config, "clickability_weight", 1.0))
    return selector_min_conf, selector_max_candidates, selector_ambiguous_gap, clickability_weight


def select_elements(
    elements: list[UIElement],
    query: str,
    *,
    observation: Observation | None = None,
    config: object | None = None,
    match: Literal["best", "all"] = "best",
    limit: int | None = None,
) -> UIElement | list[UIElement]:
    parsed = parse_selector(query)
    selector_min_conf, selector_max_candidates, ambiguous_gap, clickability_weight = _resolve_selector_config(config)

    filtered: list[tuple[float, UIElement]] = []
    for element in elements:
        if not _passes_query_filters(element, parsed, observation, selector_min_conf):
            continue

        score = (
            0.50 * float(element.confidence)
            + 0.30 * _text_similarity(parsed, element)
            + 0.10 * _area_penalty_score(element, observation)
            + 0.10 * _clickability_bonus(element) * clickability_weight
        )
        filtered.append((score, element))

    filtered.sort(key=lambda item: item[0], reverse=True)

    if not filtered:
        raise NoMatch(f"no candidate for query: {query}")

    cap = limit or selector_max_candidates
    ranked = [item[1] for item in filtered[:cap]]

    if match == "all":
        return ranked

    best = ranked[0]
    if len(filtered) > 1:
        top1 = filtered[0][0]
        top2 = filtered[1][0]
        if (top1 - top2) < ambiguous_gap:
            raise AmbiguousMatch(
                f"ambiguous best match: gap={top1 - top2:.4f} < {ambiguous_gap}",
                candidates=[filtered[0][1], filtered[1][1]],
            )

    return best


def _demo() -> None:
    sample = [
        UIElement(
            raw_id=1,
            role="button",
            text="Settings",
            label="Settings",
            bbox=(100, 100, 260, 150),
            confidence=0.92,
            source="heuristic",
            clickability_hint="high",
        ),
        UIElement(
            raw_id=2,
            role="button",
            text="Search",
            label="Search",
            bbox=(300, 100, 430, 150),
            confidence=0.88,
            source="ocr",
            clickability_hint="high",
        ),
    ]
    obs = Observation(screen_w=1920, screen_h=1080, elements=sample)
    match = select_elements(sample, "role:button text_contains:setting on_screen:true", observation=obs)
    print(match.to_json())


if __name__ == "__main__":
    _demo()
