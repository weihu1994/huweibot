from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from huweibot._pydantic_compat import BaseModel, Field

from huweibot.agent.schemas import ElementRef
from huweibot.core.observation import Observation, UIElement
from huweibot.core.selector import NoMatch, normalize_text, select_elements


class VerifyResult(BaseModel):
    ok: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    method: str
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _ResolvedElement:
    element: UIElement | None
    reason: str | None = None


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def _iter_visible_texts(
    obs: Observation,
    roi: tuple[int, int, int, int] | None = None,
) -> list[str]:
    out: list[str] = []
    for element in obs.elements:
        if roi is not None and not _bbox_intersects(element.bbox, (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))):
            continue
        for raw in (element.text, element.label):
            norm = normalize_text(raw)
            if norm:
                out.append(norm)
    return out


def _resolve_ref(
    obs: Observation,
    target_ref: ElementRef,
    selector: Any = None,
) -> _ResolvedElement:
    if target_ref.by == "id":
        for element in obs.elements:
            if element.stable_id == target_ref.value:
                return _ResolvedElement(element=element)
        return _ResolvedElement(element=None, reason="cannot_resolve")

    if target_ref.by != "query":
        return _ResolvedElement(element=None, reason="cannot_resolve")

    try:
        if selector is not None and hasattr(selector, "resolve"):
            element = selector.resolve(target_ref, obs.elements)
        elif selector is not None and callable(selector):
            element = selector(target_ref.value, obs.elements, observation=obs)
        else:
            element = select_elements(obs.elements, target_ref.value, observation=obs, match="best")
    except NoMatch:
        return _ResolvedElement(element=None, reason="cannot_resolve")
    except Exception:
        return _ResolvedElement(element=None, reason="cannot_resolve")

    return _ResolvedElement(element=element)


def verify_text_present(
    obs: Observation,
    text: str,
    roi: tuple[int, int, int, int] | None = None,
    min_match_ratio: float = 0.8,
) -> VerifyResult:
    target = normalize_text(text)
    if not target:
        return VerifyResult(
            ok=False,
            score=0.0,
            method="TEXT_PRESENT",
            details={"reason": "empty_target"},
        )

    candidates = _iter_visible_texts(obs, roi=roi)
    best_ratio = 0.0
    best_match = ""
    for value in candidates:
        if target in value:
            ratio = 1.0
        else:
            ratio = SequenceMatcher(None, target, value).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = value

    return VerifyResult(
        ok=best_ratio >= float(min_match_ratio),
        score=float(max(0.0, min(1.0, best_ratio))),
        method="TEXT_PRESENT",
        details={
            "target": target,
            "best_match": best_match,
            "best_ratio": best_ratio,
            "min_match_ratio": float(min_match_ratio),
            "roi": roi,
        },
    )


def verify_text_changed(
    before: Observation,
    after: Observation,
    roi: tuple[int, int, int, int] | None = None,
    min_delta: int = 6,
) -> VerifyResult:
    before_texts = sorted(_iter_visible_texts(before, roi=roi))
    after_texts = sorted(_iter_visible_texts(after, roi=roi))

    before_sig = "|".join(before_texts)
    after_sig = "|".join(after_texts)
    sim = SequenceMatcher(None, before_sig, after_sig).ratio() if before_sig or after_sig else 1.0

    text_delta = int(round((1.0 - sim) * 100.0))
    ui_delta = int(round(float(max(0.0, min(1.0, after.ui_change_score))) * 100.0))
    score = max(text_delta, ui_delta) / 100.0

    return VerifyResult(
        ok=(text_delta >= int(min_delta) or ui_delta >= int(min_delta)),
        score=float(max(0.0, min(1.0, score))),
        method="TEXT_CHANGED",
        details={
            "text_delta": text_delta,
            "ui_delta": ui_delta,
            "min_delta": int(min_delta),
            "before_count": len(before_texts),
            "after_count": len(after_texts),
            "roi": roi,
        },
    )


def verify_element_changed_ref(
    before: Observation,
    after: Observation,
    target_ref: ElementRef,
    selector: Any,
    **kwargs: Any,
) -> VerifyResult:
    del kwargs

    before_res = _resolve_ref(before, target_ref, selector=selector)
    after_res = _resolve_ref(after, target_ref, selector=selector)
    if before_res.element is None or after_res.element is None:
        return VerifyResult(
            ok=False,
            score=0.0,
            method="ELEMENT_CHANGED",
            details={"reason": "cannot_resolve"},
        )

    before_elem = before_res.element
    after_elem = after_res.element
    before_text = normalize_text(before_elem.text or before_elem.label)
    after_text = normalize_text(after_elem.text or after_elem.label)

    text_changed = before_text != after_text
    center_shift = _center_dist(before_elem.bbox, after_elem.bbox)
    bbox_changed = center_shift >= 4.0
    conf_changed = abs(float(before_elem.confidence) - float(after_elem.confidence)) >= 0.25

    evidences = int(text_changed) + int(bbox_changed) + int(conf_changed)
    score = min(1.0, evidences / 3.0)
    ok = text_changed or bbox_changed or conf_changed

    return VerifyResult(
        ok=ok,
        score=float(score),
        method="ELEMENT_CHANGED",
        details={
            "text_changed": text_changed,
            "bbox_shift_px": center_shift,
            "confidence_changed": conf_changed,
            "target_ref": {"by": target_ref.by, "value": target_ref.value},
        },
    )


def verify_input_effect(
    before: Observation,
    after: Observation,
    target_input: ElementRef | None,
    expected_text: str | None,
    selector: Any,
    **kwargs: Any,
) -> VerifyResult:
    requested_allow_vlm = bool(kwargs.get("allow_vlm", False))
    # Verifier stays local-only by default; VLM escalation is Router-gated.
    allow_vlm_used = False

    evidences: list[str] = []
    score = 0.0
    roi: tuple[int, int, int, int] | None = None

    before_elem = None
    after_elem = None
    if target_input is not None:
        before_res = _resolve_ref(before, target_input, selector=selector)
        after_res = _resolve_ref(after, target_input, selector=selector)
        before_elem = before_res.element
        after_elem = after_res.element

        if before_elem is not None and after_elem is not None:
            before_text = normalize_text(before_elem.text or before_elem.label)
            after_text = normalize_text(after_elem.text or after_elem.label)
            if before_text != after_text:
                score += 0.45
                evidences.append("target_text_changed")
            bx1, by1, bx2, by2 = after_elem.bbox
            pad_x = int(max(8.0, (bx2 - bx1) * 0.15))
            pad_y = int(max(8.0, (by2 - by1) * 0.25))
            roi = (
                max(0, int(round(bx1 - pad_x))),
                max(0, int(round(by1 - pad_y))),
                min(after.screen_w - 1, int(round(bx2 + pad_x))),
                min(after.screen_h - 1, int(round(by2 + pad_y))),
            )
        elif after_elem is not None:
            bx1, by1, bx2, by2 = after_elem.bbox
            roi = (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2)))

    if expected_text:
        text_check = verify_text_present(
            after,
            expected_text,
            roi=roi,
            min_match_ratio=float(kwargs.get("min_match_ratio", 0.75)),
        )
        if text_check.ok:
            score += 0.55
            evidences.append("expected_text_present")

    nearby_texts = _iter_visible_texts(after, roi=roi)
    masked = any("●" in raw or "***" in raw or "••" in raw for raw in nearby_texts)
    if masked:
        score += 0.30
        evidences.append("masked_password_indicator")

    if (after.cursor_type or "").lower() == "ibeam":
        score += 0.20
        evidences.append("cursor_ibeam")

    score = float(max(0.0, min(1.0, score)))
    ok = score >= 0.5
    if not evidences:
        reason = "no_signal"
    elif ok:
        reason = "ok"
    else:
        reason = "weak_signal"

    return VerifyResult(
        ok=ok,
        score=score,
        method="INPUT_VERIFIED",
        details={
            "reason": reason,
            "evidences": evidences,
            "target_input": None if target_input is None else {"by": target_input.by, "value": target_input.value},
            "expected_text_provided": bool(expected_text),
            "allow_vlm_requested": requested_allow_vlm,
            "allow_vlm_used": allow_vlm_used,
            "roi": roi,
        },
    )


def verify_last_action(before: Observation, after: Observation) -> VerifyResult:
    return verify_text_changed(before, after)
