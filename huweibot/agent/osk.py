from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from huweibot.agent.schemas import ElementRef
from huweibot.core.observation import Observation, UIElement
from huweibot.core.selector import normalize_text

if TYPE_CHECKING:
    from huweibot.core.loop import XBotLoop


_SPECIAL_KEYWORDS: dict[str, list[str]] = {
    "space": ["space", "空格"],
    "backspace": ["backspace", "bksp", "delete", "退格"],
    "enter": ["enter", "return", "done", "go", "确认", "完成", "换行"],
    "shift": ["shift", "caps", "⇧", "大写"],
    "lang": ["中英", "中/英", "zh", "en", "ime"],
    "mode_123": ["123", "?123", "12#"],
    "mode_abc": ["abc", "字母"],
}


def _load_keyword_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    words: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        value = normalize_text(line)
        if value:
            words.append(value)
    return words


def _element_texts(element: UIElement) -> list[str]:
    out: list[str] = []
    for value in (element.text, element.label):
        norm = normalize_text(value)
        if norm:
            out.append(norm)
    return out


def _in_keyboard_roi(element: UIElement, keyboard_roi: tuple[float, float, float, float] | None) -> bool:
    if keyboard_roi is None:
        return True
    ex1, ey1, ex2, ey2 = element.bbox
    kx1, ky1, kx2, ky2 = keyboard_roi
    return not (ex2 < kx1 or ex1 > kx2 or ey2 < ky1 or ey1 > ky2)


def _build_query_for_token(token: str, role: str = "key") -> str:
    normalized = normalize_text(token)
    if not normalized:
        normalized = token
    return f"role:{role} text_contains:{normalized}"


@dataclass
class OSKStep:
    kind: str
    ref: ElementRef
    token: str
    char: str | None = None
    verify_tokens: list[str] = field(default_factory=list)


@dataclass
class CompiledOSK:
    ok: bool
    reason: str = "ok"
    text: str = ""
    keyboard_profile: str = "EN_US"
    steps: list[OSKStep] = field(default_factory=list)
    failed_char: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "text": self.text,
            "keyboard_profile": self.keyboard_profile,
            "failed_char": self.failed_char,
            "steps": [
                {
                    "kind": step.kind,
                    "token": step.token,
                    "char": step.char,
                    "ref": {"by": step.ref.by, "value": step.ref.value},
                    "verify_tokens": list(step.verify_tokens),
                }
                for step in self.steps
            ],
        }


class OSKPlanner:
    def __init__(self, keyword_list_path: str = "assets/ui/keyboard_words.txt"):
        self.keyword_list_path = keyword_list_path
        self._file_keywords = set(_load_keyword_file(keyword_list_path))

    def _require_keyword(self, keywords: list[str]) -> str:
        for keyword in keywords:
            norm = normalize_text(keyword)
            if norm in self._file_keywords:
                return keyword
        raise ValueError("special_key_keyword_missing")

    def _resolve_from_elements(
        self,
        obs: Observation,
        token: str,
        *,
        role_hint: tuple[str, ...] = ("key", "button"),
    ) -> ElementRef:
        normalized = normalize_text(token)
        candidates: list[UIElement] = []
        for element in obs.elements:
            if element.role not in role_hint:
                continue
            if not _in_keyboard_roi(element, obs.keyboard_roi):
                continue
            texts = _element_texts(element)
            if not texts:
                continue
            if any(normalized == text or normalized in text for text in texts):
                candidates.append(element)

        candidates.sort(key=lambda item: float(item.confidence), reverse=True)
        if candidates:
            best = candidates[0]
            if best.stable_id:
                return ElementRef(by="id", value=best.stable_id)
            return ElementRef(by="query", value=_build_query_for_token(token, role=best.role))

        return ElementRef(by="query", value=_build_query_for_token(token))

    def _compile_special(self, obs: Observation, key_name: str) -> OSKStep:
        keywords = _SPECIAL_KEYWORDS.get(key_name, [])
        if key_name == "space":
            token = self._require_keyword(_SPECIAL_KEYWORDS["space"])
            ref = self._resolve_from_elements(obs, token)
            return OSKStep(kind="key", ref=ref, token=token, char=" ")

        if not keywords:
            raise ValueError(f"unsupported_special_key:{key_name}")

        token = self._require_keyword(keywords)
        ref = self._resolve_from_elements(obs, token)
        verify_tokens: list[str] = []
        if key_name in {"shift", "mode_123", "mode_abc", "lang"}:
            verify_tokens = keywords[:2]
        return OSKStep(kind="switch", ref=ref, token=token, verify_tokens=verify_tokens)

    def _compile_char(self, obs: Observation, ch: str, shift_state: str) -> list[OSKStep]:
        if not ch:
            raise ValueError("empty_char")
        if ch == " ":
            return [self._compile_special(obs, "space")]
        if ch == "\n":
            return [self._compile_special(obs, "enter")]
        if ch == "\b":
            return [self._compile_special(obs, "backspace")]
        if ord(ch) < 32 and ch not in {"\t"}:
            raise ValueError("unsupported_control_char")

        out: list[OSKStep] = []
        token = ch
        if ch.isalpha() and ch.isupper() and shift_state == "auto":
            out.append(self._compile_special(obs, "shift"))
            token = ch.lower()
        elif (ch.isdigit() or (not ch.isalnum() and not ch.isspace())) and shift_state == "auto":
            # Minimal mode-switch support for symbols/digits.
            out.append(self._compile_special(obs, "mode_123"))

        ref = self._resolve_from_elements(obs, token)
        out.append(OSKStep(kind="key", ref=ref, token=token, char=ch))
        return out

    def compile(
        self,
        text: str,
        obs: Observation,
        keyboard_profile: str = "EN_US",
        shift_state: str = "auto",
    ) -> CompiledOSK:
        if not text:
            return CompiledOSK(ok=False, reason="empty_text", text=text, keyboard_profile=keyboard_profile)
        if not obs.keyboard_mode or obs.keyboard_roi is None:
            return CompiledOSK(
                ok=False,
                reason="keyboard_not_detected",
                text=text,
                keyboard_profile=keyboard_profile,
            )

        steps: list[OSKStep] = []
        try:
            for ch in text:
                char_steps = self._compile_char(obs, ch, shift_state)
                if not char_steps:
                    return CompiledOSK(
                        ok=False,
                        reason="char_not_compiled",
                        text=text,
                        keyboard_profile=keyboard_profile,
                        failed_char=ch,
                    )
                steps.extend(char_steps)
        except ValueError:
            return CompiledOSK(
                ok=False,
                reason="char_not_compiled",
                text=text,
                keyboard_profile=keyboard_profile,
                failed_char=ch,
            )

        return CompiledOSK(
            ok=True,
            reason="ok",
            text=text,
            keyboard_profile=keyboard_profile,
            steps=steps,
        )


class OSKExecutor:
    def _obs_has_token(self, obs: Observation, tokens: list[str]) -> bool:
        if not tokens:
            return True
        norm_tokens = [normalize_text(token) for token in tokens if token]
        if not norm_tokens:
            return True
        for element in obs.elements:
            texts = _element_texts(element)
            if not texts:
                continue
            for token in norm_tokens:
                if any(token in text for text in texts):
                    return True
        return False

    def execute(
        self,
        compiled: CompiledOSK,
        loop: XBotLoop,
        target_input: ElementRef | None = None,
    ) -> dict[str, object]:
        if not compiled.ok:
            return {
                "ok": False,
                "reason": compiled.reason,
                "failed_char": compiled.failed_char,
                "compiled": compiled.to_dict(),
            }

        if target_input is not None:
            focus = loop.click_target(target_input, button="left")
            if not bool(focus.get("ok", False)):
                return {"ok": False, "reason": "focus_failed", "details": focus}

        executed = 0
        for step in compiled.steps:
            last_error: str | None = None
            for _attempt in range(2):
                result = loop.click_target(step.ref, button="left")
                if bool(result.get("ok", False)):
                    last_error = None
                    break
                last_error = str(result.get("reason", "click_failed"))
                loop.observe(force_ui=True)

            if last_error is not None:
                return {
                    "ok": False,
                    "reason": "key_click_failed",
                    "failed_char": step.char,
                    "failed_token": step.token,
                    "steps_done": executed,
                    "steps_total": len(compiled.steps),
                    "error": last_error,
                }

            executed += 1

            if step.kind == "switch":
                after = loop.observe(force_ui=True)
                if not self._obs_has_token(after, step.verify_tokens):
                    return {
                        "ok": False,
                        "reason": "switch_verify_failed",
                        "failed_char": step.char,
                        "failed_token": step.token,
                        "steps_done": executed,
                        "steps_total": len(compiled.steps),
                    }

        return {
            "ok": True,
            "reason": "ok",
            "steps_done": executed,
            "steps_total": len(compiled.steps),
            "typed_text": compiled.text,
        }
