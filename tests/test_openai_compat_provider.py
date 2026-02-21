from __future__ import annotations

import io
import json
from urllib import error as urlerror

import pytest

from huweibot.agent.router import (
    ChatMessage,
    OpenAICompatibleProvider,
    ProviderAuthError,
    ProviderRequest,
    Router,
    _normalize_base_url,
    _redact_sensitive,
)
from huweibot.config import ProviderSpec, XBotConfig
from huweibot.core.observation import Observation


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _req() -> ProviderRequest:
    return ProviderRequest(
        messages=[
            ChatMessage(role="system", content="json only"),
            ChatMessage(role="user", content="Say pong"),
        ],
        model="gpt-compat",
        stream=False,
    )


def _obs() -> Observation:
    return Observation(
        screen_w=1280,
        screen_h=720,
        cursor_xy=(100, 100),
        cursor_conf=0.9,
        cursor_type="arrow",
        elements=[],
        app_hint="",
        screen_hash="x",
        ui_change_score=0.0,
    )


def test_normalize_base_url() -> None:
    assert _normalize_base_url("https://host") == "https://host/v1"
    assert _normalize_base_url("https://host/") == "https://host/v1"
    assert _normalize_base_url("https://host/v1") == "https://host/v1"
    assert _normalize_base_url("https://host/v1/") == "https://host/v1"


def test_auto_fallback_responses_404_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    chat_payload = {
        "model": "gpt-compat",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls.append(req.full_url)
        if req.full_url.endswith("/responses"):
            raise urlerror.HTTPError(
                url=req.full_url,
                code=404,
                msg="not found",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"unknown endpoint responses"}}'),
            )
        return _FakeHTTPResponse(chat_payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="gpt-compat",
        base_url="https://host",
        api_key="sk",
        api_style="auto",
        timeout_s=5,
        max_retries=1,
    )
    response = provider.complete(_req())
    assert response.text == "pong"
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/chat/completions")
    assert provider._resolved_style == "chat_completions"


def test_auto_no_fallback_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    def _urlopen(req, timeout=0):  # noqa: ARG001
        raise urlerror.HTTPError(
            url=req.full_url,
            code=401,
            msg="unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"invalid auth"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="gpt-compat",
        base_url="https://host/v1",
        api_key="sk",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    with pytest.raises(ProviderAuthError):
        provider.complete(_req())


def test_chat_success_parse_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "model": "gpt-compat",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeHTTPResponse(payload))  # noqa: ARG005
    provider = OpenAICompatibleProvider(
        name="compat",
        model="gpt-compat",
        base_url="https://host",
        api_key="sk",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=1,
    )
    response = provider.complete(_req())
    assert response.text == "pong"
    assert response.usage.total_tokens == 7


def test_responses_success_parse_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "model": "gpt-compat",
        "output_text": "pong",
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        "status": "completed",
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeHTTPResponse(payload))  # noqa: ARG005
    provider = OpenAICompatibleProvider(
        name="compat",
        model="gpt-compat",
        base_url="https://host",
        api_key="sk",
        api_style="responses",
        timeout_s=5,
        max_retries=1,
    )
    response = provider.complete(_req())
    assert response.text == "pong"
    assert response.usage.total_tokens == 7
    assert response.finish_reason == "completed"


def test_no_authorization_header_when_key_empty() -> None:
    provider = OpenAICompatibleProvider(
        name="compat",
        model="gpt-compat",
        base_url="https://host",
        api_key="",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=1,
    )
    url, payload, headers = provider.build_http_request(_req())
    assert url.endswith("/v1/chat/completions")
    assert "Authorization" not in headers
    assert payload["model"] == "gpt-compat"


def test_redaction_masks_compat_key() -> None:
    text = "https://host/v1/responses?key=abc Authorization: Bearer secret"
    safe = _redact_sensitive(text)
    assert "abc" not in safe
    assert "secret" not in safe


def test_router_openai_compat_missing_base_or_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XBOT_PROVIDER", "openai_compat")
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    cfg = XBotConfig(
        providers=[
            ProviderSpec(name="dummy_planner", type="dummy", enabled=True),
            ProviderSpec(name="compat", type="openai_compat", enabled=True),
        ],
        default_planner_provider="compat",
    )
    with pytest.raises(RuntimeError, match="init failed"):
        Router(cfg).request_plan(
            task="demo",
            obs=_obs(),
            memory={},
            step=1,
            obs_mode="delta",
            debug_reasoning=False,
            artifacts_dir="artifacts",
        )
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://host")
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_MODEL"):
        Router(cfg).request_plan(
            task="demo",
            obs=_obs(),
            memory={},
            step=1,
            obs_mode="delta",
            debug_reasoning=False,
            artifacts_dir="artifacts",
        )
