from __future__ import annotations

import io
import json
from urllib import error as urlerror

import pytest

from huweibot.agent.router import (
    ChatMessage,
    OpenAICompatibleProvider,
    ProviderAuthError,
    ProviderHTTPError,
    ProviderRequest,
    _normalize_base_url,
    _redact_sensitive,
)


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
        model="compat-model",
        stream=False,
    )


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        ("https://x.example.com", "https://x.example.com/v1"),
        ("https://x.example.com/v1", "https://x.example.com/v1"),
    ],
)
def test_base_url_normalization(input_url: str, expected: str) -> None:
    assert _normalize_base_url(input_url) == expected


def test_forced_chat_hits_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    payload = {
        "model": "compat-model",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        seen["url"] = req.full_url
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=1,
    )
    out = provider.complete(_req())
    assert out.text == "pong"
    assert seen["url"].endswith("/v1/chat/completions")


def test_forced_responses_hits_responses_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    payload = {
        "model": "compat-model",
        "output_text": "pong",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "status": "completed",
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        seen["url"] = req.full_url
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com/v1",
        api_key="sk-test",
        api_style="responses",
        timeout_s=5,
        max_retries=1,
    )
    out = provider.complete(_req())
    assert out.text == "pong"
    assert seen["url"].endswith("/v1/responses")


def test_forced_responses_404_no_auto_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        raise urlerror.HTTPError(
            url=req.full_url,
            code=404,
            msg="not found",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"responses unsupported"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="responses",
        timeout_s=5,
        max_retries=0,
    )
    with pytest.raises(ProviderHTTPError):
        provider.complete(_req())
    assert calls["n"] == 1


def test_forced_chat_404_no_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        raise urlerror.HTTPError(
            url=req.full_url,
            code=404,
            msg="not found",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"chat unsupported"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=0,
    )
    with pytest.raises(ProviderHTTPError):
        provider.complete(_req())
    assert calls["n"] == 1


def test_auto_404_switches_to_chat_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    chat_payload = {
        "model": "compat-model",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls.append(req.full_url)
        if len(calls) == 1:
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
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    assert provider.complete(_req()).text == "pong"
    assert provider._resolved_style == "chat_completions"
    assert provider.complete(_req()).text == "pong"
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/chat/completions")
    assert calls[2].endswith("/v1/chat/completions")


def test_auto_405_switches_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    chat_payload = {
        "model": "compat-model",
        "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urlerror.HTTPError(
                url=req.full_url,
                code=405,
                msg="method not allowed",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"responses method not allowed"}}'),
            )
        return _FakeHTTPResponse(chat_payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    assert provider.complete(_req()).text == "pong"
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/chat/completions")


def test_auto_401_no_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        raise urlerror.HTTPError(
            url=req.full_url,
            code=401,
            msg="unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error":{"message":"bad auth"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    with pytest.raises(ProviderAuthError):
        provider.complete(_req())
    assert calls["n"] == 1


def test_auto_429_retries_on_responses_not_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    payload = {
        "model": "compat-model",
        "output_text": "pong",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urlerror.HTTPError(
                url=req.full_url,
                code=429,
                msg="rate",
                hdrs={"Retry-After": "0"},
                fp=io.BytesIO(b'{"error":{"message":"rate limited"}}'),
            )
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="auto",
        timeout_s=5,
        max_retries=1,
    )
    assert provider.complete(_req()).text == "pong"
    assert len(calls) == 2
    assert calls[0].endswith("/v1/responses")
    assert calls[1].endswith("/v1/responses")


def test_chat_content_parts_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "model": "compat-model",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "po"},
                        {"type": "text", "text": "ng"},
                    ]
                }
            }
        ],
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeHTTPResponse(payload))  # noqa: ARG005
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=0,
    )
    assert provider.complete(_req()).text == "pong"


def test_responses_missing_usage_still_has_usage_struct(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "model": "compat-model",
        "output": [{"content": [{"type": "output_text", "text": "pong"}]}],
        "status": "completed",
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeHTTPResponse(payload))  # noqa: ARG005
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="sk-test",
        api_style="responses",
        timeout_s=5,
        max_retries=0,
    )
    out = provider.complete(_req())
    assert out.text == "pong"
    assert hasattr(out, "usage")
    assert out.usage.input_tokens is None


def test_key_optional_and_headers_override() -> None:
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://x.example.com",
        api_key="",
        api_style="chat_completions",
        extra_headers={"Authorization": "Token custom", "x-api-key": "abc"},
        timeout_s=5,
        max_retries=0,
    )
    _, _, headers = provider.build_http_request(_req())
    assert headers.get("Authorization") == "Token custom"
    assert headers.get("x-api-key") == "abc"


def test_redaction_does_not_expose_secret() -> None:
    secret = "sk-test-SECRET"
    sample = f"https://x/v1/responses?key={secret} Authorization: Bearer {secret}"
    redacted = _redact_sensitive(sample)
    assert secret not in redacted

