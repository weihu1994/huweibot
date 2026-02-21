from __future__ import annotations

import io
import json
from urllib import error as urlerror

import pytest

from huweibot.agent.router import (
    AnthropicProvider,
    ChatMessage,
    DummyProvider,
    GeminiProvider,
    MissingAPIKeyError,
    OpenAIProvider,
    ProviderRequest,
    Router,
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


def _req(messages: list[ChatMessage], *, model: str | None = None) -> ProviderRequest:
    return ProviderRequest(messages=messages, model=model, stream=False)


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


def test_default_is_dummy_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XBOT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = XBotConfig(
        providers=[ProviderSpec(name="dummy_planner", type="dummy", enabled=True)],
        default_planner_provider="dummy_planner",
    )
    router = Router(cfg)
    step, meta = router.request_plan(
        task="demo",
        obs=_obs(),
        memory={},
        step=1,
        obs_mode="delta",
        debug_reasoning=False,
        artifacts_dir="artifacts",
    )
    assert meta["provider"] == "dummy_planner"
    assert step.action.type == "WAIT"


def test_openai_success_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    payload = {
        "id": "cmpl",
        "model": "gpt-4.1-mini",
        "choices": [{"message": {"role": "assistant", "content": '{"action":{"type":"WAIT","duration_ms":200},"verify":{"type":"NONE"}}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAIProvider(
        name="openai",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-secret-value",
        timeout_s=3,
        max_retries=2,
    )
    response = provider.complete(_req([ChatMessage(role="user", content="return strict json")]))
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert "Authorization" in headers
    assert '"type":"WAIT"' in response.text
    assert response.usage.total_tokens == 15
    assert response.finish_reason == "stop"


def test_openai_retry_429_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []
    payload = {
        "model": "gpt-4.1-mini",
        "choices": [{"message": {"role": "assistant", "content": '{"action":{"type":"DONE","reason":"ok"},"verify":{"type":"NONE"}}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }

    def _urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise urlerror.HTTPError(
                url="https://api.openai.com/v1/chat/completions",
                code=429,
                msg="rate limit",
                hdrs={"Retry-After": "1"},
                fp=io.BytesIO(b'{"error":{"message":"too many requests"}}'),
            )
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("time.sleep", lambda x: sleeps.append(float(x)))
    provider = OpenAIProvider(
        name="openai",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        timeout_s=3,
        max_retries=2,
    )
    response = provider.complete(_req([ChatMessage(role="user", content="json")]))
    assert calls["n"] == 2
    assert sleeps and abs(sleeps[0] - 1.0) < 0.001
    assert response.usage.total_tokens == 6


def test_openai_missing_key_raises() -> None:
    provider = OpenAIProvider(
        name="openai",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key="",
        timeout_s=3,
        max_retries=1,
    )
    with pytest.raises(MissingAPIKeyError):
        provider.complete(_req([ChatMessage(role="user", content="x")]))


def test_anthropic_success_and_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "content": [{"type": "text", "text": '{"action":{"type":"WAIT","duration_ms":200},"verify":{"type":"NONE"}}'}],
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "stop_reason": "end_turn",
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = AnthropicProvider(
        name="anthropic",
        model="claude-3-5-sonnet-latest",
        base_url="https://api.anthropic.com",
        api_key="ak-test",
        timeout_s=3,
        max_retries=1,
    )
    response = provider.complete(
        _req(
            [
                ChatMessage(role="system", content="json only"),
                ChatMessage(role="user", content="task"),
                ChatMessage(role="assistant", content="ack"),
            ]
        )
    )
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    body = captured["body"]
    assert isinstance(body, dict)
    assert "system" in body and body["system"] == "json only"
    assert isinstance(body.get("messages"), list)
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert response.usage.input_tokens == 12


def test_anthropic_retry_529(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    payload = {
        "model": "claude-3-5-sonnet-latest",
        "content": [{"type": "text", "text": '{"action":{"type":"DONE","reason":"ok"},"verify":{"type":"NONE"}}'}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "stop_reason": "end_turn",
    }

    def _urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise urlerror.HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=529,
                msg="overloaded",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"overloaded"}}'),
            )
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = AnthropicProvider(
        name="anthropic",
        model="claude-3-5-sonnet-latest",
        base_url="https://api.anthropic.com",
        api_key="ak-test",
        timeout_s=3,
        max_retries=2,
    )
    response = provider.complete(_req([ChatMessage(role="user", content="task")]))
    assert calls["n"] == 2
    assert response.finish_reason == "end_turn"


def test_anthropic_missing_key_raises() -> None:
    provider = AnthropicProvider(
        name="anthropic",
        model="claude-3-5-sonnet-latest",
        base_url="https://api.anthropic.com",
        api_key="",
        timeout_s=3,
        max_retries=1,
    )
    with pytest.raises(MissingAPIKeyError):
        provider.complete(_req([ChatMessage(role="user", content="task")]))


def test_gemini_success_and_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    payload = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"action":{"type":"WAIT","duration_ms":300},"verify":{"type":"NONE"}}'}]},
            }
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 6, "totalTokenCount": 15},
    }

    def _urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = GeminiProvider(
        name="gemini",
        model="gemini-1.5-pro",
        base_url="https://generativelanguage.googleapis.com",
        api_key="gk-test",
        timeout_s=3,
        max_retries=1,
    )
    response = provider.complete(
        _req(
            [
                ChatMessage(role="system", content="json only"),
                ChatMessage(role="user", content="task"),
                ChatMessage(role="assistant", content="ok"),
            ]
        )
    )
    assert "key=***" not in str(captured["url"])
    body = captured["body"]
    assert isinstance(body, dict)
    assert "contents" in body
    assert body["contents"][0]["role"] == "user"
    assert body["contents"][1]["role"] == "model"
    assert response.usage.total_tokens == 15


def test_gemini_retry_503(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    payload = {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"action":{"type":"DONE","reason":"ok"},"verify":{"type":"NONE"}}'}]}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }

    def _urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise urlerror.HTTPError(
                url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=abc",
                code=503,
                msg="unavailable",
                hdrs={},
                fp=io.BytesIO(b'{"error":{"message":"unavailable"}}'),
            )
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = GeminiProvider(
        name="gemini",
        model="gemini-1.5-pro",
        base_url="https://generativelanguage.googleapis.com",
        api_key="gk-test",
        timeout_s=3,
        max_retries=2,
    )
    response = provider.complete(_req([ChatMessage(role="user", content="task")]))
    assert calls["n"] == 2
    assert response.usage.total_tokens == 5


def test_gemini_missing_key_raises() -> None:
    provider = GeminiProvider(
        name="gemini",
        model="gemini-1.5-pro",
        base_url="https://generativelanguage.googleapis.com",
        api_key="",
        timeout_s=3,
        max_retries=1,
    )
    with pytest.raises(MissingAPIKeyError):
        provider.complete(_req([ChatMessage(role="user", content="task")]))


@pytest.mark.parametrize(
    ("provider_name", "env_key"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ],
)
def test_router_missing_key_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    env_key: str,
) -> None:
    monkeypatch.setenv("XBOT_PROVIDER", provider_name)
    monkeypatch.delenv(env_key, raising=False)
    cfg = XBotConfig(
        providers=[
            ProviderSpec(name="dummy_planner", type="dummy", enabled=True),
            ProviderSpec(name=f"{provider_name}_planner", type=provider_name, enabled=True),
        ],
        default_planner_provider=f"{provider_name}_planner",
        allow_fallback_to_dummy=False,
    )
    router = Router(cfg)
    with pytest.raises(RuntimeError, match="API key is missing"):
        router.request_plan(
            task="demo",
            obs=_obs(),
            memory={},
            step=1,
            obs_mode="delta",
            debug_reasoning=False,
            artifacts_dir="artifacts",
        )


def test_router_fallback_to_dummy_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("XBOT_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = XBotConfig(
        providers=[
            ProviderSpec(name="dummy_planner", type="dummy", enabled=True),
            ProviderSpec(name="openai_planner", type="openai", enabled=True),
        ],
        default_planner_provider="openai_planner",
        allow_fallback_to_dummy=True,
    )
    caplog.set_level("WARNING")
    router = Router(cfg)
    step, meta = router.request_plan(
        task="demo",
        obs=_obs(),
        memory={},
        step=1,
        obs_mode="delta",
        debug_reasoning=False,
        artifacts_dir="artifacts",
    )
    assert meta["provider"] == "dummy_planner"
    assert step.action.type == "WAIT"
    assert "fallback to dummy" in caplog.text.lower()
    assert "openai_api_key" not in caplog.text.lower()


def test_dummy_provider_no_network() -> None:
    provider = DummyProvider("dummy")
    response = provider.complete(_req([ChatMessage(role="user", content="task")]))
    assert response.text
    assert response.model == "dummy"


def test_redact_sensitive_masks_keys() -> None:
    text = "https://x?key=abc123&foo=bar Authorization: Bearer sk-secret"
    safe = _redact_sensitive(text)
    assert "abc123" not in safe
    assert "sk-secret" not in safe
