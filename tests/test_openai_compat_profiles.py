from __future__ import annotations

import json
from urllib import parse as urlparse

import pytest

from huweibot.agent.router import (
    ChatMessage,
    OpenAICompatibleProvider,
    ProviderHTTPError,
    ProviderRequest,
    Router,
    _redact_sensitive,
)
from huweibot.config import ProviderSpec, XBotConfig
from huweibot.core.observation import Observation


def _req() -> ProviderRequest:
    return ProviderRequest(
        messages=[
            ChatMessage(role="system", content="json only"),
            ChatMessage(role="user", content="Say pong"),
        ],
        model="compat-model",
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


def test_azure_endpoint_and_headers() -> None:
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://resource.openai.azure.com",
        api_key="az-key",
        profile="azure",
        api_style="chat_completions",
        azure_deployment="dep-a",
        azure_api_version="2024-10-21",
        timeout_s=5,
        max_retries=0,
    )
    url, _, headers = provider.build_http_request(_req())
    parsed = urlparse.urlparse(url)
    query = dict(urlparse.parse_qsl(parsed.query))
    assert parsed.path.endswith("/openai/deployments/dep-a/chat/completions")
    assert query.get("api-version") == "2024-10-21"
    assert headers.get("api-key") == "az-key"
    assert "Authorization" not in headers


def test_azure_responses_not_supported() -> None:
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://resource.openai.azure.com",
        api_key="az-key",
        profile="azure",
        api_style="responses",
        azure_deployment="dep-a",
        timeout_s=5,
        max_retries=0,
    )
    with pytest.raises(ProviderHTTPError, match="does not support responses"):
        provider.build_http_request(_req())


def test_openrouter_defaults_and_headers() -> None:
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="",
        api_key="or-key",
        profile="openrouter",
        api_style="chat_completions",
        openrouter_referer="https://example.local",
        openrouter_title="huweibot",
        timeout_s=5,
        max_retries=0,
    )
    assert provider.base_url == "https://openrouter.ai/api/v1"
    url, _, headers = provider.build_http_request(_req())
    assert url.endswith("/api/v1/chat/completions")
    assert headers.get("Authorization") == "Bearer or-key"
    assert headers.get("HTTP-Referer") == "https://example.local"
    assert headers.get("X-Title") == "huweibot"


def test_together_and_groq_defaults() -> None:
    together = OpenAICompatibleProvider(
        name="together",
        model="compat-model",
        base_url="",
        api_key="tg-key",
        profile="together",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=0,
    )
    groq = OpenAICompatibleProvider(
        name="groq",
        model="compat-model",
        base_url="",
        api_key="gq-key",
        profile="groq",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=0,
    )
    together_url, _, together_headers = together.build_http_request(_req())
    groq_url, _, groq_headers = groq.build_http_request(_req())
    assert together.base_url == "https://api.together.xyz/v1"
    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert together_url.endswith("/v1/chat/completions")
    assert groq_url.endswith("/openai/v1/chat/completions")
    assert together_headers.get("Authorization") == "Bearer tg-key"
    assert groq_headers.get("Authorization") == "Bearer gq-key"


def test_auto_initial_style_by_profile() -> None:
    generic = OpenAICompatibleProvider(
        name="generic",
        model="compat-model",
        base_url="https://host",
        api_key="k",
        profile="generic",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    openrouter = OpenAICompatibleProvider(
        name="openrouter",
        model="compat-model",
        base_url="https://host/api",
        api_key="k",
        profile="openrouter",
        api_style="auto",
        timeout_s=5,
        max_retries=0,
    )
    assert generic.resolve_api_style() == "responses"
    assert openrouter.resolve_api_style() == "chat_completions"


def test_profile_redaction_does_not_expose_key() -> None:
    secret = "sk-test-SECRET"
    provider = OpenAICompatibleProvider(
        name="compat",
        model="compat-model",
        base_url="https://openrouter.ai/api",
        api_key=secret,
        profile="openrouter",
        api_style="chat_completions",
        timeout_s=5,
        max_retries=0,
    )
    summary = provider.describe_http_request(_req())
    dumped = json.dumps(summary, ensure_ascii=False)
    assert secret not in dumped
    assert secret not in _redact_sensitive(f"Authorization: Bearer {secret}")


def test_router_azure_missing_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XBOT_PROVIDER", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_PROFILE", "azure")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://resource.openai.azure.com")
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_COMPAT_AZURE_DEPLOYMENT", raising=False)
    cfg = XBotConfig(
        providers=[
            ProviderSpec(name="dummy_planner", type="dummy", enabled=True),
            ProviderSpec(name="compat", type="openai_compat", enabled=True),
        ],
        default_planner_provider="compat",
    )
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_AZURE_DEPLOYMENT"):
        Router(cfg).request_plan(
            task="demo",
            obs=_obs(),
            memory={},
            step=1,
            obs_mode="delta",
            debug_reasoning=False,
            artifacts_dir="artifacts",
        )
