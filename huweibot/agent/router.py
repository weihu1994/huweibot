from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from huweibot._pydantic_compat import BaseModel, Field, model_to_dict, parse_obj
from huweibot.agent.schemas import NextStep
from huweibot.core.observation import Observation, UIElement

LOGGER = logging.getLogger(__name__)

_DELTA_ALLOWED_KEYS = {
    "c",
    "u",
    "la",
    "lv",
    "ed",
    "ms",
}


def _model_dump(model: BaseModel) -> dict[str, Any]:
    return model_to_dict(model)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.lower()
    return "".join(normalized.split())


def _truncate_text(value: str, policy: str) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    if "head" in policy and "tail" in policy and "~" in policy:
        try:
            left_part, right_part = policy.split("~", 1)
            head_n = int(left_part.replace("head", ""))
            tail_n = int(right_part.replace("tail", ""))
        except ValueError:
            return text
        if len(text) <= head_n + tail_n:
            return text
        return f"{text[:head_n]}~{text[-tail_n:]}"
    return text


def _estimate_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value))


def _redact_sensitive(text: str) -> str:
    if not text:
        return text
    redacted = re.sub(r"([?&](?:key|api_key)=)[^&\s]+", r"\1***", text, flags=re.IGNORECASE)
    redacted = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***", redacted, flags=re.IGNORECASE)
    return redacted


class ProviderCapabilities(BaseModel):
    supports_strict_json: bool = False
    supports_image: bool = False
    max_context_tokens: int = 4096
    cost_tier: str = "cheap"  # cheap|standard|premium


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""


class ProviderRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ProviderResponse(BaseModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    finish_reason: str | None = None


class ProviderError(RuntimeError):
    """Base provider error."""


class MissingAPIKeyError(ProviderError):
    """Raised when selected cloud provider has no API key."""


class MissingBaseURLError(ProviderError):
    """Raised when selected provider has no base url."""


class MissingModelError(ProviderError):
    """Raised when selected provider has no model."""


class MissingDeploymentError(ProviderError):
    """Raised when Azure compatible provider has no deployment."""


class ProviderAuthError(ProviderError):
    """Raised on authentication/authorization failures."""


class ProviderHTTPError(ProviderError):
    """Raised on non-retryable HTTP errors."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Provider(ABC):
    def __init__(
        self,
        name: str,
        *,
        provider_type: str = "dummy",
        model: str | None = None,
        capabilities: ProviderCapabilities | None = None,
    ):
        self.name = name
        self.provider_type = provider_type
        self.model = model
        self.capabilities = capabilities or ProviderCapabilities()

    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    def capability_dict(self) -> dict[str, Any]:
        return _model_dump(self.capabilities)

    def _build_request(self, prompt: str, *, system: str | None = None) -> ProviderRequest:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        return ProviderRequest(messages=messages, model=self.model, stream=False, metadata={})


class DummyProvider(Provider):
    def __init__(self, name: str = "dummy"):
        super().__init__(
            name=name,
            provider_type="dummy",
            model="dummy",
            capabilities=ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=2048,
                cost_tier="cheap",
            ),
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        system_text = "\n".join(msg.content for msg in request.messages if msg.role == "system")
        user_text = "\n".join(msg.content for msg in request.messages if msg.role != "system")
        prompt = f"{system_text}\n{user_text}".strip()
        lower = prompt.lower()
        if '"task"' in lower and ('"task": ""' in lower or '"task": null' in lower):
            payload = {
                "action": {"type": "DONE", "reason": "empty_task"},
                "verify": {"type": "NONE"},
                "obs_mode": "delta",
            }
        else:
            payload = {
                "action": {"type": "WAIT", "duration_ms": 500},
                "verify": {"type": "NONE"},
                "obs_mode": "delta",
            }
        text = json.dumps(payload, ensure_ascii=False)
        return ProviderResponse(
            text=text,
            raw=payload,
            model=self.model or "dummy",
            usage=ProviderUsage(input_tokens=None, output_tokens=None, total_tokens=None),
            finish_reason="stop",
        )

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        response = self.complete(self._build_request(prompt, system=system))
        return response.text


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _extract_text(value.get("text"))
        if "content" in value:
            return _extract_text(value.get("content"))
    if isinstance(value, list):
        return "".join(_extract_text(v) for v in value)
    return str(value)


def _parse_retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            dt = parsedate_to_datetime(str(raw))
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    if not value:
        return ""
    normalized = value.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _parse_json_object_env(value: str | None, *, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError(f"{name} must be valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be JSON object")
    return parsed


class _NetworkProvider(Provider):
    retryable_status_codes = {429, 500, 502, 503, 504, 529}

    def __init__(
        self,
        name: str,
        *,
        provider_type: str,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        capabilities: ProviderCapabilities | None = None,
    ):
        super().__init__(
            name=name,
            provider_type=provider_type,
            model=model,
            capabilities=capabilities
            or ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=128000,
                cost_tier="standard",
            ),
        )
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_retries = max(0, int(max_retries))

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError(
                f"{self.provider_type} selected but API key is missing; "
                f"set the required key env var or switch provider."
            )

    def _request_json(self, *, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(url=url, data=body, headers=headers, method="POST")
        attempt = 0
        while True:
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ProviderHTTPError(
                            f"{self.provider_type} returned non-JSON response"
                        ) from exc
            except urlerror.HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                body_text = ""
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                if status in {401, 403}:
                    raise ProviderAuthError(f"{self.provider_type} auth failed ({status})") from exc
                if status in self.retryable_status_codes and attempt < self.max_retries:
                    wait = _parse_retry_after(getattr(exc, "headers", None))
                    if wait is None:
                        wait = min(8.0, 0.5 * (2**attempt) + random.uniform(0, 0.25))
                    attempt += 1
                    LOGGER.warning(
                        "%s provider retrying after HTTP %s (attempt %s/%s)",
                        self.provider_type,
                        status,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait)
                    continue
                trimmed = body_text[:240].replace("\n", " ").strip()
                raise ProviderHTTPError(
                    f"{self.provider_type} HTTP {status}: {trimmed}",
                    status_code=status,
                ) from exc
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    wait = min(8.0, 0.5 * (2**attempt) + random.uniform(0, 0.25))
                    attempt += 1
                    LOGGER.warning(
                        "%s provider network retry (attempt %s/%s)",
                        self.provider_type,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(wait)
                    continue
                raise ProviderError(f"{self.provider_type} network error: {_redact_sensitive(str(exc))}") from exc

    @abstractmethod
    def build_http_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        raise NotImplementedError

    def describe_http_request(self, request: ProviderRequest) -> dict[str, Any]:
        self._ensure_key()
        url, payload, headers = self.build_http_request(request)
        safe_headers: dict[str, str] = {}
        for key, value in headers.items():
            lowered = key.lower()
            if "authorization" in lowered or "api-key" in lowered or lowered.endswith("key"):
                safe_headers[key] = "***"
            else:
                safe_headers[key] = value
        return {
            "provider": self.provider_type,
            "model": request.model or self.model,
            "url": _redact_sensitive(url),
            "headers": safe_headers,
            "payload_summary": {
                "messages": len(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stop": request.stop,
                "stream": request.stream,
            },
            "payload_preview": {
                "model": payload.get("model"),
                "stream": payload.get("stream"),
                "messages": payload.get("messages", payload.get("contents", payload.get("input", [])))[:2],
            },
        }

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        response = self.complete(self._build_request(prompt, system=system))
        return response.text


class OpenAIProvider(_NetworkProvider):
    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
        max_retries: int,
    ):
        super().__init__(
            name=name,
            provider_type="openai",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            capabilities=ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=128000,
                cost_tier="standard",
            ),
        )

    def build_http_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return url, payload, headers

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_key()
        url, payload, headers = self.build_http_request(request)
        raw = self._request_json(url=url, payload=payload, headers=headers)
        choice = ((raw.get("choices") or [{}])[0]) if isinstance(raw, dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = _extract_text(message.get("content"))
        usage_raw = raw.get("usage", {}) if isinstance(raw, dict) else {}
        usage = ProviderUsage(
            input_tokens=usage_raw.get("prompt_tokens"),
            output_tokens=usage_raw.get("completion_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )
        return ProviderResponse(
            text=text,
            raw=raw if isinstance(raw, dict) else {"raw": raw},
            model=(raw.get("model") if isinstance(raw, dict) else None) or request.model or self.model,
            usage=usage,
            finish_reason=(choice.get("finish_reason") if isinstance(choice, dict) else None),
        )


class OpenAICompatibleProvider(_NetworkProvider):
    _SUPPORTED_STYLES = {"auto", "responses", "chat_completions"}

    class ProfileAdapter:
        def __init__(
            self,
            *,
            profile: str,
            azure_deployment: str | None = None,
            azure_api_version: str = "2024-10-21",
            openrouter_referer: str | None = None,
            openrouter_title: str | None = None,
            openrouter_app_id: str | None = None,
        ):
            self.profile = profile
            self.azure_deployment = (azure_deployment or "").strip() or None
            self.azure_api_version = (azure_api_version or "2024-10-21").strip()
            self.openrouter_referer = (openrouter_referer or "").strip() or None
            self.openrouter_title = (openrouter_title or "").strip() or None
            self.openrouter_app_id = (openrouter_app_id or "").strip() or None

        def normalize_base_url(self, base_url: str) -> str:
            raw = (base_url or "").strip()
            if self.profile == "openrouter":
                raw = raw or "https://openrouter.ai/api/v1"
                base = raw.rstrip("/")
                if base.endswith("/api/v1"):
                    return base
                if base.endswith("/api"):
                    return f"{base}/v1"
                if base == "https://openrouter.ai":
                    return "https://openrouter.ai/api/v1"
                if base.endswith("/v1"):
                    return base
                return f"{base}/v1"
            if self.profile == "together":
                return _normalize_base_url(raw or "https://api.together.xyz/v1")
            if self.profile == "groq":
                raw = raw or "https://api.groq.com/openai/v1"
                base = raw.rstrip("/")
                if base.endswith("/openai/v1"):
                    return base
                if base.endswith("/openai"):
                    return f"{base}/v1"
                if base.endswith("/v1"):
                    return base
                return f"{base}/openai/v1"
            if self.profile == "azure":
                return raw.rstrip("/")
            return _normalize_base_url(raw)

        def supports_api_style(self, api_style: str) -> bool:
            if self.profile == "azure" and api_style == "responses":
                return False
            return True

        def initial_style(self, requested_style: str) -> str:
            if requested_style != "auto":
                return requested_style
            if self.profile == "generic":
                return "responses"
            return "chat_completions"

        def build_endpoint(self, base_url: str, api_style: str) -> str:
            if self.profile == "azure":
                if not self.azure_deployment:
                    raise MissingDeploymentError(
                        "azure profile requires OPENAI_COMPAT_AZURE_DEPLOYMENT"
                    )
                if api_style != "chat_completions":
                    raise ProviderHTTPError(
                        "azure profile does not support responses; use chat_completions"
                    )
                prefix = base_url.rstrip("/")
                marker = "/openai/deployments/"
                if marker in prefix:
                    deploy_prefix = prefix
                else:
                    if not prefix.endswith("/openai"):
                        prefix = f"{prefix}/openai"
                    deploy_prefix = f"{prefix}/deployments/{self.azure_deployment}"
                return f"{deploy_prefix}/chat/completions"
            if api_style == "responses":
                return f"{base_url.rstrip('/')}/responses"
            return f"{base_url.rstrip('/')}/chat/completions"

        def build_headers(self, api_key: str, user_headers: dict[str, str]) -> dict[str, str]:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.profile == "azure":
                if api_key:
                    headers["api-key"] = api_key
            else:
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
            if self.profile == "openrouter":
                if self.openrouter_referer:
                    headers["HTTP-Referer"] = self.openrouter_referer
                if self.openrouter_title:
                    headers["X-Title"] = self.openrouter_title
                if self.openrouter_app_id:
                    headers["X-App-Id"] = self.openrouter_app_id
            headers.update(user_headers)
            return headers

        def build_query(self, user_query: dict[str, str], api_style: str) -> dict[str, str]:
            query = dict(user_query)
            if self.profile == "azure":
                query.setdefault("api-version", self.azure_api_version)
            return query

        def patch_payload(self, payload: dict[str, Any], api_style: str) -> dict[str, Any]:
            patched = dict(payload)
            _ = api_style
            return patched

        def allow_auto_fallback_to_chat(self, exc: ProviderHTTPError) -> bool:
            if self.profile != "generic":
                return False
            status = int(exc.status_code or 0)
            message = str(exc).lower()
            if status in {404, 405}:
                return True
            if status == 400 and any(
                key in message for key in ("responses", "unknown endpoint", "unsupported", "not found")
            ):
                return True
            return False

        def parse_response(
            self,
            raw: dict[str, Any],
            *,
            api_style: str,
            default_model: str,
        ) -> tuple[str, ProviderUsage, str | None, str]:
            if api_style == "responses":
                direct = raw.get("output_text")
                if isinstance(direct, str) and direct.strip():
                    text = direct
                else:
                    text = ""
                    out = raw.get("output")
                    if isinstance(out, list):
                        chunks: list[str] = []
                        for item in out:
                            if not isinstance(item, dict):
                                continue
                            content = item.get("content")
                            if not isinstance(content, list):
                                continue
                            for part in content:
                                if isinstance(part, dict) and part.get("type") in {"output_text", "text", "input_text"}:
                                    chunks.append(_extract_text(part.get("text")))
                        text = "".join(chunks)
                usage_raw = raw.get("usage", {}) if isinstance(raw, dict) else {}
                input_tokens = usage_raw.get("input_tokens")
                output_tokens = usage_raw.get("output_tokens")
                total_tokens = usage_raw.get("total_tokens")
                if total_tokens is None and (input_tokens is not None or output_tokens is not None):
                    total_tokens = (input_tokens or 0) + (output_tokens or 0)
                usage = ProviderUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
                finish_reason = raw.get("status")
                model = raw.get("model") or default_model
                return text, usage, finish_reason, model

            choice = ((raw.get("choices") or [{}])[0]) if isinstance(raw, dict) else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            text = _extract_text(message.get("content"))
            if not text and isinstance(choice, dict):
                delta = choice.get("delta")
                text = _extract_text(delta.get("content") if isinstance(delta, dict) else None)
            usage_raw = raw.get("usage", {}) if isinstance(raw, dict) else {}
            usage = ProviderUsage(
                input_tokens=usage_raw.get("prompt_tokens"),
                output_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            )
            finish_reason = (
                choice.get("finish_reason")
                if isinstance(choice, dict)
                else None
            ) or raw.get("finish_reason")
            model = raw.get("model") or default_model
            return text, usage, finish_reason, model

    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        profile: str = "generic",
        api_style: str = "auto",
        extra_headers: dict[str, Any] | None = None,
        extra_query: dict[str, Any] | None = None,
        azure_deployment: str | None = None,
        azure_api_version: str = "2024-10-21",
        openrouter_referer: str | None = None,
        openrouter_title: str | None = None,
        openrouter_app_id: str | None = None,
        timeout_s: float,
        max_retries: int,
    ):
        profile_name = (profile or "generic").strip().lower().replace("-", "_")
        if profile_name == "openai_compatible":
            profile_name = "generic"
        if profile_name not in {"generic", "azure", "openrouter", "together", "groq"}:
            raise ValueError(
                "OPENAI_COMPAT_PROFILE must be one of: generic|azure|openrouter|together|groq"
            )
        adapter = self.ProfileAdapter(
            profile=profile_name,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version,
            openrouter_referer=openrouter_referer,
            openrouter_title=openrouter_title,
            openrouter_app_id=openrouter_app_id,
        )
        normalized_base = adapter.normalize_base_url(base_url)
        if not normalized_base:
            raise MissingBaseURLError("openai_compat requires OPENAI_COMPAT_BASE_URL")
        model_name = (model or "").strip()
        if not model_name:
            raise MissingModelError("openai_compat requires OPENAI_COMPAT_MODEL")
        style = (api_style or "auto").strip().lower()
        if style not in self._SUPPORTED_STYLES:
            raise ValueError("OPENAI_COMPAT_API_STYLE must be one of: auto|responses|chat_completions")
        super().__init__(
            name=name,
            provider_type="openai_compat",
            model=model_name,
            base_url=normalized_base,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            capabilities=ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=128000,
                cost_tier="standard",
            ),
        )
        self.profile = profile_name
        self.adapter = adapter
        self.api_style = style
        self.extra_headers = {str(k): str(v) for k, v in (extra_headers or {}).items()}
        self.extra_query = {str(k): str(v) for k, v in (extra_query or {}).items()}
        self._resolved_style: str | None = None

    def _ensure_key(self) -> None:
        # Some OpenAI-compatible gateways run in trusted networks with no auth key.
        return

    def _base_headers(self) -> dict[str, str]:
        return self.adapter.build_headers(self.api_key, self.extra_headers)

    def _append_query(self, url: str, profile_query: dict[str, str] | None = None) -> str:
        parsed = urlparse.urlparse(url)
        query = dict(urlparse.parse_qsl(parsed.query, keep_blank_values=True))
        if profile_query:
            query.update({k: str(v) for k, v in profile_query.items()})
        if self.extra_query:
            query.update({k: str(v) for k, v in self.extra_query.items()})
        if not query:
            return url
        encoded = urlparse.urlencode(query, doseq=True)
        return urlparse.urlunparse(parsed._replace(query=encoded))

    def _build_chat_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        payload = self.adapter.patch_payload(payload, "chat_completions")
        endpoint = self.adapter.build_endpoint(self.base_url, "chat_completions")
        query = self.adapter.build_query({}, "chat_completions")
        url = self._append_query(endpoint, query)
        return url, payload, self._base_headers()

    def _build_responses_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        if not self.adapter.supports_api_style("responses"):
            raise ProviderHTTPError(
                f"{self.profile} profile does not support responses; use chat_completions"
            )
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "input": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": bool(request.stream),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop
        payload = self.adapter.patch_payload(payload, "responses")
        endpoint = self.adapter.build_endpoint(self.base_url, "responses")
        query = self.adapter.build_query({}, "responses")
        url = self._append_query(endpoint, query)
        return url, payload, self._base_headers()

    def _resolved_or_initial_style(self) -> str:
        return self._resolved_style or self.adapter.initial_style(self.api_style)

    def build_http_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        style = self._resolved_or_initial_style()
        if style == "responses":
            return self._build_responses_request(request)
        return self._build_chat_request(request)

    def resolve_api_style(self) -> str:
        return self._resolved_or_initial_style()

    def describe_http_request(self, request: ProviderRequest) -> dict[str, Any]:
        summary = super().describe_http_request(request)
        summary["profile"] = self.profile
        summary["api_style"] = self.api_style
        summary["resolved_style"] = self.resolve_api_style()
        return summary

    def _complete_with_style(self, request: ProviderRequest, style: str) -> ProviderResponse:
        if style == "responses":
            url, payload, headers = self._build_responses_request(request)
        else:
            url, payload, headers = self._build_chat_request(request)
            style = "chat_completions"
        raw = self._request_json(url=url, payload=payload, headers=headers)
        text, usage, finish_reason, model = self.adapter.parse_response(
            raw,
            api_style=style,
            default_model=request.model or self.model,
        )
        return ProviderResponse(
            text=text,
            raw=raw,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self.api_style != "auto":
            self._resolved_style = self.api_style
            return self._complete_with_style(request, self.api_style)

        style = self._resolved_or_initial_style()
        if style == "chat_completions":
            return self._complete_with_style(request, "chat_completions")
        try:
            response = self._complete_with_style(request, "responses")
            self._resolved_style = "responses"
            return response
        except ProviderHTTPError as exc:
            if self.adapter.allow_auto_fallback_to_chat(exc):
                response = self._complete_with_style(request, "chat_completions")
                self._resolved_style = "chat_completions"
                return response
            raise


class AnthropicProvider(_NetworkProvider):
    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
        max_retries: int,
    ):
        super().__init__(
            name=name,
            provider_type="anthropic",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            capabilities=ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=200000,
                cost_tier="premium",
            ),
        )

    def build_http_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        system_parts = [m.content for m in request.messages if m.role == "system" and m.content]
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            messages.append({"role": role, "content": [{"type": "text", "text": msg.content}]})
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": int(request.max_tokens or 1024),
            "stream": bool(request.stream),
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        return f"{self.base_url}/v1/messages", payload, headers

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_key()
        url, payload, headers = self.build_http_request(request)
        raw = self._request_json(url=url, payload=payload, headers=headers)
        content = raw.get("content", []) if isinstance(raw, dict) else []
        text = ""
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        usage_raw = raw.get("usage", {}) if isinstance(raw, dict) else {}
        usage = ProviderUsage(
            input_tokens=usage_raw.get("input_tokens"),
            output_tokens=usage_raw.get("output_tokens"),
            total_tokens=(usage_raw.get("input_tokens") or 0) + (usage_raw.get("output_tokens") or 0),
        )
        return ProviderResponse(
            text=text,
            raw=raw if isinstance(raw, dict) else {"raw": raw},
            model=(raw.get("model") if isinstance(raw, dict) else None) or request.model or self.model,
            usage=usage,
            finish_reason=(raw.get("stop_reason") if isinstance(raw, dict) else None),
        )


class GeminiProvider(_NetworkProvider):
    def __init__(
        self,
        name: str,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
        max_retries: int,
    ):
        super().__init__(
            name=name,
            provider_type="gemini",
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            capabilities=ProviderCapabilities(
                supports_strict_json=True,
                supports_image=False,
                max_context_tokens=1000000,
                cost_tier="standard",
            ),
        )

    def build_http_request(self, request: ProviderRequest) -> tuple[str, dict[str, Any], dict[str, str]]:
        sys_parts = [m.content for m in request.messages if m.role == "system" and m.content]
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                continue
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        payload: dict[str, Any] = {"contents": contents}
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(sys_parts)}]}
        gen_cfg: dict[str, Any] = {}
        if request.temperature is not None:
            gen_cfg["temperature"] = request.temperature
        if request.top_p is not None:
            gen_cfg["topP"] = request.top_p
        if request.max_tokens is not None:
            gen_cfg["maxOutputTokens"] = request.max_tokens
        if request.stop:
            gen_cfg["stopSequences"] = request.stop
        if gen_cfg:
            payload["generationConfig"] = gen_cfg
        model_name = request.model or self.model
        quoted_key = urlparse.quote(self.api_key, safe="")
        url = f"{self.base_url}/v1beta/models/{model_name}:generateContent?key={quoted_key}"
        return url, payload, {"Content-Type": "application/json"}

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_key()
        url, payload, headers = self.build_http_request(request)
        raw = self._request_json(url=url, payload=payload, headers=headers)
        model_name = request.model or self.model
        candidates = raw.get("candidates", []) if isinstance(raw, dict) else []
        first = candidates[0] if isinstance(candidates, list) and candidates else {}
        content = first.get("content", {}) if isinstance(first, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage_raw = raw.get("usageMetadata", {}) if isinstance(raw, dict) else {}
        usage = ProviderUsage(
            input_tokens=usage_raw.get("promptTokenCount"),
            output_tokens=usage_raw.get("candidatesTokenCount"),
            total_tokens=usage_raw.get("totalTokenCount"),
        )
        return ProviderResponse(
            text=text,
            raw=raw if isinstance(raw, dict) else {"raw": raw},
            model=model_name,
            usage=usage,
            finish_reason=(first.get("finishReason") if isinstance(first, dict) else None),
        )


class Router:
    def __init__(self, config: object):
        self.config = config
        self.providers: dict[str, Provider] = {}
        self.last_sent_obs_digest: str | None = None
        self._last_elements_sig: dict[str, str] = {}
        self._fail_counts: dict[str, int] = {}
        self._provider_init_errors: dict[str, str] = {}
        self._task_state: dict[str, Any] = {}
        self._ambiguous_streak = 0
        self._no_match_streak = 0
        self._macro_fail_streak = 0
        self._last_app_hint = ""
        self._load_providers()
        self.reset_task_state()

    @staticmethod
    def _spec_get(spec: Any, key: str, default: Any = None) -> Any:
        if isinstance(spec, dict):
            return spec.get(key, default)
        return getattr(spec, key, default)

    @staticmethod
    def _resolve_model(spec_model: str | None, env_value: str | None, default_model: str) -> str:
        if spec_model and spec_model != "dummy-model":
            return spec_model
        if env_value:
            return env_value
        return default_model

    def _create_provider(self, *, name: str, provider_type: str, spec: Any | None = None) -> Provider:
        provider_type = provider_type.strip().lower().replace("-", "_")
        if provider_type == "openai_compatible":
            provider_type = "openai_compat"
        spec_model = self._spec_get(spec, "model", None)
        spec_base_url = self._spec_get(spec, "base_url", None)
        spec_api_key_env = self._spec_get(spec, "api_key_env", None)
        timeout_s = float(getattr(self.config, "provider_timeout_s", 30.0))
        max_retries = int(getattr(self.config, "provider_max_retries", 3))

        if provider_type == "dummy":
            return DummyProvider(name=name)
        if provider_type == "openai":
            api_key_env = spec_api_key_env or "OPENAI_API_KEY"
            return OpenAIProvider(
                name=name,
                model=self._resolve_model(spec_model, os.getenv("OPENAI_MODEL"), "gpt-4.1-mini"),
                base_url=(spec_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
                api_key=os.getenv(api_key_env),
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
        if provider_type == "anthropic":
            api_key_env = spec_api_key_env or "ANTHROPIC_API_KEY"
            return AnthropicProvider(
                name=name,
                model=self._resolve_model(spec_model, os.getenv("ANTHROPIC_MODEL"), "claude-3-5-sonnet-latest"),
                base_url=(spec_base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/"),
                api_key=os.getenv(api_key_env),
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
        if provider_type == "gemini":
            api_key_env = spec_api_key_env or "GEMINI_API_KEY"
            return GeminiProvider(
                name=name,
                model=self._resolve_model(spec_model, os.getenv("GEMINI_MODEL"), "gemini-1.5-pro"),
                base_url=(spec_base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/"),
                api_key=os.getenv(api_key_env),
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
        if provider_type == "openai_compat":
            api_key_env = spec_api_key_env or "OPENAI_COMPAT_API_KEY"
            model_env = os.getenv("OPENAI_COMPAT_MODEL")
            model_name = spec_model if spec_model and spec_model != "dummy-model" else (model_env or "")
            base_url = spec_base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or ""
            api_style = str(os.getenv("OPENAI_COMPAT_API_STYLE", "auto")).strip().lower() or "auto"
            profile = str(os.getenv("OPENAI_COMPAT_PROFILE", "generic")).strip().lower() or "generic"
            extra_headers = _parse_json_object_env(os.getenv("OPENAI_COMPAT_HEADERS_JSON"), name="OPENAI_COMPAT_HEADERS_JSON")
            extra_query = _parse_json_object_env(os.getenv("OPENAI_COMPAT_QUERY_JSON"), name="OPENAI_COMPAT_QUERY_JSON")
            return OpenAICompatibleProvider(
                name=name,
                model=model_name,
                base_url=base_url,
                api_key=os.getenv(api_key_env),
                profile=profile,
                api_style=api_style,
                extra_headers=extra_headers,
                extra_query=extra_query,
                azure_deployment=os.getenv("OPENAI_COMPAT_AZURE_DEPLOYMENT"),
                azure_api_version=os.getenv("OPENAI_COMPAT_AZURE_API_VERSION", "2024-10-21"),
                openrouter_referer=os.getenv("OPENAI_COMPAT_OR_REFERER"),
                openrouter_title=os.getenv("OPENAI_COMPAT_OR_TITLE"),
                openrouter_app_id=os.getenv("OPENAI_COMPAT_OR_APP_ID"),
                timeout_s=float(os.getenv("OPENAI_COMPAT_TIMEOUT_S", timeout_s)),
                max_retries=int(os.getenv("OPENAI_COMPAT_MAX_RETRIES", max_retries)),
            )
        raise ValueError(f"unsupported provider type: {provider_type}")

    def _load_providers(self) -> None:
        specs = getattr(self.config, "providers", [])
        for spec in specs:
            name = self._spec_get(spec, "name", None)
            provider_type = str(self._spec_get(spec, "type", "dummy")).strip().lower()
            enabled = bool(self._spec_get(spec, "enabled", True))
            if not name or not enabled:
                continue
            try:
                provider = self._create_provider(name=name, provider_type=provider_type, spec=spec)
            except Exception as exc:
                LOGGER.warning("skip provider %s (%s): %s", name, provider_type, exc)
                self._provider_init_errors[name] = str(exc)
                self._provider_init_errors[provider_type] = str(exc)
                continue
            self.providers[name] = provider
            self._fail_counts[name] = 0

        explicit = str(os.getenv("XBOT_PROVIDER", "")).strip().lower()
        explicit_key = explicit.replace("-", "_")
        if explicit and not self._find_provider_by_selector(explicit):
            try:
                provider = self._create_provider(name=explicit_key, provider_type=explicit_key, spec=None)
            except Exception as exc:
                LOGGER.warning("cannot build explicit provider %s: %s", explicit, exc)
                self._provider_init_errors[explicit] = str(exc)
            else:
                self.providers[provider.name] = provider
                self._fail_counts[provider.name] = 0

        if not self.providers:
            fallback = DummyProvider("dummy_planner")
            self.providers[fallback.name] = fallback
            self._fail_counts[fallback.name] = 0

        cache = getattr(self.config, "provider_capabilities_cache", None)
        if isinstance(cache, dict):
            cache.update(self.capabilities_report())

    def capabilities_report(self) -> dict[str, dict[str, Any]]:
        return {name: provider.capability_dict() for name, provider in self.providers.items()}

    def _provider_priority(self, provider: Provider) -> tuple[int, str]:
        tier = str(provider.capabilities.cost_tier or "standard").lower()
        order = {"cheap": 0, "standard": 1, "premium": 2}
        return (order.get(tier, 1), provider.name)

    def _find_provider_by_selector(self, selector: str) -> Provider | None:
        selector = selector.replace("-", "_")
        candidate = self.providers.get(selector)
        if candidate is not None:
            return candidate
        for provider in self.providers.values():
            if provider.provider_type == selector:
                return provider
        return None

    def _get_dummy_provider(self) -> Provider | None:
        for provider in self.providers.values():
            if provider.provider_type == "dummy":
                return provider
        return None

    def _choose_provider(self) -> Provider:
        explicit = str(os.getenv("XBOT_PROVIDER", "")).strip().lower()
        explicit = explicit.replace("-", "_")
        if explicit:
            provider = self._find_provider_by_selector(explicit)
            if provider is None:
                init_error = self._provider_init_errors.get(explicit)
                if init_error:
                    raise RuntimeError(f"provider '{explicit}' init failed: {init_error}")
                raise RuntimeError(f"provider '{explicit}' is not configured")
            return provider

        preferred = str(getattr(self.config, "default_planner_provider", "")).strip()
        provider = self._find_provider_by_selector(preferred) if preferred else None
        if provider is not None:
            if provider.provider_type != "dummy" and not self._fallback_allowed():
                return provider
            failures = self._fail_counts.get(provider.name, 0)
            if failures <= 2 or not bool(getattr(self.config, "router_enable_downgrade", True)):
                return provider

        # Minimal downgrade fallback: choose cheapest provider.
        return sorted(self.providers.values(), key=self._provider_priority)[0]

    def get_planner_provider(self) -> Provider:
        return self._choose_provider()

    def get_vlm_provider(self) -> Provider | None:
        preferred = str(getattr(self.config, "default_vlm_provider", "")).strip()
        provider = self._find_provider_by_selector(preferred) if preferred else None
        if provider is not None:
            if provider.capabilities.supports_image:
                return provider
        for provider in self.providers.values():
            if provider.capabilities.supports_image:
                return provider
        return None

    def reset_task_state(self) -> None:
        self._task_state = {
            "elements0_streak": 0,
            "vlm_calls_this_task": 0,
            "last_vlm_ts": 0.0,
            "local_boost_done": False,
        }
        self._ambiguous_streak = 0
        self._no_match_streak = 0
        self._macro_fail_streak = 0
        self._last_app_hint = ""

    @staticmethod
    def _task_state_snapshot(state: dict[str, Any]) -> dict[str, int | bool | float]:
        return {
            "elements0_streak": int(state.get("elements0_streak", 0)),
            "vlm_calls_this_task": int(state.get("vlm_calls_this_task", 0)),
            "local_boost_done": bool(state.get("local_boost_done", False)),
            "last_vlm_ts": float(state.get("last_vlm_ts", 0.0)),
        }

    def _collect_vlm_triggers(
        self,
        *,
        state: dict[str, Any],
        observation: Observation,
        elem_count: int,
        selector_status: str | None,
        macro_failed: bool,
    ) -> list[str]:
        thresholds = getattr(self.config, "vlm_trigger_thresholds", None)
        elements_min = int(getattr(thresholds, "elements_min", 3))
        streak_threshold = int(getattr(thresholds, "ambiguous_streak", 2))
        macro_fail_threshold = int(getattr(thresholds, "macro_fail_streak", 2))
        ui_change_threshold = float(getattr(thresholds, "ui_change_threshold", 0.15))

        if elem_count == 0:
            state["elements0_streak"] = int(state.get("elements0_streak", 0)) + 1
        else:
            state["elements0_streak"] = 0

        if selector_status == "ambiguous":
            self._ambiguous_streak += 1
        else:
            self._ambiguous_streak = max(0, self._ambiguous_streak - 1)
        if selector_status == "no_match":
            self._no_match_streak += 1
        else:
            self._no_match_streak = max(0, self._no_match_streak - 1)
        if macro_failed:
            self._macro_fail_streak += 1
        else:
            self._macro_fail_streak = max(0, self._macro_fail_streak - 1)

        reasons: list[str] = []
        if elem_count < elements_min:
            reasons.append("elements_below_min")
        if max(self._ambiguous_streak, self._no_match_streak) >= streak_threshold:
            reasons.append("selector_streak")
        if self._macro_fail_streak >= macro_fail_threshold:
            reasons.append("macro_fail_streak")
        app_hint = (observation.app_hint or "").strip()
        app_changed = bool(self._last_app_hint) and app_hint != self._last_app_hint
        if app_changed:
            reasons.append("app_hint_changed")
        if float(observation.ui_change_score) > ui_change_threshold:
            reasons.append("ui_change_high")
        self._last_app_hint = app_hint
        return reasons

    def should_call_vlm(
        self,
        *,
        state: dict[str, Any],
        triggers: list[str],
        now_ts: float,
        elements_count: int,
    ) -> tuple[bool, str]:
        if not triggers:
            return False, "trigger_not_met"

        if "vlm_calls_this_task" not in state and "vlm_calls" in state:
            state["vlm_calls_this_task"] = int(state.get("vlm_calls", 0))
        if "last_vlm_ts" not in state and "last_vlm_time" in state:
            state["last_vlm_ts"] = float(state.get("last_vlm_time", 0.0))

        max_per_task = int(getattr(self.config, "vlm_max_per_task", 8))
        if int(state.get("vlm_calls_this_task", 0)) >= max_per_task:
            return False, "vlm_budget_exceeded"

        cooldown_s = float(getattr(self.config, "vlm_cooldown_s", 10))
        force_allow_streak = int(getattr(self.config, "vlm_force_allow_if_elements0_streak", 3))
        requires_local_boost = bool(getattr(self.config, "vlm_force_allow_requires_local_boost", True))
        local_boost_ready = bool(state.get("local_boost_done", False) or state.get("local_boost_executed", False))
        force_break_cooldown = (
            elements_count == 0
            and int(state.get("elements0_streak", 0)) >= force_allow_streak
            and (local_boost_ready if requires_local_boost else True)
        )
        last_vlm = float(state.get("last_vlm_ts", 0.0))
        if (now_ts - last_vlm) < cooldown_s and not force_break_cooldown:
            return False, "cooldown"
        if force_break_cooldown:
            return True, "force_break_cooldown"
        return True, "ok"

    def _write_vlm_decision(
        self,
        *,
        step: int,
        artifacts_dir: str,
        trigger: str | None,
        reason: str,
        roi: tuple[float, float, float, float] | None,
        cache_hit: bool,
        state: dict[str, Any],
    ) -> None:
        if step <= 0:
            return
        self._write_step_file(
            step,
            "vlm_request.json",
            json.dumps(
                {
                    "trigger": trigger,
                    "reason": reason,
                    "deny_reason": reason if reason in {"cooldown", "vlm_budget_exceeded"} else None,
                    "allowed": reason not in {
                        "allow_vlm_false",
                        "missing_observation",
                        "missing_ui_extractor",
                        "trigger_not_met",
                        "cooldown",
                        "vlm_budget_exceeded",
                        "vlm_error",
                    },
                    "roi": roi,
                    "cache_hit": cache_hit,
                    "vlm_image_max_side": int(getattr(self.config, "vlm_image_max_side", 1280)),
                    "vlm_jpeg_quality": int(getattr(self.config, "vlm_jpeg_quality", 80)),
                    "cooldown_s": float(getattr(self.config, "vlm_cooldown_s", 10)),
                    "max_per_task": int(getattr(self.config, "vlm_max_per_task", 8)),
                    "state": self._task_state_snapshot(state),
                },
                ensure_ascii=False,
                indent=2,
            ),
            artifacts_dir,
        )

    def request_vlm(
        self,
        *,
        allow_vlm: bool,
        observation: Observation | None,
        frame_bgr: Any | None,
        ui_extractor: Any | None,
        local_elements: list[UIElement] | None = None,
        selector_status: str | None = None,
        macro_failed: bool = False,
        local_boost_done: bool = False,
        roi: tuple[float, float, float, float] | None = None,
        step: int = 0,
        artifacts_dir: str = "artifacts",
        vlm_state: dict[str, Any] | None = None,
    ) -> tuple[list[UIElement], dict[str, Any]]:
        state = self._task_state
        if isinstance(vlm_state, dict):
            state.update({k: v for k, v in vlm_state.items() if k in state})
        state["local_boost_done"] = bool(local_boost_done or state.get("local_boost_done", False))

        meta: dict[str, Any] = {
            "used": False,
            "trigger": None,
            "reason": "trigger_not_met",
            "roi": roi,
            "cache_hit": False,
            "error": None,
            "state": self._task_state_snapshot(state),
        }

        if not allow_vlm:
            meta["reason"] = "allow_vlm_false"
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=None,
                reason=meta["reason"],
                roi=roi,
                cache_hit=False,
                state=state,
            )
            return [], meta
        if observation is None or frame_bgr is None:
            meta["reason"] = "missing_observation"
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=None,
                reason=meta["reason"],
                roi=roi,
                cache_hit=False,
                state=state,
            )
            return [], meta
        if ui_extractor is None:
            meta["reason"] = "missing_ui_extractor"
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=None,
                reason=meta["reason"],
                roi=roi,
                cache_hit=False,
                state=state,
            )
            return [], meta

        elem_count = len(local_elements) if local_elements is not None else len(observation.elements)
        reasons = self._collect_vlm_triggers(
            state=state,
            observation=observation,
            elem_count=elem_count,
            selector_status=selector_status,
            macro_failed=macro_failed,
        )
        now = time.time()
        allowed, gate_reason = self.should_call_vlm(
            state=state,
            triggers=reasons,
            now_ts=now,
            elements_count=elem_count,
        )
        if not allowed:
            meta["trigger"] = ",".join(reasons) if reasons else None
            meta["reason"] = gate_reason
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=meta["trigger"],
                reason=meta["reason"],
                roi=roi,
                cache_hit=False,
                state=state,
            )
            return [], meta
        if gate_reason == "force_break_cooldown":
            reasons.append("force_break_cooldown")

        if roi is None:
            roi_hint = getattr(self.config, "app_hint_roi", (0.0, 0.0, 1.0, 0.12))
            if isinstance(roi_hint, (list, tuple)) and len(roi_hint) == 4:
                x1, y1, x2, y2 = [float(v) for v in roi_hint]
                if x2 <= 1.0 and y2 <= 1.0:
                    roi = (
                        x1 * float(observation.screen_w),
                        y1 * float(observation.screen_h),
                        x2 * float(observation.screen_w),
                        y2 * float(observation.screen_h),
                    )
                else:
                    roi = (x1, y1, x2, y2)

        extractor = ui_extractor.extract_ui_elements if hasattr(ui_extractor, "extract_ui_elements") else ui_extractor
        try:
            result = extractor(
                frame_bgr,
                ui_mode=getattr(self.config, "ui_mode", "local"),
                max_elements=int(getattr(self.config, "max_elements", 128)),
                min_text_conf=float(getattr(self.config, "min_text_conf", 0.4)),
                min_elem_conf=float(getattr(self.config, "min_elem_conf", 0.3)),
                element_merge_iou=float(getattr(self.config, "element_merge_iou", 0.5)),
                txt_trunc_policy=str(getattr(self.config, "txt_trunc_policy", "head12~tail6")),
                allow_vlm=True,
                vlm_state=state,
                vlm_roi=roi,
                ui_change_score=float(observation.ui_change_score),
                vlm_gate_passed=True,
                vlm_gate_reason=",".join(reasons),
                artifacts_dir=artifacts_dir,
                vlm_image_max_side=int(getattr(self.config, "vlm_image_max_side", 1280)),
                vlm_jpeg_quality=int(getattr(self.config, "vlm_jpeg_quality", 80)),
                return_meta=True,
            )
            elements: list[UIElement]
            extractor_meta: dict[str, Any]
            if isinstance(result, tuple):
                elements = result[0]
                extractor_meta = result[1] if isinstance(result[1], dict) else {}
            else:
                elements = result
                extractor_meta = {}
            vlm_meta = extractor_meta.get("vlm", extractor_meta) if isinstance(extractor_meta, dict) else {}
            used = bool(vlm_meta.get("called", False)) or any(getattr(elem, "source", "") == "vlm" for elem in elements)
            if used:
                state["vlm_calls_this_task"] = int(state.get("vlm_calls_this_task", 0)) + 1
                state["last_vlm_ts"] = now
                self._ambiguous_streak = max(0, self._ambiguous_streak - 1)
                self._no_match_streak = max(0, self._no_match_streak - 1)
                self._macro_fail_streak = max(0, self._macro_fail_streak - 1)
            meta.update(
                {
                    "used": used,
                    "trigger": ",".join(reasons),
                    "reason": vlm_meta.get("reason", "ok" if used else "vlm_not_called_by_extractor"),
                    "roi": roi,
                    "cache_hit": bool(vlm_meta.get("cache_hit", False)),
                    "state": self._task_state_snapshot(state),
                }
            )
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=meta.get("trigger"),
                reason=str(meta.get("reason", "ok")),
                roi=roi,
                cache_hit=bool(meta.get("cache_hit", False)),
                state=state,
            )
            return elements, meta
        except Exception as exc:
            meta["reason"] = "vlm_error"
            meta["error"] = str(exc)
            self._write_vlm_decision(
                step=step,
                artifacts_dir=artifacts_dir,
                trigger=",".join(reasons),
                reason=meta["reason"],
                roi=roi,
                cache_hit=False,
                state=state,
            )
            return [], meta

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def repair_json(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        obj = Router._extract_first_json_object(cleaned)
        if obj is not None:
            cleaned = obj
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        if '"' not in cleaned and "'" in cleaned:
            cleaned = cleaned.replace("'", '"')
        return cleaned

    @classmethod
    def parse_strict_json(cls, text: str) -> tuple[dict[str, Any], bool]:
        raw = text.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data, False
        except Exception:
            pass

        first = cls._extract_first_json_object(raw)
        if first is not None:
            try:
                data = json.loads(first)
                if isinstance(data, dict):
                    return data, False
            except Exception:
                pass

        repaired = cls.repair_json(raw)
        data = json.loads(repaired)
        if not isinstance(data, dict):
            raise ValueError("planner output json root must be object")
        return data, True

    @staticmethod
    def _element_signature(element: UIElement) -> str:
        text_norm = _normalize_text(element.text or element.label or "")
        payload = {
            "sid": element.stable_id,
            "role": element.role,
            "text": text_norm,
            "bbox": [round(float(v), 2) for v in element.bbox],
            "conf": round(float(element.confidence), 3),
            "src": element.source,
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _elements_topk(self, obs: Observation, k: int) -> list[UIElement]:
        cursor = obs.cursor_xy

        def score(e: UIElement) -> tuple[float, float]:
            conf = float(e.confidence)
            if cursor is None:
                return (conf, 0.0)
            cx = (float(e.bbox[0]) + float(e.bbox[2])) * 0.5
            cy = (float(e.bbox[1]) + float(e.bbox[3])) * 0.5
            dist = ((cx - float(cursor[0])) ** 2 + (cy - float(cursor[1])) ** 2) ** 0.5
            return (conf, -dist)

        ranked = sorted(obs.elements, key=score, reverse=True)
        return ranked[: max(1, int(k))]

    def _elements_delta(
        self,
        obs: Observation,
        max_delta: int,
        memory: dict[str, Any] | None = None,
    ) -> tuple[list[UIElement], dict[str, Any]]:
        changed: list[tuple[str, UIElement, bool]] = []
        current: dict[str, str] = {}
        prev_keys = set(self._last_elements_sig.keys())
        target_id = None
        if isinstance(memory, dict):
            last_action = memory.get("last_action")
            if isinstance(last_action, dict):
                target = None
                if isinstance(last_action.get("payload"), dict):
                    target = last_action.get("payload", {}).get("target")
                target = target or last_action.get("target")
                if isinstance(target, dict) and target.get("by") == "id":
                    target_id = str(target.get("id") or "").strip() or None

        for idx, elem in enumerate(obs.elements):
            key = elem.stable_id or f"raw_{elem.raw_id}_{idx}"
            sig = self._element_signature(elem)
            current[key] = sig
            if self._last_elements_sig.get(key) != sig:
                is_new = key not in self._last_elements_sig
                changed.append((key, elem, is_new))
        disappeared = max(0, len(prev_keys - set(current.keys())))

        cursor = obs.cursor_xy

        def score(item: tuple[str, UIElement, bool]) -> tuple[int, int, int, float, float]:
            key, elem, is_new = item
            target_bonus = 1 if target_id and elem.stable_id == target_id else 0
            near_cursor_bonus = 0
            neg_dist = 0.0
            if cursor is not None:
                cx = (float(elem.bbox[0]) + float(elem.bbox[2])) * 0.5
                cy = (float(elem.bbox[1]) + float(elem.bbox[3])) * 0.5
                dist = ((cx - float(cursor[0])) ** 2 + (cy - float(cursor[1])) ** 2) ** 0.5
                near_cursor_bonus = 1 if dist <= 180.0 else 0
                neg_dist = -dist
            high_conf_bonus = 1 if float(elem.confidence) >= 0.70 else 0
            return (target_bonus, near_cursor_bonus, high_conf_bonus + int(is_new), float(elem.confidence), neg_dist)

        changed_sorted = [item[1] for item in sorted(changed, key=score, reverse=True)]
        self._last_elements_sig = current
        kept = changed_sorted[: max(1, int(max_delta))]
        stats = {
            "raw_changed_count": len(changed_sorted),
            "kept_count": len(kept),
            "disappeared_count": disappeared,
            "sampling_rule": "near_target_or_cursor_then_new_high_conf_then_other",
        }
        return kept, stats

    @staticmethod
    def _norm_grid(value: float, bound: int) -> int:
        if bound <= 1:
            return 0
        v = max(0.0, min(float(bound - 1), float(value)))
        return int(round((v / float(bound - 1)) * 1000.0))

    def _planner_text(self, element: UIElement) -> str:
        policy = str(getattr(self.config, "txt_trunc_policy", "head12~tail6"))
        return _truncate_text(element.text or element.label or "", policy)

    def _serialize_plain_elements(self, elements: list[UIElement], max_items: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, elem in enumerate(elements[: max(1, int(max_items))]):
            x1, y1, x2, y2 = elem.bbox
            out.append(
                {
                    "stable_id": elem.stable_id,
                    "raw_id": int(elem.raw_id),
                    "role": elem.role,
                    "text": self._planner_text(elem),
                    "label": self._planner_text(elem),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(elem.confidence),
                    "source": elem.source,
                }
            )
        return out

    def _pack_obs(self, obs: Observation, elements: list[UIElement]) -> dict[str, Any]:
        role_map = {"text": "t", "button": "b", "input": "i", "icon": "n", "key": "k", "toggle": "g", "unknown": "u"}
        source_map = {"ocr": "o", "heuristic": "h", "vlm": "v"}
        use_minify = bool(getattr(self.config, "obs_key_minify", True))
        packed_elements_arr: list[list[Any]] = []
        packed_elements_obj: list[dict[str, Any]] = []
        for idx, elem in enumerate(elements):
            x1, y1, x2, y2 = elem.bbox
            bx1 = self._norm_grid(x1, obs.screen_w)
            by1 = self._norm_grid(y1, obs.screen_h)
            bx2 = self._norm_grid(x2, obs.screen_w)
            by2 = self._norm_grid(y2, obs.screen_h)
            bw = max(0, bx2 - bx1)
            bh = max(0, by2 - by1)
            text = self._planner_text(elem)
            packed_elements_arr.append(
                [
                    elem.stable_id or f"r{elem.raw_id}_{idx}",
                    role_map.get(elem.role, "u"),
                    text,
                    bx1,
                    by1,
                    bw,
                    bh,
                    int(round(float(elem.confidence) * 100.0)),
                    source_map.get(elem.source, "h"),
                ]
            )
            packed_elements_obj.append(
                {
                    "i": elem.stable_id or f"r{elem.raw_id}_{idx}",
                    "r": role_map.get(elem.role, "u"),
                    "t": text,
                    "b": [
                        bx1,
                        by1,
                        bw,
                        bh,
                    ],
                    "c": int(round(float(elem.confidence) * 100.0)),
                    "s": source_map.get(elem.source, "h"),
                }
            )

        cursor = None
        if obs.cursor_xy is not None:
            cursor = [
                self._norm_grid(obs.cursor_xy[0], obs.screen_w),
                self._norm_grid(obs.cursor_xy[1], obs.screen_h),
                int(round(float(obs.cursor_conf) * 100.0)),
                (obs.cursor_type or "unknown")[:12],
            ]
        if use_minify:
            return {
                "v": 1,
                "enc": "packed",
                "m": 1,
                "c": cursor,
                "a": _truncate_text(obs.app_hint or "", "head12~tail6"),
                "k": bool(obs.keyboard_mode),
                "u": round(float(obs.ui_change_score), 4),
                "e": packed_elements_arr,
            }
        return {
            "version": 1,
            "encoding": "packed",
            "obs_key_minify": False,
            "cursor": cursor,
            "app_hint": _truncate_text(obs.app_hint or "", "head12~tail6"),
            "keyboard_mode": bool(obs.keyboard_mode),
            "ui_change_score": round(float(obs.ui_change_score), 4),
            "elements": packed_elements_obj,
        }

    def build_planner_observation(self, obs: Observation, *, mode: str, elements: list[UIElement]) -> dict[str, Any]:
        max_items = int(getattr(self.config, "elements_delta_max", 10))
        element_slice = elements[: max(1, max_items)]
        if str(getattr(self.config, "obs_model_encoding", "packed")).lower() == "packed":
            return self._pack_obs(obs, element_slice)
        plain_elements = self._serialize_plain_elements(element_slice, max_items=max_items)
        if mode == "full":
            return {
                "cursor": obs.cursor_xy,
                "cursor_conf": float(obs.cursor_conf),
                "cursor_type": obs.cursor_type,
                "app_hint": _truncate_text(obs.app_hint or "", "head12~tail6"),
                "keyboard_mode": bool(obs.keyboard_mode),
                "keyboard_roi": obs.keyboard_roi,
                "ui_change_score": float(obs.ui_change_score),
                "elements_topk": plain_elements,
            }
        return {
            "cursor": obs.cursor_xy,
            "cursor_conf": float(obs.cursor_conf),
            "cursor_type": obs.cursor_type,
            "ui_change_score": float(obs.ui_change_score),
            "elements_delta": plain_elements,
        }

    def prune_delta_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, int]]:
        pruned = dict(payload)
        removed_keys: list[str] = []
        removed_sizes: dict[str, int] = {}

        for key in ("task", "constraints", "history", "system", "memory", "full_elements"):
            if key in pruned:
                removed_keys.append(key)
                removed_sizes[key] = _estimate_size(pruned.get(key))
                pruned.pop(key, None)

        allowed = _DELTA_ALLOWED_KEYS
        for key in list(pruned.keys()):
            if key in allowed:
                continue
            removed_keys.append(key)
            removed_sizes[key] = _estimate_size(pruned.get(key))
            pruned.pop(key, None)

        if isinstance(pruned.get("ms"), dict):
            macro_state = dict(pruned["ms"])
            for noisy in ("task", "constraints", "history", "long_text"):
                if noisy in macro_state and _estimate_size(macro_state[noisy]) > 128:
                    removed_keys.append(f"ms.{noisy}")
                    removed_sizes[f"ms.{noisy}"] = _estimate_size(macro_state[noisy])
                    macro_state.pop(noisy, None)
            pruned["ms"] = macro_state

        return pruned, removed_keys, removed_sizes

    def _write_last_delta_clip(
        self,
        *,
        removed_keys: list[str],
        kept_keys: list[str],
        source: str = "planner",
    ) -> None:
        if not removed_keys:
            return
        artifacts_dir = str(getattr(self.config, "artifacts_dir", "artifacts"))
        payload = {
            "time": time.time(),
            "mode": "delta",
            "source": source,
            "reason": "delta_lock",
            "removed_keys": list(removed_keys),
            "kept_keys": list(kept_keys),
        }
        try:
            p = Path(artifacts_dir) / "last_delta_clip.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("failed to write last_delta_clip.json: %s", exc)

    def _build_payload(
        self,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        obs_mode: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool, str]:
        max_delta = int(getattr(self.config, "elements_delta_max", 10))
        clipped = False
        clip_reason = ""

        if obs_mode == "full":
            topk = self._elements_topk(obs, max_delta)
            planner_obs = self.build_planner_observation(obs, mode="full", elements=topk)
            payload = {
                "task": task,
                "constraints": [
                    "pure_mouse_only",
                    "no_system_api_injection",
                    "type_text_requires_osk",
                    "click_scroll_should_verify",
                ],
                "obs": planner_obs,
                "memory": {
                    "last_action": memory.get("last_action"),
                    "last_verify": memory.get("last_verify"),
                    "macro_state": memory.get("macro_state"),
                },
            }
            packed = self._pack_obs(obs, topk)
            digest_meta = {
                "mode": "full",
                "elements_topk": len(topk),
                "sampling_rule": "topk_by_conf_and_cursor_distance",
            }
        else:
            changed, delta_stats = self._elements_delta(obs, max_delta, memory=memory)
            planner_obs = self.build_planner_observation(obs, mode="delta", elements=changed)
            if str(getattr(self.config, "obs_model_encoding", "packed")).lower() == "packed":
                delta_elements = planner_obs.get("e", planner_obs.get("elements", []))
                delta_cursor = planner_obs.get("c")
            else:
                delta_elements = planner_obs.get("elements_delta", planner_obs.get("elements", []))
                delta_cursor = {
                    "xy": list(obs.cursor_xy) if obs.cursor_xy is not None else None,
                    "cf": float(obs.cursor_conf),
                    "t": obs.cursor_type,
                }
            payload_raw = {
                "task": task,
                "constraints": [
                    "pure_mouse_only",
                    "no_system_api_injection",
                    "type_text_requires_osk",
                ],
                "system": "delta_lock",
                "full_elements": obs.elements,
                "la": memory.get("last_action"),
                "lv": memory.get("last_verify"),
                "c": delta_cursor,
                "u": obs.ui_change_score,
                "ed": delta_elements,
                "ms": memory.get("macro_state"),
            }
            payload, removed_keys, removed_sizes = self.prune_delta_payload(payload_raw)
            if removed_keys:
                clipped = True
                clip_reason = f"removed:{','.join(removed_keys)}"
                self._write_last_delta_clip(
                    removed_keys=removed_keys,
                    kept_keys=sorted(payload.keys()),
                    source="planner",
                )
            packed = payload
            digest_meta = {
                "mode": "delta",
                "elements_delta": len(changed),
                "sampling_rule": "changed_elements_limit_elements_delta_max",
                "elements_delta_sampling": delta_stats,
                "delta_prune": {
                    "clipped": bool(removed_keys),
                    "reason": "delta_lock",
                    "allowlist": sorted(_DELTA_ALLOWED_KEYS),
                    "kept_keys": sorted(payload.keys()),
                    "removed_keys": removed_keys,
                    "removed_sizes": removed_sizes,
                    "dropped_bytes_estimate": int(sum(removed_sizes.values())) if removed_sizes else 0,
                },
            }

        digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        digest_meta["last_sent_obs_digest"] = self.last_sent_obs_digest
        digest_meta["new_obs_digest"] = digest
        digest_meta["clipped"] = clipped
        digest_meta["clip_reason"] = clip_reason
        if obs_mode == "delta" and "delta_prune" not in digest_meta:
            digest_meta["delta_prune"] = {"clipped": False, "removed_keys": [], "removed_sizes": {}}
        self.last_sent_obs_digest = digest
        return payload, packed, digest_meta, clipped, clip_reason

    def build_planner_payload(
        self,
        *,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        obs_mode: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool, str]:
        return self._build_payload(task=task, obs=obs, memory=memory, obs_mode=obs_mode)

    def _write_step_file(self, step: int, suffix: str, content: str, artifacts_dir: str) -> None:
        p = Path(artifacts_dir) / f"step_{step:04d}_{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def _write_router_artifacts(
        self,
        *,
        step: int,
        obs_mode: str,
        payload: dict[str, Any],
        packed: dict[str, Any],
        digest_meta: dict[str, Any],
        raw: str,
        parsed: dict[str, Any],
        artifacts_dir: str,
    ) -> None:
        planner_in_payload = payload
        if obs_mode == "delta":
            planner_in_payload = {
                "last_action": payload.get("la"),
                "last_verify": payload.get("lv"),
                "cursor": payload.get("c"),
                "ui_change_score": payload.get("u"),
                "elements_delta": payload.get("ed"),
                "macro_state": payload.get("ms"),
            }
        self._write_step_file(
            step,
            f"planner_in_{obs_mode}.json",
            json.dumps(planner_in_payload, ensure_ascii=False, indent=2),
            artifacts_dir,
        )
        self._write_step_file(
            step,
            "planner_obs_packed.json",
            json.dumps(packed, ensure_ascii=False, indent=2),
            artifacts_dir,
        )
        self._write_step_file(
            step,
            "obs_digest.json",
            json.dumps(digest_meta, ensure_ascii=False, indent=2),
            artifacts_dir,
        )
        if obs_mode == "delta":
            delta_prune = digest_meta.get("delta_prune")
            if isinstance(delta_prune, dict) and delta_prune.get("clipped"):
                self._write_step_file(
                    step,
                    "delta_prune.json",
                    json.dumps(delta_prune, ensure_ascii=False, indent=2),
                    artifacts_dir,
                )
        self._write_step_file(step, "planner_llm_raw.txt", raw, artifacts_dir)
        self._write_step_file(
            step,
            "planner_out.json",
            json.dumps(parsed, ensure_ascii=False, indent=2),
            artifacts_dir,
        )

    def _build_system_prompt(self, debug_reasoning: bool) -> str:
        base = (
            "You are huweibot planner. Output strict JSON only.\n"
            "No markdown/code fences.\n"
            "Pure mouse only: never inject keyboard/system APIs.\n"
            "TYPE_TEXT must use method='osk'.\n"
            "CLICK/SCROLL should include verify unless action is WAIT/DONE.\n"
            "Use action types: CLICK, TYPE_TEXT, OPEN_APP, OPEN_OSK, SCROLL, WAIT, DONE.\n"
        )
        encoding = str(getattr(self.config, "obs_model_encoding", "plain")).lower()
        key_minify = bool(getattr(self.config, "obs_key_minify", False))
        if encoding == "packed" or key_minify:
            base += (
                "Packed keys: c=cursor,u=ui_change_score,la=last_action,lv=last_verify,ed=elements_delta,ms=macro_state.\n"
                "Cursor: c=[x,y,cf,t]. Elements: e=[[id,r,txt,bx,by,bw,bh,cf,src],...].\n"
                "role: b/i/t/k/g/n/u=button/input/text/key/toggle/icon/unknown. src: o/h/v=ocr/heuristic/vlm.\n"
                "bbox bx/by/bw/bh are 0..999 grid ints relative to screen.\n"
            )
        if debug_reasoning:
            base += "reasoning_hint allowed <=120 chars.\n"
        else:
            base += "Do not output reasoning_hint.\n"
        return base

    def _fallback_allowed(self) -> bool:
        return bool(getattr(self.config, "allow_fallback_to_dummy", False))

    def _should_fallback(self, provider: Provider, exc: Exception) -> bool:
        if provider.provider_type == "dummy":
            return False
        if not self._fallback_allowed():
            return False
        return isinstance(exc, ProviderError)

    def request_plan(
        self,
        *,
        task: str,
        obs: Observation,
        memory: dict[str, Any],
        step: int,
        obs_mode: str = "delta",
        debug_reasoning: bool = False,
        artifacts_dir: str = "artifacts",
    ) -> tuple[NextStep, dict[str, Any]]:
        provider = self._choose_provider()
        payload, packed, digest_meta, _clipped, _clip_reason = self.build_planner_payload(
            task=task,
            obs=obs,
            memory=memory,
            obs_mode=obs_mode,
        )
        if obs_mode == "full":
            prompt_obj = {"task": task, "obs_mode": obs_mode, "input": payload}
        else:
            prompt_obj = {"obs_mode": obs_mode, "input": payload}
        prompt = json.dumps(prompt_obj, ensure_ascii=False)
        digest_meta["prompt_chars"] = len(prompt)
        digest_meta["packed_elements"] = len(packed.get("elements", packed.get("e", packed.get("ed", []))))
        system = self._build_system_prompt(debug_reasoning)
        request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=prompt),
            ],
            model=getattr(provider, "model", None),
            stream=False,
            metadata={"step": step, "task": task, "obs_mode": obs_mode},
        )
        response: ProviderResponse
        try:
            response = provider.complete(request)
        except Exception as exc:
            if self._should_fallback(provider, exc):
                fallback = self._get_dummy_provider()
                if fallback is None:
                    raise RuntimeError(
                        f"provider '{provider.name}' failed and fallback dummy is unavailable: {exc}"
                    ) from exc
                LOGGER.warning(
                    "provider %s failed (%s), fallback to dummy because allow_fallback_to_dummy=true",
                    provider.name,
                    _redact_sensitive(str(exc)),
                )
                provider = fallback
                request.model = getattr(provider, "model", None)
                response = provider.complete(request)
            else:
                raise RuntimeError(f"provider '{provider.name}' request failed: {exc}") from exc
        raw = response.text
        repair_used = False
        try:
            parsed, repair_used = self.parse_strict_json(raw)
            step_obj = parse_obj(NextStep, parsed)
            if not debug_reasoning:
                step_obj.reasoning_hint = None
            elif step_obj.reasoning_hint is not None and len(step_obj.reasoning_hint) > 120:
                step_obj.reasoning_hint = step_obj.reasoning_hint[:120]
            parsed = _model_dump(step_obj)
            self._fail_counts[provider.name] = 0
        except Exception as exc:
            self._fail_counts[provider.name] = self._fail_counts.get(provider.name, 0) + 1
            raise RuntimeError(f"planner_json_parse_failed: {exc}") from exc

        self._write_router_artifacts(
            step=step,
            obs_mode=obs_mode,
            payload=payload,
            packed=packed,
            digest_meta=digest_meta,
            raw=raw,
            parsed=parsed,
            artifacts_dir=artifacts_dir,
        )

        meta = {
            "task": task,
            "provider": provider.name,
            "model": getattr(provider, "model", provider.name),
            "capabilities": provider.capability_dict(),
            "repair_used": repair_used,
            "obs_mode": obs_mode,
            "prompt_chars": len(prompt),
            "packed_elements": len(packed.get("elements", packed.get("e", packed.get("ed", [])))),
            "usage": _model_dump(response.usage),
            "finish_reason": response.finish_reason,
            "error": None,
            "output": parsed,
        }
        return step_obj, meta
