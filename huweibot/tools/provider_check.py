from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from huweibot.agent.router import (
    AnthropicProvider,
    ChatMessage,
    DummyProvider,
    GeminiProvider,
    MissingAPIKeyError,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
    ProviderRequest,
    _NetworkProvider,
)


def _json_print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _build_provider(name: str) -> Provider:
    provider_name = name.strip().lower().replace("-", "_")
    if provider_name == "dummy":
        return DummyProvider("dummy")
    raise RuntimeError("provider args required")


def _build_provider_from_args(args: argparse.Namespace) -> Provider:
    provider_name = str(args.provider).strip().lower().replace("-", "_")
    if provider_name == "dummy":
        return _build_provider("dummy")
    if provider_name == "openai":
        return OpenAIProvider(
            name="openai_check",
            model=args.model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
            base_url=(args.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
            api_key=(args.api_key if args.api_key is not None else os.getenv("OPENAI_API_KEY")),
            timeout_s=float(args.timeout_s),
            max_retries=int(args.max_retries),
        )
    if provider_name == "anthropic":
        return AnthropicProvider(
            name="anthropic_check",
            model=args.model or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            base_url=(args.base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/"),
            api_key=(args.api_key if args.api_key is not None else os.getenv("ANTHROPIC_API_KEY")),
            timeout_s=float(args.timeout_s),
            max_retries=int(args.max_retries),
        )
    if provider_name == "gemini":
        return GeminiProvider(
            name="gemini_check",
            model=args.model or os.getenv("GEMINI_MODEL") or "gemini-1.5-pro",
            base_url=(args.base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com").rstrip("/"),
            api_key=(args.api_key if args.api_key is not None else os.getenv("GEMINI_API_KEY")),
            timeout_s=float(args.timeout_s),
            max_retries=int(args.max_retries),
        )
    if provider_name == "openai_compat":
        headers = {}
        query = {}
        if args.headers_json:
            headers = json.loads(args.headers_json)
            if not isinstance(headers, dict):
                raise RuntimeError("--headers-json must be a JSON object")
        if args.query_json:
            query = json.loads(args.query_json)
            if not isinstance(query, dict):
                raise RuntimeError("--query-json must be a JSON object")
        return OpenAICompatibleProvider(
            name="openai_compat_check",
            model=args.model or os.getenv("OPENAI_COMPAT_MODEL") or "",
            base_url=args.base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or "",
            api_key=(args.api_key if args.api_key is not None else os.getenv("OPENAI_COMPAT_API_KEY")),
            profile=args.profile or os.getenv("OPENAI_COMPAT_PROFILE") or "generic",
            api_style=args.api_style or os.getenv("OPENAI_COMPAT_API_STYLE") or "auto",
            extra_headers=headers or json.loads(os.getenv("OPENAI_COMPAT_HEADERS_JSON", "{}")),
            extra_query=query or json.loads(os.getenv("OPENAI_COMPAT_QUERY_JSON", "{}")),
            azure_deployment=args.azure_deployment or os.getenv("OPENAI_COMPAT_AZURE_DEPLOYMENT"),
            azure_api_version=args.azure_api_version or os.getenv("OPENAI_COMPAT_AZURE_API_VERSION") or "2024-10-21",
            openrouter_referer=args.or_referer or os.getenv("OPENAI_COMPAT_OR_REFERER"),
            openrouter_title=args.or_title or os.getenv("OPENAI_COMPAT_OR_TITLE"),
            openrouter_app_id=args.or_app_id or os.getenv("OPENAI_COMPAT_OR_APP_ID"),
            timeout_s=float(args.timeout_s),
            max_retries=int(args.max_retries),
        )
    raise RuntimeError(f"unsupported provider: {provider_name}")


def _build_request(args: argparse.Namespace, provider: Provider) -> ProviderRequest:
    messages = []
    if args.system:
        messages.append(ChatMessage(role="system", content=str(args.system)))
    messages.append(ChatMessage(role="user", content=str(args.message)))
    return ProviderRequest(
        messages=messages,
        model=(args.model or getattr(provider, "model", None)),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stream=False,
        metadata={"source": "provider_check"},
    )


def run_dry(provider: Provider, request: ProviderRequest) -> int:
    if isinstance(provider, DummyProvider):
        _json_print(
            {
                "provider": "dummy",
                "mode": "dry-run",
                "network": False,
                "message": "huweibot dummy provider uses local response only",
                "messages": len(request.messages),
            }
        )
        return 0
    if isinstance(provider, _NetworkProvider):
        try:
            summary = provider.describe_http_request(request)
        except MissingAPIKeyError as exc:
            print(f"[error] {exc}")
            return 2
        summary["mode"] = "dry-run"
        summary["network"] = False
        _json_print(summary)
        return 0
    print("[error] unknown provider type")
    return 2


def run_ping(provider: Provider, request: ProviderRequest) -> int:
    try:
        response = provider.complete(request)
    except MissingAPIKeyError as exc:
        print(f"[error] {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[error] provider ping failed: {exc}")
        return 3
    payload = {
        "provider": provider.provider_type,
        "mode": "ping",
        "text": response.text,
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "finish_reason": response.finish_reason,
    }
    _json_print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="huweibot provider health check")
    parser.add_argument("--provider", default="dummy", choices=["dummy", "openai", "anthropic", "gemini", "openai_compat"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-style", default=None, choices=["auto", "responses", "chat_completions"])
    parser.add_argument("--profile", default=None, choices=["generic", "azure", "openrouter", "together", "groq"])
    parser.add_argument("--azure-deployment", default=None)
    parser.add_argument("--azure-api-version", default=None)
    parser.add_argument("--or-referer", default=None)
    parser.add_argument("--or-title", default=None)
    parser.add_argument("--or-app-id", default=None)
    parser.add_argument("--headers-json", default=None, help="Extra headers JSON object")
    parser.add_argument("--query-json", default=None, help="Extra query JSON object")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--message", default="Say 'pong'.")
    parser.add_argument("--system", default="Return plain text.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--ping", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provider = _build_provider_from_args(args)
    request = _build_request(args, provider)
    if args.dry_run:
        return run_dry(provider, request)
    return run_ping(provider, request)


if __name__ == "__main__":
    raise SystemExit(main())
