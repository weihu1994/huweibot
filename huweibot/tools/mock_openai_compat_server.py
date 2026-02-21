from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

Mode = Literal[
    "chat",
    "responses",
    "both",
    "chat_list_content",
    "no_usage",
    "auth_required",
    "azure_chat",
    "openrouter_chat",
    "openrouter_require_headers",
    "together_chat",
    "groq_chat",
]


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _chat_payload(mode: Mode) -> dict[str, Any]:
    if mode == "chat_list_content":
        content: Any = [{"type": "text", "text": "po"}, {"type": "text", "text": "ng"}]
    else:
        content = "pong"
    payload: dict[str, Any] = {
        "id": "chatcmpl-mock",
        "model": "mock-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if mode != "no_usage":
        payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    return payload


def _responses_payload(mode: Mode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "resp-mock",
        "model": "mock-model",
        "output_text": "pong",
        "status": "completed",
    }
    if mode != "no_usage":
        payload["usage"] = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
    return payload


def build_server(host: str, port: int, mode: Mode) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A003
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            data = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth_ok(self) -> bool:
            if mode != "auth_required":
                if mode == "azure_chat":
                    return self.headers.get("api-key", "") == "sk-test-SECRET"
                return True
            auth = self.headers.get("Authorization", "")
            return auth == "Bearer sk-test-SECRET"

        def _openrouter_headers_ok(self) -> bool:
            if mode != "openrouter_require_headers":
                return True
            return bool(self.headers.get("HTTP-Referer")) and bool(self.headers.get("X-Title"))

        def do_POST(self):  # noqa: N802
            if not self._auth_ok():
                self._send_json(401, {"error": {"message": "unauthorized"}})
                return
            if not self._openrouter_headers_ok():
                self._send_json(400, {"error": {"message": "missing openrouter headers"}})
                return

            if self.path.startswith("/openai/deployments/") and "/chat/completions" in self.path:
                if mode != "azure_chat":
                    self._send_json(404, {"error": {"message": "azure endpoint not available"}})
                    return
                self._send_json(200, _chat_payload(mode))
                return

            if self.path.startswith("/api/v1/chat/completions"):
                if mode not in {"openrouter_chat", "openrouter_require_headers"}:
                    self._send_json(404, {"error": {"message": "openrouter endpoint not available"}})
                    return
                self._send_json(200, _chat_payload(mode))
                return

            if self.path.startswith("/openai/v1/chat/completions"):
                if mode != "groq_chat":
                    self._send_json(404, {"error": {"message": "groq endpoint not available"}})
                    return
                self._send_json(200, _chat_payload(mode))
                return

            if self.path.startswith("/v1/chat/completions"):
                if mode in {"responses"}:
                    self._send_json(404, {"error": {"message": "chat endpoint not available"}})
                    return
                if mode == "together_chat":
                    self._send_json(200, _chat_payload(mode))
                    return
                if mode in {"chat", "both", "chat_list_content", "no_usage", "auth_required"}:
                    self._send_json(200, _chat_payload(mode))
                    return
                self._send_json(404, {"error": {"message": "together endpoint not available"}})
                return

            if self.path.startswith("/v1/responses"):
                if mode in {"chat", "chat_list_content", "together_chat", "azure_chat", "openrouter_chat", "openrouter_require_headers", "groq_chat"}:
                    self._send_json(404, {"error": {"message": "responses endpoint not available"}})
                    return
                self._send_json(200, _responses_payload(mode))
                return

            self._send_json(404, {"error": {"message": "unknown endpoint"}})

    return ThreadingHTTPServer((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible mock server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--mode",
        choices=[
            "chat",
            "responses",
            "both",
            "chat_list_content",
            "no_usage",
            "auth_required",
            "azure_chat",
            "openrouter_chat",
            "openrouter_require_headers",
            "together_chat",
            "groq_chat",
        ],
        default="both",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = build_server(str(args.host), int(args.port), args.mode)
    print(f"mock_openai_compat_server listening on http://{args.host}:{args.port} mode={args.mode}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
