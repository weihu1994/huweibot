from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from huweibot.tools.mock_openai_compat_server import build_server


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str


def _run_provider_check(args: Sequence[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "huweibot.tools.provider_check", *args]
    env = dict(os.environ)
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _ensure_pong(output: str) -> bool:
    return "pong" in output.lower()


def _run_case(mode: str, args: Sequence[str], *, expect_success: bool, needs_pong: bool = True) -> CaseResult:
    server = build_server("127.0.0.1", 18080, mode=mode)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        code, output = _run_provider_check(args)
        ok = (code == 0) if expect_success else (code != 0)
        if ok and expect_success and needs_pong:
            ok = _ensure_pong(output)
            if not ok:
                return CaseResult(name=f"{mode}:{' '.join(args)}", ok=False, detail="missing pong in output")
        return CaseResult(name=f"{mode}:{' '.join(args)}", ok=ok, detail=output.strip().splitlines()[-1] if output.strip() else "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def run_all() -> list[CaseResult]:
    base = "http://127.0.0.1:18080"
    results: list[CaseResult] = []

    results.append(
        _run_case(
            "chat",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "auto", "--ping"],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "chat",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "responses", "--ping"],
            expect_success=False,
            needs_pong=False,
        )
    )
    results.append(
        _run_case(
            "responses",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "responses", "--ping"],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "responses",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "chat_completions", "--ping"],
            expect_success=False,
            needs_pong=False,
        )
    )
    results.append(
        _run_case(
            "chat_list_content",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "chat_completions", "--ping"],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "no_usage",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "responses", "--ping"],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "auth_required",
            ["--provider", "openai_compat", "--base-url", base, "--model", "mock-model", "--api-style", "auto", "--ping"],
            expect_success=False,
            needs_pong=False,
        )
    )
    results.append(
        _run_case(
            "auth_required",
            [
                "--provider",
                "openai_compat",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "auto",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    return results


def run_profile_matrix() -> list[CaseResult]:
    base = "http://127.0.0.1:18080"
    results: list[CaseResult] = []

    results.append(
        _run_case(
            "azure_chat",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "azure",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "chat_completions",
                "--azure-deployment",
                "dep-a",
                "--azure-api-version",
                "2024-10-21",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "openrouter_chat",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "openrouter",
                "--base-url",
                f"{base}/api",
                "--model",
                "mock-model",
                "--api-style",
                "chat_completions",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "openrouter_require_headers",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "openrouter",
                "--base-url",
                f"{base}/api",
                "--model",
                "mock-model",
                "--api-style",
                "chat_completions",
                "--api-key",
                "sk-test-SECRET",
                "--or-referer",
                "https://example.local",
                "--or-title",
                "huweibot-e2e",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "together_chat",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "together",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "auto",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "groq_chat",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "groq",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "auto",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "chat",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "generic",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "auto",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    results.append(
        _run_case(
            "responses",
            [
                "--provider",
                "openai_compat",
                "--profile",
                "generic",
                "--base-url",
                base,
                "--model",
                "mock-model",
                "--api-style",
                "auto",
                "--api-key",
                "sk-test-SECRET",
                "--ping",
            ],
            expect_success=True,
        )
    )
    return results


def _print_summary(results: list[CaseResult]) -> int:
    rows = []
    failed = 0
    for r in results:
        if not r.ok:
            failed += 1
        rows.append({"case": r.name, "result": "PASS" if r.ok else "FAIL", "detail": r.detail})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"summary: {len(results)-failed}/{len(results)} passed")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end openai_compat local validation (offline).")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-profile-matrix", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_all:
        results = run_all()
        return _print_summary(results)
    if args.run_profile_matrix:
        results = run_profile_matrix()
        return _print_summary(results)
    print("Use --run-all or --run-profile-matrix")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
