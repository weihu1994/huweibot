# Providers Manual Validation

`huweibot` defaults to `DummyProvider` (no network).

## OpenAI-compatible gateways

Use any OpenAI API compatible endpoint (official/OpenRouter/proxy/self-hosted):

```bash
export XBOT_PROVIDER=openai_compat
export OPENAI_COMPAT_BASE_URL="https://your-gateway.example.com"   # with or without /v1
export OPENAI_COMPAT_API_KEY="***"                                  # optional for trusted internal gateways
export OPENAI_COMPAT_MODEL="gpt-4.1-mini"
```

Optional controls:
- `OPENAI_COMPAT_API_STYLE=auto|responses|chat_completions` (default `auto`)
- `OPENAI_COMPAT_PROFILE=generic|azure|openrouter|together|groq` (default `generic`)
- `OPENAI_COMPAT_HEADERS_JSON='{"x-api-key":"..."}'`
- `OPENAI_COMPAT_QUERY_JSON='{"tenant":"abc"}'`
- `OPENAI_COMPAT_TIMEOUT_S=30`
- `OPENAI_COMPAT_MAX_RETRIES=3`

Profile extras:
- Azure:
  - `OPENAI_COMPAT_PROFILE=azure`
  - `OPENAI_COMPAT_AZURE_DEPLOYMENT=...` (required)
  - `OPENAI_COMPAT_AZURE_API_VERSION=2024-10-21` (default)
- OpenRouter:
  - `OPENAI_COMPAT_PROFILE=openrouter`
  - optional headers: `OPENAI_COMPAT_OR_REFERER`, `OPENAI_COMPAT_OR_TITLE`, `OPENAI_COMPAT_OR_APP_ID`

OpenAI-compatible behavior:
- `OPENAI_COMPAT_API_STYLE=auto`: try `/v1/responses` first; fallback to `/v1/chat/completions` only on 404/405/supported 400 endpoint errors.
- 401/403 never trigger fallback.

Base URL examples (all accepted):
- `https://host`
- `https://host/`
- `https://host/v1`
- `https://host/v1/`
- `https://openrouter.ai/api` (normalized to `/api/v1`)
- `https://api.together.xyz` (normalized to `/v1`)
- `https://api.groq.com` (normalized to `/openai/v1`)

## Environment

- OpenAI:
  - `XBOT_PROVIDER=openai`
  - `OPENAI_API_KEY=...`
  - optional: `OPENAI_MODEL`, `OPENAI_BASE_URL`
- Anthropic:
  - `XBOT_PROVIDER=anthropic`
  - `ANTHROPIC_API_KEY=...`
  - optional: `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`
- Gemini:
  - `XBOT_PROVIDER=gemini`
  - `GEMINI_API_KEY=...`
  - optional: `GEMINI_MODEL`, `GEMINI_BASE_URL`

## Health Check CLI

- Dry run (no request sent):
  - `python3 -m huweibot.tools.provider_check --provider openai --dry-run`
- Ping:
  - `python3 -m huweibot.tools.provider_check --provider openai --ping`
  - `python3 -m huweibot.tools.provider_check --provider anthropic --ping`
  - `python3 -m huweibot.tools.provider_check --provider gemini --ping`

Notes:
- CI does not run real cloud pings.
- Missing key fails immediately with readable error.
- Fallback to dummy is disabled by default unless `allow_fallback_to_dummy=true`.

## Offline compatibility E2E

Run local mock server matrix (no internet):

```bash
python3 -m huweibot.tools.e2e_openai_compat --run-all
python3 -m huweibot.tools.e2e_openai_compat --run-profile-matrix
```
