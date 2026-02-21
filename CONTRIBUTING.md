# Contributing to huweibot

Thanks for contributing.

## Development Setup

1. Create and activate a virtual environment.
2. Install project and development dependencies:

```bash
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -q
```

## Build Check

```bash
python3 -m build
```

## Code Style and Scope

- Keep changes minimal and focused.
- Avoid unrelated formatting/reordering.
- Preserve dual-machine safety constraints documented in `README.md`.
- Do not commit local/generated files (`.venv*`, `dist/`, `build/`, `artifacts/`, caches).

## Pull Request Process

1. Create a branch from `main`.
2. Make focused commits with clear messages.
3. Ensure tests and build pass locally.
4. Open a PR with:
   - What changed
   - Why it changed
   - How it was validated

## Reporting Issues

Use GitHub Issues for bugs and feature requests.
For security issues, follow `SECURITY.md`.
