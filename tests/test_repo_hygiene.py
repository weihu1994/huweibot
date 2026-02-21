from __future__ import annotations

import subprocess
from pathlib import Path


def test_no_generated_or_local_files_tracked() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def is_forbidden(path: str) -> bool:
        lowered = path.lower()
        return (
            path.startswith("dist/")
            or path.startswith("build/")
            or path.startswith("artifacts/")
            or path.startswith("logs/")
            or path.startswith("captures/")
            or path.startswith("venv/")
            or path.startswith("env/")
            or path.startswith(".venv")
            or "/__pycache__/" in path
            or path.endswith("/__pycache__")
            or path.endswith(".pyc")
            or ".egg-info/" in path
            or lowered.endswith(".ds_store")
        )

    hits = [p for p in paths if is_forbidden(p)]
    assert not hits, "Tracked local/generated files found (showing up to 20):\n" + "\n".join(hits[:20])
