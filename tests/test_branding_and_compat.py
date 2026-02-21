from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


def test_xbot_shim_imports_huweibot() -> None:
    import huweibot
    import xbot

    assert hasattr(huweibot, "__version__")
    assert xbot.__version__ == huweibot.__version__


def test_xbot_shim_submodule_import() -> None:
    from xbot.core.actions import DoneAction, action_from_json, action_to_json

    obj = DoneAction(reason="compat")
    round_trip = action_from_json(action_to_json(obj))
    assert round_trip.type == "DONE"


def test_no_chinese_in_web_ui() -> None:
    web_root = Path(__file__).resolve().parents[1] / "huweibot" / "web"
    cjk = re.compile(r"[\u4e00-\u9fff]")
    files = []
    for p in web_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".html", ".js", ".css", ".py", ".md", ".txt"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if cjk.search(text):
            files.append(str(p))
    assert not files, f"Chinese text found in web UI files: {files}"


def test_scroll_action_default_verify_is_valid() -> None:
    from huweibot.core.actions import ScrollAction

    action = ScrollAction(delta=120)
    assert action.verify.type == "TEXT_CHANGED"
    assert bool((action.verify.text or "").strip())


def test_actions_module_self_test_entrypoint() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "huweibot.core.actions"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "OK" in proc.stdout
