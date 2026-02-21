"""Compatibility shim for deprecated `xbot` package.

Use `huweibot` instead.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

_new_pkg = import_module("huweibot")

__version__ = getattr(_new_pkg, "__version__", "0.1.0")
__all__ = getattr(_new_pkg, "__all__", [])

# Keep this package path and include huweibot package path so imports like
# `xbot.core.*` resolve to `huweibot/core/*` without duplicating code.
__path__ = [str(Path(__file__).resolve().parent)]
for _p in getattr(_new_pkg, "__path__", []):
    if _p not in __path__:
        __path__.append(_p)


def __getattr__(name: str):
    return getattr(_new_pkg, name)
