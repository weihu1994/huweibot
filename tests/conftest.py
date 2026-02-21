from __future__ import annotations

from urllib import request as urlrequest

import pytest


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("allow_network"):
        return

    def _blocked(*_args, **_kwargs):
        raise AssertionError("Real outbound network is blocked in tests; use mocked urlopen.")

    monkeypatch.setattr(urlrequest, "urlopen", _blocked)

