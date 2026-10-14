"""End-user attribution: the ``user_id`` Hermes passes on ``pre_api_request`` (e.g. from the API
server's ``X-Hermes-User-Id`` header, for a frontend proxying many people through one deployment)
lands on the trace next to ``session_id``, enabling Langfuse's per-user views. Only the SDK client
is replaced by a recorder.
"""
from __future__ import annotations

import contextlib
import importlib
import sys

import pytest


class _Span:
    def update(self, **kw):
        pass

    def set_trace_io(self, **kw):
        pass

    def end(self, **kw):
        pass

    def start_observation(self, **kw):
        return _Span()


class _Client:
    def __init__(self, roots):
        self._roots = roots

    def create_trace_id(self, seed=None):
        return f"trace::{seed}"

    def start_observation(self, **kw):
        self._roots.append(kw)
        return _Span()

    def flush(self):
        pass


@pytest.fixture
def traced(monkeypatch):
    """Plugin wired to a recording client; returns (module, roots, propagated)."""
    sys.modules.pop("langfuse_plugin", None)
    mod = importlib.import_module("langfuse_plugin")
    roots: list = []
    propagated: list = []

    @contextlib.contextmanager
    def _propagate(**kw):
        propagated.append(kw)
        yield

    monkeypatch.setattr(mod, "_get_langfuse", lambda: _Client(roots))
    monkeypatch.setattr(mod, "propagate_attributes", _propagate)
    mod._TRACE_STATE.clear()
    return mod, roots, propagated


def _open_turn(mod, session, **extra):
    mod.on_pre_llm_request(
        task_id=session, session_id=session, model="m", provider="p", api_mode="chat", api_call_count=1,
        request_messages=[{"role": "user", "content": "hi"}], turn_id=f"{session}:t1",
        api_request_id=f"{session}:t1:api:1", **extra,
    )


def test_user_id_is_set_on_trace_context_and_propagated(traced):
    mod, roots, propagated = traced
    _open_turn(mod, "sess-uid", user_id="alice@example.com")
    assert roots[0]["trace_context"]["user_id"] == "alice@example.com"
    assert roots[0]["trace_context"]["session_id"] == "sess-uid"
    # Children re-apply the root's propagated attributes, so they carry the user too.
    assert propagated[0]["user_id"] == "alice@example.com"
    (state,) = mod._TRACE_STATE.values()
    assert state.propagated["user_id"] == "alice@example.com"


def test_absent_user_id_is_omitted(traced):
    """No user_id -> key absent entirely, not an empty string (an empty Langfuse user_id is still a
    value, and would group every anonymous turn under one "user")."""
    mod, roots, propagated = traced
    _open_turn(mod, "sess-nouid")
    assert "user_id" not in roots[0]["trace_context"]
    assert "user_id" not in propagated[0]


def test_legacy_pre_llm_call_also_carries_user_id(traced):
    mod, roots, _ = traced
    mod.on_pre_llm_call(task_id="s3", session_id="s3", model="m", provider="p", api_mode="chat",
                        messages=[{"role": "user", "content": "hi"}], turn_id="s3:t1", user_id="bob")
    assert roots[0]["trace_context"]["user_id"] == "bob"
