"""Probe/control separation and backend process liveness during readiness."""

import time

import pytest
from test_di_native_ray import config
from test_di_native_ray import runtime as native_runtime

from Magpie.di.native_ray import NativeRayDISession, _NativeRole


@pytest.fixture
def runtime(monkeypatch):
    return native_runtime.__wrapped__(monkeypatch)


def test_only_http_probes_use_dedicated_concurrency_group(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    assert [event[1] for event in runtime.events if event[0] == "class_options"] == [
        {"concurrency_groups": {"probe": 1}}
    ]
    probe_options = [
        event[1] for event in runtime.events if event[0] == "probe_options"
    ]
    assert probe_options and all(
        value == {"concurrency_group": "probe"} for value in probe_options
    )
    assert all(
        "max_concurrency" not in event[1]
        for event in runtime.events
        if event[0] == "actor_options"
    )
    # The base class remains Ray-import/decorator-free until session startup.
    assert not hasattr(_NativeRole.request, "__ray_concurrency_group__")
    session.stop()


def test_backend_exiting_after_ready_fails_with_router_still_alive(
    runtime, tmp_path, monkeypatch
):
    raw = config()
    raw["disaggregation"]["startup_timeout"] = 600
    session = NativeRayDISession(raw, tmp_path)
    command = session._router_command
    seen = []
    captured = {}

    def router_command():
        prefill = session.actors["prefill-0"]
        decode = session.actors["decode-0"]
        captured.update(prefill=prefill, decode=decode)
        request = prefill.request.remote

        def response(name, path, *args, **kwargs):
            seen.append(path)
            if path == "/health":
                decode.returncodes["engine"] = 7
            return request(name, path, *args, **kwargs)

        prefill.request.remote = response
        return command()

    monkeypatch.setattr(session, "_router_command", router_command)
    with pytest.raises(RuntimeError, match="decode-0 engine exited with 7"):
        session.start()
    assert "/health" in seen and "/v1/completions" not in seen
    assert "router" in captured["prefill"].commands
    assert captured["prefill"].returncodes.get("router") is None
    assert session.state["status"] == "stopped"
    assert not runtime.groups and not runtime.actors


@pytest.mark.parametrize("interruption", ["backend-exit", "cancel"])
def test_pending_long_probe_polls_control_liveness_and_allows_cleanup(
    runtime, tmp_path, monkeypatch, interruption
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    prefill = session.actors["prefill-0"]
    decode = session.actors["decode-0"]
    pending = object()
    request = prefill.request.remote
    prefill.request.remote = lambda name, path, *args, **kwargs: (
        pending if path == "/v1/completions" else request(name, path, *args, **kwargs)
    )
    wait = runtime.ray.wait
    polls = []
    cancelled = []

    def cancel_check():
        if cancelled:
            raise RuntimeError("cancel during long probe")

    session.check_cancelled = cancel_check

    def wait_for_probe(refs, **kwargs):
        if refs == [pending]:
            polls.append(True)
            if interruption == "backend-exit":
                decode.returncodes["engine"] = 9
            else:
                cancelled.append(True)
            return [], refs
        return wait(refs, **kwargs)

    monkeypatch.setattr(runtime.ray, "wait", wait_for_probe)
    message = (
        "decode-0 engine exited with 9"
        if interruption == "backend-exit"
        else "cancel during long probe"
    )
    try:
        with pytest.raises(RuntimeError, match=message):
            session._probe(
                "prefill-0",
                "router",
                "/v1/completions",
                time.monotonic() + 600,
                {"prompt": "Hello"},
            )
        assert len(polls) == 1
    finally:
        # A pending probe must not prevent control cleanup, including while
        # the owning evaluation's cancellation hook remains raised.
        session.stop()
    assert not runtime.groups and not runtime.actors
