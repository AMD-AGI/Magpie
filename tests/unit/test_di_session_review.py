"""Second review: session teardown, shared startup budget and direct API env."""

import json
import os
import signal
from types import SimpleNamespace

import pytest
from test_di_native_ray import config
from test_di_native_ray import runtime as native_runtime

from Magpie.di import native_ray
from Magpie.di.native_ray import NativeRayDISession, _NativeRole


@pytest.fixture
def runtime(monkeypatch):
    return native_runtime.__wrapped__(monkeypatch)


@pytest.mark.parametrize("body_error", [False, True])
def test_session_diagnostic_failure_is_best_effort_and_stop_is_strict(
    runtime, tmp_path, monkeypatch, body_error
):
    session = NativeRayDISession(config(), tmp_path)
    monkeypatch.setattr(
        session,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("diagnostic unavailable")),
    )
    original = RuntimeError("body failed") if body_error else None
    if body_error:
        with pytest.raises(RuntimeError, match="body failed") as caught, session:
            raise original
        assert caught.value is original
    else:
        with session:
            pass
    assert session.state["status"] == "stopped"
    assert (
        json.loads(session.path.read_text())["generation"]["collection_error"]
        == "diagnostic unavailable"
    )
    assert not runtime.groups and not runtime.actors


def test_session_collect_error_does_not_mask_strict_stop_failure(
    runtime, tmp_path, monkeypatch
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    monkeypatch.setattr(
        session,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("diagnostics failed")),
    )
    session.actors["decode-0"].fail_cleanup = True
    with pytest.raises(RuntimeError, match="group still active"):
        session.__exit__(None, None, None)
    assert runtime.groups and session.state["status"] == "cleanup_failed"


@pytest.mark.parametrize("failure", ["body", "stop", "success"])
def test_pending_sigint_during_teardown_preserves_original_or_strict_error(
    runtime, tmp_path, monkeypatch, failure
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    events = []
    original_stop = session.stop

    def collect():
        os.kill(os.getpid(), signal.SIGINT)
        events.append("diagnostic-finished")

    def stop():
        events.append("stop-entered")
        if failure == "stop":
            raise RuntimeError("strict cleanup failed")
        original_stop()
        events.append("stop-finished")

    monkeypatch.setattr(session, "collect", collect)
    monkeypatch.setattr(session, "stop", stop)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if failure == "body":
            with pytest.raises(RuntimeError, match="body error"):
                try:
                    raise RuntimeError("body error")
                finally:
                    session.__exit__(RuntimeError, None, None)
        elif failure == "stop":
            with pytest.raises(RuntimeError, match="strict cleanup failed"):
                session.__exit__(None, None, None)
        else:
            with pytest.raises(KeyboardInterrupt):
                session.__exit__(None, None, None)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert events[:2] == ["diagnostic-finished", "stop-entered"]
    if failure != "stop":
        assert events[-1] == "stop-finished" and not runtime.groups


def test_startup_failure_defers_sigint_until_cleanup_preserving_original(
    runtime, tmp_path, monkeypatch
):
    session = NativeRayDISession(config(), tmp_path)
    original = RuntimeError("startup failed")
    monkeypatch.setattr(session, "_ready", lambda *_: (_ for _ in ()).throw(original))

    def collect():
        os.kill(os.getpid(), signal.SIGINT)
        raise RuntimeError("optional diagnostics failed")

    monkeypatch.setattr(session, "collect", collect)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(RuntimeError, match="startup failed") as caught:
            session.start()
    finally:
        signal.signal(signal.SIGINT, previous)
    assert caught.value is original and session.state["status"] == "stopped"
    assert (
        session.state["generation"]["collection_error"] == "optional diagnostics failed"
    )
    assert not runtime.groups and not runtime.actors


def test_startup_budget_includes_supervisor_identity_before_launch(
    runtime, tmp_path, monkeypatch
):
    raw = config()
    raw["disaggregation"]["startup_timeout"] = 10
    session = NativeRayDISession(raw, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(native_ray.time, "monotonic", lambda: clock[0])
    original_remote = runtime.ray.remote

    def remote(cls=None, **options):
        if cls is None:
            return lambda klass: remote(klass, **options)
        actor_class = original_remote(cls, **options)
        options = actor_class.options

        def configure(**kwargs):
            configured = options(**kwargs)
            create = configured.remote

            def actor(*args):
                obj = create(*args)
                identity = obj.identity.remote

                def slow():
                    clock[0] = 11
                    return identity()

                obj.identity.remote = slow
                return obj

            return SimpleNamespace(remote=actor)

        actor_class.options = configure
        return actor_class

    monkeypatch.setattr(runtime.ray, "remote", remote)
    with pytest.raises(TimeoutError, match="serving startup timed out"):
        session.start()
    assert not any(event[0] == "launch" for event in runtime.events)
    assert session.state["status"] == "stopped" and not runtime.groups


def test_startup_http_and_rpc_use_shared_remaining_budget_over_fixed_caps(
    runtime, tmp_path, monkeypatch
):
    raw = config()
    raw["disaggregation"]["startup_timeout"] = 100
    session = NativeRayDISession(raw, tmp_path)
    clock = [0.0]
    timeouts = []
    monkeypatch.setattr(native_ray.time, "monotonic", lambda: clock[0])
    original_rpc = session._rpc

    def rpc(reference, timeout=30, cancellable=True, checkpoint=None):
        if session.actors and cancellable:
            timeouts.append(timeout)
        return original_rpc(reference, timeout, cancellable, checkpoint=checkpoint)

    monkeypatch.setattr(session, "_rpc", rpc)
    original_command = session._router_command

    def router_command():
        actor = session.actors["prefill-0"]
        request = actor.request.remote

        def slow_probe(name, path, port, data=None, timeout=10):
            if path == "/v1/completions":
                assert timeout == 100
                clock[0] += 40
            return request(name, path, port, data, timeout=timeout)

        actor.request.remote = slow_probe
        return original_command()

    monkeypatch.setattr(session, "_router_command", router_command)
    session.start()
    assert 60 in timeouts and max(timeouts) == 100
    assert session.state["status"] == "ready"
    session.stop()


def test_native_http_timeout_is_forwarded(monkeypatch):
    actor = _NativeRole.__new__(_NativeRole)
    actor.processes = {"engine": SimpleNamespace(poll=lambda: None)}
    calls = []
    monkeypatch.setattr(
        native_ray, "_http", lambda url, data, timeout: calls.append(timeout) or {}
    )
    actor.request("engine", "/get_server_info", 30000, timeout=45)
    assert calls == [45]


def test_direct_session_applies_common_ray_env_and_role_overrides_everywhere(
    runtime, tmp_path
):
    raw = config()
    raw["ray_config"] = {"env_vars": {"COMMON": "direct", "ROLE": "ray"}}
    raw["envs"]["ROLE"] = "common"
    raw["disaggregation"]["prefill"]["envs"] = {
        "ROLE": "prefill",
        "PREFILL_ONLY": "engine",
    }
    session = NativeRayDISession(raw, tmp_path)
    session.start()
    options = [event[1] for event in runtime.events if event[0] == "actor_options"]
    for option in options:
        env = option["runtime_env"]["env_vars"]
        assert env == {"COMMON": "direct", "ROLE": "ray"}
    prefill = session.actors["prefill-0"]
    assert prefill.launch_envs["engine"]["COMMON"] == "direct"
    assert prefill.launch_envs["engine"]["ROLE"] == "prefill"
    assert prefill.launch_envs["router"]["COMMON"] == "direct"
    assert prefill.launch_envs["router"]["ROLE"] == "common"
    assert prefill.launch_envs["engine"]["PREFILL_ONLY"] == "engine"
    assert "PREFILL_ONLY" not in prefill.launch_envs["router"]
    assert (
        "MAGPIE_DI_SERVING_ENV"
        not in json.loads(session.path.read_text())["spec"]["instances"][0][
            "serving_env"
        ]
    )
    session.stop()


def test_replacement_ignores_diagnostic_failure_but_applies_candidate(
    runtime, tmp_path, monkeypatch
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    previous = session.id
    monkeypatch.setattr(
        session,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("optional evidence unavailable")),
    )
    candidate = config()
    candidate["envs"]["DI_REPLACEMENT"] = "yes"
    session.replace(candidate)
    assert session.id != previous and session.state["status"] == "ready"
    assert (
        session.state["generations"][0]["collection_error"]
        == "optional evidence unavailable"
    )
    assert session.actors["prefill-0"].launch_envs["engine"]["DI_REPLACEMENT"] == "yes"
    session.stop()


def test_replacement_sigint_finishes_old_cleanup_without_new_generation(
    runtime, tmp_path, monkeypatch
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    previous = session.id
    monkeypatch.setattr(session, "collect", lambda: os.kill(os.getpid(), signal.SIGINT))
    original = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            session.replace(config())
    finally:
        signal.signal(signal.SIGINT, original)
    assert len(session.state["generations"]) == 1 and session.id == previous
    assert (
        session.state["status"] == "stopped"
        and not runtime.groups
        and not runtime.actors
    )


def test_replacement_strict_stop_failure_prevents_new_generation(
    runtime, tmp_path, monkeypatch
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    previous = session.id
    monkeypatch.setattr(
        session,
        "collect",
        lambda: (_ for _ in ()).throw(RuntimeError("optional evidence unavailable")),
    )
    session.actors["decode-0"].fail_cleanup = True
    with pytest.raises(RuntimeError, match="group still active"):
        session.replace(config())
    assert len(session.state["generations"]) == 1 and session.id == previous
    assert session.state["status"] == "cleanup_failed" and runtime.groups
