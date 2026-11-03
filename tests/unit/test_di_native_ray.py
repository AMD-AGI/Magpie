"""Native Ray DI contracts: placement, commands, durable cleanup and client guards."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from Magpie.core.executor import EndpointClientCleanupError
from Magpie.di.native_ray import NativeRayDISession, _NativeRole


def config(prefill=1, decode=1):
    return {
        "framework": "sglang",
        "run_mode": "ray",
        "model": "/models/test",
        "disaggregation": {
            "prefill": {"instances": prefill},
            "decode": {"instances": decode, "envs": {"EP": 2}},
            "startup_timeout": 0.05,
        },
        "envs": {
            "TP": 2,
            "EP": 1,
            "EXTRA_SGLANG_ARGS": "--context-length 1024",
            "CONC": 1,
        },
    }


@pytest.fixture
def runtime(monkeypatch):
    events, actors, groups = [], {}, {}
    ray = ModuleType("ray")
    ray.is_initialized = lambda: True
    ray.get_runtime_context = lambda: SimpleNamespace(namespace="magpie-native-test")
    ray.wait = lambda refs, **kwargs: (refs, [])
    ray.get = lambda value: value
    ray.get_actor = lambda name, namespace: (
        actors[name] if name in actors else missing()
    )
    ray.kill = lambda actor, **kwargs: (
        events.append(("kill", actor.instance["id"])),
        actors.pop(actor.name, None),
    )
    nodes = [
        {
            "NodeID": f"node-{index}",
            "NodeManagerAddress": f"10.0.0.{index + 1}",
            "Alive": True,
            "Resources": {"CPU": 2, "GPU": 2, f"node:10.0.0.{index + 1}": 1},
        }
        for index in range(4)
    ]
    ray.nodes = lambda: nodes
    available = {node["NodeID"]: dict(node["Resources"]) for node in nodes}
    state = ModuleType("ray._private.state")
    state.available_resources_per_node = lambda: available
    private = ModuleType("ray._private")
    private.state = state
    ray._private = private
    monkeypatch.setitem(sys.modules, "ray._private", private)
    monkeypatch.setitem(sys.modules, "ray._private.state", state)

    def missing():
        raise ValueError("not found")

    class Actor:
        def __init__(self, owner, instance, directory, options):
            self.instance, self.name, self.owner = instance, options["name"], owner
            strategy = options["scheduling_strategy"]
            self.index = strategy.placement_group_bundle_index
            role_bundles = strategy.placement_group.bundle_specs[
                self.index : self.index + instance["tp"] * instance["pp"]
            ]
            self.rank_nodes = {
                key
                for rank_bundle in role_bundles
                for key in rank_bundle
                if key.startswith("node:")
            }
            bundle = strategy.placement_group.bundle_specs[self.index]
            node_key = next(key for key in bundle if key.startswith("node:"))
            self.host = node_key.removeprefix("node:")
            self.node_id = next(
                node["NodeID"]
                for node in nodes
                if node["NodeManagerAddress"] == self.host
            )
            self.commands = {}
            self.launch_envs = {}
            self.identity = SimpleNamespace(remote=self.identify)
            self.launch = SimpleNamespace(remote=self.launch_impl)
            self.request = SimpleNamespace(remote=self.request_impl)
            self.stop = SimpleNamespace(remote=self.stop_impl)
            self.inspect = SimpleNamespace(remote=lambda: deepcopy(self.commands))
            self.fail_cleanup = False

        def identify(self):
            return {
                "owner": self.owner,
                "instance": self.instance["id"],
                "node_id": self.node_id,
                "host": self.host,
                "uid": 123,
                "gpu_ids": [],
                "python": "/worker/bin/python",
            }

        def launch_impl(self, name, argv, env, ports):
            events.append(("launch", self.instance["id"], name))
            self.commands[name] = argv
            self.launch_envs[name] = dict(env)
            if "/models/missing" in argv:
                raise RuntimeError("candidate model missing")
            return {"status": "started"}

        def request_impl(self, name, path, port, data=None):
            if path == "/get_server_info":
                return {
                    "model_path": "/models/test",
                    "disaggregation_mode": self.instance["role"],
                    "tp_size": self.instance["tp"],
                    "ep_size": self.instance["ep"],
                    "pp_size": self.instance["pp"],
                    "nnodes": len(self.rank_nodes),
                    "context_length": 1024,
                    "disaggregation_transfer_backend": "mooncake",
                }
            if path == "/health":
                return {}
            return {
                "choices": [{"text": "mock completion"}],
                "usage": {"completion_tokens": 1},
            }

        def stop_impl(self):
            events.append(("stop", self.instance["id"]))
            if self.fail_cleanup:
                raise RuntimeError("group still active")
            return {"cleanup_verified": True}

    class ActorClass:
        def options(self, **options):
            events.append(("actor_options", options))

            def create(owner, instance, directory):
                actor = Actor(owner, instance, directory, options)
                actors[options["name"]] = actor
                return actor

            return SimpleNamespace(remote=create)

    ray.remote = lambda klass: ActorClass()
    placement = ModuleType("ray.util.placement_group")

    def create(bundles, **options):
        events.append(("placement", bundles, options))
        group = SimpleNamespace(
            ready=lambda: True,
            name=options["name"],
            bundle_specs=bundles,
            id=SimpleNamespace(
                hex=lambda: hashlib.sha256(options["name"].encode()).hexdigest()[:36]
            ),
        )
        groups[group.name] = group
        return group

    placement.placement_group = create
    placement.get_placement_group = lambda name: (
        groups[name] if name in groups else missing()
    )
    placement.remove_placement_group = lambda group: (
        events.append(("release", group.name)),
        groups.pop(group.name, None),
    )
    strategy = ModuleType("ray.util.scheduling_strategies")
    strategy.PlacementGroupSchedulingStrategy = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ModuleType("ray.util"))
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategy)
    return SimpleNamespace(
        ray=ray,
        actors=actors,
        groups=groups,
        events=events,
        nodes=nodes,
        available=available,
    )


def test_native_xpyd_placement_commands_and_cleanup_journal(runtime, tmp_path):
    session = NativeRayDISession(config(2, 1), tmp_path)
    assert session.start() == "http://10.0.0.1:18000"
    placement = next(event for event in runtime.events if event[0] == "placement")
    assert len(placement[1]) == 6
    assert all(bundle["GPU"] == 1 for bundle in placement[1])
    assert sum(bundle.get("CPU", 0) for bundle in placement[1]) == 3
    assert placement[2]["strategy"] == "PACK"
    options = [event[1] for event in runtime.events if event[0] == "actor_options"]
    assert all(o["num_gpus"] == 0 and o["lifetime"] == "detached" for o in options)
    assert [o["scheduling_strategy"].placement_group_bundle_index for o in options] == [
        0,
        2,
        4,
    ]
    command = session.actors["decode-0"].commands["engine"]
    assert command[0] == "/worker/bin/python"
    assert command[command.index("--ep-size") + 1] == "2"
    assert command[command.index("--pp-size") + 1] == "1"
    router = session.actors["prefill-0"].commands["router"]
    assert router.count("--prefill") == 2 and router.count("--decode") == 1
    assert router[router.index("--prefill") + 1 : router.index("--prefill") + 3] == [
        "http://10.0.0.1:30000",
        "8998",
    ]
    assert not any("docker" in argument for argument in command)
    readiness = session.state["generation"]["readiness"]
    assert readiness["routed_completion"]["choices"]
    session.stop()
    record = json.loads((tmp_path / "native-session.json").read_text())
    assert record["generation"]["cleanup_verified"] and record["status"] == "stopped"
    first_kill = next(i for i, e in enumerate(runtime.events) if e[0] == "kill")
    assert all(e[0] != "stop" for e in runtime.events[first_kill:])
    assert not runtime.actors and not runtime.groups


def test_native_failed_cleanup_retains_reservations(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    session.actors["decode-0"].fail_cleanup = True
    with pytest.raises(RuntimeError, match="group still active"):
        session.stop()
    assert runtime.groups and runtime.actors
    assert not session.state["generation"]["cleanup_verified"]
    assert not any(e[0] == "kill" for e in runtime.events)


def test_native_recovery_verifies_identity_and_does_not_reacquire(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    placements = sum(e[0] == "placement" for e in runtime.events)
    recovered = NativeRayDISession.load(tmp_path)
    assert recovered.id == session.id
    with pytest.raises(RuntimeError, match="inspection and cleanup"):
        recovered.start()
    recovered.stop()
    assert sum(e[0] == "placement" for e in runtime.events) == placements


def test_native_missing_launcher_fails_closed(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    runtime.actors.pop(session.state["generation"]["actors"]["decode-0"])
    recovered = NativeRayDISession.load(tmp_path)
    with pytest.raises(RuntimeError, match="launchers missing"):
        recovered.stop()
    assert runtime.groups


def test_native_failed_replacement_restores_new_generation(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    first_id = session.id
    candidate = config()
    candidate["model"] = "/models/missing"
    with pytest.raises(RuntimeError, match="previous candidate restored"):
        session.replace(candidate)
    assert session.id != first_id and session.state["status"] == "ready"
    assert session.spec["model"] == "/models/test"
    assert session.state["replacement"]["restored"] is True
    assert len(session.state["generations"]) == 3
    session.stop()


def test_native_client_run_journals_active_work_and_retains_uncertain_cleanup(
    runtime, tmp_path
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    with (
        pytest.raises(EndpointClientCleanupError),
        session.client_run("evaluation-task", tmp_path / "measurement"),
    ):
        persisted = json.loads(session.path.read_text())
        assert persisted["client_cleanup_pending"] == {
            "task_id": "evaluation-task",
            "output_dir": str((tmp_path / "measurement").resolve()),
        }
        with pytest.raises(EndpointClientCleanupError):
            session.replace(config())
        with pytest.raises(EndpointClientCleanupError):
            session.stop()
        raise EndpointClientCleanupError("client group still active")
    assert runtime.groups
    recovered = NativeRayDISession.load(tmp_path)
    assert recovered.state["client_cleanup_pending"]["task_id"] == "evaluation-task"
    with pytest.raises(EndpointClientCleanupError):
        recovered.stop()
    assert runtime.groups


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_native_client_run_clears_confirmed_client_before_reuse(
    runtime, tmp_path, failure
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()

    def measure():
        with session.client_run("evaluation-task", tmp_path / "measurement"):
            assert session.state["client_cleanup_pending"]
            if failure is not None:
                raise failure("client exited after confirmed process cleanup")

    if failure is None:
        measure()
    else:
        with pytest.raises(failure):
            measure()
    assert session.state["client_cleanup_pending"] is None
    assert json.loads(session.path.read_text())["client_cleanup_pending"] is None
    session.stop()
    assert not runtime.groups and not runtime.actors


def test_native_client_run_rejects_work_before_ready(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    with (
        pytest.raises(RuntimeError, match="ready"),
        session.client_run("evaluation-task", tmp_path / "measurement"),
    ):
        pytest.fail("must not enter client before serving readiness")
    assert session.state["client_cleanup_pending"] is None
    assert not runtime.groups


def test_actor_owns_real_process_group_and_log(tmp_path):
    actor = _NativeRole("owner", {"id": "prefill-0"}, tmp_path)
    receipt = actor.launch(
        "engine",
        [
            sys.executable,
            "-c",
            'import time; print("native-test",flush=True); time.sleep(30)',
        ],
        {},
        [],
    )
    assert receipt["pgid"] == receipt["pid"]
    try:
        assert actor.inspect()["engine"]["returncode"] is None
        assert actor.stop()["cleanup_verified"]
        from Magpie.modes.benchmark.endpoint_client import _group_exists

        assert not _group_exists(receipt["pgid"])
    finally:
        actor.stop()


def test_partial_actor_creation_failure_cleans_created_actor_and_placement(
    runtime, tmp_path, monkeypatch
):
    original = runtime.ray.remote

    def remote(klass):
        actor_class = original(klass)
        options = actor_class.options

        def set_options(**kwargs):
            if kwargs["name"].endswith("decode-0"):
                raise RuntimeError("actor creation rejected")
            return options(**kwargs)

        actor_class.options = set_options
        return actor_class

    monkeypatch.setattr(runtime.ray, "remote", remote)
    session = NativeRayDISession(config(), tmp_path)
    with pytest.raises(RuntimeError, match="actor creation rejected"):
        session.start()
    assert session.state["status"] == "stopped"
    assert session.state["generation"]["cleanup_verified"]
    assert not runtime.groups and not runtime.actors
    assert not any(event[0] == "launch" for event in runtime.events)


def test_recovery_rejects_changed_gpu_identity(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    actor = session.actors["decode-0"]
    original = actor.identity.remote
    actor.identity.remote = lambda: {**original(), "gpu_ids": ["4", "5"]}
    with pytest.raises(RuntimeError, match="identity changed"):
        NativeRayDISession.load(tmp_path)
    assert runtime.groups
    actor.identity.remote = original
    session.stop()


def test_startup_configuration_mismatch_collects_then_stops(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    original = session._ready

    def wrong_settings(instance, deadline):
        if instance["role"] == "decode":
            raise RuntimeError("configuration mismatch")
        return original(instance, deadline)

    session._ready = wrong_settings
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        session.start()
    assert session.state["status"] == "stopped"
    assert (
        Path(session.state["generation"]["directory"]) / "inspection.json"
    ).is_file()
    assert not runtime.actors and not runtime.groups


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_path", "/models/wrong"),
        ("tp_size", 8),
        ("ep_size", 8),
        ("pp_size", 2),
        ("nnodes", 99),
        ("context_length", 2048),
        ("disaggregation_mode", "decode"),
    ],
)
def test_native_readiness_validates_running_settings(runtime, tmp_path, field, value):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    actor = session.actors["prefill-0"]
    original = actor.request.remote
    actor.request.remote = lambda *args, **kwargs: {
        **original(*args, **kwargs),
        field: value,
    }
    import time

    with pytest.raises(RuntimeError, match="configuration mismatch"):
        session._ready(session.spec["instances"][0], time.monotonic() + 1)
    session.stop()


def test_router_health_accepts_plain_text(monkeypatch):
    from contextlib import nullcontext

    from Magpie.di.native_ray import _http

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: nullcontext(
            SimpleNamespace(status=200, read=lambda: b"healthy")
        ),
    )
    assert _http("http://localhost:18000/health") == {"healthy": True}


@pytest.mark.parametrize("init_fails", [False, True])
def test_native_connect_preserves_custom_cancellation_handlers(
    monkeypatch, tmp_path, init_fails
):
    import signal
    import threading

    session = NativeRayDISession({"benchmark": config()}, tmp_path)
    cancelled = threading.Event()

    def caller(*_args):
        cancelled.set()

    handlers = {signal.SIGTERM: caller, signal.SIGINT: caller}
    monkeypatch.setattr(signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(
        signal, "signal", lambda sig, callback: handlers.__setitem__(sig, callback)
    )

    def initialize(**kwargs):
        handlers.update({signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL})
        if init_fails:
            raise RuntimeError("initialization failed")

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: False,
            init=initialize,
            get_runtime_context=lambda: SimpleNamespace(namespace="magpie-di"),
        ),
    )
    if init_fails:
        with pytest.raises(RuntimeError, match="initialization failed"):
            session._connect()
    else:
        session._connect()
    assert all(callback is caller for callback in handlers.values())
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert cancelled.is_set()


@pytest.mark.parametrize("mode", ["default", "reused", "background"])
def test_native_connect_respects_default_reused_and_background_handlers(
    monkeypatch, tmp_path, mode
):
    import signal
    import threading

    session = NativeRayDISession(config(), tmp_path)
    handlers = {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.default_int_handler,
    }
    writes = []
    monkeypatch.setattr(signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(signal, "signal", lambda *args: writes.append(args))
    if mode in {"reused", "background"}:
        monkeypatch.setattr(
            signal,
            "getsignal",
            lambda _sig: pytest.fail("unexpected signal inspection"),
        )
    if mode == "background":
        monkeypatch.setattr(threading, "current_thread", object)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: mode == "reused",
            init=lambda **kwargs: None,
            get_runtime_context=lambda: SimpleNamespace(namespace="magpie-di"),
        ),
    )
    session._connect()
    assert writes == []


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"namespace": "wrong"}, "different DI namespace"),
        ({"expected_node_ids": ["missing"]}, "membership differs"),
    ],
)
def test_native_connection_rejects_wrong_namespace_or_membership(
    runtime, tmp_path, settings, message
):
    raw = config()
    raw["ray_config"] = settings
    session = NativeRayDISession(raw, tmp_path)
    with pytest.raises(ValueError, match=message):
        session.start()
    assert not runtime.events


def test_native_load_rejects_journal_source_mismatch_before_connect(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    record = json.loads(session.path.read_text())
    record["raw"]["model"] = "/models/changed"
    session.path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="raw benchmark and spec disagree"):
        NativeRayDISession.load(tmp_path)
    assert not runtime.events


def test_json_recovery_stop_updates_canonical_history(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    replacement = config()
    session.replace(replacement)
    # Fresh JSON objects must be rebound; sharing the original Python object
    # would conceal a stale generation-history entry during recovered cleanup.
    saved = json.loads(session.path.read_text())
    assert saved["generation"] is not saved["generations"][-1]
    recovered = NativeRayDISession.load(tmp_path)
    assert recovered.state["generation"] is recovered.state["generations"][-1]
    recovered.stop()
    final = json.loads(session.path.read_text())
    assert final["generation"] == final["generations"][-1]
    assert all(
        g["status"] == "stopped" and g["cleanup_verified"] for g in final["generations"]
    )


@pytest.mark.parametrize("corruption", ["duplicate", "not_tail", "contradictory"])
def test_recovery_rejects_ambiguous_generation_history(runtime, tmp_path, corruption):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    record = json.loads(session.path.read_text())
    if corruption == "duplicate":
        record["generations"].append(deepcopy(record["generation"]))
    elif corruption == "not_tail":
        record["generations"].append({**record["generation"], "id": "other-generation"})
    else:
        record["generations"][-1]["status"] = "stopped"
    session.path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="generation history"):
        NativeRayDISession.load(tmp_path)
    assert runtime.groups


def test_ambiguous_pg_creation_reconciles_journaled_name(
    runtime, tmp_path, monkeypatch
):
    placement = sys.modules["ray.util.placement_group"]
    original = placement.placement_group

    def accepted_without_handle(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("placement submission reply lost")

    monkeypatch.setattr(placement, "placement_group", accepted_without_handle)
    session = NativeRayDISession(config(), tmp_path)
    with pytest.raises(RuntimeError, match="submission reply lost"):
        session.start()
    assert not runtime.groups
    assert session.state["status"] == "stopped"
    assert session.state["generation"]["cleanup_verified"]
    assert not any(event[0] == "launch" for event in runtime.events)


def test_unconfirmed_pg_creation_with_no_named_handle_stays_uncertain(
    runtime, tmp_path, monkeypatch
):
    placement = sys.modules["ray.util.placement_group"]

    def lost_submission(*args, **kwargs):
        raise RuntimeError("placement submission transport unknown")

    monkeypatch.setattr(placement, "placement_group", lost_submission)
    session = NativeRayDISession(config(), tmp_path)
    with pytest.raises(RuntimeError, match="placement group creation is unconfirmed"):
        session.start()
    assert session.state["status"] == "cleanup_failed"
    assert session.state["generation"]["cleanup_errors"]


def test_inherited_role_settings_reach_launch_commands_and_environments(
    runtime, tmp_path
):
    settings = config()
    settings["envs"]["SGLANG_USE_AITER"] = "0"
    settings["disaggregation"]["prefill"]["envs"] = {
        "SGLANG_USE_AITER": "1",
        "EXTRA_SGLANG_ARGS": "--context-length 1024 --disable-cuda-graph",
    }
    with NativeRayDISession(settings, tmp_path) as session:
        prefill = session.actors["prefill-0"]
        decode = session.actors["decode-0"]
        assert (
            prefill.launch_envs["engine"]["TP"]
            == decode.launch_envs["engine"]["TP"]
            == "2"
        )
        assert prefill.launch_envs["engine"]["EP"] == "1"
        assert decode.launch_envs["engine"]["EP"] == "2"
        assert (
            prefill.launch_envs["engine"]["PP"]
            == decode.launch_envs["engine"]["PP"]
            == "1"
        )
        assert prefill.launch_envs["engine"]["SGLANG_USE_AITER"] == "1"
        assert decode.launch_envs["engine"]["SGLANG_USE_AITER"] == "0"
        assert "--disable-cuda-graph" in prefill.commands["engine"]
        assert "--disable-cuda-graph" not in decode.commands["engine"]
        assert prefill.launch_envs["router"]["SGLANG_USE_AITER"] == "0"


@pytest.mark.parametrize(
    "key",
    [
        "MAGPIE_BENCHMARK_PYTHON",
        "MAGPIE_BENCHMARK_BACKEND",
        "MAGPIE_BENCHMARK_ENDPOINT",
        "MAGPIE_CLIENT_TOKENIZER_MODE",
        "HYPERLOOM_CLIENT_TOKENIZER_MODE",
        "MAGPIE_TRUST_REMOTE_CODE",
        "MAGPIE_ACCURACY_REPORT_PYTHON",
        "MAGPIE_INFX_CLIENT_PYTHON",
        "MAGPIE_INFERENCEX_ROOT",
    ],
)
def test_role_env_cannot_override_client_controls_before_session_creation(
    tmp_path, key
):
    settings = config()
    settings["disaggregation"]["prefill"]["envs"] = {key: "override"}
    state = tmp_path / "rejected-session"
    with pytest.raises(
        ValueError, match="role envs cannot override benchmark client settings"
    ):
        NativeRayDISession(settings, state)
    assert not state.exists()


@pytest.mark.parametrize("role", [None, "decode"])
def test_disaggregation_rejects_existing_benchmark_profile_switch(tmp_path, role):
    settings = config()
    envs = (
        settings["envs"]
        if role is None
        else settings["disaggregation"][role].setdefault("envs", {})
    )
    envs["PROFILE"] = "1"
    with pytest.raises(
        ValueError, match="coordinated native DI profiling is unsupported"
    ):
        NativeRayDISession(settings, tmp_path / "rejected")


@pytest.mark.parametrize("tp,pp", [(4, 1), (2, 2)])
def test_sglang_owns_rank_gpus_across_multiple_nodes(runtime, tmp_path, tp, pp):
    settings = config()
    settings["disaggregation"]["prefill"]["envs"] = {"TP": tp, "PP": pp}
    session = NativeRayDISession(settings, tmp_path)
    session.start()
    generation = session.state["generation"]
    layout = generation["rank_layout"]
    assert layout["prefill-0"]["nodes"] == ["node-0", "node-1"]
    assert layout["decode-0"]["nodes"] == ["node-2"]
    assert layout["prefill-0"]["bundle_indices"] == [0, 1, 2, 3]
    assert layout["decode-0"]["bundle_indices"] == [4, 5]
    bundles = next(event[1] for event in runtime.events if event[0] == "placement")
    assert len(bundles) == 6 and sum(bundle["GPU"] for bundle in bundles) == 6
    assert [bundle.get("CPU", 0) for bundle in bundles] == [1, 0, 0, 0, 1, 0]
    assert [
        next(key for key in bundle if key.startswith("node:")) for bundle in bundles
    ] == [
        "node:10.0.0.1",
        "node:10.0.0.1",
        "node:10.0.0.2",
        "node:10.0.0.2",
        "node:10.0.0.3",
        "node:10.0.0.3",
    ]
    frontend_options = [
        event[1] for event in runtime.events if event[0] == "actor_options"
    ]
    assert all(options["num_gpus"] == 0 for options in frontend_options)
    assert all(
        identity["gpu_ids"] == [] for identity in generation["identities"].values()
    )
    command = session.actors["prefill-0"].commands["engine"]
    assert command[:3] == ["/worker/bin/python", "-m", "Magpie.di.sglang_ray"]
    assert command[command.index("--pg-id") + 1] == generation["placement_group_id"]
    assert command[command.index("--bundle-indices") + 1] == "0,1,2,3"
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--tensor-parallel-size") + 1] == str(tp)
    assert command[command.index("--pp-size") + 1] == str(pp)
    assert "--use-ray" in command
    assert len(generation["schedulers"]["prefill-0"]) == 4
    assert len(generation["schedulers"]["decode-0"]) == 2
    session.stop()
    assert not runtime.groups and not runtime.actors


def test_rank_planner_uses_available_gpus_instead_of_total_node_capacity(runtime):
    from Magpie.di.native_ray import _rank_layout

    runtime.available["node-0"]["GPU"] = 0
    instances = [
        {"id": "prefill-0", "tp": 4, "pp": 1},
        {"id": "decode-0", "tp": 2, "pp": 1},
    ]
    bundles, layout = _rank_layout(instances, runtime.nodes, runtime.available)
    assert layout["prefill-0"]["nodes"] == ["node-1", "node-2"]
    assert layout["decode-0"]["nodes"] == ["node-3"]
    assert not any("node:10.0.0.1" in bundle for bundle in bundles)


def install_scheduler_doubles(session, runtime, *, write_journal=True):
    generation = session.state["generation"]
    installed = []
    for role, names in generation["schedulers"].items():
        ids = []
        for index, name in enumerate(names):
            identity = f"{role}-rank-{index}"
            actor = SimpleNamespace(
                name=name,
                instance={"id": name},
                _actor_id=SimpleNamespace(hex=lambda identity=identity: identity),
            )
            runtime.actors[name] = actor
            ids.append(identity)
            installed.append(name)
        if write_journal:
            path = Path(generation["directory"]) / role / "ray-engine.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "pg_id": generation["placement_group_id"],
                        "namespace": session.spec["ray_config"]["namespace"],
                        "scheduler_names": names,
                        "actor_ids": ids,
                    }
                )
            )
    return installed


@pytest.mark.parametrize("write_journal", [True, False])
def test_scheduler_cleanup_is_verified_after_frontends_and_before_pg_release(
    runtime, tmp_path, write_journal
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    schedulers = install_scheduler_doubles(
        session, runtime, write_journal=write_journal
    )
    unrelated = SimpleNamespace(name="unrelated", instance={"id": "unrelated"})
    runtime.actors["unrelated"] = unrelated
    session.stop()
    stops = [index for index, event in enumerate(runtime.events) if event[0] == "stop"]
    rank_kills = [
        index
        for index, event in enumerate(runtime.events)
        if event[0] == "kill" and event[1] in schedulers
    ]
    released = next(
        index for index, event in enumerate(runtime.events) if event[0] == "release"
    )
    assert max(stops) < min(rank_kills) < max(rank_kills) < released
    assert all(name not in runtime.actors for name in schedulers)
    assert runtime.actors == {"unrelated": unrelated}
    assert session.state["generation"]["scheduler_cleanup_verified"]
    assert not runtime.groups


@pytest.mark.parametrize(
    "corruption",
    [
        "pg_id",
        "namespace",
        "scheduler_names",
        "missing_ids",
        "wrong_id",
        "compatibility_error",
    ],
)
def test_scheduler_journal_identity_failure_retains_gpu_reservations(
    runtime, tmp_path, corruption
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    names = install_scheduler_doubles(session, runtime)
    journal = (
        Path(session.state["generation"]["directory"]) / "prefill-0/ray-engine.json"
    )
    record = json.loads(journal.read_text())
    if corruption == "missing_ids":
        record["actor_ids"] = record["actor_ids"][:-1]
    elif corruption == "wrong_id":
        record["actor_ids"][0] = "different-actor"
    else:
        record[corruption] = (
            ["other-scheduler"] if corruption == "scheduler_names" else "different"
        )
    journal.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="identit|compatibility"):
        session.stop()
    assert runtime.groups
    assert session.state["status"] == "cleanup_failed"
    assert not session.state["generation"]["cleanup_verified"]
    assert not any(event[0] == "release" for event in runtime.events)
    assert all(name in runtime.actors for name in names)


def test_uncertain_remote_scheduler_exit_holds_entire_topology(
    runtime, tmp_path, monkeypatch
):
    from Magpie.di import native_ray

    session = NativeRayDISession(config(), tmp_path)
    session.start()
    names = install_scheduler_doubles(session, runtime)
    original_kill = runtime.ray.kill

    def kill(actor, **kwargs):
        if actor.name in names:
            runtime.events.append(("kill-stuck", actor.name))
        else:
            original_kill(actor, **kwargs)

    runtime.ray.kill = kill
    clock = [0]
    monkeypatch.setattr(
        native_ray,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + 31),
        ),
    )
    with pytest.raises(RuntimeError, match="scheduler cleanup was not confirmed"):
        session.stop()
    assert session.state["status"] == "cleanup_failed"
    assert runtime.groups and all(name in runtime.actors for name in names)
    assert not any(event[0] == "release" for event in runtime.events)


def test_recovery_rejects_replaced_named_placement_group(runtime, tmp_path):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    original_id = session.pg.id.hex
    session.pg.id.hex = lambda: "replacement-group-id"
    before = list(runtime.events)
    with pytest.raises(RuntimeError, match="placement group identity changed"):
        NativeRayDISession.load(tmp_path)
    assert runtime.events == before
    assert runtime.groups
    session.pg.id.hex = original_id
    session.stop()


@pytest.mark.parametrize(
    "corruption", ["missing-instance", "duplicate-rank", "out-of-range", "shared-node"]
)
def test_recovery_rejects_invalid_rank_ownership_before_ray_operations(
    runtime, tmp_path, corruption
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    original = session.path.read_text()
    record = json.loads(original)
    layout = record["generation"]["rank_layout"]
    if corruption == "missing-instance":
        layout.pop("decode-0")
    elif corruption == "duplicate-rank":
        layout["decode-0"]["bundle_indices"] = list(
            layout["prefill-0"]["bundle_indices"]
        )
    elif corruption == "out-of-range":
        layout["decode-0"]["bundle_indices"][0] = 1000
    else:
        layout["decode-0"]["nodes"] = list(layout["prefill-0"]["nodes"])
    record["generations"][-1] = deepcopy(record["generation"])
    session.path.write_text(json.dumps(record))
    before = list(runtime.events)
    with pytest.raises(RuntimeError, match="rank"):
        NativeRayDISession.load(tmp_path)
    assert runtime.events == before
    assert runtime.groups
    session.path.write_text(original)
    session.stop()


@pytest.mark.parametrize("corruption", ["missing", "empty"])
def test_missing_launched_scheduler_ownership_cannot_confirm_cleanup(
    runtime, tmp_path, corruption
):
    session = NativeRayDISession(config(), tmp_path)
    session.start()
    names = install_scheduler_doubles(session, runtime)
    generation = session.state["generation"]
    if corruption == "missing":
        generation["schedulers"].pop("prefill-0")
    else:
        generation["schedulers"]["prefill-0"] = []
    session._save()
    with pytest.raises(RuntimeError, match="scheduler|ownership"):
        recovered = NativeRayDISession.load(tmp_path)
        recovered.stop()
    assert runtime.groups
    assert not any(event[0] == "release" for event in runtime.events)
    assert any(name in runtime.actors for name in names)
