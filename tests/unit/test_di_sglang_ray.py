"""The HTTP bridge passes one existing group to pinned SGLang, without GPUs."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from Magpie.di import sglang_ray


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("MAGPIE_DI_SERVING_ENV", "{}")
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    ):
        monkeypatch.delenv(key, raising=False)
    pg = SimpleNamespace(id=SimpleNamespace(hex=lambda: "12345678abcdef"))
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "actor-id"))
    actors = [
        actor,
        SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "actor-id-2")),
    ]
    monkeypatch.setattr(sglang_ray, "version", lambda package: "0.5.18")
    args = SimpleNamespace(
        use_ray=True,
        dp_size=1,
        tp_size=2,
        pp_size=1,
        encoder_only=False,
        smg_grpc_mode=False,
        grpc_mode=False,
        grpc_port=None,
        nnodes=1,
        dist_init_addr=None,
    )
    journal = tmp_path / "schedulers.json"
    ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: events.append(("init", kwargs)),
        shutdown=lambda: events.append(("shutdown",)),
        get_runtime_context=lambda: SimpleNamespace(
            namespace="evaluation", get_job_id=lambda: "job-id"
        ),
        util=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.1"),
        kill=lambda actor, **kwargs: events.append(("kill", actor, kwargs)),
    )

    def get_actor(name, namespace):
        names = sglang_ray.scheduler_names(
            "12345678abcdef", "10.0.0.1", [3, 4], args.tp_size
        )
        if name not in names:
            raise ValueError("not found")
        return actors[names.index(name)]

    ray.get_actor = get_actor
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(
        sys.modules,
        "ray.util.placement_group",
        SimpleNamespace(
            get_placement_group=lambda name: events.append(("lookup", name)) or pg
        ),
    )
    for name in ["sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.ray"]:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins",
        SimpleNamespace(load_plugins=lambda: events.append(("plugins",))),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.server_args",
        SimpleNamespace(
            prepare_server_args=lambda argv: events.append(("args", argv)) or args
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.engine",
        SimpleNamespace(
            init_tokenizer_manager=object(),
            run_detokenizer_process=object(),
            run_scheduler_process=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils",
        SimpleNamespace(
            kill_process_tree=lambda pid, **kwargs: events.append(
                ("local-cleanup", kwargs)
            )
        ),
    )

    def startup(server_args, *, placement_group, **kwargs):
        saved = json.loads(journal.read_text())
        assert saved["status"] == "starting"
        assert saved["scheduler_names"] == sglang_ray.scheduler_names(
            "12345678abcdef", "10.0.0.1", [3, 4], 2
        )
        assert placement_group is pg and server_args is args
        events.append(("engine", placement_group))
        return (
            "tokenizer",
            "templates",
            "ports",
            SimpleNamespace(scheduler_actors=actors, scheduler_infos=[{"ready": True}]),
            "watchdog",
            [],
        )

    engine = SimpleNamespace(_launch_subprocesses=startup)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.ray.engine", SimpleNamespace(RayEngine=engine)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.http_server",
        SimpleNamespace(
            _execute_server_warmup="warmup",
            _setup_and_run_http_server=lambda *args, **kwargs: events.append(
                ("http", args, kwargs)
            ),
        ),
    )
    return SimpleNamespace(
        events=events,
        ray=ray,
        pg=pg,
        actor=actor,
        actors=actors,
        args=args,
        engine=engine,
        journal=journal,
    )


def launch(runtime):
    return sglang_ray.launch(
        "reserved-topology",
        "12345678abcdef",
        "evaluation",
        [3, 4],
        runtime.journal,
        ["--use-ray", "--model-path", "model"],
    )


def test_bridge_uses_reserved_group_and_stock_http_with_durable_names(
    runtime, monkeypatch
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    launch(runtime)
    import os

    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["SGLANG_RAY_BUNDLE_INDICES"] == "3,4"
    names = [event[0] for event in runtime.events]
    assert names == [
        "init",
        "lookup",
        "plugins",
        "args",
        "engine",
        "http",
        "kill",
        "kill",
        "local-cleanup",
        "shutdown",
    ]
    http = next(event for event in runtime.events if event[0] == "http")
    assert http[1][1:] == (
        "tokenizer",
        "templates",
        "ports",
        [{"ready": True}],
        "watchdog",
    )
    assert http[2] == {"execute_warmup_func": "warmup"}
    record = json.loads(runtime.journal.read_text())
    assert record["actor_ids"] == ["actor-id", "actor-id-2"]
    assert record["job_id"] == "job-id" and record["pg_id"] == "12345678abcdef"
    assert record["cleanup_requested"] and record["cleanup_errors"] == []
    assert "cleanup_verified" not in record


def test_partial_startup_keeps_expected_names_and_cleans_local_children(runtime):
    def startup(server_args, *, placement_group, **kwargs):
        assert runtime.journal.is_file()
        raise RuntimeError("scheduler initialization failed")

    runtime.engine._launch_subprocesses = startup
    with pytest.raises(RuntimeError, match="scheduler initialization failed"):
        launch(runtime)
    saved = json.loads(runtime.journal.read_text())
    assert len(saved["scheduler_names"]) == 2 and saved["actor_ids"] == []
    assert [event[0] for event in runtime.events][-2:] == ["local-cleanup", "shutdown"]
    assert not any(event[0] == "http" for event in runtime.events)


@pytest.mark.parametrize(
    "failure", ["identity", "signature", "http-signature", "mode", "rank-count"]
)
def test_incompatible_setup_fails_before_scheduler_launch(runtime, failure):
    if failure == "identity":
        runtime.pg.id.hex = lambda: "other-placement-group"
    elif failure == "signature":
        runtime.engine._launch_subprocesses = lambda server_args: None
    elif failure == "http-signature":
        sys.modules["sglang.srt.entrypoints.http_server"]._setup_and_run_http_server = (
            lambda server_args: None
        )
    elif failure == "mode":
        runtime.args.use_ray = False
    else:
        runtime.args.tp_size = 4
    with pytest.raises((RuntimeError, ValueError)):
        launch(runtime)
    assert not any(event[0] in {"engine", "http"} for event in runtime.events)
    assert not runtime.journal.exists()
    assert runtime.events[-1] == ("shutdown",)


def test_borrowed_wrong_namespace_is_not_disconnected(runtime):
    runtime.ray.is_initialized = lambda: True
    runtime.ray.get_runtime_context = lambda: SimpleNamespace(namespace="unrelated")
    with pytest.raises(ValueError, match="namespace"):
        launch(runtime)
    assert runtime.events == []


def test_cli_metadata_is_separate_from_stock_server_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sglang_ray, "launch", lambda *args: calls.append(args))
    sglang_ray.main(
        [
            "--pg-name",
            "group",
            "--pg-id",
            "abcdef",
            "--namespace",
            "ns",
            "--bundle-indices",
            "0,2",
            "--journal-path",
            str(tmp_path / "journal.json"),
            "--",
            "--use-ray",
            "--tp",
            "2",
        ]
    )
    assert calls == [
        (
            "group",
            "abcdef",
            "ns",
            [0, 2],
            str(tmp_path / "journal.json"),
            ["--use-ray", "--tp", "2"],
        )
    ]


def test_scheduler_names_match_pinned_custom_group_rank_formula():
    assert sglang_ray.scheduler_names(
        "12345678abcdef", "10.0.0.1", [5, 7, 9, 11], 2
    ) == [
        "sglang_scheduler_node10.0.0.1_dp0_pp0_tp0_pg12345678_bundle5",
        "sglang_scheduler_node10.0.0.1_dp0_pp0_tp1_pg12345678_bundle7",
        "sglang_scheduler_node10.0.0.1_dp0_pp1_tp0_pg12345678_bundle9",
        "sglang_scheduler_node10.0.0.1_dp0_pp1_tp1_pg12345678_bundle11",
    ]


@pytest.mark.parametrize("existing", [None, "10.0.0.1:22000"])
def test_multinode_rendezvous_is_ready_before_port_initialization(
    runtime, monkeypatch, existing
):
    import socket

    runtime.args.nnodes = 2
    runtime.args.dist_init_addr = existing
    binds = []
    listener = SimpleNamespace(
        bind=lambda address: binds.append(address),
        getsockname=lambda: ("10.0.0.1", 21000),
    )

    class Socket:
        def __enter__(self):
            return listener

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(socket, "socket", lambda family: Socket())
    original = runtime.engine._launch_subprocesses

    def startup(server_args, *, placement_group, **kwargs):
        assert server_args.dist_init_addr == (existing or "10.0.0.1:21000")
        return original(server_args, placement_group=placement_group, **kwargs)

    runtime.engine._launch_subprocesses = startup
    launch(runtime)
    assert binds == ([] if existing else [("10.0.0.1", 0)])


def test_pp_two_uses_all_rank_bundles_with_correct_scheduler_names(runtime):
    runtime.args.tp_size = 1
    runtime.args.pp_size = 2

    def startup(server_args, *, placement_group, **kwargs):
        record = json.loads(runtime.journal.read_text())
        assert record["scheduler_names"] == sglang_ray.scheduler_names(
            "12345678abcdef", "10.0.0.1", [3, 4], 1
        )
        return (
            "tokenizer",
            "templates",
            "ports",
            SimpleNamespace(scheduler_actors=runtime.actors, scheduler_infos=[{}]),
            "watchdog",
            [],
        )

    runtime.engine._launch_subprocesses = startup
    launch(runtime)


@pytest.mark.parametrize("installed", ["0.5.17", "0.5.19"])
def test_different_framework_version_rejected_before_gpu_actor_creation(
    runtime, monkeypatch, installed
):
    monkeypatch.setattr(sglang_ray, "version", lambda package: installed)
    with pytest.raises(RuntimeError, match="requires SGLang 0.5.18"):
        launch(runtime)
    assert not any(event[0] in {"engine", "http"} for event in runtime.events)
    assert not runtime.journal.exists()


@pytest.mark.parametrize("mismatch", ["identity", "count"])
def test_unexpected_scheduler_identity_fails_before_http_and_kills_returned_handles(
    runtime,
    mismatch,
):
    if mismatch == "identity":
        runtime.ray.get_actor = lambda name, namespace: SimpleNamespace(
            _actor_id=SimpleNamespace(hex=lambda: "different-actor")
        )
    else:
        runtime.actors.pop()
    with pytest.raises(RuntimeError, match="ownership API is incompatible"):
        launch(runtime)
    record = json.loads(runtime.journal.read_text())
    assert record["actor_ids"] == (
        ["actor-id", "actor-id-2"] if mismatch == "identity" else ["actor-id"]
    )
    assert record["compatibility_error"]
    assert not any(event[0] == "http" for event in runtime.events)
    assert sum(event[0] == "kill" for event in runtime.events) == len(runtime.actors)


@pytest.mark.parametrize("role_flag", ["0", "1"])
def test_gpu_scheduler_job_receives_effective_role_env_and_model_credentials(
    runtime, monkeypatch, role_flag
):
    import os

    role_env = {
        "SGLANG_USE_AITER": role_flag,
        "EP": "2",
        "NCCL_DEBUG": "WARN",
        "CUDA_VISIBLE_DEVICES": "7",
        "rocr_visible_devices": "3",
    }
    monkeypatch.setenv("MAGPIE_DI_SERVING_ENV", json.dumps(role_env))
    monkeypatch.setenv("HF_TOKEN", "dummy-model-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "dummy-legacy-token")
    monkeypatch.setenv("HF_HOME", "/prepared/model-cache")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-propagate")
    launch(runtime)
    supplied = next(event[1] for event in runtime.events if event[0] == "init")[
        "runtime_env"
    ]["env_vars"]
    assert supplied == {
        "SGLANG_USE_AITER": role_flag,
        "EP": "2",
        "NCCL_DEBUG": "WARN",
        "HF_TOKEN": "dummy-model-token",
        "HUGGING_FACE_HUB_TOKEN": "dummy-legacy-token",
        "HF_HOME": "/prepared/model-cache",
        "SGLANG_RAY_BUNDLE_INDICES": "3,4",
    }
    assert "MAGPIE_DI_SERVING_ENV" not in os.environ
    assert "dummy-model-token" not in runtime.journal.read_text()
    assert "UNRELATED_SECRET" not in supplied


def test_role_credential_precedes_inherited_credential(runtime, monkeypatch):
    monkeypatch.setenv(
        "MAGPIE_DI_SERVING_ENV", json.dumps({"HF_TOKEN": "dummy-role-token"})
    )
    monkeypatch.setenv("HF_TOKEN", "dummy-parent-token")
    launch(runtime)
    supplied = next(event[1] for event in runtime.events if event[0] == "init")[
        "runtime_env"
    ]["env_vars"]
    assert supplied["HF_TOKEN"] == "dummy-role-token"


def test_existing_driver_cannot_silently_ignore_per_role_runtime_env(runtime):
    runtime.ray.is_initialized = lambda: True
    with pytest.raises(RuntimeError, match="fresh Ray frontend driver"):
        launch(runtime)
    assert runtime.events == []


def test_invalid_serving_environment_handoff_fails_before_ray_connect(
    runtime, monkeypatch
):
    monkeypatch.setenv("MAGPIE_DI_SERVING_ENV", "[]")
    with pytest.raises(TypeError, match="must be a mapping"):
        launch(runtime)
    assert runtime.events == []
    assert not runtime.journal.exists()
