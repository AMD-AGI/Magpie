"""Native topology and public dispatch contracts; no Ray cluster is started."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from Magpie.di.native_config import normalize_native_di


@pytest.fixture
def benchmark():
    return {
        "framework": "sglang",
        "run_mode": "ray",
        "model": "org/prepared-model",
        "envs": {
            "TP": 8,
            "EP": 1,
            "ISL": 1024,
            "OSL": 100,
            "CONC": 4,
            "NUM_PROMPTS": 16,
            "NCCL_DEBUG": "WARN",
            "EXTRA_SGLANG_ARGS": "--context-length 8192",
        },
        "ray_config": {
            "cluster_address": "auto",
            "shared_storage_path": "/shared/results",
        },
        "disaggregation": {
            "prefill": {"instances": 1},
            "decode": {"instances": 1, "envs": {"EP": 8}},
        },
    }


def test_normalizes_explicit_roles_without_mutating_workload(benchmark):
    original = deepcopy(benchmark)
    benchmark["disaggregation"]["prefill"]["instances"] = 2
    normalized = normalize_native_di(benchmark)
    assert [item["id"] for item in normalized["instances"]] == [
        "prefill-0",
        "prefill-1",
        "decode-0",
    ]
    assert {
        key: normalized["instances"][0][key] for key in ("id", "role", "tp", "ep", "pp")
    } == {
        "id": "prefill-0",
        "role": "prefill",
        "tp": 8,
        "ep": 1,
        "pp": 1,
    }
    assert normalized["serving_env"] == {"TP": "8", "EP": "1", "NCCL_DEBUG": "WARN"}
    assert normalized["instances"][0]["server_args"] == ["--context-length", "8192"]
    assert normalized["model"] == original["model"]
    assert normalized["server_port"] == 30000 and normalized["router_port"] == 18000
    assert (
        normalized["bootstrap_port"] == 8998 and normalized["startup_timeout"] == 1800
    )
    normalized["ray_config"]["cluster_address"] = "changed"
    assert benchmark["ray_config"] == original["ray_config"]
    assert benchmark["envs"] == original["envs"]


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("prefill", "TP", 0),
        ("decode", "EP", -1),
        ("prefill", "TP", True),
        ("decode", "PP", 0),
        ("prefill", "EP", 1.5),
    ],
)
def test_rejects_invalid_role_sizes(benchmark, role, field, value):
    benchmark["disaggregation"][role].setdefault("envs", {})[field] = value
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


@pytest.mark.parametrize("value", [0, -1, True, "2", 1.5])
def test_rejects_invalid_instance_count(benchmark, value):
    benchmark["disaggregation"]["decode"]["instances"] = value
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


def test_parallelism_and_instance_counts_default_to_one(benchmark):
    benchmark["envs"].pop("TP")
    benchmark["envs"].pop("EP")
    benchmark["disaggregation"] = {"prefill": {}, "decode": {}}
    normalized = normalize_native_di(benchmark)
    assert [
        (item["tp"], item["ep"], item["pp"]) for item in normalized["instances"]
    ] == [(1, 1, 1), (1, 1, 1)]


@pytest.mark.parametrize("key", ["TP", "EP", "PP"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_rejects_invalid_common_parallelism(benchmark, key, value):
    benchmark["envs"][key] = value
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


@pytest.mark.parametrize(
    "arguments",
    [
        "--tp 8",
        "--model other",
        "--config /tmp/server.yaml",
        "--ep-size=4",
        "--pp-size 1",
        "--model-path /other",
        "--host 0.0.0.0",
        "--port=1234",
        "--disaggregation-mode prefill",
        "--disaggregation-bootstrap-port 8999",
    ],
)
def test_rejects_topology_overrides_in_extra_args(benchmark, arguments):
    benchmark["envs"]["EXTRA_SGLANG_ARGS"] = arguments
    with pytest.raises(ValueError, match="topology"):
        normalize_native_di(benchmark)


@pytest.mark.parametrize("placement", ["envs", "ray"])
def test_rejects_gpu_visibility_override(benchmark, placement):
    env = (
        benchmark["envs"]
        if placement == "envs"
        else benchmark["ray_config"].setdefault("env_vars", {})
    )
    env["rocr_visible_devices"] = "0"
    with pytest.raises(ValueError, match="visibility"):
        normalize_native_di(benchmark)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"docker_image": "candidate:other"}, "prepared Ray worker image"),
        ({"run_mode": "local"}, "run_mode=ray"),
        ({"framework": "vllm"}, "framework=sglang"),
        ({"profiler": {"torch_profiler": {"enabled": True}}}, "profiling"),
        ({"agentx": {"enabled": True}}, "AgentX"),
    ],
)
def test_rejects_unsupported_native_modes(benchmark, change, match):
    benchmark.update(change)
    with pytest.raises(ValueError, match=match):
        normalize_native_di(benchmark)


@pytest.mark.parametrize(
    "options",
    [
        {"server_port": 0},
        {"router_port": 30000},
        {"bootstrap_port": 65536},
        {"startup_timeout": float("inf")},
    ],
)
def test_rejects_invalid_ports_and_timeouts(benchmark, options):
    benchmark["disaggregation"].update(options)
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


def test_client_workload_and_accuracy_changes_do_not_change_serving(benchmark):
    initial = normalize_native_di(benchmark)
    benchmark["envs"].update(
        CONC=16, RUN_EVAL="true", MAGPIE_EVAL_TASKS="gsm8k", EVAL_LIMIT=100
    )
    changed = normalize_native_di(benchmark)
    assert changed["serving_env"] == initial["serving_env"]
    assert benchmark["envs"]["MAGPIE_EVAL_TASKS"] == "gsm8k"


def test_native_context_and_nic_are_explicit_runtime_settings(benchmark):
    benchmark["envs"]["MAX_MODEL_LEN"] = "8192"
    benchmark["disaggregation"]["nic"] = "mlx5_0"
    normalized = normalize_native_di(benchmark)
    assert normalized["instances"][0]["server_args"] == ["--context-length", "8192"]
    assert normalized["nic"] == "mlx5_0"
    benchmark["envs"]["EXTRA_SGLANG_ARGS"] = "--disable-radix-cache"
    assert normalize_native_di(benchmark)["instances"][0]["server_args"][-2:] == [
        "--context-length",
        "8192",
    ]
    benchmark["envs"]["MAX_MODEL_LEN"] = 4096
    benchmark["envs"]["EXTRA_SGLANG_ARGS"] = "--context-length 8192"
    with pytest.raises(ValueError, match="conflicts"):
        normalize_native_di(benchmark)


def test_nested_image_is_rejected_before_unknown_option_error(benchmark):
    benchmark["disaggregation"]["docker_image"] = "another:runtime"
    with pytest.raises(ValueError, match="prepared Ray worker image"):
        normalize_native_di(benchmark)


def test_main_ordinary_benchmark_uses_existing_execution_path(
    benchmark, tmp_path, monkeypatch
):
    from Magpie import main
    from Magpie.modes import benchmark as mode_module

    ordinary = deepcopy(benchmark)
    ordinary.pop("disaggregation")
    ordinary["envs"]["TP"] = 8
    path = tmp_path / "ordinary.yaml"
    path.write_text(yaml.safe_dump({"benchmark": ordinary}))
    calls = []

    class Mode:
        def __init__(self, config, output_dir):
            calls.append(config.envs["TP"])

        def run(self):
            return SimpleNamespace(
                success=True, workspace_dir=tmp_path, get_summary=lambda: "ordinary"
            )

    monkeypatch.setattr(mode_module, "BenchmarkMode", Mode)
    assert (
        main.run_benchmark(
            SimpleNamespace(benchmark_config=path, output_dir=tmp_path), {}
        )
        == 0
    )
    assert calls == [8]


def test_native_constructor_creates_normalized_session_without_ray(benchmark, tmp_path):
    from Magpie.di.native_ray import NativeRayDISession

    session = NativeRayDISession(benchmark, tmp_path / "constructed-session")
    assert isinstance(session, NativeRayDISession)
    assert session.state["status"] == "created"
    assert session.actors == {} and session.pg is None
    assert session.spec["model"] == benchmark["model"]
    assert [
        (item["id"], item["tp"], item["ep"], item["pp"])
        for item in session.spec["instances"]
    ] == [("prefill-0", 8, 1, 1), ("decode-0", 8, 8, 1)]
    assert (
        json.loads((tmp_path / "constructed-session/native-session.json").read_text())[
            "status"
        ]
        == "created"
    )


def test_benchmark_config_native_roundtrip_keeps_namespace_and_safe_ray_defaults(
    benchmark,
):
    from Magpie.modes.benchmark import BenchmarkConfig

    benchmark["ray_config"].update(
        namespace="native-contract",
        expected_node_ids=["node-a", "node-b"],
        placement_timeout_seconds=27,
    )
    before = deepcopy(benchmark)
    config = BenchmarkConfig.from_dict(benchmark)
    result = config.to_dict()
    assert config.disaggregation == benchmark["disaggregation"]
    assert result["ray_config"]["namespace"] == "native-contract"
    assert result["ray_config"]["expected_node_ids"] == ["node-a", "node-b"]
    assert result["ray_config"]["placement_timeout_seconds"] == 27
    assert config.ray_config.entrypoint_num_cpus == 1
    assert config.ray_config.install_magpie is False
    assert BenchmarkConfig.from_dict(result).to_dict() == result
    assert benchmark == before


def test_native_empty_env_defaults_do_not_create_global_tp(benchmark):
    from Magpie.modes.benchmark import BenchmarkConfig

    benchmark["envs"] = {}
    config = BenchmarkConfig.from_dict(benchmark)
    assert "TP" not in config.envs
    assert config.envs["CONC"] == 32
    assert not config.profiler.torch_profiler.enabled
    assert not config.profiler.gpu_monitor.enabled
    ordinary = BenchmarkConfig.from_dict(
        {"framework": "sglang", "model": "normal", "run_mode": "ray"}
    )
    assert (
        ordinary.disaggregation is None and "disaggregation" not in ordinary.to_dict()
    )
    assert ordinary.envs["TP"] == 1
    assert ordinary.ray_config.entrypoint_num_cpus == 16
    assert ordinary.ray_config.install_magpie is True
    assert ordinary.profiler.torch_profiler.enabled


def test_native_driver_uses_existing_ray_execution_with_intact_topology(
    benchmark, tmp_path, monkeypatch
):
    from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode

    mode = BenchmarkMode(BenchmarkConfig.from_dict(benchmark), output_dir=str(tmp_path))
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        mode, "_execute_ray_benchmark", lambda: calls.append("ray") or sentinel
    )
    monkeypatch.setattr(
        mode,
        "_execute_native_di_benchmark",
        lambda *args, **kwargs: pytest.fail("driver must not launch serving locally"),
    )
    assert mode.run() is sentinel
    assert calls == ["ray"]
    task, error = mode._build_ray_benchmark_task()
    assert error is None
    assert (
        task.to_dict()["mode_config"]["benchmark_config"]["disaggregation"]
        == benchmark["disaggregation"]
    )
    assert task.to_dict()["mode_config"]["benchmark_config"]["envs"]["CONC"] == 4


def test_remote_di_task_enters_worker_phase_without_recursive_run(
    benchmark, tmp_path, monkeypatch
):
    from Magpie.modes import benchmark as module
    from Magpie.remote.tasks import _run_benchmark

    events = []

    class Mode:
        def __init__(self, config, output_dir):
            events.append(("construct", config, output_dir))

        def _execute_native_di_benchmark(self, task_id=None):
            events.append(("di-worker", task_id))
            return SimpleNamespace(to_dict=lambda: {"success": True})

        def run(self, *args, **kwargs):
            pytest.fail(
                "DI worker must use explicit phase, not re-enter run/Ray dispatch"
            )

    monkeypatch.setattr(module, "BenchmarkMode", Mode)
    result = _run_benchmark(
        {"benchmark_config": deepcopy(benchmark)},
        {"shared_storage_path": str(tmp_path)},
        "one-task",
    )
    assert result == {"success": True}
    assert [event[0] for event in events] == ["construct", "di-worker"]
    assert events[0][1].run_mode == "ray"
    assert events[0][1].disaggregation == benchmark["disaggregation"]
    assert events[1][1] == "one-task"


def test_native_worker_runs_existing_client_phase_inside_single_session(
    benchmark, tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from Magpie.di import native_ray
    from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode, BenchmarkResult

    events = []

    class Session:
        endpoint = "http://prefill-fixture:18000"

        def __init__(self, raw, state_dir, check_cancelled=None):
            assert raw["disaggregation"] == benchmark["disaggregation"]
            self.state_dir = state_dir
            self.state_dir.mkdir(parents=True)
            self.path = state_dir / "native-session.json"
            self.state = {"status": "created"}
            events.append("construct")

        def __enter__(self):
            events.append("serving-ready")
            return self

        @contextmanager
        def client_run(self, task_id, output_dir):
            assert task_id == "single-ray-task"
            events.append("client-enter")
            try:
                yield
            finally:
                events.append("client-exit")

        def __exit__(self, *args):
            self.state["status"] = "stopped"
            events.append("serving-stopped")

    monkeypatch.setattr(native_ray, "NativeRayDISession", Session)
    mode = BenchmarkMode(BenchmarkConfig.from_dict(benchmark), output_dir=str(tmp_path))
    from Magpie.modes.benchmark import benchmarker

    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_get_runner_type", lambda: "mi355x")
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda runner: "compatible.sh")
    original = mode.config.to_dict()

    def phase(endpoint=None, workspace=None):
        assert endpoint == Session.endpoint
        assert workspace.is_dir()
        assert (workspace / "di-session").is_dir()
        events.append("existing-phase")
        result = BenchmarkResult(framework="sglang", model=benchmark["model"])
        result.success = True
        result.workspace_dir = workspace
        return result

    monkeypatch.setattr(mode, "_run_benchmark", phase)
    monkeypatch.setattr(
        mode,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "worker phase must not recursively call run"
        ),
    )
    result = mode._execute_native_di_benchmark(task_id="single-ray-task")
    assert result.success
    assert mode.config.to_dict() == original
    assert events == [
        "construct",
        "serving-ready",
        "client-enter",
        "existing-phase",
        "client-exit",
        "serving-stopped",
    ]


def test_async_native_di_uses_existing_executor_submission(benchmark, tmp_path):
    from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode

    submitted = []
    executor = SimpleNamespace(
        submit=lambda task: submitted.append(task) or "one-ray-task"
    )
    mode = BenchmarkMode(BenchmarkConfig.from_dict(benchmark), output_dir=str(tmp_path))
    before = mode.config.to_dict()
    result = mode.submit_ray_benchmark(executor)
    assert result.success is True
    assert result.metadata["task_id"] == "one-ray-task"
    assert len(submitted) == 1
    assert (
        submitted[0].to_dict()["mode_config"]["benchmark_config"]["disaggregation"]
        == benchmark["disaggregation"]
    )
    assert mode.config.to_dict() == before


def test_benchmark_cancellation_restores_callers_signal_handlers():
    import signal

    from Magpie.di.native_ray import cancellation_signals

    previous = signal.getsignal(signal.SIGTERM)
    with cancellation_signals() as check_cancelled:
        check_cancelled()
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        with pytest.raises(RuntimeError, match="interrupted"):
            check_cancelled()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_public_session_api_exports_native_implementation():
    import importlib

    import Magpie.di
    from Magpie.di.native_ray import NativeRayDISession

    assert importlib.reload(Magpie.di).DISession is NativeRayDISession


def public_document(benchmark):
    return {"benchmark": deepcopy(benchmark)}


def test_session_constructor_accepts_existing_full_document(benchmark, tmp_path):
    from Magpie.di.native_ray import NativeRayDISession

    document = public_document(benchmark)
    before = deepcopy(document)
    session = NativeRayDISession(document, tmp_path / "full-document")
    assert [(item["tp"], item["ep"]) for item in session.spec["instances"]] == [
        (8, 1),
        (8, 8),
    ]
    assert session.spec["model"] == benchmark["model"]
    assert document == before


def test_roles_inherit_common_env_and_override_independently(benchmark):
    benchmark["envs"].update(TP="8", EP="1", MAX_MODEL_LEN="8192")
    benchmark["disaggregation"]["decode"]["envs"].update(
        TP="4", NCCL_DEBUG="INFO", EXTRA_SGLANG_ARGS="--disable-radix-cache"
    )
    spec = normalize_native_di(benchmark)
    prefill, decode = spec["instances"]
    assert (prefill["tp"], prefill["ep"], prefill["pp"]) == (8, 1, 1)
    assert (decode["tp"], decode["ep"], decode["pp"]) == (4, 8, 1)
    assert prefill["server_args"] == ["--context-length", "8192"]
    assert decode["server_args"] == [
        "--disable-radix-cache",
        "--context-length",
        "8192",
    ]
    assert prefill["serving_env"]["NCCL_DEBUG"] == "WARN"
    assert decode["serving_env"]["NCCL_DEBUG"] == "INFO"
    assert {key: decode["serving_env"][key] for key in ("TP", "EP", "PP")} == {
        "TP": "4",
        "EP": "8",
        "PP": "1",
    }
    assert spec["serving_env"]["NCCL_DEBUG"] == "WARN"
    assert "CONC" not in decode["serving_env"]


@pytest.mark.parametrize(
    "common_key,role_key", [("tp", "TP"), ("TP", "tp"), ("Tp", "tP")]
)
def test_role_env_case_matches_ordinary_benchmark_precedence(
    benchmark, common_key, role_key
):
    from Magpie.modes.benchmark import BenchmarkConfig

    benchmark["envs"] = {common_key: 4, "ep": 1, "nccl_debug": "WARN"}
    benchmark["disaggregation"] = {
        "prefill": {"instances": 1},
        "decode": {
            "instances": 1,
            "envs": {role_key: 2, "eP": 2, "NCCL_DEBUG": "INFO"},
        },
    }
    original = deepcopy(benchmark)
    ordinary = deepcopy(benchmark)
    ordinary.pop("disaggregation")
    expected = BenchmarkConfig.from_dict(ordinary).get_env_vars()
    spec = normalize_native_di(benchmark)
    prefill, decode = spec["instances"]
    assert prefill["tp"] == int(expected["TP"]) == 4
    assert prefill["ep"] == int(expected["EP"]) == 1
    assert prefill["serving_env"]["NCCL_DEBUG"] == expected["NCCL_DEBUG"]
    assert (decode["tp"], decode["ep"]) == (2, 2)
    assert decode["serving_env"]["TP"] == "2"
    assert decode["serving_env"]["EP"] == "2"
    assert decode["serving_env"]["NCCL_DEBUG"] == "INFO"
    assert "tp" not in decode["serving_env"]
    assert benchmark == original


@pytest.mark.parametrize(
    "key",
    ["CONC", "ISL", "RUN_EVAL", "MAGPIE_EVAL_TASKS", "BENCHMARK_BASE_URL", "PORT"],
)
def test_rejects_role_client_workload_and_endpoint_overrides(benchmark, key):
    benchmark["disaggregation"]["decode"]["envs"][key] = "other"
    with pytest.raises(
        ValueError,
        match="role envs cannot override benchmark client settings|owns its endpoint",
    ):
        normalize_native_di(benchmark)


@pytest.mark.parametrize(
    "env",
    [
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"EXTRA_SGLANG_ARGS": "--tp 4"},
        {"ENABLE_PROFILE": "true"},
        {"MAX_MODEL_LEN": 4096},
    ],
)
def test_role_settings_apply_same_serving_safety_checks(benchmark, env):
    benchmark["disaggregation"]["decode"]["envs"].update(env)
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


@pytest.mark.parametrize(
    "location", ["top-level-di", "role-replicas", "role-tp", "nested-benchmark-di"]
)
def test_legacy_topology_schema_is_not_accepted(benchmark, location):
    if location == "top-level-di":
        benchmark["di"] = benchmark.pop("disaggregation")
    elif location == "role-replicas":
        benchmark["disaggregation"]["prefill"]["replicas"] = 2
    elif location == "role-tp":
        benchmark["disaggregation"]["prefill"]["tp"] = 8
    else:
        benchmark["di"] = {"prefill": {"tp": 8, "ep": 1}, "decode": {"tp": 8, "ep": 8}}
    with pytest.raises(ValueError):
        normalize_native_di(benchmark)


def test_main_public_yaml_preserves_nested_disaggregation(
    benchmark, tmp_path, monkeypatch
):
    from Magpie import main
    from Magpie.modes import benchmark as mode_module

    path = tmp_path / "di.yaml"
    path.write_text(yaml.safe_dump(public_document(benchmark)))
    seen = []

    class Mode:
        def __init__(self, config, output_dir):
            seen.append(config)

        def run(self):
            return SimpleNamespace(
                success=True, workspace_dir=tmp_path, get_summary=lambda: "native"
            )

    monkeypatch.setattr(mode_module, "BenchmarkMode", Mode)
    assert (
        main.run_benchmark(
            SimpleNamespace(benchmark_config=path, output_dir=tmp_path), {}
        )
        == 0
    )
    assert len(seen) == 1
    assert seen[0].disaggregation == benchmark["disaggregation"]
    assert seen[0].envs["TP"] == 8 and seen[0].envs["EP"] == 1


def test_pipeline_parallelism_is_preserved_for_serving_not_multiplied_into_tp(
    benchmark,
):
    benchmark["disaggregation"]["prefill"]["envs"] = {"TP": 4, "PP": 2}
    spec = normalize_native_di(benchmark)
    prefill = spec["instances"][0]
    assert (prefill["tp"], prefill["pp"]) == (4, 2)
    assert prefill["serving_env"]["TP"] == "4"
    assert prefill["serving_env"]["PP"] == "2"


@pytest.mark.parametrize(
    "key", ["PROFILE", "profile", "ENABLE_PROFILE", "enable_profile"]
)
@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_ray_runtime_env_cannot_bypass_di_profiling_restriction(benchmark, key, value):
    benchmark["ray_config"]["env_vars"] = {key: value}
    with pytest.raises(ValueError, match="profiling is unsupported"):
        normalize_native_di(benchmark)


def test_ray_runtime_disabled_profiling_does_not_change_non_di_configuration(benchmark):
    from Magpie.modes.benchmark import BenchmarkConfig

    benchmark["ray_config"]["env_vars"] = {"PROFILE": "0", "ENABLE_PROFILE": "false"}
    assert (
        normalize_native_di(benchmark)["ray_config"]["env_vars"]
        == benchmark["ray_config"]["env_vars"]
    )
    ordinary = deepcopy(benchmark)
    ordinary.pop("disaggregation")
    ordinary["ray_config"]["env_vars"] = {"PROFILE": "1"}
    assert BenchmarkConfig.from_dict(ordinary).ray_config.env_vars == {"PROFILE": "1"}


@pytest.mark.parametrize("placement", ["common", "role", "ray"])
def test_caller_cannot_override_internal_serving_env_handoff(benchmark, placement):
    if placement == "common":
        env = benchmark["envs"]
    elif placement == "role":
        env = benchmark["disaggregation"]["decode"]["envs"]
    else:
        env = benchmark["ray_config"].setdefault("env_vars", {})
    env["magpie_di_serving_env"] = "{}"
    with pytest.raises(ValueError, match="internal DI"):
        normalize_native_di(benchmark)
