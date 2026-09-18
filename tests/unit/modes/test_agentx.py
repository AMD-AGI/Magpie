import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Magpie.modes.benchmark import agentx
from Magpie.modes.benchmark.agentx import (
    ensure_agentx_dependencies,
    resolve_agentx_recipe,
)
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.result import ResultParser
from Magpie.utils.gpu import GPUVendor


def _minimal_config(**overrides):
    data = {
        "framework": "sglang",
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "precision": "fp4",
        "agentx": "enabled",
        "docker_image": "sglang:pinned",
        "benchmark_script": "single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh",
    }
    data.update(overrides)
    return BenchmarkConfig.from_dict(data)


def _fake_inferencex(tmp_path: Path) -> Path:
    root = tmp_path / "InferenceX"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "amd-master.yaml").write_text(
        yaml.safe_dump(
            {
                "dsv4-fp4-mi355x-sglang-agentic-mtp": {
                    "image": "sglang:test",
                    "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
                    "model-prefix": "dsv4",
                    "runner": "cluster:mi355x-amds",
                    "precision": "fp4",
                    "framework": "sglang",
                    "multinode": False,
                    "scenarios": {
                        "agentic-coding": [
                            {
                                "dram-utilization": 0.8,
                                "search-space": [
                                    {
                                        "tp": 8,
                                        "ep": 1,
                                        "conc-list": [16, 32, 48],
                                        "spec-decoding": "mtp",
                                        "kv-offloading": "dram",
                                        "kv-offload-backend": {"name": "hicache"},
                                        "router": {
                                            "name": "vllm-router",
                                            "version": "test",
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "configs" / "runners.yaml").write_text(
        yaml.safe_dump(
            {
                "hardware": {
                    "cluster:mi355x-amds": {
                        "available-cpu-dram-mib": 3_095_781,
                        "gpus-per-node": 8,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_agentx_example_pins_latest_public_assets():
    root = Path(__file__).resolve().parents[3]
    example = (
        root
        / "examples"
        / "benchmarks"
        / "benchmark_sglang_deepseek_v4_pro_fp4_mi355x_agentx.yaml"
    )
    data = yaml.safe_load(example.read_text(encoding="utf-8"))["benchmark"]

    config = BenchmarkConfig.from_dict(data)

    assert config.is_agentx is True
    assert config.model == "deepseek-ai/DeepSeek-V4-Pro-0813"
    assert config.docker_image.endswith("v0.5.19-rocm720-mi35x-20260914")
    assert config.benchmark_script == (
        "single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh"
    )


@pytest.mark.parametrize("value", [True, "true", "enable", "enabled"])
def test_agentx_one_line_config_enables_safe_defaults(value):
    config = _minimal_config(agentx=value)

    assert config.is_agentx is True
    assert config.scenario == "agentx"
    assert config.envs == {"CONC": 32}
    assert config.profiler.torch_profiler.enabled is False
    assert config.profiler.gpu_monitor.enabled is False
    assert config.agentx is not None
    assert config.agentx.mode == "canonical"


def test_agentx_keeps_default_concurrency_with_other_environment_values():
    config = _minimal_config(envs={"MODEL_PATH": "/models/dsv4"})

    assert config.envs == {"MODEL_PATH": "/models/dsv4", "CONC": 32}


@pytest.mark.parametrize("value", [False, "false", "disable", "disabled"])
def test_agentx_disabled_values_preserve_normal_benchmark(value):
    config = _minimal_config(agentx=value)

    assert config.is_agentx is False
    assert config.scenario == "fixed-seq-len"
    assert config.envs["ISL"] == 1024


def test_agentx_rejects_magpie_profiler():
    with pytest.raises(ValueError, match="torch_profiler"):
        _minimal_config(
            profiler={"torch_profiler": {"enabled": True}},
        )


def test_agentx_requires_pinned_script_and_docker_image():
    with pytest.raises(ValueError, match="benchmark_script"):
        _minimal_config(benchmark_script=None)

    with pytest.raises(ValueError, match="docker_image"):
        _minimal_config(docker_image=None)


def test_resolve_agentx_uses_inferencex_recipe(tmp_path):
    root = _fake_inferencex(tmp_path)
    config = _minimal_config(inferencex_path=str(root))

    spec = resolve_agentx_recipe(config, str(root), runner_type="mi355x")

    assert spec.recipe == "dsv4-fp4-mi355x-sglang-agentic-mtp"
    assert config.docker_image == "sglang:pinned"
    assert config.envs["MODEL_PREFIX"] == "dsv4"
    assert config.envs["TP"] == 8
    assert config.envs["KV_OFFLOADING"] == "dram"
    assert config.envs["KV_OFFLOAD_BACKEND"] == "hicache"
    assert config.envs["TOTAL_CPU_DRAM_GB"] == 2399
    assert config.envs["ROUTER_METADATA"] == ('{"name":"vllm-router","version":"test"}')
    assert len(config.envs["RECIPE_FINGERPRINT"]) == 64
    assert spec.entry["image"] == "sglang:pinned"
    assert spec.entry["run-eval"] is False
    assert spec.entry["recipe-fingerprint"] == config.envs["RECIPE_FINGERPRINT"]
    assert config.envs["RECIPE_FINGERPRINT"] == (
        "f21963118866248a3f35d3ef991e86f02f071ca8f9c512473949882a6cf55691"
    )
    assert config.timeout_seconds == 7200

    another_point = _minimal_config(
        inferencex_path=str(root),
        envs={"CONC": 16},
    )
    resolve_agentx_recipe(another_point, str(root), runner_type="mi355x")
    assert another_point.envs["RECIPE_FINGERPRINT"] == config.envs["RECIPE_FINGERPRINT"]

    selected = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "selector": {"kv_offload_backend": "hicache"},
        },
    )
    resolve_agentx_recipe(selected, str(root), runner_type="mi355x")
    assert selected.agentx is not None
    assert selected.agentx.resolved["kv-offload-backend"]["name"] == "hicache"


def test_agentx_recipe_must_match_detected_gpu(tmp_path):
    root = _fake_inferencex(tmp_path)
    config = _minimal_config(inferencex_path=str(root))

    with pytest.raises(ValueError, match="no single-node AgentX recipe"):
        resolve_agentx_recipe(config, str(root), runner_type="b200")

    explicit = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )
    with pytest.raises(ValueError, match="not detected runner"):
        resolve_agentx_recipe(explicit, str(root), runner_type="b200")


def test_explicit_agentx_recipe_must_match_benchmark_identity(tmp_path):
    root = _fake_inferencex(tmp_path)
    config = _minimal_config(
        model="another/model",
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )

    with pytest.raises(ValueError, match="does not match the benchmark identity"):
        resolve_agentx_recipe(config, str(root), runner_type="mi355x")


def test_agentx_external_config_file_is_reported_as_absolute_path(tmp_path):
    root = _fake_inferencex(tmp_path)
    external = tmp_path / "external-agentx.yaml"
    external.write_text(
        (root / "configs" / "amd-master.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
            "config_file": str(external),
        },
    )

    spec = resolve_agentx_recipe(config, str(root), runner_type="mi355x")

    assert spec.config_file == str(external)


def test_agentx_rejects_invalid_concurrency_range(tmp_path):
    root = _fake_inferencex(tmp_path)
    config_file = root / "configs" / "amd-master.yaml"
    recipes = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    arm = recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"]["scenarios"]["agentic-coding"][
        0
    ]["search-space"][0]
    arm.pop("conc-list")
    arm.update({"conc-start": 0, "conc-end": 32})
    config_file.write_text(yaml.safe_dump(recipes), encoding="utf-8")

    with pytest.raises(ValueError, match="0 < start <= end"):
        resolve_agentx_recipe(
            _minimal_config(inferencex_path=str(root)),
            str(root),
            runner_type="mi355x",
        )


def test_agentx_concurrency_helpers_reject_invalid_values():
    with pytest.raises(ValueError, match="must contain integers"):
        agentx._concurrency_values({"conc-list": ["invalid"]})
    with pytest.raises(ValueError, match="must be positive"):
        agentx._concurrency_values({"conc-list": [0, 1]})
    with pytest.raises(ValueError, match="bounds must be integers"):
        agentx._concurrency_values({"conc-start": "invalid", "conc-end": 8})

    assert agentx._concurrency_values({}) == []
    assert agentx._concurrency_values({"conc-start": 3, "conc-end": 10}) == [3, 6, 10]


def test_agentx_recipe_lookup_rejects_missing_and_ambiguous_recipes(tmp_path):
    root = _fake_inferencex(tmp_path)
    missing = _minimal_config(
        inferencex_path=str(root),
        agentx={"enabled": True, "recipe": "missing-recipe"},
    )
    with pytest.raises(ValueError, match="was not found"):
        resolve_agentx_recipe(missing, str(root), runner_type="mi355x")

    amd_config = root / "configs" / "amd-master.yaml"
    nvidia_config = root / "configs" / "nvidia-master.yaml"
    nvidia_config.write_text(amd_config.read_text(encoding="utf-8"), encoding="utf-8")
    explicit = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )
    with pytest.raises(ValueError, match="is ambiguous"):
        resolve_agentx_recipe(explicit, str(root), runner_type="mi355x")

    nvidia_config.unlink()
    recipes = yaml.safe_load(amd_config.read_text(encoding="utf-8"))
    recipes["duplicate-agentx-recipe"] = deepcopy(
        recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"]
    )
    amd_config.write_text(yaml.safe_dump(recipes), encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple InferenceX AgentX recipes"):
        resolve_agentx_recipe(
            _minimal_config(inferencex_path=str(root)),
            str(root),
            runner_type="mi355x",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("recipe-not-object", "is not a YAML object"),
        ("multinode", "supports single-node recipes only"),
        ("missing-field", "missing required fields"),
        ("missing-scenario", "has no agentic-coding scenario"),
        ("invalid-parallelism", "invalid parallelism metadata"),
        ("zero-parallelism", "parallelism values must be positive"),
        ("missing-hardware", "has no hardware metadata"),
        ("invalid-dram", "DRAM offload metadata is incomplete"),
        ("invalid-utilization", "dram-utilization must be between 0 and 1"),
        ("oversized-topology", "topology needs 16 GPUs"),
    ],
)
def test_agentx_recipe_schema_validation(tmp_path, mutation, message):
    root = _fake_inferencex(tmp_path)
    config_file = root / "configs" / "amd-master.yaml"
    recipes = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    recipe = recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"]
    scenario = recipe["scenarios"]["agentic-coding"][0]
    arm = scenario["search-space"][0]
    runners_file = root / "configs" / "runners.yaml"
    runners = yaml.safe_load(runners_file.read_text(encoding="utf-8"))

    if mutation == "recipe-not-object":
        recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"] = []
    elif mutation == "multinode":
        recipe["multinode"] = True
    elif mutation == "missing-field":
        recipe.pop("image")
    elif mutation == "missing-scenario":
        recipe["scenarios"] = {}
    elif mutation == "invalid-parallelism":
        arm["tp"] = "invalid"
    elif mutation == "zero-parallelism":
        arm["ep"] = 0
    elif mutation == "missing-hardware":
        runners["hardware"] = {}
    elif mutation == "invalid-dram":
        runners["hardware"]["cluster:mi355x-amds"]["gpus-per-node"] = "invalid"
    elif mutation == "invalid-utilization":
        scenario["dram-utilization"] = 1.1
    elif mutation == "oversized-topology":
        arm["pp"] = 2

    config_file.write_text(yaml.safe_dump(recipes), encoding="utf-8")
    runners_file.write_text(yaml.safe_dump(runners), encoding="utf-8")
    config = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )
    with pytest.raises(ValueError, match=message):
        resolve_agentx_recipe(
            config,
            str(root),
            runner_type="mi355x",
        )


def test_agentx_recipe_selector_rejects_zero_or_multiple_points(tmp_path):
    root = _fake_inferencex(tmp_path)
    config_file = root / "configs" / "amd-master.yaml"
    recipes = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    search_space = recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"]["scenarios"][
        "agentic-coding"
    ][0]["search-space"]
    search_space.append(deepcopy(search_space[0]))
    config_file.write_text(yaml.safe_dump(recipes), encoding="utf-8")

    with pytest.raises(ValueError, match="resolves to multiple points"):
        resolve_agentx_recipe(
            _minimal_config(inferencex_path=str(root)),
            str(root),
            runner_type="mi355x",
        )

    no_match = _minimal_config(
        inferencex_path=str(root),
        agentx={"enabled": True, "selector": {"tp": 7}},
    )
    with pytest.raises(ValueError, match="matching selector"):
        resolve_agentx_recipe(no_match, str(root), runner_type="mi355x")


def test_agentx_resolution_requires_enabled_config_and_runner(tmp_path):
    root = _fake_inferencex(tmp_path)
    with pytest.raises(ValueError, match="configuration is missing"):
        resolve_agentx_recipe(
            _minimal_config(agentx=False, inferencex_path=str(root)), str(root)
        )
    with pytest.raises(ValueError, match="runner_type is required"):
        resolve_agentx_recipe(_minimal_config(inferencex_path=str(root)), str(root))

    missing_concurrency = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )
    missing_concurrency.envs.clear()
    with pytest.raises(ValueError, match="concurrency.*is required"):
        resolve_agentx_recipe(missing_concurrency, str(root), runner_type="mi355x")


def test_agentx_rejects_unsupported_resolved_framework(tmp_path):
    root = _fake_inferencex(tmp_path)
    config_file = root / "configs" / "amd-master.yaml"
    recipes = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    recipes["dsv4-fp4-mi355x-sglang-agentic-mtp"]["framework"] = "unsupported"
    config_file.write_text(yaml.safe_dump(recipes), encoding="utf-8")
    config = _minimal_config(
        inferencex_path=str(root),
        agentx={
            "enabled": True,
            "recipe": "dsv4-fp4-mi355x-sglang-agentic-mtp",
        },
    )
    config.framework = "unsupported"

    with pytest.raises(ValueError, match="supports vllm, sglang, and atom"):
        resolve_agentx_recipe(config, str(root), runner_type="mi355x")


def test_agentx_mapping_and_no_offload_helpers(tmp_path):
    invalid_mapping = tmp_path / "invalid.yaml"
    invalid_mapping.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must contain a YAML mapping"):
        agentx._load_mapping(invalid_mapping, "test config")

    assert agentx._agentic_dram_offload_gb({}, {}, "unused", {}) == 0
    assert agentx._describe_point({"tp": 1}) == "tp=1"


def test_ensure_agentx_dependencies_initializes_pinned_submodule(monkeypatch, tmp_path):
    root = tmp_path / "InferenceX"
    requirements = root / "utils" / "agentic-benchmark" / "requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.touch()

    def initialize(*args, **kwargs):
        project = root / "utils" / "aiperf" / "pyproject.toml"
        project.parent.mkdir(parents=True)
        project.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agentx.subprocess, "run", initialize)
    ensure_agentx_dependencies(str(root))
    assert (root / "utils" / "aiperf" / "pyproject.toml").is_file()
    ensure_agentx_dependencies(str(root))


def test_ensure_agentx_dependencies_reports_missing_and_failed_checkout(
    monkeypatch, tmp_path
):
    root = tmp_path / "InferenceX"
    root.mkdir()
    with pytest.raises(RuntimeError, match="does not contain AgentX support"):
        ensure_agentx_dependencies(str(root))

    requirements = root / "utils" / "agentic-benchmark" / "requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.touch()
    monkeypatch.setattr(
        agentx.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="submodule failed"
        ),
    )
    with pytest.raises(RuntimeError, match="submodule failed"):
        ensure_agentx_dependencies(str(root))


def test_agentx_docker_command_passes_inferencex_workspace(monkeypatch, tmp_path):
    root = tmp_path / "InferenceX"
    root.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    launcher = root / "benchmarks" / "single_node" / "agentic" / "launcher.sh"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    config = _minimal_config(
        inferencex_path=str(root),
        docker_image="sglang:test",
        benchmark_script="single_node/agentic/launcher.sh",
        envs={"CONC": 32, "MODEL_PATH": str(model_path)},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._task_id = "agentx-test"
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx950"),
    )

    command = mode._build_docker_command(
        docker_image="sglang:test",
        workspace=tmp_path / "workspace",
        runner_type="mi355x",
    )
    env_args = [
        command[index + 1] for index, item in enumerate(command) if item == "-e"
    ]

    assert "INFMAX_CONTAINER_WORKSPACE=/opt/InferenceX" in env_args
    assert "AGENTIC_OUTPUT_DIR=/workspace" in env_args
    assert "AIPERF_RUNTIME_DIR=/tmp/inferencex-agentx" in env_args
    assert f"{model_path}:{model_path}" in command
    assert "PROFILE=1" not in env_args
    assert command[-1].endswith("bash benchmarks/single_node/agentic/launcher.sh")


def test_agentx_gpu_selection_accounts_for_pp_and_pcp(monkeypatch, tmp_path):
    requested = []
    config = _minimal_config(
        envs={"CONC": 32, "TP": 2, "PP_SIZE": 2, "PCP_SIZE": 3},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.find_idle_gpus",
        lambda count, **kwargs: requested.append(count) or list(range(count)),
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx950"),
    )

    mode._apply_gpu_selection()

    assert requested == [12]
    assert config.envs["ROCR_VISIBLE_DEVICES"] == ",".join(map(str, range(12)))


def test_parse_agentx_aggregate_and_publication_status(tmp_path):
    aggregate = {
        "scenario_type": "agentic-coding",
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "framework": "sglang",
        "precision": "fp4",
        "recipe_fingerprint": "a" * 64,
        "num_requests_total": 12,
        "num_requests_successful": 10,
        "request_accounting": {
            "records_total": 12,
            "records_profiled": 10,
            "records_error_dropped": 1,
            "records_warmup_dropped": 1,
        },
        "request_metrics": {
            "qps": {"mean": 0.5},
            "throughput": {
                "input": {"tokens_per_second": 100.0},
                "output": {"tokens_per_second": 20.0},
                "total": {"tokens_per_second": 120.0},
                "duration_seconds": 1000.0,
            },
            "latency": {
                "ttft": {"mean": 1.25, "p50": 1.0, "std": 0.25},
                "tpot": {"mean": 0.02, "p50": 0.01, "std": 0.005},
            },
        },
        "server_metrics": {"cache": {}},
    }
    result_file = tmp_path / "inferencex_result.json"
    result_file.write_text(json.dumps(aggregate), encoding="utf-8")

    parsed = ResultParser.parse_inferencex_result(
        result_file,
        framework="sglang",
        model="deepseek-ai/DeepSeek-V4-Pro-0813",
        scenario="agentx",
        agentx_mode="canonical",
    )

    assert parsed.success is True
    assert parsed.benchmark_valid is True
    assert parsed.publishable is True
    assert parsed.throughput.total_token_throughput == 120.0
    assert parsed.latency.ttft_mean == 1250.0
    assert parsed.agentx_metrics["requests"]["error_rate"] == pytest.approx(1 / 11)
    assert parsed.agentx_metrics["requests"]["profiled_total"] == 11
    assert "AgentX:" in parsed.get_summary()
    assert "Throughput:" not in parsed.get_summary()

    fast = ResultParser.parse_inferencex_result(
        result_file,
        scenario="agentx",
        agentx_mode="fast",
    )
    assert fast.success is True
    assert fast.publishable is False

    aggregate.pop("recipe_fingerprint")
    result_file.write_text(json.dumps(aggregate), encoding="utf-8")
    unpinned = ResultParser.parse_inferencex_result(
        result_file,
        scenario="agentx",
        agentx_mode="canonical",
    )
    assert unpinned.benchmark_valid is True
    assert unpinned.publishable is False


def test_parse_agentx_rejects_excessive_request_errors(tmp_path):
    aggregate = {
        "scenario_type": "agentic-coding",
        "num_requests_total": 10,
        "num_requests_successful": 7,
        "request_accounting": {"records_error_dropped": 3},
        "request_metrics": {
            "qps": {"mean": 1.0},
            "throughput": {
                "total": {"tokens_per_second": 10.0},
                "duration_seconds": 900.0,
            },
            "latency": {},
        },
    }
    result_file = tmp_path / "inferencex_result.json"
    result_file.write_text(json.dumps(aggregate), encoding="utf-8")

    parsed = ResultParser.parse_inferencex_result(
        result_file,
        scenario="agentx",
        failed_request_threshold=0.1,
    )

    assert parsed.success is False
    assert parsed.benchmark_valid is False
    assert parsed.publishable is False
    assert "error rate exceeded" in parsed.errors[0]
