"""Materialize single-node native recipes without a Slurm deployment.

InferenceX still owns point validation and golden acceptance. Magpie owns the
literal server command and its lifecycle; unsupported deployment semantics are
rejected instead of silently discarded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .config import BenchmarkConfig

CLIENT_SCRIPT = "srt_agentic.sh"

# These upstream APIs perform no submission and import srtctl only in other,
# unused functions. A separate interpreter avoids altering Magpie's sys.path or
# reusing a module cached from a different checkout.
_VALIDATE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from infx.srt_slurm.single_node import validate_recipe
from infx.srt_slurm.synthetic_acceptance import build_overrides
data = json.load(sys.stdin)
matches, errors = [], []
for name, recipe in data['variants']:
    try:
        validate_recipe(recipe, data['environment'])
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f'{name}: {exc}')
    else:
        matches.append((name, recipe))
if len(matches) != 1:
    raise ValueError('Expected exactly one matching native recipe: ' +
                     ('; '.join(errors) if not matches else str([x[0] for x in matches])))
name, recipe = matches[0]
overrides = build_overrides(recipe, data['environment']['FRAMEWORK'], data['environment'])
print(json.dumps({'variant': name, 'recipe': recipe, 'overrides': overrides}))
"""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _variants(raw: dict[str, Any], selector: str) -> list[tuple[str, dict[str, Any]]]:
    if "base" not in raw:
        if selector:
            raise ValueError("Native recipe selector requires a base/override recipe")
        return [("", raw)]
    if any(key.startswith("zip_override_") for key in raw):
        raise ValueError("Magpie does not support zipped native recipe sweeps")
    if not isinstance(raw["base"], dict):
        raise ValueError("Native recipe base must be a mapping")
    names = sorted(key for key in raw if key.startswith("override_"))
    if selector == "base" or not names:
        return [("base", raw["base"])]
    if selector:
        if selector not in names:
            raise ValueError(f"Unknown native recipe variant: {selector}")
        names = [selector]
    if any(not isinstance(raw[name], dict) for name in names):
        raise ValueError("Native recipe overrides must be mappings")
    return [(name, _merge(raw["base"], raw[name])) for name in names]


def _strings(value: Any, description: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        or not isinstance(item, (str, int, float, bool))
        for key, item in value.items()
    ):
        raise ValueError(f"{description} must map environment names to scalar values")
    return {
        key: str(item).lower() if isinstance(item, bool) else str(item)
        for key, item in value.items()
    }


def _apply_golden(recipe: dict[str, Any], overrides: list[str]) -> None:
    if len(overrides) % 2:
        raise ValueError("Malformed upstream golden acceptance overrides")
    for operation, raw in zip(overrides[::2], overrides[1::2]):
        path, _, value = raw.partition("=")
        keys = path.split(".")
        if keys[0] not in {"roles", "environment"}:
            raise ValueError(f"Unexpected upstream golden acceptance target: {path}")
        target = recipe
        for key in keys[:-1]:
            if not isinstance(target, dict):
                raise ValueError("Native recipe roles/environment must be mappings")
            target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError("Native recipe roles/environment must be mappings")
        if operation == "--set":
            target[keys[-1]] = json.loads(value)
        elif operation == "--unset":
            target.pop(keys[-1], None)
        else:
            raise ValueError(
                f"Unexpected upstream golden acceptance operation: {operation}"
            )


def _flags(arguments: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for name, value in arguments.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][\w-]*", name):
            raise ValueError(f"Invalid native server option: {name!r}")
        if value is None or value is False:
            continue
        tokens.append("--" + name)
        if value is True:
            continue
        if isinstance(value, dict):
            tokens.append(json.dumps(value, separators=(",", ":")))
        elif isinstance(value, list):
            if any(not isinstance(item, (str, int, float)) for item in value):
                raise ValueError(f"Unsupported native server list argument: {name}")
            tokens.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            tokens.append(str(value))
        else:
            raise ValueError(f"Unsupported native server argument: {name}")
    return tokens


def _port(config: BenchmarkConfig) -> int:
    raw = config.envs.get("PORT", 8888)
    if isinstance(raw, bool) or not str(raw).isdigit() or not 0 < int(raw) < 65536:
        raise ValueError("AgentX PORT must be an integer between 1 and 65535")
    return int(raw)


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    try:
        return {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    except OSError as exc:
        raise ValueError(
            f"Required AgentX protocol/recipe source is unreadable: {exc}"
        ) from exc


def resolve_client_sources(root: Path) -> dict[str, str]:
    """Bind the official client, shared shell, aggregation, and acceptance gate."""
    files = [
        root / "benchmarks/srt_agentic.sh",
        root / "benchmarks/benchmark_lib.sh",
        root / "benchmarks/runtime_settings.sh",
        root / "runners/srt-slurm/hooks/common.sh",
        root / "infx/results/agentic/process_agentic_result.py",
        root / "infx/results/agentic/validate_agentic_result.py",
        root / "infx/__init__.py",
        root / "infx/results/__init__.py",
        root / "infx/results/metadata.py",
        root / "infx/results/topology.py",
    ]
    files += sorted((root / "infx/results/agentic").rglob("*.py"))
    files += sorted((root / "infx/results/power").rglob("*.py"))
    return _file_hashes(files)


def _client_revisions(root: Path) -> dict[str, str]:
    """Record checkout and harness pins when resolving an actual Git checkout."""
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if head.returncode or not re.fullmatch(r"[0-9a-f]{40}", head.stdout.strip()):
        return {}
    result = {"client_revision": head.stdout.strip()}
    harness = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "HEAD", "--", "utils/aiperf"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    match = re.fullmatch(
        r"160000 commit ([0-9a-f]{40})\tutils/aiperf", harness.stdout.strip()
    )
    if harness.returncode == 0 and match:
        result["aiperf_revision"] = match[1]
    return result


def _client_defaults(root: Path, config: BenchmarkConfig) -> dict[str, str]:
    result = {}
    for line in (root / "benchmarks/runtime_settings.sh").read_text().splitlines():
        match = re.fullmatch(r"export ([A-Z_]+)='([^']*)'", line)
        if match and (
            match[1].startswith(("AIPERF_", "AGENTIC_"))
            or match[1] == "ENABLE_AGENTX_POWER"
        ):
            result[match[1]] = match[2]
    required = {
        "AIPERF_PYTHON_VERSION",
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD",
        "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS",
        "AIPERF_WARMUP_REQUESTS_PER_LANE",
        "AGENTIC_WARMUP_GRACE_PERIOD",
        "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
        "AIPERF_UNSAFE_OVERRIDE",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING",
        "ENABLE_AGENTX_POWER",
    }
    if any(not result.get(name) for name in required):
        raise ValueError(
            "InferenceX client runtime settings are missing literal AgentX defaults"
        )
    assert config.agentx is not None
    result["AIPERF_FAILED_REQUEST_THRESHOLD"] = str(
        config.agentx.failed_request_threshold
    )
    result["AIPERF_EXPERIMENTAL_FAST"] = "1" if config.agentx.mode == "fast" else "0"
    return result


def _server_environment(config: BenchmarkConfig) -> dict[str, str]:
    # Materialization writes replay metadata back to config.envs. Do not feed
    # those fields into the server or change spec identity on snapshot reload.
    if any(key.upper().startswith("SGLANG_SIMULATE_ACC_") for key in config.envs):
        raise ValueError("AgentX golden acceptance environment is owned by InferenceX")
    client_names = {
        "MODEL",
        "MODEL_PATH",
        "MODEL_PREFIX",
        "FRAMEWORK",
        "IMAGE",
        "PRECISION",
        "EXP_NAME",
        "TP",
        "PP",
        "PP_SIZE",
        "EP",
        "EP_SIZE",
        "DCP_SIZE",
        "PCP_SIZE",
        "DP_ATTENTION",
        "CONC",
        "SPEC_DECODING",
        "DISAGG",
        "IS_AGENTIC",
        "IS_MULTINODE",
        "KV_OFFLOADING",
        "KV_OFFLOAD_BACKEND",
        "KV_OFFLOAD_BACKEND_METADATA",
        "ROUTER_METADATA",
        "KV_P2P_TRANSFER",
        "TOTAL_CPU_DRAM_GB",
        "DURATION",
        "RUN_EVAL",
        "EVAL_ONLY",
        "RECIPE_FINGERPRINT",
        "MAX_MODEL_LEN",
        "WEKA_LOADER_OVERRIDE",
        "PORT",
        "GPU_COUNT",
        "THINKING_MODE",
        "ENABLE_AGENTX_POWER",
        "REQUIRE_POWER",
    }
    return _strings(
        {
            key: value
            for key, value in config.envs.items()
            if key.upper() not in client_names
            and not (key.upper().startswith("EXTRA_") and key.upper().endswith("_ARGS"))
            and not key.upper().startswith(
                ("AIPERF_", "AGENTIC_", "AGENTX_", "SCENARIO_")
            )
        },
        "Explicit server environment",
    )


def native_server_spec(
    config: BenchmarkConfig, root: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one YAML variant, preserving upstream validation and acceptance."""
    filename, _, selector = str(entry["srt-recipe"]).partition(":")
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(
            f"Native AgentX recipe is missing or outside InferenceX: {filename}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Native AgentX recipe must be a mapping")
    environment = {
        "MODEL": entry["model"],
        "MODEL_PREFIX": entry["model-prefix"],
        "FRAMEWORK": entry["framework"],
        "PRECISION": entry["precision"],
        "IMAGE": config.docker_image or entry["image"],
        "TP": str(entry["tp"]),
        "EP_SIZE": str(entry["ep"]),
        "PP_SIZE": str(entry.get("pp", 1)),
        "DCP_SIZE": str(entry.get("dcp-size", 1)),
        "PCP_SIZE": str(entry.get("pcp-size", 1)),
        "GPU_COUNT": str(entry["tp"] * entry.get("pp", 1) * entry.get("pcp-size", 1)),
        "DP_ATTENTION": str(entry.get("dp-attn", False)).lower(),
        "CONC": str(entry["conc"]),
        "KV_OFFLOADING": entry["kv-offloading"],
        "TOTAL_CPU_DRAM_GB": str(entry["total-cpu-dram-gb"]),
        "SPEC_DECODING": entry["spec-decoding"],
        "EVAL_ONLY": "false",
        "IS_AGENTIC": "1",
        # This is the official single-node workflow default, not an inferred AL.
        "THINKING_MODE": str(config.envs.get("THINKING_MODE", "thinking_on")),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _VALIDATE, str(root)],
        input=json.dumps(
            {"variants": _variants(raw, selector), "environment": environment}
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError(
            "InferenceX native recipe validation failed: " + completed.stderr.strip()
        )
    selected = json.loads(completed.stdout)
    recipe = selected["recipe"]
    _apply_golden(recipe, selected["overrides"])
    allowed = {
        "schema",
        "name",
        "model",
        "resources",
        "frontend",
        "observability",
        "engine",
        "health_check",
        "roles",
        "benchmark",
        "environment",
        "setup_script",
    }
    if set(recipe) - allowed or recipe.get("schema") != 2:
        raise ValueError("Unsupported native AgentX recipe deployment fields or schema")
    engine = recipe["engine"]
    if isinstance(engine, dict):
        if set(engine) - {"type", "connector"} or engine.get("connector") is not None:
            raise ValueError("Magpie does not support native engine connectors")
        engine = engine["type"]
    if engine not in {"sglang", "vllm"}:
        raise ValueError("Magpie-owned AgentX supports SGLang and vLLM only")
    frontend = recipe["frontend"]
    if (
        frontend.get("type") != engine
        or frontend.get("enable_multiple_frontends", False)
        or frontend.get("args")
        or frontend.get("env")
    ):
        raise ValueError(
            "Magpie-owned AgentX does not support router or multiple frontends"
        )
    observability = recipe.get("observability", {})
    if observability.get("enabled") or observability.get("tachometer", {}).get(
        "enabled"
    ):
        raise ValueError(
            "Magpie-owned AgentX does not support recipe observability services"
        )
    role = recipe["roles"]["agg"]
    if set(role) - {"nodes", "workers", "gpus", "args", "env"}:
        raise ValueError("Unsupported native AgentX worker deployment fields")
    arguments = dict(role["args"])
    for name in ("host", "port", "model", "model-path"):
        if name in arguments:
            raise ValueError(f"Native recipe must leave runtime-owned {name} to Magpie")
    env = _strings(role.get("env", {}), "Native worker environment")
    env.update(_strings(recipe.get("environment", {}), "Native global environment"))
    env.update(_server_environment(config))
    port = _port(config)
    model_path = str(config.envs.get("MODEL_PATH") or config.model)
    if engine == "sglang":
        argv = ["python3", "-m", "sglang.launch_server", "--model-path", model_path]
    else:
        argv = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
        ]
    argv += ["--host", "0.0.0.0", "--port", str(port), *_flags(arguments)]
    setup = []
    files = [
        path,
        root / "infx/srt_slurm/single_node.py",
        root / "infx/srt_slurm/synthetic_acceptance.py",
        root / "infx/golden_al_distribution/__init__.py",
        root / "infx/golden_al_distribution/curves.py",
    ]
    files += sorted((root / "infx/golden_al_distribution").glob("*.yaml"))
    if recipe.get("setup_script"):
        if recipe["setup_script"] != "pip-runtime-deps.sh":
            raise ValueError(
                "Unsupported native AgentX setup_script: " + str(recipe["setup_script"])
            )
        script = (
            root / "benchmarks/multi_node/srt-slurm-recipes/configs/pip-runtime-deps.sh"
        )
        files.append(script)
        setup = [["bash", str(script)]]
    client = _client_defaults(root, config)
    client.update(
        _strings(recipe["benchmark"].get("env", {}), "Native client environment")
    )
    client.update(environment)
    client.update(
        PORT=str(port),
        AIPERF_SERVER_URL=f"http://127.0.0.1:{port}",
        AIPERF_SERVER_METRICS_URLS=f"http://127.0.0.1:{port}/metrics",
    )
    context = arguments.get("context-length" if engine == "sglang" else "max-model-len")
    if context is not None:
        client["AIPERF_MAX_CONTEXT_LENGTH"] = str(context)
    health = recipe.get("health_check", {})
    return {
        **_client_revisions(root),
        "version": 1,
        "framework": engine,
        "argv": argv,
        "env": env,
        "setup_commands": setup,
        "port": port,
        "health_path": "/health",
        "health_timeout_seconds": int(health.get("interval_seconds", 10))
        * int(health.get("max_attempts", 180)),
        "model_metadata": None,
        "client_env": client,
        "source_files": {**_file_hashes(files), **resolve_client_sources(root)},
        "recipe_variant": selected["variant"],
        "golden_acceptance_overrides": selected["overrides"],
    }


def custom_server_spec(
    config: BenchmarkConfig, root: Path, entry: dict[str, Any]
) -> dict[str, Any]:
    """Explicit custom models share the official replay, without a fake recipe."""
    port = _port(config)
    model = str(config.envs.get("MODEL_PATH") or config.model)
    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
        config.precision, config.precision
    )
    if dtype in {"fp8", "fp4", "mxfp4"}:
        dtype = "auto"
    if config.framework == "sglang":
        argv = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--tensor-parallel-size",
            str(entry["tp"]),
            "--expert-parallel-size",
            str(entry["ep"]),
            "--context-length",
            str(entry["max-model-len"]),
            "--enable-metrics",
        ]
    else:
        argv = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--tensor-parallel-size",
            str(entry["tp"]),
            "--max-model-len",
            str(entry["max-model-len"]),
        ]
        if entry["ep"] > 1:
            argv.append("--enable-expert-parallel")
    argv += [
        "--served-model-name",
        config.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        dtype,
    ]
    if entry.get("model-revision"):
        argv += ["--revision", entry["model-revision"]]
    return {
        **_client_revisions(root),
        "version": 1,
        "framework": config.framework,
        "argv": argv,
        "env": _server_environment(config),
        "setup_commands": [],
        "port": port,
        "health_path": "/health",
        "health_timeout_seconds": 1800,
        "model_metadata": {
            "source": entry["model-config-source"],
            "sha256": entry["model-config-sha256"],
            "native_context_length": entry["native-context-length"],
            "max_model_len": entry["max-model-len"],
            "revision": entry.get("model-revision"),
        },
        "client_env": {
            **_client_defaults(root, config),
            "AIPERF_MAX_CONTEXT_LENGTH": str(entry["max-model-len"]),
            "WEKA_LOADER_OVERRIDE": entry["trace-loader"],
            "PORT": str(port),
            "AIPERF_SERVER_URL": f"http://127.0.0.1:{port}",
            "AIPERF_SERVER_METRICS_URLS": f"http://127.0.0.1:{port}/metrics",
        },
        "source_files": resolve_client_sources(root),
        "recipe_variant": "custom",
        "golden_acceptance_overrides": [],
    }
