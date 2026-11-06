"""Validate native Ray DI topology before reserving resources or launching roles."""

import math
import shlex
from copy import deepcopy

_MANAGED_FLAGS = {
    "--model-path",
    "--model",
    "--served-model-name",
    "--config",
    "--config-file",
    "--host",
    "--port",
    "--tp",
    "--tp-size",
    "--tensor-parallel-size",
    "--tensor-parallel",
    "--ep",
    "--ep-size",
    "--expert-parallel-size",
    "--expert-parallel",
    "--pp",
    "--pp-size",
    "--pipeline-parallel-size",
    "--pipeline-parallel",
    "--nnodes",
    "--node-rank",
    "--dist-init-addr",
    "--dp",
    "--dp-size",
    "--data-parallel-size",
    "--num-nodes",
    "--device",
    "--base-gpu-id",
    "--gpu-id-step",
    "--bootstrap-port",
    "--disaggregation-bootstrap-port",
}
_CLIENT_ENVS = {
    "ISL",
    "OSL",
    "CONC",
    "NUM_PROMPTS",
    "RUN_EVAL",
    "EVAL_ONLY",
    "RANDOM_RANGE_RATIO",
    "PORT",
    "MODEL_PATH",
    "MODEL_NAME",
    "RESULT_DIR",
    "EVAL_RESULT_DIR",
    "GPU_METRICS_CSV",
    "SERVER_LOG",
    "REQUEST_RATE",
    "MAX_CONCURRENCY",
    "MAGPIE_BENCHMARK_PYTHON",
    "MAGPIE_BENCHMARK_BACKEND",
    "MAGPIE_BENCHMARK_ENDPOINT",
    "MAGPIE_CLIENT_TOKENIZER_MODE",
    "HYPERLOOM_CLIENT_TOKENIZER_MODE",
    "MAGPIE_TRUST_REMOTE_CODE",
    "MAGPIE_ACCURACY_REPORT_PYTHON",
    "MAGPIE_INFX_CLIENT_PYTHON",
    "MAGPIE_INFERENCEX_ROOT",
}
_GPU_VISIBILITY = {
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "NVIDIA_VISIBLE_DEVICES",
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _instances(disaggregation, common_env, ray_config):
    common_env = {str(key).upper(): value for key, value in common_env.items()}
    instances = []
    for role in ("prefill", "decode"):
        settings = disaggregation.get(role)
        if not isinstance(settings, dict) or set(settings) - {"instances", "envs"}:
            raise ValueError(f"disaggregation.{role} accepts instances and envs")
        overrides = settings.get("envs", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"disaggregation.{role}.envs must be a mapping")
        overrides = {str(key).upper(): value for key, value in overrides.items()}
        if _CLIENT_ENVS & overrides.keys() or any(
            key.startswith(("MAGPIE_EVAL_", "EVAL_")) for key in overrides
        ):
            raise ValueError("role envs cannot override benchmark client settings")
        env = {**common_env, **overrides}
        parallelism = {}
        for name in ("TP", "EP", "PP"):
            value = env.get(name, 1)
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError as error:
                    raise ValueError(
                        f"disaggregation.{role}.envs.{name} must be a positive integer"
                    ) from error
            parallelism[name.lower()] = _positive_integer(
                value, f"disaggregation.{role}.envs.{name}"
            )
        arguments, serving_env = _serving_settings(env, ray_config)
        serving_env.update(
            {name.upper(): str(value) for name, value in parallelism.items()}
        )
        count = _positive_integer(
            settings.get("instances", 1), f"disaggregation.{role}.instances"
        )
        instances.extend(
            {
                "id": f"{role}-{index}",
                "role": role,
                **parallelism,
                "server_args": arguments,
                "serving_env": serving_env,
            }
            for index in range(count)
        )
    return instances


def _context_arguments(arguments, env):
    if "MAX_MODEL_LEN" not in env:
        return arguments
    value = env["MAX_MODEL_LEN"]
    if isinstance(value, bool):
        raise TypeError("MAX_MODEL_LEN must be a positive integer")
    try:
        context = int(str(value))
    except ValueError as error:
        raise ValueError("MAX_MODEL_LEN must be a positive integer") from error
    if context <= 0:
        raise ValueError("MAX_MODEL_LEN must be a positive integer")
    pinned = []
    for index, argument in enumerate(arguments):
        if argument == "--context-length":
            if index + 1 == len(arguments):
                raise ValueError("--context-length requires a value")
            pinned.append(int(arguments[index + 1]))
        elif argument.startswith("--context-length="):
            pinned.append(int(argument.split("=", 1)[1]))
    if pinned and any(value != context for value in pinned):
        raise ValueError("MAX_MODEL_LEN conflicts with explicit --context-length")
    return arguments if pinned else [*arguments, "--context-length", str(context)]


def _serving_settings(env, ray_config):
    upper = {str(key).upper(): value for key, value in env.items()}
    visibility = _GPU_VISIBILITY & upper.keys()
    ray_env = ray_config.get("env_vars") or {}
    if not isinstance(ray_env, dict):
        raise TypeError("ray_config.env_vars must be a mapping")
    if visibility or _GPU_VISIBILITY & {str(key).upper() for key in ray_env}:
        raise ValueError(
            "GPU visibility is owned by native Ray placement; remove visibility overrides"
        )
    ray_upper = {str(key).upper(): value for key, value in ray_env.items()}
    if "MAGPIE_DI_SERVING_ENV" in upper or "MAGPIE_DI_SERVING_ENV" in ray_upper:
        raise ValueError("MAGPIE_DI_SERVING_ENV is an internal DI environment handoff")
    if any(
        str(source.get(key, "")).lower() in {"1", "true", "yes"}
        for source in (upper, ray_upper)
        for key in ("ENABLE_PROFILE", "PROFILE")
    ):
        raise ValueError("coordinated native DI profiling is unsupported")
    if upper.get("BENCHMARK_BASE_URL") or upper.get("MAGPIE_RUN_PHASE"):
        raise ValueError(
            "native DI owns its endpoint; remove externally managed endpoint settings"
        )
    extra_args = upper.get("EXTRA_SGLANG_ARGS") or ""
    if not isinstance(extra_args, str):
        raise TypeError("EXTRA_SGLANG_ARGS must be a shell-style argument string")
    arguments = shlex.split(extra_args)
    if any(
        argument.split("=", 1)[0] in _MANAGED_FLAGS
        or argument.startswith("--disaggregation-")
        for argument in arguments
    ):
        raise ValueError(
            "EXTRA_SGLANG_ARGS cannot override native DI topology/model/ports"
        )
    arguments = _context_arguments(arguments, upper)
    serving_env = {
        key: str(value)
        for key, value in upper.items()
        if key not in _CLIENT_ENVS | {"EXTRA_SGLANG_ARGS"}
        and not key.startswith(("MAGPIE_EVAL_", "EVAL_"))
    }
    return arguments, serving_env


def normalize_native_di(benchmark):
    """Return an isolated normalized native configuration; never mutate caller YAML."""
    if not isinstance(benchmark, dict):
        raise TypeError("native DI requires a benchmark mapping")
    raw = deepcopy(benchmark.get("benchmark", benchmark))
    if "di" in raw:
        raise ValueError("use benchmark.disaggregation with instances and role envs")
    di = raw.pop("disaggregation", None)
    if raw.get("docker_image") or (isinstance(di, dict) and di.get("docker_image")):
        raise ValueError(
            "native DI uses the prepared Ray worker image; docker_image cannot change an existing pod image"
        )
    allowed = {
        "prefill",
        "decode",
        "server_port",
        "router_port",
        "bootstrap_port",
        "startup_timeout",
        "nic",
    }
    if not isinstance(di, dict) or set(di) - allowed:
        raise ValueError(
            "benchmark.disaggregation requires prefill/decode and optional ports/startup_timeout/nic"
        )
    if raw.get("framework") != "sglang" or raw.get("run_mode") != "ray":
        raise ValueError("native DI requires framework=sglang and run_mode=ray")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "native DI requires an actual model path or Hugging Face model ID"
        )
    profiler = raw.get("profiler") or {}
    if not isinstance(profiler, dict) or any(
        item is True or (isinstance(item, dict) and item.get("enabled"))
        for item in profiler.values()
    ):
        raise ValueError("coordinated native DI profiling is unsupported")
    agentx = raw.get("agentx")
    if (
        agentx and not (isinstance(agentx, dict) and agentx.get("enabled") is False)
    ) or raw.get("scenario"):
        raise ValueError(
            "AgentX/scenario has a separate lifecycle and is unsupported by native DI"
        )
    if (raw.get("server_lifecycle") or {}).get("enabled") or raw.get("server_reuse"):
        raise ValueError(
            "native DI owns serving; disable the separate server lifecycle/reuse"
        )
    env = raw.get("envs") or {}
    ray_config = raw.get("ray_config") or {}
    if not isinstance(env, dict) or not isinstance(ray_config, dict):
        raise TypeError("envs and ray_config must be mappings")
    _, serving_env = _serving_settings(env, ray_config)
    ports = {
        key: _positive_integer(di.get(key, default), f"disaggregation.{key}")
        for key, default in (
            ("server_port", 30000),
            ("router_port", 18000),
            ("bootstrap_port", 8998),
        )
    }
    if any(port > 65535 for port in ports.values()) or len(set(ports.values())) != len(
        ports
    ):
        raise ValueError("native DI ports must be distinct and within 1..65535")
    timeout = di.get("startup_timeout", 1800)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("disaggregation.startup_timeout must be positive and finite")
    nic = di.get("nic")
    if nic is not None and (not isinstance(nic, str) or not nic.strip()):
        raise ValueError(
            "disaggregation.nic must be a nonempty device string when supplied"
        )
    return {
        "model": model,
        "instances": _instances(di, env, ray_config),
        "serving_env": serving_env,
        "ray_config": deepcopy(ray_config),
        **ports,
        "startup_timeout": timeout,
        **({"nic": nic} if nic is not None else {}),
    }
