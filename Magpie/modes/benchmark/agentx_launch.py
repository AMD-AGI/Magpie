"""Versioned AgentX launch requests and independently persisted launch evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BenchmarkConfig

PROTECTED_ARGS = frozenset(
    {
        "--model",
        "--model-path",
        "--profiler-config",
        "--enable-profile-cuda-graph",
        "--enable-shape-discovery-for-cuda-graph-profile",
        "--revision",
        "--code-revision",
        "--tokenizer-revision",
        "--max-model-len",
        "--context-length",
        "--served-model-name",
        "--host",
        "--port",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--pp",
        "--pp-size",
        "--pipeline-parallel-size",
        "--ep",
        "--ep-size",
        "--expert-parallel-size",
        "--dp",
        "--dp-size",
        "--data-parallel-size",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-size-local",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--enable-dp-attention",
        "--enable-expert-parallel",
        "--decode-context-parallel-size",
        "--prefill-context-parallel-size",
        "--dcp-size",
        "--pcp-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
    }
)
PROTECTED_ENV = frozenset(
    {
        "PROFILE",
        "SGLANG_TORCH_PROFILER_DIR",
        "SGLANG_PROFILE_WITH_STACK",
        "SGLANG_PROFILE_RECORD_SHAPES",
        "SGLANG_GRAPH_BATCH_CAPTURE",
        "VLLM_TORCH_PROFILER_DIR",
        "MODEL",
        "MODEL_PATH",
        "MAX_MODEL_LEN",
        "MODEL_NAME",
        "SERVED_MODEL_NAME",
        "MODEL_PREFIX",
        "TP",
        "PP_SIZE",
        "EP_SIZE",
        "DP_ATTENTION",
        "DCP_SIZE",
        "PCP_SIZE",
        "CONC",
        "DURATION",
        "PORT",
        "RUN_EVAL",
        "EVAL_ONLY",
        "IS_AGENTIC",
        "SCENARIO_TYPE",
        "SCENARIO_SUBDIR",
        "RECIPE_FINGERPRINT",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    }
)
PROTECTED_ENV_PREFIXES = ("AGENTX_", "AGENTIC_", "AIPERF_", "SGLANG_SIMULATE_ACC_")
FIELDS = {
    "version",
    "append_args",
    "remove_args",
    "replace_args",
    "env",
    "unset_env",
    "source_files",
    "absent_source_files",
    "executable",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def validate_overrides(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("AgentX launch_overrides requires version: 1")
    if unknown := value.keys() - FIELDS:
        raise ValueError(f"Unknown AgentX launch override fields: {sorted(unknown)}")
    result = {
        "version": 1,
        "append_args": [],
        "remove_args": [],
        "replace_args": False,
        "env": {},
        "unset_env": [],
        "source_files": {},
        "absent_source_files": [],
        "executable": None,
        **value,
    }
    for key in ("append_args", "remove_args", "unset_env", "absent_source_files"):
        if not isinstance(result[key], list) or any(
            not isinstance(item, str) or "\0" in item for item in result[key]
        ):
            raise ValueError(f"{key} must be a list of strings without NUL bytes")
    if type(result["replace_args"]) is not bool:
        raise ValueError("replace_args must be a boolean")
    for key in ("env", "source_files"):
        if not isinstance(result[key], dict) or any(
            not isinstance(name, str)
            or not isinstance(item, str)
            or "\0" in name
            or "\0" in item
            for name, item in result[key].items()
        ):
            raise ValueError(f"{key} must map strings to strings without NUL bytes")
    if len(set(result["remove_args"])) != len(result["remove_args"]):
        raise ValueError("remove_args must not contain duplicates")
    for name in result["remove_args"]:
        if not re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"remove_args must contain long option names: {name!r}")
        if name.replace("_", "-") in PROTECTED_ARGS:
            raise ValueError(f"Cannot remove protocol option {name}")
    names = set(result["env"]) | set(result["unset_env"])
    if set(result["env"]) & set(result["unset_env"]):
        raise ValueError("An environment variable cannot be both set and unset")
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if name in PROTECTED_ENV or name.startswith(PROTECTED_ENV_PREFIXES):
            raise ValueError(f"Cannot override protocol environment variable {name}")
    for name, checksum in result["source_files"].items():
        if not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(
                "source_files requires absolute paths and lowercase SHA256 values"
            )
    for name in result["absent_source_files"]:
        if not Path(name).is_absolute() or name in result["source_files"]:
            raise ValueError("absent_source_files requires distinct absolute paths")
    executable = result["executable"]
    if executable is not None and (
        not isinstance(executable, str)
        or "\0" in executable
        or not Path(executable).is_absolute()
    ):
        raise ValueError("executable must be an absolute path or null")
    return result


def option_groups(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    """Separate fixed executable/positionals and unambiguous long options."""
    prefix: list[str] = []
    groups: list[list[str]] = []
    for token in argv:
        if token.startswith("--"):
            name = token.partition("=")[0]
            if not re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Ambiguous server option: {token!r}")
            groups.append([token])
        elif groups:
            # Short options are ambiguous with values and cannot safely be
            # removed as part of the preceding long option. Negative numbers
            # are ordinary values. Equals-style options already own a value.
            if "=" in groups[-1][0] or (
                token.startswith("-")
                and not re.fullmatch(r"-\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token)
            ):
                raise ValueError(f"Ambiguous server option value: {token!r}")
            groups[-1].append(token)
        else:
            prefix.append(token)
    return prefix, groups


def apply_launch_args(
    argv: list[str],
    overrides: dict[str, Any],
    framework: str,
) -> list[str]:
    """Apply the v1 argv contract without reading files or the process environment."""
    overrides = validate_overrides(overrides)
    if framework not in {"sglang", "vllm"} or not argv:
        raise ValueError(
            "AgentX launch extension supports nonempty SGLang/vLLM commands"
        )
    effective = list(argv)
    # Empty requests do not parse/rewrite an existing command. This preserves
    # every canonical launcher token, including options unfamiliar to v1.
    if (
        overrides["append_args"]
        or overrides["remove_args"]
        or overrides["replace_args"]
    ):
        prefix, groups = option_groups(argv)
        extra_prefix, extra = option_groups(overrides["append_args"])
        if extra_prefix:
            raise ValueError(
                "append_args cannot inject an executable or positional arguments"
            )
        if any(
            group[0].partition("=")[0].replace("_", "-") in PROTECTED_ARGS
            for group in extra
        ):
            raise ValueError(
                "append_args cannot change model, topology, host or port; profiling is also protected"
            )
        for name in overrides["remove_args"]:
            matches = [group for group in groups if group[0].partition("=")[0] == name]
            if len(matches) != 1:
                raise ValueError(
                    f"remove_args must match exactly one existing option: {name}"
                )
            groups.remove(matches[0])
        if overrides["replace_args"]:
            # Keep the executable/model positionals and all protocol options;
            # only the optional serving configuration is replaced.
            groups = [
                group
                for group in groups
                if group[0].partition("=")[0].replace("_", "-") in PROTECTED_ARGS
            ]
        effective = prefix + [token for group in groups + extra for token in group]
    if overrides["executable"] is not None:
        effective[0] = overrides["executable"]
    return effective


def _validate_golden_launch(spec: dict[str, Any], effective: list[str]) -> None:
    """A candidate cannot change the acceptance curve without re-resolution."""
    if not spec.get("golden_acceptance_overrides"):
        return

    def values(argv: list[str]) -> dict[str, list[list[str]]]:
        result: dict[str, list[list[str]]] = {}
        for group in option_groups(argv)[1]:
            key, equal, inline = group[0].partition("=")
            result.setdefault(key, []).append(([inline] if equal else []) + group[1:])
        return result

    base, changed = values(spec["argv"]), values(effective)
    if spec["framework"] == "sglang":
        keys = {
            "--speculative-algorithm",
            "--speculative-algo",
            "--speculative-num-steps",
            "--speculative-dspark-block-size",
            "--speculative-draft-model-path",
        }
        if any(base.get(key) != changed.get(key) for key in keys):
            raise ValueError(
                "Changing AgentX golden acceptance inputs requires a new resolved recipe"
            )
    elif spec["framework"] == "vllm":

        def parameters(options):
            groups = options.get("--speculative-config", [])
            if not groups:
                return {}
            if len(groups) != 1 or len(groups[0]) != 1:
                raise ValueError(
                    "AgentX requires exactly one speculative-config JSON object"
                )
            value = json.loads(groups[0][0])
            if not isinstance(value, dict):
                raise ValueError("AgentX speculative-config must be a JSON object")
            return value

        old, new = parameters(base), parameters(changed)
        keys = {
            "method",
            "model",
            "num_speculative_tokens",
            "rejection_sample_method",
            "synthetic_acceptance_length",
        }
        if any(old.get(key) != new.get(key) for key in keys):
            raise ValueError("AgentX golden acceptance parameters cannot be overridden")


def launch_environment(
    config: BenchmarkConfig,
    workspace: Path,
    *,
    docker: bool = False,
) -> dict[str, str]:
    """Persist the exact normalized request and bind this run's output path."""
    result = {
        "AGENTX_LAUNCH_OVERRIDES_FILE": "",
        "AGENTX_LAUNCH_OVERRIDES_SHA256": "",
        "AGENTX_SERVER_LAUNCH_FILE": "",
    }
    if config.agentx is None or config.agentx.launch_overrides is None:
        return result
    overrides = validate_overrides(config.agentx.launch_overrides)
    if overrides != config.agentx.launch_overrides:
        raise ValueError("AgentX launch overrides changed after config normalization")
    workspace.mkdir(parents=True, exist_ok=True)
    request = workspace / "agentx_launch_overrides.json"
    request.write_text(json.dumps(overrides, indent=2) + "\n", encoding="utf-8")
    # A stale receipt must never make a failed new launch appear verified.
    (workspace / "agentx_server_launch.json").unlink(missing_ok=True)
    runtime_workspace = Path("/workspace") if docker else workspace.resolve()
    result.update(
        {
            "AGENTX_LAUNCH_OVERRIDES_FILE": str(runtime_workspace / request.name),
            "AGENTX_LAUNCH_OVERRIDES_SHA256": digest(overrides),
            "AGENTX_SERVER_LAUNCH_FILE": str(
                runtime_workspace / "agentx_server_launch.json"
            ),
        }
    )
    return result


def read_launch_evidence(config: BenchmarkConfig, workspace: Path) -> dict[str, Any]:
    """Verify the receipt against the immutable request persisted for this run."""
    assert config.agentx is not None
    overrides = validate_overrides(config.agentx.launch_overrides or {"version": 1})
    request = json.loads(
        (workspace / "agentx_launch_overrides.json").read_text(encoding="utf-8")
    )
    if request != overrides:
        raise ValueError(
            "Persisted AgentX launch request differs from the benchmark config"
        )
    evidence = json.loads(
        (workspace / "agentx_server_launch.json").read_text(encoding="utf-8")
    )
    if not isinstance(evidence, dict):
        raise ValueError(
            "AgentX server launch evidence must be an object"
        )  # noqa: TRY004
    payload = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    if evidence.get("evidence_sha256") != digest(payload):
        raise ValueError("AgentX server launch evidence hash mismatch")
    if (
        type(evidence.get("version")) is not int
        or evidence.get("version") != 1
        or evidence.get("framework") != config.framework
    ):
        raise ValueError("AgentX server launch evidence identity mismatch")
    if evidence.get("overrides_sha256") != digest(overrides):
        raise ValueError("AgentX server did not apply this run's launch overrides")
    if evidence.get("source_files") != overrides["source_files"]:
        raise ValueError("AgentX source file launch evidence mismatch")
    if evidence.get("absent_source_files") != overrides["absent_source_files"]:
        raise ValueError("AgentX deleted source file launch evidence mismatch")
    names = sorted(set(overrides["env"]) | set(overrides["unset_env"]))
    expected_env = {name: overrides["env"].get(name) for name in names}
    if evidence.get("effective_env") != expected_env:
        raise ValueError("AgentX effective server environment mismatch")
    if not isinstance(evidence.get("base_env"), dict) or set(
        evidence["base_env"]
    ) != set(names):
        raise ValueError("AgentX base server environment evidence is incomplete")
    for key in ("base_argv", "effective_argv"):
        argv = evidence.get(key)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise ValueError(f"AgentX {key} evidence must contain server argv tokens")
    if (
        overrides["executable"]
        and evidence["effective_argv"][0] != overrides["executable"]
    ):
        raise ValueError("AgentX server executable override was not applied")
    executable = evidence.get("resolved_executable")
    if not isinstance(executable, str) or not Path(executable).is_absolute():
        raise ValueError("AgentX resolved executable evidence is missing")
    runtime = evidence.get("runtime_environment")
    if not isinstance(runtime, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in runtime.items()
    ):
        raise ValueError("AgentX runtime environment evidence is incomplete")
    spec = (config.agentx.resolved or {}).get("server-launch-spec")
    if spec is not None:
        from .agentx_profile_config import profile_server_spec, profile_settings

        settings = profile_settings(config)
        profiling = evidence.get("torch_profiler")
        if settings is not None:
            if not isinstance(profiling, dict):
                raise ValueError("AgentX profiler launch evidence is missing")
            spec = profile_server_spec(
                spec,
                settings,
                workspace,
                profiling.get("capture_id"),
                capabilities=profiling.get("capabilities"),
                launch_overrides=config.agentx.launch_overrides,
            )
            if settings.get("detailed_annotations") and not profiling.get(
                "capabilities"
            ):
                raise ValueError(
                    "AgentX detailed annotation capability evidence is missing"
                )
            if profiling != spec["torch_profiler"]:
                raise ValueError("AgentX profiler configuration evidence mismatch")
            if (
                config.framework == "sglang"
                and runtime.get("SGLANG_TORCH_PROFILER_DIR")
                != spec["env"]["SGLANG_TORCH_PROFILER_DIR"]
            ):
                raise ValueError("AgentX profiler output environment mismatch")
            if config.framework == "sglang" and settings.get("detailed_annotations"):
                for name in (
                    "SGLANG_PROFILE_WITH_STACK",
                    "SGLANG_PROFILE_RECORD_SHAPES",
                    "SGLANG_GRAPH_BATCH_CAPTURE",
                ):
                    if runtime.get(name) != spec["env"][name]:
                        raise ValueError(
                            f"AgentX annotation environment mismatch: {name}"
                        )
        elif profiling is not None:
            raise ValueError("Unexpected AgentX profiler launch evidence")
        if evidence.get("owner") != "magpie":
            raise ValueError("AgentX server launch evidence is not Magpie-owned")
        if evidence.get("server_spec_sha256") != digest(spec):
            raise ValueError("AgentX server launch specification identity mismatch")
        if evidence.get("recipe_source_files") != spec.get("source_files", {}):
            raise ValueError("AgentX recipe source evidence is incomplete")
        if evidence["effective_argv"] != apply_launch_args(
            evidence["base_argv"], overrides, config.framework
        ):
            raise ValueError("AgentX effective argv does not match this candidate")
        _validate_golden_launch(spec, evidence["effective_argv"])
        if evidence["base_argv"] != spec.get("argv"):
            raise ValueError("AgentX base argv does not match the resolved recipe")
    return evidence


def prepare_server_launch(
    argv: list[str],
    environment: dict[str, str],
    overrides: dict[str, Any] | None,
    framework: str,
    workspace: Path,
    *,
    server_spec: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Attest a Magpie-owned server in its actual runtime, after setup.

    The caller immediately spawns the returned argv/environment. This function
    must run inside the serving container for Docker jobs. The client never
    invokes it, so client setup cannot overwrite the server's receipt.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    receipt = workspace / "agentx_server_launch.json"
    receipt.unlink(missing_ok=True)
    request = validate_overrides(overrides or {"version": 1})
    if not isinstance(argv, list) or any(
        not isinstance(token, str) or "\0" in token for token in argv
    ):
        raise ValueError("AgentX server argv must contain literal string tokens")
    if any(
        not isinstance(name, str)
        or not isinstance(value, str)
        or "\0" in name
        or "\0" in value
        for name, value in environment.items()
    ):
        raise ValueError("AgentX server environment must contain strings without NUL")
    if server_spec is not None and server_spec.get("argv") != argv:
        raise ValueError("AgentX server argv differs from its resolved specification")
    effective = apply_launch_args(argv, request, framework)
    if server_spec is not None:
        _validate_golden_launch(server_spec, effective)
    child_env = dict(environment)
    touched = sorted(set(request["env"]) | set(request["unset_env"]))
    base_env = {name: environment.get(name) for name in touched}
    for name in request["unset_env"]:
        child_env.pop(name, None)
    child_env.update(request["env"])

    sources = {}
    for name, expected in request["source_files"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"AgentX source changed before server launch: {name}")
        sources[name] = actual
    recipe_sources = {}
    for name, expected in (server_spec or {}).get("source_files", {}).items():
        if not Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("AgentX recipe sources require absolute paths and SHA256")
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"AgentX recipe input changed before server launch: {name}"
            )
        recipe_sources[name] = actual
    for name in request["absent_source_files"]:
        if os.path.lexists(name):
            raise ValueError(f"AgentX deleted source reappeared before launch: {name}")
    resolved_executable = shutil.which(effective[0], path=child_env.get("PATH", ""))
    if resolved_executable is None:
        raise ValueError(f"AgentX server executable not found: {effective[0]}")
    evidence: dict[str, Any] = {
        "version": 1,
        "owner": "magpie",
        "framework": framework,
        "base_argv": list(argv),
        "effective_argv": effective,
        "base_env": base_env,
        "effective_env": {name: child_env.get(name) for name in touched},
        "overrides_sha256": digest(request),
        "source_files": sources,
        "absent_source_files": request["absent_source_files"],
        "resolved_executable": str(Path(resolved_executable).resolve()),
        "runtime_environment": {
            name: value
            for name, value in sorted(child_env.items())
            if (
                name
                in {
                    "PATH",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "LIBRARY_PATH",
                    "GPU_ARCHS",
                }
                or name.startswith(
                    (
                        "SGLANG_",
                        "VLLM_",
                        "AITER_",
                        "TRITON_",
                        "TORCH_",
                        "PYTORCH_",
                        "HIP_",
                        "ROCR_",
                        "HSA_",
                        "NCCL_",
                        "RCCL_",
                        "OMP_",
                    )
                )
            )
            and not any(
                word in name.upper()
                for word in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")
            )
        },
    }
    if server_spec is not None:
        if server_spec.get("torch_profiler"):
            evidence["torch_profiler"] = server_spec["torch_profiler"]
        evidence["server_spec_sha256"] = digest(server_spec)
        evidence["recipe_source_files"] = recipe_sources
    evidence["evidence_sha256"] = digest(evidence)
    (workspace / "agentx_launch_overrides.json").write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )
    temporary = receipt.with_name(receipt.name + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return effective, child_env, evidence
