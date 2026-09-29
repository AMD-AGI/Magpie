"""Versioned AgentX launch requests and independently persisted launch evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BenchmarkConfig

PROTECTED_ARGS = frozenset(
    {
        "--model",
        "--model-path",
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
        "MODEL",
        "MODEL_PATH",
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
        if name in PROTECTED_ARGS:
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
    assert config.agentx is not None and config.agentx.launch_overrides is not None
    overrides = config.agentx.launch_overrides
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
        raise ValueError("AgentX server launch evidence must be an object")  # noqa: TRY004
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
    return evidence
