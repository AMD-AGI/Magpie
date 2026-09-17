###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""InferenceX AgentX recipe resolution and launch preparation."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from .config import AgentXConfig, BenchmarkConfig

logger = logging.getLogger(__name__)

_BYTES_PER_MIB = 1024 * 1024
_BYTES_PER_GB = 1_000_000_000
_MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB = 2_861_022
_DEFAULT_AGENTIC_DURATION_SECONDS = 3600


@dataclass(frozen=True)
class AgentXLaunchSpec:
    """One concrete single-node AgentX point resolved from InferenceX."""

    recipe: str
    config_file: str
    entry: Dict[str, Any]


def _recipe_fingerprint(entry: Dict[str, Any]) -> str:
    """Match InferenceX's concurrency-independent recipe fingerprint."""

    recipe = {
        key: value
        for key, value in entry.items()
        if key not in {"conc", "exp-name", "recipe-fingerprint"}
    }
    canonical = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_config_files(
    inferencex_root: Path,
    configured_file: Optional[str],
) -> Iterable[Path]:
    if configured_file:
        path = Path(configured_file).expanduser()
        if not path.is_absolute():
            path = inferencex_root / path
        yield path
        return

    yield inferencex_root / "configs" / "amd-master.yaml"
    yield inferencex_root / "configs" / "nvidia-master.yaml"


def _find_recipe_config(
    inferencex_root: Path,
    agentx: AgentXConfig,
) -> Path:
    if not agentx.recipe:
        raise ValueError("agentx.recipe is required for recipe resolution")
    matches = []
    checked = []
    for config_file in _candidate_config_files(inferencex_root, agentx.config_file):
        checked.append(str(config_file))
        if not config_file.is_file():
            continue
        try:
            with config_file.open(encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Unable to read InferenceX config {config_file}: {exc}"
            ) from exc
        if isinstance(data, dict) and agentx.recipe in data:
            matches.append(config_file)

    if not matches:
        locations = ", ".join(checked) or "<none>"
        raise ValueError(
            f"AgentX recipe '{agentx.recipe}' was not found in: {locations}"
        )
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"AgentX recipe '{agentx.recipe}' is ambiguous; found in: "
            f"{locations}. Set agentx.config_file explicitly."
        )
    return matches[0]


def _infer_recipe(
    config: BenchmarkConfig,
    inferencex_root: Path,
    runner_type: str,
) -> Path:
    """Infer one single-node AgentX recipe from public benchmark fields."""

    assert config.agentx is not None
    matches = []
    checked = []
    for config_file in _candidate_config_files(
        inferencex_root, config.agentx.config_file
    ):
        checked.append(str(config_file))
        if not config_file.is_file():
            continue
        try:
            with config_file.open(encoding="utf-8") as handle:
                recipes = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Unable to read InferenceX config {config_file}: {exc}"
            ) from exc
        if not isinstance(recipes, dict):
            continue
        for recipe, value in recipes.items():
            if not isinstance(value, dict):
                continue
            scenarios = value.get("scenarios")
            runner = str(value.get("runner", "")).lower()
            if (
                value.get("model") == config.model
                and str(value.get("framework", "")).lower() == config.framework
                and str(value.get("precision", "")).lower() == config.precision
                and isinstance(scenarios, dict)
                and "agentic-coding" in scenarios
                and not bool(value.get("multinode", False))
                and runner_type.lower() in runner
            ):
                matches.append((str(recipe), config_file))

    if not matches:
        locations = ", ".join(checked) or "<none>"
        raise ValueError(
            "InferenceX has no single-node AgentX recipe matching "
            f"model={config.model!r}, framework={config.framework!r}, "
            f"precision={config.precision!r}, gpu={runner_type!r} in: "
            f"{locations}"
        )
    if len(matches) > 1:
        names = ", ".join(recipe for recipe, _ in matches)
        raise ValueError(
            "Multiple InferenceX AgentX recipes match this benchmark: "
            f"{names}. Set agentx.recipe to choose one."
        )

    config.agentx.recipe = matches[0][0]
    logger.info(
        "Selected InferenceX AgentX recipe %s from %s",
        matches[0][0],
        matches[0][1],
    )
    return matches[0][1]


def _selector_matches(entry: Dict[str, Any], selector: Dict[str, Any]) -> bool:
    for raw_key, expected in selector.items():
        key = raw_key.replace("_", "-")
        actual = entry.get(key)
        if isinstance(actual, dict) and isinstance(expected, str):
            actual = actual.get("name")
        if actual != expected:
            return False
    return True


def _describe_point(entry: Dict[str, Any]) -> str:
    fields = (
        "tp",
        "pp",
        "ep",
        "conc",
        "spec-decoding",
        "kv-offloading",
        "kv-offload-backend",
    )
    values = []
    for key in fields:
        if key not in entry:
            continue
        value = entry[key]
        if isinstance(value, dict):
            value = value.get("name", value)
        values.append(f"{key}={value}")
    return ", ".join(values)


def _load_mapping(path: Path, description: str) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"Unable to read InferenceX {description} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"InferenceX {description} must contain a YAML mapping: {path}"
        )
    return value


def _concurrency_values(arm: Dict[str, Any]) -> list[int]:
    raw_list = arm.get("conc-list")
    if isinstance(raw_list, list):
        try:
            values = [int(value) for value in raw_list]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "InferenceX AgentX conc-list must contain integers"
            ) from exc
        if any(value <= 0 for value in values):
            raise ValueError("InferenceX AgentX conc-list values must be positive")
        return values

    if "conc-start" not in arm or "conc-end" not in arm:
        return []
    try:
        start = int(arm["conc-start"])
        end = int(arm["conc-end"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "InferenceX AgentX concurrency bounds must be integers"
        ) from exc
    if start <= 0 or end <= 0 or start > end:
        raise ValueError(
            "InferenceX AgentX concurrency bounds must satisfy 0 < start <= end"
        )
    values = []
    current = start
    while current <= end:
        values.append(current)
        if current == end:
            break
        current *= 2
        if current > end:
            current = end
    return values


def _agentic_dram_offload_gb(
    scenario: Dict[str, Any],
    arm: Dict[str, Any],
    runner: str,
    runners: Dict[str, Any],
) -> int:
    if arm.get("kv-offloading", "none") != "dram":
        return 0

    hardware = runners.get("hardware")
    node = hardware.get(runner) if isinstance(hardware, dict) else None
    if not isinstance(node, dict):
        raise ValueError(
            f"InferenceX runners.yaml has no hardware metadata for {runner!r}"
        )
    try:
        available_mib = min(
            int(node["available-cpu-dram-mib"]),
            _MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB,
        )
        gpus_per_node = int(node["gpus-per-node"])
        utilization = Decimal(str(scenario["dram-utilization"]))
        gpu_count = int(arm["tp"]) * int(arm.get("pp", 1)) * int(arm.get("pcp-size", 1))
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "InferenceX AgentX DRAM offload metadata is incomplete"
        ) from exc
    if not Decimal("0") <= utilization <= Decimal("1"):
        raise ValueError("InferenceX AgentX dram-utilization must be between 0 and 1")
    if gpus_per_node <= 0 or gpu_count <= 0 or gpu_count > gpus_per_node:
        raise ValueError(
            f"InferenceX AgentX topology needs {gpu_count} GPUs on a "
            f"{gpus_per_node}-GPU node"
        )
    proportional_bytes = (
        Decimal(available_mib)
        * _BYTES_PER_MIB
        * utilization
        * gpu_count
        / gpus_per_node
    )
    return int(proportional_bytes / _BYTES_PER_GB)


def _expand_single_node_agentx_entries(
    inferencex_root: Path,
    config_file: Path,
    recipe: str,
    concurrency: int,
) -> list[Dict[str, Any]]:
    """Expand the small subset of the InferenceX matrix used by AgentX.

    Calling InferenceX's generator directly would make a normal Magpie host
    install InferenceX's development dependency on pydantic. Reading the
    public YAML contract here keeps Magpie's runtime dependency-free while the
    actual replay remains entirely owned by the pinned InferenceX script.
    """

    recipes = _load_mapping(config_file, "recipe config")
    value = recipes.get(recipe)
    if not isinstance(value, dict):
        raise ValueError(f"AgentX recipe {recipe!r} is not a YAML object")
    if bool(value.get("multinode", False)):
        raise ValueError("Magpie AgentX v1 supports single-node recipes only")

    required_recipe_fields = {
        "image",
        "model",
        "model-prefix",
        "precision",
        "framework",
        "runner",
    }
    missing_recipe_fields = sorted(required_recipe_fields.difference(value))
    if missing_recipe_fields:
        raise ValueError(
            f"AgentX recipe {recipe!r} is missing required fields: "
            + ", ".join(missing_recipe_fields)
        )

    scenarios = value.get("scenarios")
    agentic = scenarios.get("agentic-coding") if isinstance(scenarios, dict) else None
    if not isinstance(agentic, list):
        raise ValueError(f"AgentX recipe {recipe!r} has no agentic-coding scenario")

    runners_file = inferencex_root / "configs" / "runners.yaml"
    runners = _load_mapping(runners_file, "runner config")
    runner = str(value.get("runner", ""))
    entries: list[Dict[str, Any]] = []
    for scenario in agentic:
        if not isinstance(scenario, dict):
            continue
        search_space = scenario.get("search-space")
        if not isinstance(search_space, list):
            continue
        for arm in search_space:
            if not isinstance(arm, dict) or concurrency not in _concurrency_values(arm):
                continue
            try:
                tp = int(arm["tp"])
                pp = int(arm.get("pp", 1))
                dcp_size = int(arm.get("dcp-size", 1))
                pcp_size = int(arm.get("pcp-size", 1))
                ep = int(arm.get("ep", 1))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"AgentX recipe {recipe!r} has invalid parallelism metadata"
                ) from exc
            if min(tp, pp, dcp_size, pcp_size, ep) <= 0:
                raise ValueError(
                    f"AgentX recipe {recipe!r} parallelism values must be positive"
                )
            kv_offloading = str(arm.get("kv-offloading", "none"))
            backend = arm.get("kv-offload-backend")
            backend_name = backend.get("name", "") if isinstance(backend, dict) else ""
            spec_decoding = str(arm.get("spec-decoding", "none"))
            entry: Dict[str, Any] = {
                "image": value["image"],
                "model": value["model"],
                "model-prefix": value["model-prefix"],
                "precision": value["precision"],
                "framework": value["framework"],
                "runner": runner,
                "tp": tp,
                "pp": pp,
                "dcp-size": dcp_size,
                "pcp-size": pcp_size,
                "ep": ep,
                "dp-attn": bool(arm.get("dp-attn", False)),
                "spec-decoding": spec_decoding,
                "conc": concurrency,
                "kv-offloading": kv_offloading,
                "total-cpu-dram-gb": _agentic_dram_offload_gb(
                    scenario, arm, runner, runners
                ),
                "duration": _DEFAULT_AGENTIC_DURATION_SECONDS,
                # InferenceX's generator injects this default before computing
                # the recipe fingerprint. Keep it in the resolved entry even
                # though AgentX does not run the fixed-sequence eval pipeline.
                "run-eval": False,
                "exp-name": (
                    f"{value['model-prefix']}_tp{tp}_conc{concurrency}_"
                    f"kv{kv_offloading}"
                    + (f"-{backend_name}" if backend_name else "")
                    + (f"_spec-{spec_decoding}" if spec_decoding != "none" else "")
                ),
                "scenario-type": "agentic-coding",
            }
            if backend is not None:
                entry["kv-offload-backend"] = backend
            for key in ("router", "kv-p2p-transfer"):
                component = arm.get(key, value.get(key))
                if component is not None:
                    entry[key] = component
            entries.append(entry)
    return entries


def resolve_agentx_recipe(
    config: BenchmarkConfig,
    inferencex_path: str,
    runner_type: Optional[str] = None,
) -> AgentXLaunchSpec:
    """Resolve and apply one AgentX matrix point to ``config``."""

    if config.agentx is None:
        raise ValueError("AgentX configuration is missing")

    inferencex_root = Path(inferencex_path).resolve()
    agentx = config.agentx
    if not agentx.recipe:
        if not runner_type:
            raise ValueError("runner_type is required when agentx.recipe is not set")
        config_file = _infer_recipe(
            config,
            inferencex_root,
            runner_type,
        )
    else:
        config_file = _find_recipe_config(inferencex_root, agentx)
    recipe = agentx.recipe
    assert recipe is not None
    concurrency = agentx.concurrency
    if concurrency is None:
        raw_concurrency = config.envs.get("CONC", config.envs.get("conc"))
        if raw_concurrency is None:
            raise ValueError("agentx.concurrency or benchmark.envs.CONC is required")
        concurrency = int(raw_concurrency)
        agentx.concurrency = concurrency
    entries = _expand_single_node_agentx_entries(
        inferencex_root,
        config_file,
        recipe,
        concurrency,
    )
    entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("scenario-type") == "agentic-coding"
        and _selector_matches(entry, agentx.selector)
    ]
    if not entries:
        selector = json.dumps(agentx.selector, sort_keys=True)
        raise ValueError(
            f"AgentX recipe '{recipe}' has no concurrency "
            f"{concurrency} point matching selector {selector}"
        )
    if len(entries) > 1:
        candidates = "; ".join(_describe_point(entry) for entry in entries)
        raise ValueError(
            f"AgentX recipe '{recipe}' resolves to multiple points. "
            f"Set agentx.selector to disambiguate: {candidates}"
        )

    entry = dict(entries[0])
    if "prefill" in entry or "decode" in entry:
        raise ValueError(
            "Magpie AgentX v1 supports single-node recipes only; the resolved "
            "point is multi-node/disaggregated"
        )
    framework = str(entry.get("framework", "")).lower()
    if framework not in {"vllm", "sglang", "atom"}:
        raise ValueError(
            "Magpie AgentX v1 supports vllm, sglang, and atom single-node "
            f"recipes; got framework={framework!r}"
        )

    identity_mismatches = []
    expected_identity = {
        "model": config.model,
        "framework": config.framework,
        "precision": config.precision,
    }
    for field, expected in expected_identity.items():
        actual = str(entry.get(field, ""))
        if field in {"framework", "precision"}:
            actual = actual.lower()
            expected = str(expected).lower()
        if actual != expected:
            identity_mismatches.append(f"{field}={actual!r} (expected {expected!r})")
    if identity_mismatches:
        raise ValueError(
            f"AgentX recipe {recipe!r} does not match the benchmark identity: "
            + ", ".join(identity_mismatches)
        )

    resolved_runner = str(entry.get("runner", "")).lower()
    if runner_type and runner_type.lower() not in resolved_runner:
        raise ValueError(
            f"AgentX recipe {recipe!r} targets runner {resolved_runner!r}, "
            f"not detected runner {runner_type!r}"
        )

    # The public benchmark config explicitly owns the runtime image. Include
    # that pin in the effective InferenceX recipe fingerprint so customized
    # registries/images remain reproducible rather than being mislabeled as
    # the image recorded in the source recipe.
    if config.docker_image:
        entry["image"] = config.docker_image
    entry["recipe-fingerprint"] = _recipe_fingerprint(entry)

    _apply_launch_entry(config, entry)
    try:
        config_file_label = str(config_file.relative_to(inferencex_root))
    except ValueError:
        config_file_label = str(config_file)
    spec = AgentXLaunchSpec(
        recipe=recipe,
        config_file=config_file_label,
        entry=entry,
    )
    agentx.resolved = entry
    logger.info("Resolved AgentX recipe %s: %s", recipe, _describe_point(entry))
    return spec


def _apply_launch_entry(config: BenchmarkConfig, entry: Dict[str, Any]) -> None:
    assert config.agentx is not None
    config.framework = str(entry["framework"]).lower()
    config.model = str(entry["model"])
    config.precision = str(entry["precision"]).lower()
    # The benchmark YAML owns the runtime image. The recipe image is useful
    # metadata/default information, but must not silently replace a pinned
    # runtime supplied by the user.
    config.docker_image = config.docker_image or str(entry["image"])

    backend = entry.get("kv-offload-backend")
    backend_name = backend.get("name", "") if isinstance(backend, dict) else ""
    runtime_env: Dict[str, Any] = dict(config.envs)
    runtime_env.update(
        {
            "MODEL_PREFIX": entry.get("model-prefix", ""),
            "FRAMEWORK": config.framework,
            "IMAGE": config.docker_image,
            "EXP_NAME": entry.get("exp-name", config.agentx.recipe),
            "TP": entry.get("tp", 1),
            "PP_SIZE": entry.get("pp", 1),
            "DCP_SIZE": entry.get("dcp-size", 1),
            "PCP_SIZE": entry.get("pcp-size", 1),
            "EP_SIZE": entry.get("ep", 1),
            "DP_ATTENTION": str(entry.get("dp-attn", False)).lower(),
            "CONC": entry["conc"],
            "SPEC_DECODING": entry.get("spec-decoding", "none"),
            "DISAGG": str(entry.get("disagg", False)).lower(),
            "SCENARIO_TYPE": "agentic-coding",
            "SCENARIO_SUBDIR": "agentic/",
            "IS_AGENTIC": "1",
            "KV_OFFLOADING": entry.get("kv-offloading", "none"),
            "KV_OFFLOAD_BACKEND": backend_name,
            "KV_OFFLOAD_BACKEND_METADATA": (
                json.dumps(backend, separators=(",", ":"))
                if isinstance(backend, dict)
                else ""
            ),
            "ROUTER_METADATA": (
                json.dumps(entry["router"], separators=(",", ":"))
                if isinstance(entry.get("router"), dict)
                else ""
            ),
            "KV_P2P_TRANSFER": entry.get("kv-p2p-transfer", ""),
            "TOTAL_CPU_DRAM_GB": entry.get("total-cpu-dram-gb", 0),
            "DURATION": entry.get("duration", 3600),
            "RUN_EVAL": "false",
            "EVAL_ONLY": "false",
            "RECIPE_FINGERPRINT": entry["recipe-fingerprint"],
            "AIPERF_EXPERIMENTAL_FAST": ("1" if config.agentx.mode == "fast" else "0"),
            "AIPERF_FAILED_REQUEST_THRESHOLD": (config.agentx.failed_request_threshold),
        }
    )
    config.envs = runtime_env

    # Loading a large model plus the canonical one-hour profile can exceed the
    # fixed-sequence default. Respect an explicit larger timeout.
    minimum_timeout = 2400 if config.agentx.mode == "fast" else 7200
    config.timeout_seconds = max(config.timeout_seconds, minimum_timeout)


def ensure_agentx_dependencies(inferencex_path: str) -> None:
    """Initialize the pinned AIPerf submodule required by AgentX."""

    root = Path(inferencex_path).resolve()
    requirements = root / "utils" / "agentic-benchmark" / "requirements.txt"
    aiperf_project = root / "utils" / "aiperf" / "pyproject.toml"
    if not requirements.is_file():
        raise RuntimeError(
            "InferenceX checkout does not contain AgentX support: "
            f"missing {requirements}"
        )
    if aiperf_project.is_file():
        return

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "utils/aiperf",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0 or not aiperf_project.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Unable to initialize InferenceX utils/aiperf submodule: "
            f"{detail or 'pyproject.toml is still missing'}"
        )
