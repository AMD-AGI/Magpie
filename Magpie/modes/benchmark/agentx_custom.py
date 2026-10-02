"""Resolve explicit custom-model AgentX points without modifying InferenceX."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from .config import BenchmarkConfig


def native_context_length(config: dict[str, Any]) -> int:
    text = config.get("text_config", config.get("language_config", config))
    if not isinstance(text, dict):
        raise ValueError("Model text configuration must be an object")  # noqa: TRY004
    values = [
        text.get(name)
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        )
    ]
    confirmed = [value for value in values if type(value) is int and value > 0]
    if not confirmed:
        raise ValueError(
            "config.json does not declare a positive native context length"
        )
    return min(confirmed)


def _model_metadata(
    config: BenchmarkConfig, *, pin_remote: bool = False
) -> tuple[dict[str, Any], str, str, str | None]:
    envs = {key.upper(): value for key, value in config.envs.items()}
    local = str(envs.get("MODEL_PATH") or "")
    revision = None
    if local:
        path = Path(local).expanduser() / "config.json"
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"Custom AgentX requires readable local model metadata: {path}"
            ) from exc
        source = str(path.resolve())
    elif Path(config.model).is_dir():
        path = Path(config.model) / "config.json"
        contents = path.read_bytes()
        source = str(path.resolve())
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config.model):
            raise ValueError(
                "Custom AgentX model must be a local directory or HuggingFace model ID"
            )
        saved = (config.agentx.resolved if config.agentx else None) or {}
        revision = (
            saved.get("model-revision")
            if saved.get("custom")
            else envs.get("AGENTX_MODEL_REVISION")
        )
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("Custom AgentX model revision must be an immutable commit")
        source = f"https://huggingface.co/{quote(config.model, safe='/')}/resolve/{revision or 'main'}/config.json"
        headers = {"Accept": "application/json"}
        try:
            with urlopen(Request(source, headers=headers), timeout=30) as response:
                contents = response.read(8 * 1024 * 1024 + 1)
                observed = getattr(response, "headers", {}).get("X-Repo-Commit")
                if observed:
                    if revision and revision != observed:
                        raise ValueError("Custom model metadata revision changed")
                    revision = observed
        except OSError as exc:
            raise ValueError(
                f"Cannot read public model config metadata for {config.model}; "
                "gated/private models require a local MODEL_PATH"
            ) from exc
        if len(contents) > 8 * 1024 * 1024:
            raise ValueError("Custom AgentX model config metadata exceeds 8 MiB")
        if pin_remote and not re.fullmatch(r"[0-9a-f]{40}", str(revision or "")):
            raise ValueError(
                "Custom AgentX remote config must identify its immutable HF revision"
            )
        if revision:
            source = f"https://huggingface.co/{quote(config.model, safe='/')}/resolve/{revision}/config.json"
    data = json.loads(contents)
    if not isinstance(data, dict):
        raise ValueError(
            "Custom AgentX config.json must contain an object"
        )  # noqa: TRY004
    return data, hashlib.sha256(contents).hexdigest(), source, revision


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError(f"Custom AgentX {name} must be an explicit positive integer")
    return int(value)


def resolve_custom_entry(
    config: BenchmarkConfig,
    root: Path,
    runner_type: str,
) -> dict[str, Any] | None:
    """Return a declared custom point, or None when no generic capability exists."""
    assert config.agentx is not None
    name = f"custom-{config.framework}-{runner_type}"
    manifest = root / "configs/agentx-launchers.json"
    owned = (root / "benchmarks/srt_agentic.sh").is_file()
    if owned:
        if runner_type not in {"mi300x", "mi325x", "mi355x"}:
            raise ValueError(
                "Custom AgentX supports AMD MI300X, MI325X, and MI355X only"
            )
        row = {
            "framework": config.framework,
            "runner_type": runner_type,
            "launch_overrides_version": 1,
            "max_gpus": 8,
        }
    else:
        if not manifest.is_file():
            return None
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported AgentX launcher manifest")
        generic = data.get("generic", {})
        if not isinstance(generic, dict):
            raise ValueError(
                "AgentX generic launcher manifest must be an object"
            )  # noqa: TRY004
        row = generic.get(name)
        if row is None:
            return None
    if (
        not isinstance(row, dict)
        or row.get("framework") != config.framework
        or row.get("runner_type") != runner_type
    ):
        raise ValueError(f"Generic AgentX capability identity mismatch: {name}")
    if row.get("launch_overrides_version") != 1:
        raise ValueError("Generic AgentX requires the verified launch extension v1")
    if not config.docker_image:
        raise ValueError(
            "Custom AgentX requires an explicit docker_image, including local runs"
        )
    if config.framework not in {"sglang", "vllm"}:
        raise ValueError("Custom AgentX supports SGLang and vLLM only")
    if any(character.isspace() for character in config.model):
        raise ValueError(
            "Custom AgentX model identifier cannot contain whitespace; use MODEL_PATH for local files"
        )
    envs = {key.upper(): value for key, value in config.envs.items()}
    if envs.get("SERVED_MODEL_NAME", config.model) != config.model:
        raise ValueError(
            "Custom AgentX SERVED_MODEL_NAME must match the model identity"
        )
    tp = _positive_int(envs.get("TP"), "TP")
    ep = _positive_int(envs.get("EP_SIZE", envs.get("EP")), "EP_SIZE")
    if "EP" in envs and _positive_int(envs["EP"], "EP") != ep:
        raise ValueError("Custom AgentX EP and EP_SIZE disagree")
    conc = _positive_int(config.agentx.concurrency or envs.get("CONC"), "CONC")
    if tp > int(row.get("max_gpus", 8)) or tp % ep:
        raise ValueError("Custom AgentX EP must divide TP within one declared GPU node")
    if config.framework == "vllm" and ep not in {1, tp}:
        raise ValueError("Custom vLLM AgentX supports EP=1 or EP=TP")
    for key in ("PP", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "DP_SIZE", "NNODES"):
        if str(envs.get(key, 1)) != "1":
            raise ValueError(f"Custom AgentX supports {key}=1 only")
    for key in ("DP_ATTENTION", "DISAGG"):
        if str(envs.get(key, False)).lower() not in {"false", "0", ""}:
            raise ValueError(f"Custom AgentX does not support {key}")
    if str(envs.get("KV_OFFLOADING", "none")) != "none" or envs.get(
        "KV_OFFLOAD_BACKEND"
    ):
        raise ValueError("Custom AgentX currently supports KV_OFFLOADING=none only")
    if str(envs.get("SPEC_DECODING", "none")) != "none":
        raise ValueError("Custom AgentX does not infer speculative decoding recipes")
    precision = config.precision.lower()
    if precision not in {
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "fp32",
        "float32",
        "auto",
        "fp8",
        "fp4",
        "mxfp4",
    }:
        raise ValueError(f"Unsupported custom AgentX precision: {precision}")
    metadata, checksum, source, revision = _model_metadata(config, pin_remote=owned)
    if precision in {"fp8", "fp4", "mxfp4"} and not metadata.get("quantization_config"):
        raise ValueError(
            "Quantized custom AgentX requires quantization_config in model config.json"
        )
    native = native_context_length(metadata)
    maximum = _positive_int(envs.get("MAX_MODEL_LEN", native), "MAX_MODEL_LEN")
    if maximum > native:
        raise ValueError(
            f"Custom AgentX MAX_MODEL_LEN={maximum} exceeds confirmed native context {native}"
        )
    config.agentx.recipe = name
    config.agentx.concurrency = conc
    result = {
        "custom": True,
        "model": config.model,
        "framework": config.framework,
        "precision": precision,
        "image": config.docker_image,
        "model-prefix": "custom-"
        + re.sub(r"[^A-Za-z0-9_.-]+", "-", config.model.rsplit("/", 1)[-1]),
        "runner": runner_type,
        "tp": tp,
        "pp": 1,
        "ep": ep,
        "dcp-size": 1,
        "pcp-size": 1,
        "dp-attn": False,
        "conc": conc,
        "spec-decoding": "none",
        "kv-offloading": "none",
        "total-cpu-dram-gb": 0,
        "duration": 3600,
        "run-eval": False,
        "scenario-type": "agentic-coding",
        "exp-name": f"{name}_tp{tp}_conc{conc}",
        "native-context-length": native,
        "max-model-len": maximum,
        "model-config-sha256": checksum,
        "model-config-source": source,
        "trace-loader": "semianalysis_cc_traces_weka_062126_256k",
    }
    if revision:
        result["model-revision"] = revision
    return result
