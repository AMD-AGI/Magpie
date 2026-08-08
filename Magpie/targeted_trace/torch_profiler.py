"""Streaming adapter from PyTorch/Chrome profiler traces to the trace contract."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

from .config import TargetSpec, TargetedTraceConfig
from .sampling import stable_key
from .schema import (
    LaunchSemantics,
    RuntimeEvidence,
    SourceEvidence,
    TargetedTraceManifest,
    TargetedTraceRecord,
    TensorEvidence,
    TraceContext,
    TraceIdentity,
)
from .writer import TraceShardWriter, default_shard_path, write_manifest


TRACE_EVENTS_KEY = '"traceEvents"'
RANK_RE = re.compile(r"(?:^|[/_.-])rank[_-]?(\d+)(?:[/_.-]|$)", re.IGNORECASE)


def _open_trace(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_trace_events(path: Path, *, chunk_size: int = 64 * 1024) -> Iterator[Mapping[str, Any]]:
    """Yield ``traceEvents`` entries without materializing the full trace object."""

    decoder = json.JSONDecoder()
    with _open_trace(Path(path)) as stream:
        buffer = ""
        array_started = False
        eof = False
        while not array_started:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"traceEvents array not found in {path}")
            buffer += chunk
            key_index = buffer.find(TRACE_EVENTS_KEY)
            if key_index < 0:
                buffer = buffer[-len(TRACE_EVENTS_KEY) :]
                continue
            array_index = buffer.find("[", key_index + len(TRACE_EVENTS_KEY))
            while array_index < 0:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"traceEvents array is truncated in {path}")
                buffer += chunk
                array_index = buffer.find("[", key_index + len(TRACE_EVENTS_KEY))
            buffer = buffer[array_index + 1 :]
            array_started = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer and eof:
                raise ValueError(f"traceEvents array is truncated in {path}")
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(chunk_size)
                if not chunk:
                    if eof:
                        raise ValueError(f"invalid traceEvents JSON in {path}")
                    eof = True
                else:
                    buffer += chunk
                continue
            buffer = buffer[end:]
            if isinstance(value, Mapping):
                yield value


def _first(args: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in args.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _vector(args: Mapping[str, Any], prefix: str) -> Optional[Tuple[int, ...]]:
    direct = _first(args, [prefix, f"{prefix} size", f"{prefix}_size"])
    if isinstance(direct, (list, tuple)):
        try:
            return tuple(int(item) for item in direct)
        except (TypeError, ValueError):
            return None
    values: List[int] = []
    for axis in ("x", "y", "z"):
        value = _first(args, [f"{prefix}_{axis}", f"{prefix} {axis}"])
        if value is None:
            if values:
                values.append(1)
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return None
    return tuple(values) if values else None


def _rank(event: Mapping[str, Any], path: Path) -> int:
    args = event.get("args", {})
    if isinstance(args, Mapping):
        value = _first(args, ["rank", "global rank", "distributed rank"])
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    match = RANK_RE.search(path.as_posix())
    return int(match.group(1)) if match else 0


def _stage(event: Mapping[str, Any], path: Path) -> str:
    args = event.get("args", {})
    if isinstance(args, Mapping):
        value = _first(args, ["stage", "inference stage", "phase"])
        if value is not None:
            return str(value).lower()
    lowered = path.as_posix().lower()
    for stage in ("prefilldecode", "decode", "prefill", "mixed"):
        if stage in lowered:
            return stage
    return "unknown"


def _tensor_semantics(
    args: Mapping[str, Any],
) -> Tuple[Tuple[TensorEvidence, ...], Tuple[str, ...]]:
    dims = _first(args, ["Input Dims", "Input Shapes", "shapes"])
    dtypes = _first(args, ["Input type", "Input Types", "dtypes"])
    strides = _first(args, ["Input Strides", "strides"])
    if not isinstance(dims, (list, tuple)):
        return (), ("torch_profiler_missing_tensor_shapes",)
    if dims and not isinstance(dims[0], (list, tuple)):
        dims = [dims]
    dtype_items = list(dtypes) if isinstance(dtypes, (list, tuple)) else []
    stride_items = list(strides) if isinstance(strides, (list, tuple)) else []
    tensors: List[TensorEvidence] = []
    warnings: List[str] = []
    for index, shape in enumerate(dims):
        if not isinstance(shape, (list, tuple)):
            continue
        dtype = str(dtype_items[index]) if index < len(dtype_items) else "unknown"
        stride_value = stride_items[index] if index < len(stride_items) else None
        stride = tuple(stride_value) if isinstance(stride_value, (list, tuple)) else None
        if stride is None:
            warnings.append(f"arg{index}:torch_profiler_missing_stride")
        tensors.append(
            TensorEvidence(
                name=f"arg{index}",
                shape=tuple(shape),
                dtype=dtype,
                stride=stride,
            )
        )
    return tuple(tensors), tuple(warnings)


def _source(spec: TargetSpec) -> Optional[SourceEvidence]:
    if not spec.source:
        return None
    return SourceEvidence.from_dict(spec.source)


def _is_kernel_event(event: Mapping[str, Any]) -> bool:
    if str(event.get("ph", "X")) not in {"X", ""}:
        return False
    category = str(event.get("cat", "")).lower()
    args = event.get("args", {})
    return (
        any(token in category for token in ("kernel", "gpu", "cuda", "hip"))
        or isinstance(args, Mapping)
        and _first(args, ["stream", "grid", "grid_x", "device"]) is not None
    )


def _matching_targets(symbol: str, targets: Sequence[TargetSpec]) -> Iterator[TargetSpec]:
    for target in targets:
        if target.matches(symbol):
            yield target


def adapt_torch_profiler_traces(
    trace_paths: Iterable[Path],
    output_dir: Path,
    *,
    config: TargetedTraceConfig,
    run_id: str,
    framework: str,
    framework_version: Optional[str] = None,
    image: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> TargetedTraceManifest:
    """Stream selected Torch profiler events into checksummed targeted shards."""

    if not config.enabled:
        raise ValueError("targeted trace adapter requires config.enabled=true")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[Tuple[int, int], TraceShardWriter] = {}
    occurrences: defaultdict[str, int] = defaultdict(int)
    input_paths = [Path(path) for path in trace_paths]
    non_capture_paths = [
        path
        for path in input_paths
        if "capture_traces" not in {part.lower() for part in path.parts}
    ]
    selected_paths = sorted(non_capture_paths or input_paths)
    adapter_warnings: List[str] = []

    def get_writer(rank: int, pid: int) -> TraceShardWriter:
        key = (rank, pid)
        if key not in writers:
            writers[key] = TraceShardWriter(
                default_shard_path(output_dir, rank=rank, pid=pid),
                run_id=run_id,
                rank=rank,
                pid=pid,
                run_seed=config.run_seed,
                sample_rate=config.sample_rate,
                max_records=config.max_records_per_shard,
                header_metadata={
                    "framework": framework,
                    "framework_version": framework_version,
                    "capture_backend": "torch_profiler",
                },
            )
        return writers[key]

    for trace_path in selected_paths:
        try:
            events = iter_trace_events(trace_path)
            for event in events:
                if not _is_kernel_event(event):
                    continue
                symbol = str(event.get("name", ""))
                matches = list(_matching_targets(symbol, config.targets))
                if not matches:
                    continue
                rank = _rank(event, trace_path)
                try:
                    pid = max(0, int(event.get("pid", 0)))
                except (TypeError, ValueError):
                    pid = 0
                writer = get_writer(rank, pid)
                args = event.get("args", {})
                args = args if isinstance(args, Mapping) else {}
                tensors, tensor_warnings = _tensor_semantics(args)
                runtime = RuntimeEvidence(
                    cpu_uid=(
                        str(_first(args, ["cpu uid", "external id"]))
                        if _first(args, ["cpu uid", "external id"]) is not None
                        else None
                    ),
                    correlation_id=(
                        str(_first(args, ["correlation", "correlation id"]))
                        if _first(args, ["correlation", "correlation id"])
                        is not None
                        else None
                    ),
                    gpu_uid=stable_key(
                        {
                            "name": symbol,
                            "rank": rank,
                            "tid": event.get("tid"),
                            "ts": event.get("ts"),
                        }
                    ),
                    gpu_symbol=symbol,
                    grid=_vector(args, "grid"),
                    block=_vector(args, "block"),
                    stream=(
                        str(_first(args, ["stream", "stream id"]))
                        if _first(args, ["stream", "stream id"]) is not None
                        else None
                    ),
                    duration_us=(
                        float(event["dur"])
                        if event.get("dur") is not None
                        else None
                    ),
                    timestamp_us=(
                        float(event["ts"])
                        if event.get("ts") is not None
                        else None
                    ),
                )
                graph_id = _first(args, ["graph id", "graph_id", "cuda graph id"])
                execution_mode = "graph" if graph_id is not None else "unknown"
                stage = _stage(event, trace_path)
                for target in matches:
                    base_parts = {
                        "target_id": target.target_id,
                        "symbol": symbol,
                        "rank": rank,
                        "stage": stage,
                        "grid": list(runtime.grid) if runtime.grid else None,
                        "block": list(runtime.block) if runtime.block else None,
                        "tensors": [
                            {
                                "shape": list(tensor.shape),
                                "dtype": tensor.dtype,
                                "stride": (
                                    list(tensor.stride) if tensor.stride else None
                                ),
                            }
                            for tensor in tensors
                        ],
                    }
                    token = stable_key(base_parts)
                    occurrence = occurrences[token]
                    occurrences[token] += 1
                    warnings = list(tensor_warnings)
                    if target.source is None:
                        warnings.append("torch_profiler_missing_launch_source")
                    if stage == "unknown":
                        warnings.append("torch_profiler_missing_phase")
                    if runtime.grid is None:
                        warnings.append("torch_profiler_missing_launch_grid")
                    if runtime.correlation_id is None:
                        warnings.append("torch_profiler_missing_runtime_correlation")
                    try:
                        record = TargetedTraceRecord(
                            kind="torch_profiler_kernel",
                            stable_event_key=stable_key(
                                base_parts, occurrence=occurrence
                            ),
                            identity=TraceIdentity(
                                run_id=run_id,
                                target_id=target.target_id,
                                variant_id=target.variant_id,
                                package=target.package,
                                image=image,
                                source_hashes=target.source_hashes,
                                provenance_hashes=target.provenance_hashes,
                            ),
                            context=TraceContext(
                                framework=framework,
                                framework_version=framework_version,
                                rank=rank,
                                pid=pid,
                                stage=stage,
                                execution_mode=execution_mode,
                                graph_id=(
                                    str(graph_id) if graph_id is not None else None
                                ),
                            ),
                            semantics=LaunchSemantics(
                                source=_source(target),
                                tensors=tensors,
                                meta={
                                    "profiler_category": str(event.get("cat", "")),
                                },
                            ),
                            runtime=runtime,
                            timestamp_ns=(
                                int(float(event["ts"]) * 1000)
                                if event.get("ts") is not None
                                else None
                            ),
                            warnings=tuple(warnings),
                        )
                        writer.submit(record)
                    except Exception:
                        writer.note_failed_observation("serialization_error")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            adapter_warnings.append(f"{trace_path}: {exc}")
            continue

    receipts = tuple(writer.close() for _, writer in sorted(writers.items()))
    manifest = TargetedTraceManifest(
        run_id=run_id,
        acquisition_backend="torch_profiler",
        targets=tuple(target.to_dict() for target in config.targets),
        shards=receipts,
        provenance={
            **dict(provenance or {}),
            "framework": framework,
            "framework_version": framework_version,
            "image": image,
            "input_traces": [str(path) for path in selected_paths],
            "adapter_warnings": adapter_warnings,
        },
    )
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest
