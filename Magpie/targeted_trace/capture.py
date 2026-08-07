"""Generic Python runtime capture API for Triton and Python-visible HIP launches."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .sampling import stable_key
from .schema import (
    RuntimeEvidence,
    TargetedTraceRecord,
    TraceContext,
    TraceIdentity,
)
from .serialization import invocation_semantics, source_evidence
from .writer import TraceShardWriter, default_shard_path, merge_runtime_manifest


class TargetedTraceRecorder:
    """Capture semantic launch evidence through an explicit, framework-neutral API.

    The recorder does not patch packages or dispatchers.  Callers inject these
    methods at a known Python-visible launch/wrapper boundary, keeping framework
    discovery separate from the durable artifact contract.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        run_id: str,
        run_seed: str,
        framework: str,
        rank: int = 0,
        pid: Optional[int] = None,
        world_size: Optional[int] = None,
        framework_version: Optional[str] = None,
        execution_mode: str = "unknown",
        stage: str = "unknown",
        sample_rate: float = 1.0,
        max_records: int = 100_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.framework = framework
        self.rank = rank
        self.pid = os.getpid() if pid is None else pid
        self.world_size = world_size
        self.framework_version = framework_version
        self.execution_mode = execution_mode
        self.stage = stage
        self._occurrences: defaultdict[str, int] = defaultdict(int)
        self._targets: dict[tuple[str, str], dict[str, Any]] = {}
        self._manifest_written = False
        self.writer = TraceShardWriter(
            default_shard_path(self.output_dir, rank=rank, pid=self.pid),
            run_id=run_id,
            rank=rank,
            pid=self.pid,
            run_seed=run_seed,
            sample_rate=sample_rate,
            max_records=max_records,
            header_metadata={
                "framework": framework,
                "framework_version": framework_version,
                "world_size": world_size,
                "capture_backend": "python_runtime",
            },
        )

    def _record(
        self,
        *,
        kind: str,
        target_id: str,
        kernel_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        grid: Any,
        source_path: Optional[str],
        source_line: Optional[int],
        source_function: Optional[str],
        positional_names: Optional[Sequence[str]],
        meta_names: Iterable[str],
        constexpr_names: Iterable[str],
        variant_id: str,
        package: Optional[str],
        image: Optional[str],
        source_hashes: Optional[Mapping[str, str]],
        provenance_hashes: Optional[Mapping[str, str]],
        runtime: Optional[RuntimeEvidence],
        stage: Optional[str],
        graph_id: Optional[str],
    ) -> bool:
        target_key = (target_id, variant_id)
        if target_key not in self._targets:
            self._targets[target_key] = {
                "target_id": target_id,
                "variant_id": variant_id,
                "name_patterns": [],
                "package": package,
                "image": image,
                "source": (
                    {
                        "path": source_path,
                        "line": source_line,
                        "function": source_function,
                    }
                    if source_path
                    else None
                ),
                "source_hashes": dict(source_hashes or {}),
                "provenance_hashes": dict(provenance_hashes or {}),
            }
        patterns = self._targets[target_key]["name_patterns"]
        if kernel_name not in patterns:
            patterns.append(kernel_name)
        try:
            semantics, warnings = invocation_semantics(
                args=args,
                kwargs=kwargs,
                positional_names=positional_names,
                meta_names=meta_names,
                constexpr_names=constexpr_names,
                python_grid=grid,
                source=source_evidence(
                    source_path,
                    line=source_line,
                    function=source_function,
                ),
            )
            base_parts = {
                "kind": kind,
                "target_id": target_id,
                "kernel_name": kernel_name,
                "source": semantics.source.to_dict() if semantics.source else None,
                "tensors": [
                    {
                        "name": tensor.name,
                        "shape": list(tensor.shape),
                        "dtype": tensor.dtype,
                        "stride": list(tensor.stride) if tensor.stride else None,
                    }
                    for tensor in semantics.tensors
                ],
                "scalars": dict(semantics.named_scalars),
                "constexpr": dict(semantics.constexpr),
                "meta": dict(semantics.meta),
                "grid": semantics.python_grid,
            }
            base_token = stable_key(base_parts)
            occurrence = self._occurrences[base_token]
            self._occurrences[base_token] += 1
            event_key = stable_key(base_parts, occurrence=occurrence)
            runtime_evidence = runtime or RuntimeEvidence(gpu_symbol=kernel_name)
            record = TargetedTraceRecord(
                kind=kind,
                stable_event_key=event_key,
                identity=TraceIdentity(
                    run_id=self.run_id,
                    target_id=target_id,
                    variant_id=variant_id,
                    package=package,
                    image=image,
                    source_hashes=dict(source_hashes or {}),
                    provenance_hashes=dict(provenance_hashes or {}),
                ),
                context=TraceContext(
                    framework=self.framework,
                    framework_version=self.framework_version,
                    rank=self.rank,
                    pid=self.pid,
                    world_size=self.world_size,
                    stage=stage or self.stage,
                    execution_mode=self.execution_mode,
                    graph_id=graph_id,
                ),
                semantics=semantics,
                runtime=runtime_evidence,
                timestamp_ns=time.time_ns(),
                warnings=warnings,
            )
            return self.writer.submit(record)
        except Exception:
            self.writer.note_failed_observation("serialization_error")
            return False

    def record_triton_launch(
        self,
        *,
        target_id: str,
        kernel_name: str,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        grid: Any = None,
        source_path: Optional[str] = None,
        source_line: Optional[int] = None,
        source_function: Optional[str] = None,
        positional_names: Optional[Sequence[str]] = None,
        meta_names: Iterable[str] = (),
        constexpr_names: Iterable[str] = (),
        variant_id: str = "baseline",
        package: Optional[str] = None,
        image: Optional[str] = None,
        source_hashes: Optional[Mapping[str, str]] = None,
        provenance_hashes: Optional[Mapping[str, str]] = None,
        runtime: Optional[RuntimeEvidence] = None,
        stage: Optional[str] = None,
        graph_id: Optional[str] = None,
    ) -> bool:
        """Record one Python-visible Triton launch."""

        return self._record(
            kind="triton_launch",
            target_id=target_id,
            kernel_name=kernel_name,
            args=args,
            kwargs=kwargs or {},
            grid=grid,
            source_path=source_path,
            source_line=source_line,
            source_function=source_function,
            positional_names=positional_names,
            meta_names=meta_names,
            constexpr_names=constexpr_names,
            variant_id=variant_id,
            package=package,
            image=image,
            source_hashes=source_hashes,
            provenance_hashes=provenance_hashes,
            runtime=runtime,
            stage=stage,
            graph_id=graph_id,
        )

    def record_python_hip_launch(
        self,
        *,
        target_id: str,
        kernel_name: str,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        grid: Any = None,
        source_path: Optional[str] = None,
        source_line: Optional[int] = None,
        source_function: Optional[str] = None,
        positional_names: Optional[Sequence[str]] = None,
        meta_names: Iterable[str] = (),
        constexpr_names: Iterable[str] = (),
        variant_id: str = "baseline",
        package: Optional[str] = None,
        image: Optional[str] = None,
        source_hashes: Optional[Mapping[str, str]] = None,
        provenance_hashes: Optional[Mapping[str, str]] = None,
        runtime: Optional[RuntimeEvidence] = None,
        stage: Optional[str] = None,
        graph_id: Optional[str] = None,
    ) -> bool:
        """Record one Python wrapper call that launches a HIP/custom op."""

        return self._record(
            kind="python_hip_launch",
            target_id=target_id,
            kernel_name=kernel_name,
            args=args,
            kwargs=kwargs or {},
            grid=grid,
            source_path=source_path,
            source_line=source_line,
            source_function=source_function,
            positional_names=positional_names,
            meta_names=meta_names,
            constexpr_names=constexpr_names,
            variant_id=variant_id,
            package=package,
            image=image,
            source_hashes=source_hashes,
            provenance_hashes=provenance_hashes,
            runtime=runtime,
            stage=stage,
            graph_id=graph_id,
        )

    def close(self):
        """Close the underlying shard and return its receipt."""

        receipt = self.writer.close()
        if not self._manifest_written:
            merge_runtime_manifest(
                self.output_dir / "manifest.json",
                run_id=self.run_id,
                receipt=receipt,
                targets=list(self._targets.values()),
                provenance={
                    "framework": self.framework,
                    "framework_version": self.framework_version,
                    "world_size": self.world_size,
                    "capture_backend": "python_runtime",
                },
            )
            self._manifest_written = True
        return receipt

    def __enter__(self) -> "TargetedTraceRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
