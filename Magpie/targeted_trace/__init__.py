"""TargetedKernelTrace acquisition contract and adapters."""

from .capture import TargetedTraceRecorder
from .config import TargetSpec, TargetedTraceConfig
from .postprocess import postprocess_trace_dir, validate_shard
from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    LaunchSemantics,
    RuntimeEvidence,
    ShardCounters,
    ShardReceipt,
    SourceEvidence,
    TargetedTraceManifest,
    TargetedTraceRecord,
    TensorEvidence,
    TraceContext,
    TraceIdentity,
    TraceValidationError,
)
from .torch_profiler import adapt_torch_profiler_traces, iter_trace_events

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "LaunchSemantics",
    "RuntimeEvidence",
    "ShardCounters",
    "ShardReceipt",
    "SourceEvidence",
    "TargetSpec",
    "TargetedTraceConfig",
    "TargetedTraceManifest",
    "TargetedTraceRecord",
    "TargetedTraceRecorder",
    "TensorEvidence",
    "TraceContext",
    "TraceIdentity",
    "TraceValidationError",
    "adapt_torch_profiler_traces",
    "iter_trace_events",
    "postprocess_trace_dir",
    "validate_shard",
]
