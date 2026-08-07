"""Side-effect-free serialization of Python-visible launch metadata."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import LaunchSemantics, SourceEvidence, TensorEvidence


MAX_DEPTH = 3
MAX_ITEMS = 64


def is_trace_unsafe_proxy(value: Any) -> bool:
    """Return whether inspecting *value* can perturb Torch tracing/compilation."""

    typ = type(value)
    module = getattr(typ, "__module__", "")
    name = getattr(typ, "__name__", "")
    markers = ("torch.fx", "proxy_tensor", "fake_tensor", "torch._subclasses")
    return any(marker in module for marker in markers) or "Proxy" in name or "FakeTensor" in name


def is_tensor_like(value: Any) -> bool:
    """Recognize tensors without importing Torch or reading device memory."""

    return (
        not is_trace_unsafe_proxy(value)
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and callable(getattr(value, "stride", None))
    )


def _dimension(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def tensor_evidence(name: str, value: Any) -> TensorEvidence:
    """Capture shape/dtype/stride while never touching tensor contents."""

    shape = tuple(_dimension(item) for item in list(getattr(value, "shape", ())))
    stride: Optional[Tuple[Any, ...]]
    try:
        stride = tuple(_dimension(item) for item in list(value.stride()))
    except Exception:
        stride = None
    requires_grad: Optional[bool]
    try:
        requires_grad = bool(getattr(value, "requires_grad"))
    except Exception:
        requires_grad = None
    return TensorEvidence(
        name=name,
        shape=shape,
        dtype=str(getattr(value, "dtype", "unknown")),
        stride=stride,
        device=(
            str(getattr(value, "device"))
            if getattr(value, "device", None) is not None
            else None
        ),
        layout=(
            str(getattr(value, "layout"))
            if getattr(value, "layout", None) is not None
            else None
        ),
        requires_grad=requires_grad,
    )


def json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert host metadata to bounded deterministic JSON without raw pointers."""

    if depth > MAX_DEPTH:
        return {"type": type(value).__name__, "truncated": True}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if is_trace_unsafe_proxy(value):
        typ = type(value)
        return {
            "type": getattr(typ, "__name__", "proxy"),
            "module": getattr(typ, "__module__", ""),
            "unavailable": "torch_tracing_proxy",
        }
    if is_tensor_like(value):
        return tensor_evidence("value", value).to_dict()
    if isinstance(value, Mapping):
        items = list(value.items())
        result = {
            str(key): json_safe(item, depth=depth + 1)
            for key, item in items[:MAX_ITEMS]
        }
        if len(items) > MAX_ITEMS:
            result["_truncated"] = len(items) - MAX_ITEMS
        return result
    if isinstance(value, tuple):
        values = list(value)
        return {
            "type": "tuple",
            "items": [json_safe(item, depth=depth + 1) for item in values[:MAX_ITEMS]],
            **({"truncated": len(values) - MAX_ITEMS} if len(values) > MAX_ITEMS else {}),
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
        return [json_safe(item, depth=depth + 1) for item in values[:MAX_ITEMS]]
    if callable(value):
        return {
            "type": "callable",
            "module": getattr(value, "__module__", ""),
            "qualified_name": getattr(value, "__qualname__", getattr(value, "__name__", "")),
        }
    return {
        "type": type(value).__name__,
        "module": type(value).__module__,
    }


def source_evidence(
    path: Optional[str], *, line: Optional[int] = None, function: Optional[str] = None
) -> Optional[SourceEvidence]:
    """Build source evidence and hash a readable file without requiring it."""

    if not path:
        return None
    digest = _cached_source_sha256(str(path))
    return SourceEvidence(path=str(path), line=line, function=function, sha256=digest)


@lru_cache(maxsize=256)
def _cached_source_sha256(path: str) -> Optional[str]:
    """Hash each immutable launch source at most once per tracing process."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def invocation_semantics(
    *,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    positional_names: Optional[Sequence[str]] = None,
    meta_names: Iterable[str] = (),
    constexpr_names: Iterable[str] = (),
    python_grid: Any = None,
    source: Optional[SourceEvidence] = None,
) -> Tuple[LaunchSemantics, Tuple[str, ...]]:
    """Split invocation values into tensors, scalars, constexpr, and meta evidence."""

    names = list(positional_names or ())
    tensors: List[TensorEvidence] = []
    named_scalars: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    constexpr: Dict[str, Any] = {}
    warnings: List[str] = []
    meta_set = set(meta_names)
    constexpr_set = set(constexpr_names)

    for index, value in enumerate(args):
        name = names[index] if index < len(names) else f"arg{index}"
        if is_trace_unsafe_proxy(value):
            warnings.append(f"{name}:torch_tracing_proxy")
        if is_tensor_like(value):
            tensors.append(tensor_evidence(name, value))
            continue
        serialized = json_safe(value)
        if name in constexpr_set:
            constexpr[name] = serialized
        elif name in meta_set:
            meta[name] = serialized
        else:
            named_scalars[name] = serialized

    for name, value in kwargs.items():
        if is_trace_unsafe_proxy(value):
            warnings.append(f"{name}:torch_tracing_proxy")
        if is_tensor_like(value):
            tensors.append(tensor_evidence(str(name), value))
            continue
        serialized = json_safe(value)
        if name in constexpr_set:
            constexpr[str(name)] = serialized
        elif name in meta_set:
            meta[str(name)] = serialized
        else:
            named_scalars[str(name)] = serialized

    return (
        LaunchSemantics(
            source=source,
            tensors=tuple(tensors),
            named_scalars=named_scalars,
            constexpr=constexpr,
            meta=meta,
            python_grid=json_safe(python_grid),
        ),
        tuple(warnings),
    )
