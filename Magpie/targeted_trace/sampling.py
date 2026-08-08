"""Deterministic sampling keyed only by run seed and stable event key."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .schema import canonical_json


def sampling_value(run_seed: str, stable_event_key: str) -> float:
    """Map ``{run_seed, stable_event_key}`` to a reproducible value in [0, 1).

    Python's process-randomized ``hash()`` and mutable PRNG state are deliberately
    excluded so the same semantic event makes the same decision across processes
    and reruns.
    """

    digest = hashlib.sha256(
        f"{run_seed}\x00{stable_event_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def should_sample(run_seed: str, stable_event_key: str, sample_rate: float) -> bool:
    """Return the deterministic sampling decision for one event."""

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be between 0 and 1, got {sample_rate}")
    if sample_rate == 0.0:
        return False
    if sample_rate == 1.0:
        return True
    return sampling_value(run_seed, stable_event_key) < sample_rate


def stable_key(parts: Mapping[str, Any], *, occurrence: int = 0) -> str:
    """Build a stable event key from canonical semantic parts and occurrence."""

    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    payload = {"parts": dict(parts), "occurrence": occurrence}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
