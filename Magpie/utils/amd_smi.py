###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Shared read-only ``amd-smi`` JSON probes.

``rocm-smi`` is deprecated from ROCm 7.0 and is no longer built as part of the
standard ROCm 10.1 build, so the AMD probes in ``gpu.py`` and
``gpu_monitor.py`` read ``amd-smi``'s ``--json`` output instead of scraping
``rocm-smi``'s text tables. There is no ``rocm-smi`` fallback here: every
ROCm version Magpie supports ships ``amd-smi``.

``amd-smi``'s JSON has moved between releases: a metric may be a bare number
or a ``{"value", "unit"}`` object, a field may be the literal string ``"N/A"``
instead of simply missing, and whole keys appear and disappear between
majors (observed firsthand across tool versions 26.2.2, 26.5.0, and 27.1.0).
Both ``gpu.py`` and ``gpu_monitor.py`` need the same power/clock/
temperature/memory readings, so the parsing is written once here instead of
twice -- a second, slightly different parser in one of the two call sites is
exactly how the "N/A" handling would drift and crash on only one of them.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TIMEOUT_SEC = 10.0

#: MiB per unit amd-smi may label a memory size with. Its "MB" is mebibytes, same convention
#: rocm-smi used for byte counts.
_MIB_PER_UNIT: Dict[str, float] = {
    "B": 1.0 / 1024**2,
    "KB": 1.0 / 1024,
    "KIB": 1.0 / 1024,
    "MB": 1.0,
    "MIB": 1.0,
    "GB": 1024.0,
    "GIB": 1024.0,
    "TB": 1024.0**2,
    "TIB": 1024.0**2,
}


def run_json(args: Sequence[str], *, timeout_s: float = _TIMEOUT_SEC) -> Optional[Any]:
    """Run a read-only ``amd-smi`` subcommand with ``--json`` and parse its stdout.

    Args:
        args: Subcommand and flags, without ``amd-smi`` and without ``--json``.
        timeout_s: Seconds to wait before giving up on the probe.

    Returns:
        The parsed payload -- a list for ``list``/``process``, a dict with a
        ``gpu_data`` list for ``metric``/``static`` -- or ``None`` when
        amd-smi is absent, failed, timed out, or printed something that is
        not its JSON.
    """
    cmd = ["amd-smi", *args, "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        logger.debug("amd-smi not found on PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.debug("amd-smi %s timed out", " ".join(args))
        return None
    except Exception as e:
        logger.debug("amd-smi %s could not run: %s", " ".join(args), e)
        return None
    if result.returncode != 0:
        logger.debug(
            "amd-smi %s exited %d: %s",
            " ".join(args),
            result.returncode,
            (result.stderr or "").strip()[:200],
        )
        return None
    try:
        return json.loads(result.stdout)
    except ValueError:
        # Also the no-driver case: amd-smi logs "Unable to detect any GPU devices" to stderr
        # and still exits 0, so an absent GPU arrives here as empty stdout rather than a raise.
        logger.debug("amd-smi %s printed no parseable JSON", " ".join(args))
        return None


def gpu_rows(payload: Any) -> List[Dict[str, Any]]:
    """Return the per-GPU dicts in an amd-smi payload, whether it wraps them in ``gpu_data`` or is a bare list."""
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict)]


def scalar(block: Any) -> Optional[float]:
    """Read a metric that may be a bare number, a ``{"value", "unit"}`` object, or absent/``"N/A"``.

    Args:
        block: The raw field, as amd-smi renders it: a number, a unit-wrapped object, or the
            string ``"N/A"`` when the part does not expose it.

    Returns:
        The numeric value, or ``None`` when it is absent or not reported.
    """
    if isinstance(block, dict):
        block = block.get("value")
    if isinstance(block, bool) or not isinstance(block, (int, float)):
        return None
    return float(block)


def lookup(mapping: Any, *names: str) -> Any:
    """Return the first of ``names`` present in ``mapping`` with a reportable value.

    amd-smi renames and drops keys between releases, so a caller names every spelling it will
    accept (hottest/most-preferred first) rather than one.

    Args:
        mapping: The block to read, which need not be a dict.
        *names: Candidate keys, most-preferred first.

    Returns:
        The first value found that is neither missing nor the ``"N/A"`` amd-smi prints for an
        unsupported field; ``None`` when none of them qualifies.
    """
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        if value is None or (
            isinstance(value, str) and value.strip().upper() in ("", "N/A")
        ):
            continue
        return value
    return None


def mib(block: Any) -> Optional[float]:
    """Convert an amd-smi memory field to MiB, honouring whichever unit it labels itself with."""
    value = scalar(block)
    if value is None:
        return None
    unit = (
        str(block.get("unit") or "").strip().upper()
        if isinstance(block, dict)
        else "MB"
    )
    scale = _MIB_PER_UNIT.get(unit)
    if scale is None:
        return None
    return value * scale


def clock_mhz(clock: Any, *domain_names: str) -> Optional[float]:
    """Read a clock frequency in MHz from a ``metric --clock`` block.

    Per-domain entries (``gfx_0``, ``mem_0``) nest the reading under a ``clk`` sub-key and read
    ``"N/A"`` on APUs; the APU aggregate entries (``apu_average_gfxclk_frequency``,
    ``apu_average_uclk_frequency``) are themselves a plain scalar block. Both shapes are tried,
    most-preferred first.

    Args:
        clock: The ``"clock"`` block from one ``metric --clock`` GPU row.
        *domain_names: Candidate keys, most-preferred first.

    Returns:
        The frequency in MHz, or ``None`` when none of ``domain_names`` reported one.
    """
    for name in domain_names:
        block = clock.get(name) if isinstance(clock, dict) else None
        if block is None:
            continue
        value = (
            scalar(block.get("clk"))
            if isinstance(block, dict) and "clk" in block
            else scalar(block)
        )
        if value is not None:
            return value
    return None


def gpu_count() -> int:
    """Return how many GPUs ``amd-smi`` enumerates, or 0 when it cannot be asked."""
    payload = run_json(["list"], timeout_s=5.0)
    if payload is None:
        return 0
    return len(gpu_rows(payload))


__all__ = [
    "clock_mhz",
    "gpu_count",
    "gpu_rows",
    "lookup",
    "mib",
    "run_json",
    "scalar",
]
