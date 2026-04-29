"""
Simple Triton vector-add kernel for Magpie's 0-overhead Latency harness.

Exposes two BLOCK_SIZE variants of the *same* kernel so the
``compare_triton_blocksize.yaml`` example can demonstrate why
``primary_metric: kernel_median_ms`` matters for kernel-config autotuning
(both variants have identical dispatch overhead; only the kernel duration
differs).

Module contract for ``Magpie/bench/_runner.py``:
  - One or more ``callable`` symbols (here: ``triton_vector_add_block256``,
    ``triton_vector_add_block1024``).
  - ``get_inputs() -> (args, kwargs)`` — the runner calls ``callable(*args, **kwargs)``.

You can also run this module directly to sanity-check the kernel and print
a ``MAGPIE_LATENCY_JSON: {...}`` line — that doubles as both:
  * the user-harness escape hatch (``method: cuda_graph`` without
    ``bench_target``),
  * a quick local benchmark for development.

Usage:
  pip install triton torch
  python -m Magpie analyze -k examples/simple_triton_test/analyze_triton_latency.yaml
"""

from __future__ import annotations

import json
import sys
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise ImportError(
        "This example requires Triton. Install with: pip install triton"
    ) from e


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.jit
def _vector_add_kernel(
    a_ptr, b_ptr, c_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def _launch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, block_size: int) -> None:
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    _vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=block_size)


# ---------------------------------------------------------------------------
# Magpie bench_target callables
# ---------------------------------------------------------------------------


def triton_vector_add_block256(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> None:
    _launch(a, b, c, block_size=256)


def triton_vector_add_block1024(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> None:
    _launch(a, b, c, block_size=1024)


# Default callable used by ``analyze_triton_latency.yaml``.
def triton_vector_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> None:
    _launch(a, b, c, block_size=1024)


# ---------------------------------------------------------------------------
# Inputs factory (called once by the runner *after* torch.manual_seed(seed))
# ---------------------------------------------------------------------------


N = 1 << 20  # 1M elements — small enough that dispatch overhead matters,
             # which is exactly the scenario where kernel_median_ms beats
             # wall_median_ms for autotuning rankings.


def get_inputs() -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
    """Magpie runner contract: returns (args, kwargs)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA / HIP is required for the Triton example")
    device = "cuda"
    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    c = torch.empty_like(a)
    return (a, b, c), {}


# ---------------------------------------------------------------------------
# CLI: --check (correctness, used by testcase_command)
#      --bench (user-harness latency mode, prints MAGPIE_LATENCY_JSON line)
# ---------------------------------------------------------------------------


def _check() -> int:
    (a, b, c), _ = get_inputs()
    triton_vector_add(a, b, c)
    torch.cuda.synchronize()
    expected = a + b
    if torch.allclose(c, expected, atol=1e-5, rtol=1e-5):
        print(f"PASSED: vector_add OK on {N} elements")
        return 0
    print("FAILED: vector_add mismatch")
    return 1


def _bench() -> int:
    """User-harness escape hatch — emits the MAGPIE_LATENCY_JSON marker."""
    from Magpie.bench import MAGPIE_LATENCY_JSON_MARKER, do_bench_cudagraph

    (a, b, c), _ = get_inputs()
    stats = do_bench_cudagraph(
        lambda: triton_vector_add(a, b, c),
        rep=20,
        n_retries=5,
        estimate_reps=5,
    )
    print(
        f"{MAGPIE_LATENCY_JSON_MARKER} "
        + json.dumps({"mode": "cuda_graph", "stats": stats.to_dict()})
    )
    return 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        sys.exit(_bench())
    if "--check" in sys.argv:
        sys.exit(_check())
    # Default: behave like a smoke test.
    sys.exit(_check())
