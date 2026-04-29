###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Subprocess harness for Magpie's import-based latency benchmarking.

Spawned by ``Magpie/eval/latency.py``; never imported by the rest of Magpie.

Two modes (selected by ``--profile``):

1. (default) **CUDA-graph timing.** Runs ``magpie.bench.do_bench_cudagraph``
   on the user's callable and prints one line:

       MAGPIE_LATENCY_JSON: {"stats": {...}, "n_repeat": ..., ...}

2. **--profile** (kernel-trace harness mode). Runs a tight
   ``for _ in range(N): fn(); torch.cuda.synchronize()`` loop sized to
   roughly ``rep_ms`` so the outer ``rocprofv3 --kernel-trace`` invocation
   captures clean per-dispatch HW kernel durations. Still prints the
   marker line, but ``stats`` is ``null`` (the wrapper parses the rocprofv3
   CSV instead).

Inputs are read from environment variables (kept on env, not argv, so the
runner is invoked uniformly under both ``python _runner.py`` and
``rocprofv3 ... -- python _runner.py --profile``):

  - MAGPIE_BENCH_MODULE:        importable module path (required)
  - MAGPIE_BENCH_CALLABLE:      attribute of the callable (required)
  - MAGPIE_BENCH_INPUTS_FUNC:   attribute of the inputs factory
                                (default: "get_inputs")
  - MAGPIE_BENCH_REP_MS:        target measurement window in ms (default: 20)
  - MAGPIE_BENCH_N_RETRIES:     retries (default: 5)
  - MAGPIE_BENCH_ESTIMATE_REPS: estimate-graph reps (default: 5)
  - MAGPIE_BENCH_WARMUP_ITERS:  eager warmup iters before timing (default: 5)
  - MAGPIE_BENCH_SEED:          torch.manual_seed value (default: 42)
  - MAGPIE_BENCH_PROFILE_REP_MS: only used with --profile (default: rep_ms*5)

Reproducibility note: the seed is set BEFORE inputs are materialized so
tensor shapes/values are stable across runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Tuple


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


def _seed_everything(seed: int) -> None:
    """Make tensor shapes/values reproducible BEFORE inputs are materialized."""
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        import numpy as np  # noqa: F401

        np.random.seed(seed)
    except Exception:
        pass


def _resolve_target(
    module_name: str, callable_name: str, inputs_func_name: str
) -> Tuple[Callable[..., Any], Callable[[], Any]]:
    """Import the user's module and look up the callable + inputs factory."""
    module = importlib.import_module(module_name)
    if not hasattr(module, callable_name):
        raise AttributeError(
            f"module {module_name!r} has no attribute {callable_name!r}"
        )
    if not hasattr(module, inputs_func_name):
        raise AttributeError(
            f"module {module_name!r} has no attribute {inputs_func_name!r} "
            f"(expected an inputs factory)"
        )
    fn = getattr(module, callable_name)
    inputs_factory = getattr(module, inputs_func_name)
    if not callable(fn):
        raise TypeError(f"{module_name}.{callable_name} is not callable")
    if not callable(inputs_factory):
        raise TypeError(f"{module_name}.{inputs_func_name} is not callable")
    return fn, inputs_factory


def _normalize_inputs(raw: Any) -> Tuple[tuple, dict]:
    """Coerce the inputs factory return value into ``(args, kwargs)``."""
    if raw is None:
        return tuple(), {}
    # Accept (args, kwargs)
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        first = raw[0]
        if isinstance(first, (list, tuple)):
            return tuple(first), dict(raw[1])
    # Accept positional iterable
    if isinstance(raw, (list, tuple)):
        return tuple(raw), {}
    # Accept dict-as-kwargs
    if isinstance(raw, dict):
        return tuple(), dict(raw)
    # Single positional value
    return (raw,), {}


def _emit(payload: dict) -> None:
    """Print the canonical marker line."""
    sys.stdout.write(
        "MAGPIE_LATENCY_JSON: " + json.dumps(payload, default=str) + "\n"
    )
    sys.stdout.flush()


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        prog="magpie.bench._runner",
        description="Magpie 0-overhead latency benchmark runner",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run in kernel-trace harness mode (tight loop, no graph capture). "
        "Intended to be wrapped by rocprofv3 --kernel-trace.",
    )
    parser.add_argument(
        "--module", type=str, default=None,
        help="Override MAGPIE_BENCH_MODULE",
    )
    parser.add_argument(
        "--callable", type=str, default=None,
        help="Override MAGPIE_BENCH_CALLABLE",
    )
    parser.add_argument(
        "--get-inputs", type=str, default=None,
        help="Override MAGPIE_BENCH_INPUTS_FUNC",
    )
    args = parser.parse_args(argv)

    module_name = args.module or _env_str("MAGPIE_BENCH_MODULE", "")
    callable_name = args.callable or _env_str("MAGPIE_BENCH_CALLABLE", "")
    inputs_func_name = (
        args.get_inputs or _env_str("MAGPIE_BENCH_INPUTS_FUNC", "get_inputs")
    )

    if not module_name or not callable_name:
        _emit({
            "stats": None,
            "error": "MAGPIE_BENCH_MODULE and MAGPIE_BENCH_CALLABLE are required",
        })
        return 2

    rep_ms = _env_int("MAGPIE_BENCH_REP_MS", 20)
    n_retries = _env_int("MAGPIE_BENCH_N_RETRIES", 5)
    estimate_reps = _env_int("MAGPIE_BENCH_ESTIMATE_REPS", 5)
    warmup_iters = _env_int("MAGPIE_BENCH_WARMUP_ITERS", 5)
    seed = _env_int("MAGPIE_BENCH_SEED", 42)
    profile_rep_ms = _env_int("MAGPIE_BENCH_PROFILE_REP_MS", max(rep_ms * 5, 50))

    try:
        _seed_everything(seed)
        fn_raw, inputs_factory = _resolve_target(
            module_name, callable_name, inputs_func_name
        )
        raw_inputs = inputs_factory()
        args_tuple, kwargs = _normalize_inputs(raw_inputs)

        def call_fn() -> None:
            fn_raw(*args_tuple, **kwargs)

        for _ in range(max(0, warmup_iters)):
            call_fn()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        if args.profile:
            # Kernel-trace harness mode: tight loop, no graph capture.
            # Pre-size N from a quick wall-clock estimate so the loop runs
            # roughly profile_rep_ms milliseconds without CUDA event overhead.
            est_calls = max(estimate_reps, 1)
            t0 = time.perf_counter()
            for _ in range(est_calls):
                call_fn()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            t1 = time.perf_counter()
            per_call_ms = ((t1 - t0) * 1000.0) / est_calls
            if per_call_ms <= 0:
                n_iter = 1000
            else:
                n_iter = max(1, int(profile_rep_ms / per_call_ms))

            # Tight, dispatch-only loop. No CUDA events. The outer rocprofv3
            # --kernel-trace is what produces timing.
            t_start = time.perf_counter()
            for _ in range(n_iter):
                call_fn()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            t_end = time.perf_counter()

            _emit(
                {
                    "mode": "profile",
                    "stats": None,
                    "n_iter": n_iter,
                    "per_call_estimate_ms": per_call_ms,
                    "wall_loop_ms": (t_end - t_start) * 1000.0,
                    "module": module_name,
                    "callable": callable_name,
                    "seed": seed,
                }
            )
            return 0

        # Default mode: CUDA-graph based wall-clock timing.
        from Magpie.bench import do_bench_cudagraph  # type: ignore

        stats = do_bench_cudagraph(
            call_fn,
            rep=rep_ms,
            n_retries=n_retries,
            estimate_reps=estimate_reps,
        )

        _emit(
            {
                "mode": "cuda_graph",
                "stats": stats.to_dict(),
                "module": module_name,
                "callable": callable_name,
                "seed": seed,
            }
        )
        return 0

    except SystemExit:
        raise
    except BaseException as e:
        _emit(
            {
                "stats": None,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "module": module_name,
                "callable": callable_name,
            }
        )
        return 1


if __name__ == "__main__":
    # Allow ``python -m Magpie.bench._runner`` and ``python _runner.py``.
    # The fallback handles being launched directly (no parent package) by
    # adding the repo root to sys.path so ``import Magpie.bench`` works.
    try:
        from Magpie.bench import do_bench_cudagraph  # noqa: F401
    except ImportError:
        # When invoked as ``python /path/to/Magpie/bench/_runner.py``, the
        # parent ``Magpie`` package may not be on sys.path. Walk up two dirs
        # and prepend so the import succeeds.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    sys.exit(main(sys.argv[1:]))
