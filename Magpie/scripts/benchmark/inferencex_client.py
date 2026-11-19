"""Translate Magpie's single-node client arguments to the packaged InferenceX CLI."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def fixed_argv(
    arguments: list[str], env: dict[str, str]
) -> tuple[list[str], int | None]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", default=env.get("PORT", "8888"))
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--input-len", required=True)
    parser.add_argument("--output-len", required=True)
    parser.add_argument("--random-range-ratio", required=True)
    parser.add_argument("--num-prompts", required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--server-pid", type=int)
    args, extra = parser.parse_known_args(arguments)
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    base_url = env.get("BENCHMARK_BASE_URL") or f"http://127.0.0.1:{args.port}"
    result = args.result_filename
    if not result.endswith(".json"):
        result += ".json"
    argv = [
        sys.executable,
        "-m",
        "infx.bench_serving.benchmark_serving",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--random-input-len",
        args.input_len,
        "--random-output-len",
        args.output_len,
        "--random-range-ratio",
        args.random_range_ratio,
        "--num-prompts",
        args.num_prompts,
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups",
        str(2 * args.max_concurrency),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        args.result_dir,
        "--result-filename",
        result,
    ]
    if env.get("PROFILE") == "1":
        argv.append("--profile")
    return [*argv, *extra], args.server_pid


def run(argv: list[str], server_pid: int | None) -> int:
    """Relay cancellation, and stop only the client when its owned server exits."""
    child = subprocess.Popen(argv)
    handlers = {}

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, forward)
        while child.poll() is None:
            if server_pid is not None:
                try:
                    os.kill(server_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError("Inference server exited during benchmark")
            time.sleep(0.1)
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> int:
    argv, server_pid = fixed_argv(sys.argv[1:], dict(os.environ))
    try:
        return run(argv, server_pid)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
