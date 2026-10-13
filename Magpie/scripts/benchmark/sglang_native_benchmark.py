"""Opt-in SGLang native benchmark protocol and literal server environment."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
COMPILE_MARKER = re.compile(
    r"\b(?:compiling|building)\s+(?:the\s+)?(?:triton|aiter|tilelang|torchinductor|module_)"
    r"|(?:\[(?:aiter|tilelang)\]|triton|torchinductor).*?\b(?:compiling|building)\b"
    r"|\bJIT compiling\b"
    r"|\b(?:start|finish|failed)\s+(?:jit\s+)?build\s+\[module_"
    r"|\[aiter\]\s+build\s+\[module_.*?\bfailed\b",
    re.IGNORECASE,
)


def server_environment(inherited: dict[str, str]) -> dict[str, str]:
    """Apply overrides after the shell's hardware defaults, without shell eval."""
    result = dict(inherited)
    for name in inherited.get("SGLANG_UNSET_ENVS", "").split():
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid variable name in SGLANG_UNSET_ENVS: {name}")
        result.pop(name, None)
    overrides = json.loads(inherited.get("SGLANG_SERVER_ENVS_JSON") or "{}")
    if not isinstance(overrides, dict):
        raise TypeError("SGLANG_SERVER_ENVS_JSON must be an object")
    for name, value in overrides.items():
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid server environment name: {name}")
        if value is None:
            result.pop(name, None)
        elif isinstance(value, str) and "\0" not in value:
            result[name] = value
        else:
            raise ValueError(f"Server environment {name} must be a string or null")
    return result


def integer(name: str, default: int | None = None, minimum: int = 1) -> int:
    value = os.environ.get(name, str(default) if default is not None else "")
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def run_client(command: list[str], log: Path) -> int:
    """Keep each invocation's output while retaining the normal console log."""
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise


def load_result(path: Path, expected: int, isl: int, osl: int) -> dict:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected one native result record in {path}")
    result = records[0]
    if result.get("failed") or result.get("error"):
        raise ValueError("Native benchmark reported failed requests or an error")
    if result.get("completed") != expected:
        raise ValueError(f"Completed {result.get('completed')} of {expected} requests")
    errors = result.get("errors")
    if not isinstance(errors, list) or len(errors) != expected or any(errors):
        raise ValueError("Native request errors are missing, incomplete, or nonempty")
    for phase, length in (("input", isl), ("output", osl)):
        lengths = result.get(f"{phase}_lens")
        total = result.get(f"total_{phase}_tokens", result.get(f"total_{phase}"))
        if (
            not isinstance(lengths, list)
            or len(lengths) != expected
            or any(observed != length for observed in lengths)
            or total != expected * length
        ):
            raise ValueError(
                f"Native {phase} tokens do not match the fixed-length workload"
            )
    for name in ("duration", "output_throughput", "total_throughput"):
        value = result.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid native metric: {name}")
    return result


def scan_compilation(path: Path, offset: int = 0) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(offset)
        lines = (
            ANSI_ESCAPE.sub("", line.decode("utf-8", errors="replace")).strip()
            for line in handle
        )
        return [line for line in lines if COMPILE_MARKER.search(line)]


def client() -> int:
    if os.environ.get("PROFILE") == "1":
        raise ValueError(
            "sglang-native is for unprofiled benchmarks; PROFILE must not be 1"
        )
    if float(os.environ["RANDOM_RANGE_RATIO"]) != 1.0:
        raise ValueError(
            "sglang-native requires RANDOM_RANGE_RATIO=1 for fixed lengths"
        )
    conc = integer("CONC")
    isl, osl = integer("ISL"), integer("OSL")
    prompts = integer("NUM_PROMPTS", conc * 10)
    warmup = max(
        integer("SGLANG_NATIVE_FULL_WARMUP_MINIMUM", 0, 0),
        integer("SGLANG_NATIVE_FULL_WARMUP_MULTIPLIER", 2, 0) * conc,
    )
    if warmup == 0:
        raise ValueError(
            "sglang-native requires at least one full-output warmup request"
        )
    seed = integer("SGLANG_BENCHMARK_SEED", 42, 0)
    temperature = float(os.environ.get("SGLANG_BENCHMARK_TEMPERATURE", "0"))
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("SGLANG_BENCHMARK_TEMPERATURE must be finite and non-negative")
    model = os.environ["MODEL"]
    filename = os.environ["RESULT_FILENAME"]
    if not filename or Path(filename).name != filename or filename in (".", ".."):
        raise ValueError("RESULT_FILENAME must be a basename")
    stem = filename.removesuffix(".json")
    result_dir = Path(os.environ.get("RESULT_DIR", "/workspace"))
    result_dir.mkdir(parents=True, exist_ok=True)
    destination = result_dir / f"{stem}.json"
    evidence = result_dir / f"{stem}.native"
    # A retry must use a fresh result identity; never mix appended native records.
    if destination.exists() or evidence.exists():
        raise ValueError(f"Native benchmark results already exist: {stem}")
    evidence.mkdir()
    base_url = os.environ.get("BENCHMARK_BASE_URL") or (
        "http://127.0.0.1:" + os.environ.get("PORT", "8888")
    )
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--model",
        model,
        "--tokenizer",
        os.environ.get("TOKENIZER") or model,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--warmup-requests",
        str(conc),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--output-details",
        "--disable-tqdm",
    ]
    manifest = {
        "client": "sglang-native",
        "base_command": command,
        "full_output_warmup_requests": warmup,
        "measurement_requests": prompts,
        "internal_warmup_requests_per_invocation": conc,
        "status": "running",
        "started_at": time.time(),
    }
    manifest_file = evidence / "protocol.json"
    server_log_name = os.environ.get("SERVER_LOG")
    server_log = Path(server_log_name) if server_log_name else None
    try:
        for phase, count in (("warmup", warmup), ("measurement", prompts)):
            raw = evidence / f"{phase}.jsonl"
            log = evidence / f"{phase}.log"
            invocation = [
                *command,
                "--num-prompts",
                str(count),
                "--output-file",
                str(raw),
            ]
            offset = (
                server_log.stat().st_size
                if server_log and server_log.is_file()
                else None
            )
            if phase == "measurement":
                manifest["measurement_started_at"] = time.time()
                manifest["server_log"] = str(server_log) if server_log else None
                manifest["server_log_start_offset"] = offset
            print(f"[sglang-native] {phase}: {count} full-output requests", flush=True)
            code = run_client(invocation, log)
            if code:
                manifest.update(
                    status="failed",
                    failed_phase=phase,
                    exit_code=code,
                    finished_at=time.time(),
                )
                return code if code > 0 else 128 - code
            data = load_result(raw, count, isl, osl)
            if phase == "measurement":
                markers = scan_compilation(log)
                if offset is not None:
                    if not server_log.is_file() or server_log.stat().st_size < offset:
                        raise ValueError(
                            "Server log disappeared or rotated during measurement"
                        )
                    markers.extend(scan_compilation(server_log, offset))
                manifest["compile_markers"] = markers
                manifest["compile_check_scope"] = (
                    "Explicit compiler messages during the measurement client invocation, "
                    "including its internal warmup; this is not proof of all cache hits."
                )
                if markers:
                    raise ValueError(
                        "Compilation observed during measurement; raw results retained"
                    )
        data["total_token_throughput"] = data["total_throughput"]
        data.setdefault("total_input_tokens", data.get("total_input", 0))
        data.setdefault("total_output_tokens", data.get("total_output", 0))
        for statistic in ("mean", "median", "p99", "std"):
            data[f"{statistic}_e2el_ms"] = data.get(f"{statistic}_e2e_latency_ms", 0.0)
        manifest.update(status="completed", finished_at=time.time())
        data["native_protocol"] = manifest
        destination.write_text(json.dumps(data) + "\n")
        return 0
    except Exception as error:
        manifest.update(status="failed", error=str(error), finished_at=time.time())
        raise
    finally:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    try:
        if sys.argv[1:] and sys.argv[1] == "server" and len(sys.argv) > 2:
            os.execvpe(sys.argv[2], sys.argv[2:], server_environment(dict(os.environ)))
        if sys.argv[1:] == ["client"]:
            return client()
        raise ValueError(
            "Usage: sglang_native_benchmark.py client | server COMMAND ..."
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
