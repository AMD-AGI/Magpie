# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect bounded AgentX server traces after AIPerf enters its measured phase.

The bounded-memory trace reader is adapted from Hyperloom's MIT-licensed
``inference_optimizer/assets/agentx/aiperf_phase_gate.py``. This module has no
Magpie imports so the standalone container worker can use it directly.
"""

from __future__ import annotations

import gzip
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

_POLL_SECONDS = 0.2
_HTTP_SECONDS = 2.0
_STARTUP_TRACE_DIRS = {"graph_capture_profile", "capture_traces"}
_TRACE_RANK_PATTERNS = (
    re.compile(r"(?:^|[-_.])rank[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE),
    re.compile(r"^r(\d+)(?=[-.])", re.IGNORECASE),
)
_LOCAL_TP_RANK_PATTERN = re.compile(r"(?:^|[-_.])tp[-_](\d+)(?=[-_.]|$)", re.IGNORECASE)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


_TRACE_READ_CHARS = 64 * 1024
_MAX_JSON_VALUE_CHARS = 1024 * 1024
_MAX_JSON_DEPTH = 128
_JSON_DECODER = json.JSONDecoder(parse_constant=_reject_json_constant)
_JSON_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')


class _TraceValueTooLarge(ValueError):
    """Switch from whole-value decoding to field-wise streaming."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("trace check exceeded its remaining budget")


class _TraceJSONReader:
    """Read a complete trace document without retaining its event array."""

    def __init__(
        self, handle: Any, deadline: float, check_alive: Callable[[], None]
    ) -> None:
        self.handle = handle
        self.deadline = deadline
        self.check_alive = check_alive
        self.buffer = ""
        self.position = 0
        self.eof = False

    def refill(self) -> None:
        self.check_alive()
        _check_deadline(self.deadline)
        self.buffer = self.buffer[self.position :]
        self.position = 0
        if len(self.buffer) >= _MAX_JSON_VALUE_CHARS:
            raise _TraceValueTooLarge("trace JSON value requires field-wise streaming")
        chunk = self.handle.read(
            min(_TRACE_READ_CHARS, _MAX_JSON_VALUE_CHARS - len(self.buffer))
        )
        self.check_alive()
        _check_deadline(self.deadline)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self) -> str:
        self.check_alive()
        _check_deadline(self.deadline)
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position] in " \t\r\n"
            ):
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.refill()

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"expected {character!r} in trace JSON")
        self.position += 1

    def value(self) -> Any:
        if not self.peek():
            raise ValueError("incomplete trace JSON value")
        while True:
            self.check_alive()
            _check_deadline(self.deadline)
            try:
                value, end = _JSON_DECODER.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.refill()
                continue
            # A number can end at a chunk boundary before its exponent arrives.
            if end == len(self.buffer) and not self.eof:
                self.refill()
                continue
            if end < len(self.buffer) and self.buffer[end] not in " \t\r\n,]}":
                if (
                    type(value) in (int, float)
                    and self.buffer[end] in ".eE"
                    and not self.eof
                ):
                    self.refill()
                    continue
                raise ValueError("invalid trace JSON value delimiter")
            self.position = end
            return value

    def string(self) -> str | None:
        """Validate strings of any length, retaining only short metadata values."""
        self.expect('"')
        parts = []
        length = 0
        while True:
            self.check_alive()
            _check_deadline(self.deadline)
            match = _JSON_STRING_SPECIAL.search(self.buffer, self.position)
            end = match.start() if match else len(self.buffer)
            if length <= 64:
                length += end - self.position
                if length <= 64:
                    parts.append(self.buffer[self.position : end])
                else:
                    parts.clear()
            self.position = end
            if match is None:
                if self.eof:
                    raise ValueError("incomplete trace JSON string")
                self.refill()
                continue
            character = self.buffer[self.position]
            self.position += 1
            if character == '"':
                return "".join(parts) if length <= 64 else None
            if character != "\\":
                raise ValueError("unescaped control character in trace JSON string")
            while len(self.buffer) - self.position < 1 and not self.eof:
                self.refill()
            if self.position == len(self.buffer):
                raise ValueError("incomplete trace JSON escape")
            escape_length = 5 if self.buffer[self.position] == "u" else 1
            while len(self.buffer) - self.position < escape_length and not self.eof:
                self.refill()
            escaped = self.buffer[self.position : self.position + escape_length]
            decoded = _JSON_DECODER.decode('"\\' + escaped + '"')
            self.position += escape_length
            length += len(decoded)
            if length <= 64:
                parts.append(decoded)
            else:
                parts.clear()

    def event(self) -> bool:
        try:
            event = self.value()
        except _TraceValueTooLarge:
            if self.peek() != "{":
                self.skip()
                return False
            fields = {}
            for key in self.members():
                if key in {"cat", "ph"} and self.peek() == '"':
                    fields[key] = self.string()
                else:
                    if key in {"cat", "ph"}:
                        fields[key] = None
                    self.skip()
            return fields.get("cat") == "kernel" and fields.get("ph") == "X"
        return (
            isinstance(event, dict)
            and event.get("cat") == "kernel"
            and event.get("ph") == "X"
        )

    def members(self):
        self.expect("{")
        if self.peek() == "}":
            self.position += 1
            return
        while True:
            if self.peek() != '"':
                raise ValueError("trace JSON object key must be a string")
            key = self.string()
            self.expect(":")
            yield key
            if self.peek() == "}":
                self.position += 1
                return
            self.expect(",")

    def skip(self, depth: int = 0) -> None:
        if depth >= _MAX_JSON_DEPTH:
            raise ValueError("trace JSON nesting exceeds the parsing limit")
        character = self.peek()
        if character == "{":
            for _key in self.members():
                self.skip(depth + 1)
        elif character == "[":
            self.position += 1
            if self.peek() == "]":
                self.position += 1
                return
            while True:
                self.skip(depth + 1)
                if self.peek() == "]":
                    self.position += 1
                    return
                self.expect(",")
        elif character == '"':
            self.string()
        else:
            self.value()


def _trace_metadata(
    path: Path, *, deadline: float, check_alive: Callable[[], None]
) -> dict[str, Any]:
    rank = None
    for token in (path.name, path.parent.name):
        for pattern in _TRACE_RANK_PATTERNS:
            match = pattern.search(token)
            if match:
                rank = int(match.group(1))
                break
        if rank is not None:
            break
    global_filename_rank = rank
    if rank is None:
        # SGLang's TP component is local to a pipeline stage. Prefer the global
        # process rank in PyTorch's distributedInfo when that metadata exists.
        for token in (path.name, path.parent.name):
            match = _LOCAL_TP_RANK_PATTERN.search(token)
            if match:
                rank = int(match.group(1))
                break
    opener = gzip.open if path.suffix == ".gz" else open
    has_kernel = False
    seen = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = _TraceJSONReader(handle, deadline, check_alive)
        for key in reader.members():
            if key in {"traceEvents", "distributedInfo"}:
                if key in seen:
                    raise ValueError("duplicate trace metadata field")
                seen.add(key)
            if key == "traceEvents":
                reader.expect("[")
                if reader.peek() == "]":
                    reader.position += 1
                    continue
                while True:
                    if reader.event():
                        has_kernel = True
                    if reader.peek() == "]":
                        reader.position += 1
                        break
                    reader.expect(",")
            elif key == "distributedInfo" and reader.peek() == "{":
                rank_seen = False
                for field in reader.members():
                    if field == "rank":
                        header_rank = reader.value()
                        if (
                            rank_seen
                            or type(header_rank) is not int
                            or (
                                global_filename_rank is not None
                                and global_filename_rank != header_rank
                            )
                        ):
                            raise ValueError("conflicting trace rank")
                        rank = header_rank
                        rank_seen = True
                    else:
                        reader.skip()
            else:
                reader.skip()
        if reader.peek():
            raise ValueError("trailing content after trace JSON document")
    return {"rank": rank, "has_kernel": has_kernel and "traceEvents" in seen}


def _trace_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.trace.json*")
        if path.is_file()
        and not path.is_symlink()
        and path.name.endswith((".trace.json", ".trace.json.gz"))
        and not path.name.startswith(("graph_capture_", "merged-"))
        and not (_STARTUP_TRACE_DIRS | {"trace_split"}).intersection(
            path.relative_to(directory).parts
        )
    )


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino


def _complete_traces(
    directory: Path,
    expected_ranks: int,
    deadline: float,
    check_alive: Callable[[], None],
    cache: dict[Path, tuple[tuple[int, ...], dict[str, Any] | None]],
) -> list[str] | None:
    files = _trace_files(directory)
    ranks: set[int] = set()
    valid = []
    for path in files:
        check_alive()
        _check_deadline(deadline)
        signature = _signature(path)
        cached = cache.get(path)
        if cached is not None and cached[0] == signature:
            metadata = cached[1]
        else:
            try:
                metadata = _trace_metadata(
                    path, deadline=deadline, check_alive=check_alive
                )
            except (EOFError, ValueError, UnicodeError, RecursionError, zlib.error):
                metadata = None
            except OSError as exc:
                if isinstance(exc, TimeoutError):
                    raise
                metadata = None
            cache[path] = signature, metadata
        if _signature(path) != signature or metadata is None:
            return None
        if not metadata["has_kernel"]:
            continue
        rank = metadata["rank"]
        if rank is None and expected_ranks == 1:
            rank = 0
        if type(rank) is not int or rank not in range(expected_ranks):
            return None
        # A worker may export several stage/schedule traces. They add evidence
        # for that rank, but cannot substitute for another missing worker.
        ranks.add(rank)
        valid.append(str(path))
    if ranks == set(range(expected_ranks)) and files == _trace_files(directory):
        return valid
    return None


def _http(url: str, *, timeout: float, body: dict[str, Any] | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("AgentX profiler control response exceeds 1 MiB")
    return payload


def _wait_phase(
    progress_url: str,
    deadline: float,
    check_alive: Callable[[], None],
    client_started_ns: int | None,
) -> int:
    last_error = ""
    while True:
        check_alive()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"AgentX profiling phase did not start: {last_error}")
        try:
            payload = json.loads(
                _http(progress_url, timeout=min(_HTTP_SECONDS, remaining)),
                parse_constant=_reject_json_constant,
            )
            phases = payload.get("phases") if isinstance(payload, dict) else None
            stats = phases.get("profiling") if isinstance(phases, dict) else None
            if not isinstance(stats, dict):
                raise ValueError("AIPerf progress has no profiling phase")  # noqa: TRY004
            start = stats.get("start_ns")
            if (
                type(start) is int
                and start > 0
                and client_started_ns is not None
                and start < client_started_ns
            ):
                raise ValueError("AIPerf progress belongs to an earlier client")
            if stats.get("was_cancelled") is True:
                raise RuntimeError("AIPerf profiling phase was cancelled")
            if type(start) is int and start > 0:
                if stats.get("requests_end_ns") is not None:
                    raise RuntimeError(
                        "AIPerf profiling phase ended before capture started"
                    )
                return start
            last_error = "AIPerf is still preparing or warming up"
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last_error = str(exc)
        time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"AgentX {name} must be a positive finite number")  # noqa: TRY004
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"AgentX {name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"AgentX {name} must be a positive finite number")
    return result


def _save_capture(directory: Path, capture: dict[str, Any]) -> None:
    capture["recorded_at_ns"] = time.time_ns()
    temporary = directory / ".capture.json.tmp"
    temporary.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    temporary.replace(directory / "capture.json")


def _check_fresh_trace_directory(directory: Path) -> None:
    for path in directory.rglob("*"):
        if not path.is_symlink():
            if path.is_dir():
                continue
            if _STARTUP_TRACE_DIRS.intersection(path.relative_to(directory).parts[:-1]):
                continue
        raise ValueError(
            "AgentX profile capture requires a fresh unique trace directory; "
            "only empty directories and startup trace directories may exist"
        )


def capture_profile(
    *,
    framework: str,
    server_url: str,
    progress_url: str,
    trace_dir: Path,
    settings: dict[str, Any],
    expected_ranks: int,
    phase_timeout_seconds: float,
    check_alive: Callable[[], None],
    client_started_ns: int | None = None,
) -> dict[str, Any]:
    """Wait for the measured replay, trigger bounded profiling and verify its files.

    The caller configures vLLM's worker step limit before launching the server.
    ``check_alive`` must raise on cancellation, failed child processes or the
    overall deadline; a successful client exit must still permit trace flushing.
    The trace directory must be unique to this capture. Empty directories and
    known server-startup graph trace directories may already exist in it.
    """
    if framework not in {"sglang", "vllm"}:
        raise ValueError("AgentX profiling supports SGLang and vLLM")
    if type(expected_ranks) is not int or expected_ranks <= 0:
        raise ValueError("AgentX expected_ranks must be a positive integer")
    steps = settings.get("num_steps", 20)
    if type(steps) is not int or steps <= 0:
        raise ValueError("AgentX num_steps must be a positive integer")
    phase_timeout = _positive_seconds(phase_timeout_seconds, "phase_timeout_seconds")
    capture_timeout = _positive_seconds(
        settings.get("capture_timeout_seconds", 300), "capture_timeout_seconds"
    )
    flush_timeout = _positive_seconds(
        settings.get("flush_timeout_seconds", 1800), "flush_timeout_seconds"
    )
    if client_started_ns is not None and (
        type(client_started_ns) is not int or client_started_ns <= 0
    ):
        raise ValueError("AgentX client_started_ns must be a positive integer")
    trace_dir = Path(trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    _check_fresh_trace_directory(trace_dir)
    capture: dict[str, Any] = {
        "version": 1,
        "capture_id": trace_dir.name,
        "status": "failed",
        "framework": framework,
        "num_steps": steps,
        "expected_ranks": expected_ranks,
        "phase_start_ns": None,
        "trace_files": [],
    }
    attempted = False
    failure: BaseException | None = None
    try:
        capture["phase_start_ns"] = _wait_phase(
            progress_url,
            time.monotonic() + phase_timeout,
            check_alive,
            client_started_ns,
        )
        check_alive()
        body = {}
        if framework == "sglang":
            body = {
                "num_steps": steps,
                "output_dir": str(trace_dir),
                "activities": ["CPU", "GPU"],
                "profile_prefix": trace_dir.name,
            }
        deadline = time.monotonic() + capture_timeout
        capture["capture_started_ns"] = time.time_ns()
        attempted = True
        _http(
            server_url.rstrip("/") + "/start_profile",
            timeout=capture_timeout,
            body=body,
        )
        check_alive()
        if time.monotonic() >= deadline:
            raise TimeoutError("AgentX profiling start exceeded capture timeout")
        while not _trace_files(trace_dir):
            check_alive()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "AgentX bounded profiling produced no trace before capture timeout"
                )
            time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
        capture["flush_started_ns"] = time.time_ns()
        deadline = time.monotonic() + flush_timeout
        cache: dict[Path, tuple[tuple[int, ...], dict[str, Any] | None]] = {}
        while True:
            check_alive()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "AgentX trace flush did not produce complete GPU traces for "
                    f"ranks 0..{expected_ranks - 1}"
                )
            complete = _complete_traces(
                trace_dir, expected_ranks, deadline, check_alive, cache
            )
            if complete is not None:
                capture.update(status="complete", trace_files=complete)
                return capture
            time.sleep(min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
    except BaseException as exc:
        # Persist failure and stop an uncertain capture even on cancellation;
        # the original exception still propagates to the runtime's owner.
        failure = exc
        capture["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if failure is not None and attempted:
            try:
                _http(
                    server_url.rstrip("/") + "/stop_profile",
                    timeout=min(_HTTP_SECONDS, flush_timeout),
                    body={},
                )
                capture["cleanup"] = "stop_requested"
            except (OSError, ValueError, http.client.HTTPException) as exc:
                capture["cleanup"] = f"stop request: {type(exc).__name__}: {exc}"
        _save_capture(trace_dir, capture)
