"""Capture bounded SGLang expert totals without putting tensors in the report.

Run with the server's Python (which has torch), outside the client Python shim.
Counts retain source rank and physical expert axes: they are not globally reduced.
The last expert map is only valid throughout the capture with EPLB disabled.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile

SUMMARY = "expert_load_distribution.json"
ARTIFACT = "expert_distribution.json"
STATE = ".expert_distribution_state.json"
SCOPE = "client_including_warmup"
DUMP_WAIT_SECONDS = 10


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False) + "\n")
    temporary.replace(path)


def _request(base_url, action):
    request = Request(base_url.rstrip("/") + f"/{action}_expert_distribution_record", method="POST")
    timeout = 40 if action == "dump" else 30
    with urlopen(request, timeout=timeout) as response:
        response.read(4096)


def _get_server_info(base_url):
    with urlopen(base_url.rstrip("/") + "/get_server_info", timeout=30) as response:
        body = response.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("SGLang server information exceeds 1 MiB")
    return json.loads(body)


def _expected_dump_count(info):
    # SGLang exposes ServerArgs at the top level; some versions nest the fields.
    args = info.get("server_args", info)
    for name in ("tp_size", "pp_size", "dp_size"):
        if type(args.get(name)) is not int or args[name] < 1:
            raise ValueError(f"SGLang server information has no valid {name}")
    if args["pp_size"] != 1:
        raise ValueError("Expert-count capture currently requires pp_size=1")
    if args["dp_size"] > 1 and args.get("enable_dp_attention") is not True:
        raise ValueError("Expert-count capture does not support standalone DP replicas")
    if args.get("expert_distribution_recorder_mode") != "stat":
        raise ValueError("SGLang must use expert_distribution_recorder_mode=stat")
    if args.get("expert_distribution_recorder_buffer_size") != 1:
        raise ValueError("SGLang must use expert_distribution_recorder_buffer_size=1")
    if args.get("enable_eplb"):
        raise ValueError("Expert-count capture requires EPLB disabled for a stable map")
    return args["tp_size"]


def _matrix(value):
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, list) or not rows or not rows[0]:
        raise ValueError("Expected a nonempty layer by physical-expert matrix")
    width = len(rows[0])
    if any(
        not isinstance(row, list)
        or len(row) != width
        or any(type(item) is not int or item < 0 for item in row)
        for row in rows
    ):
        raise ValueError("Expected rectangular nonnegative integer expert counts/map")
    return rows


def _load_dump(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def export_aggregate_dumps(paths, destination, load_dump=None, *, allow_empty=False):
    """Publish compact, unreduced worker totals only after native plugin proof."""
    load_dump = load_dump or _load_dump
    workers, dimensions, record_count = [], None, 0
    for path in paths:
        dump = load_dump(path)
        if dump.get("schema_version") != 2 or dump.get("aggregation") != "sum_over_capture":
            raise ValueError("Native aggregate recorder plugin is unavailable on a worker")
        if dump.get("gatherer_key") != "primary":
            raise ValueError("Aggregate recorder requires the primary expert gatherer")
        counts = _matrix(dump["global_physical_count"])
        mapping = _matrix(dump["last_physical_to_logical_map"])
        shape = (len(counts), len(counts[0]))
        if (len(mapping), len(mapping[0])) != shape or (
            dimensions is not None and dimensions != shape
        ):
            raise ValueError("Worker aggregate counts/maps have different dimensions")
        dimensions = shape
        rank, passes = dump["rank"], dump["forward_pass_count"]
        first, last = dump["first_forward_pass_id"], dump["last_forward_pass_id"]
        if type(rank) is not int or rank < 0 or type(passes) is not int or passes < 0:
            raise ValueError("Invalid native aggregate rank/forward-pass count")
        if passes == 0:
            if first is not None or last is not None or any(any(row) for row in counts):
                raise ValueError("Empty aggregate worker has nonzero counts or pass IDs")
        elif (
            type(first) is not int
            or type(last) is not int
            or first < 0
            or last < first
            or last - first + 1 < passes
        ):
            raise ValueError("Invalid aggregate forward-pass ID range/count")
        workers.append(
            {
                "source_file": path.name,
                "rank": rank,
                "forward_pass_count": passes,
                "first_forward_pass_id": first,
                "last_forward_pass_id": last,
                "global_physical_count": counts,
                "last_physical_to_logical_map": mapping,
            }
        )
        record_count += passes
    if not workers:
        raise ValueError("No native aggregate recorder dumps were written")
    if not allow_empty and not any(
        any(any(row) for row in worker["global_physical_count"]) for worker in workers
    ):
        raise ValueError("Aggregate capture has no expert assignments")
    _write_json(
        destination,
        {
            "schema_version": 2,
            "source": "sglang",
            "recorder_mode": "aggregate",
            "aggregation": "sum_over_capture",
            "scope": SCOPE,
            "count_axes": ["layer", "physical_expert"],
            "rank_reduction": "none",
            "stream_identity": "source_file",
            "rank_identity": "native_recorder_rank",
            "ranks": workers,
        },
    )
    return {"rank_count": len(workers), "record_count": record_count}


def _native_files(recorder_dir):
    return {path.name for path in recorder_dir.glob("expert_distribution_recorder_*.pt")}


def _remove_owned(paths):
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


def _export_when_ready(recorder_dir, existing, destination, expected_count, *, allow_empty=False):
    deadline = time.monotonic() + DUMP_WAIT_SECONDS
    last_error = None
    while True:
        paths = sorted(
            path
            for path in recorder_dir.glob("expert_distribution_recorder_*.pt")
            if path.name not in existing
        )
        if len(paths) > expected_count:
            raise ValueError(f"Expected {expected_count} worker dumps, found {len(paths)}")
        if len(paths) == expected_count:
            try:
                return export_aggregate_dumps(paths, destination, allow_empty=allow_empty), paths
            except (EOFError, BadZipFile, RuntimeError) as error:
                # torch.save writes directly to the final filename. Rank 0's HTTP
                # response can arrive while another worker's ZIP is still open.
                if not isinstance(error, (EOFError, BadZipFile)) and not any(
                    message in str(error).lower()
                    for message in (
                        "failed reading zip archive",
                        "failed finding central directory",
                        "unexpected eof",
                        "unexpected end of file",
                        "invalid header or archive is corrupted",
                    )
                ):
                    raise
                last_error = error
        if time.monotonic() >= deadline:
            detail = f"; last read: {last_error}" if last_error is not None else ""
            raise TimeoutError(
                f"Expected {expected_count} complete worker dumps; found {len(paths)}{detail}"
            )
        time.sleep(0.25)


def capture(action, base_url, output_dir, recorder_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / STATE
    stop_attempted = False
    stop_completed = False
    start_attempted = False
    try:
        if action == "start":
            for name in (STATE, ARTIFACT, SUMMARY):
                (output_dir / name).unlink(missing_ok=True)
            existing = _native_files(recorder_dir)
            expected_count = _expected_dump_count(_get_server_info(base_url))
            # A fresh dump from every worker proves actual activation. A marker
            # file or package install alone could outlive the serving process.
            start_attempted = True
            _request(base_url, "start")
            _request(base_url, "stop")
            _request(base_url, "dump")
            probe = output_dir / ".expert_distribution_probe.json"
            try:
                _, probe_paths = _export_when_ready(
                    recorder_dir,
                    existing,
                    probe,
                    expected_count,
                    allow_empty=True,
                )
            finally:
                probe.unlink(missing_ok=True)
            _remove_owned(probe_paths)
            # The probe's reset must not contribute counts to this capture.
            existing = _native_files(recorder_dir)
            _request(base_url, "start")
            _write_json(state_path, {"existing": sorted(existing), "expected_count": expected_count})
            return
        else:
            state = json.loads(state_path.read_text())
            stop_attempted = True
            _request(base_url, "stop")
            stop_completed = True
            _request(base_url, "dump")
            metadata, paths = _export_when_ready(
                recorder_dir,
                set(state["existing"]),
                output_dir / ARTIFACT,
                state["expected_count"],
            )
            summary = {"status": "available", "scope": SCOPE, "mode": "aggregate",
                       "artifact_path": ARTIFACT, **metadata}
            _write_json(output_dir / SUMMARY, summary)
            # Retain native evidence until portable counts and metadata are saved.
            _remove_owned(paths)
            return summary
    except Exception:
        # Even a timed-out start may have enabled recording on a reused server.
        if start_attempted or (stop_attempted and not stop_completed):
            try:
                _request(base_url, "stop")
            except Exception:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "finish"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--recorder-dir", required=True, type=Path)
    args = parser.parse_args()
    capture(args.action, args.base_url, args.output_dir, args.recorder_dir)


if __name__ == "__main__":
    main()
