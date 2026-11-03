"""Run an endpoint-only client with a bounded, owned process group."""

from __future__ import annotations

import os
import signal
import subprocess
import time

from ...core.executor import EndpointClientCleanupError


def _group_exists(pgid):
    # Darwin may report EPERM rather than ESRCH for a now-empty group. Check
    # numeric process groups before the final kill; never infer absence from
    # a permission error against a group which still has members.
    result = subprocess.run(
        ["ps", "-axo", "pgid=,stat="],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return any(
        fields[0] == str(pgid) and not fields[1].startswith("Z")
        for line in result.stdout.splitlines()
        if len(fields := line.split()) >= 2
    )


def execute(argv, env, timeout, check_cancelled):
    timeout = float(timeout)
    if not 0 < timeout < float("inf"):
        raise ValueError("endpoint client timeout must be positive and finite")
    check_cancelled()
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    stdout, stderr = "", ""
    try:
        while True:
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    argv, timeout, output=stdout, stderr=stderr
                )
            try:
                stdout, stderr = process.communicate(timeout=min(1, remaining))
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired as error:
                stdout = error.output or ""
                stderr = error.stderr or ""
                continue
    finally:
        # Group cleanup covers descendants even if the shell leader exited.
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                if _group_exists(process.pid):
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while _group_exists(process.pid):
                if time.monotonic() >= deadline:
                    raise EndpointClientCleanupError(
                        "endpoint client process group is still active"
                    )
                time.sleep(0.05)
        except (OSError, subprocess.SubprocessError) as error:
            raise EndpointClientCleanupError(
                "endpoint client process cleanup was not confirmed"
            ) from error
