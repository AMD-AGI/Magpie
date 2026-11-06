"""Run an endpoint-only client with a bounded, owned process group."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager

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


@contextmanager
def _defer_sigint():
    """Let Ray cancellation run after process ownership is recorded/cleaned."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    pending = []
    signal.signal(signal.SIGINT, lambda signum, frame: pending.append((signum, frame)))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if pending and previous != signal.SIG_IGN:
            handler = previous if callable(previous) else signal.default_int_handler
            handler(*pending[0])


def execute(argv, env, timeout, check_cancelled):
    timeout = float(timeout)
    if not 0 < timeout < float("inf"):
        raise ValueError("endpoint client timeout must be positive and finite")
    check_cancelled()
    deadline = time.monotonic() + timeout
    process = None
    stdout, stderr = "", ""
    try:
        # The finally guard exists before launch. Defer Ray's SIGINT handler
        # through Popen's return/assignment without masking signals in the child.
        with _defer_sigint():
            process = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
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
        if process is None:
            raise EndpointClientCleanupError(
                "endpoint client launch ownership was not confirmed"
            )
        # Group cleanup covers descendants even if the shell leader exited.
        cleaned = False
        try:
            with _defer_sigint():
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
                cleaned = True
        except BaseException as error:
            if cleaned:
                raise
            raise EndpointClientCleanupError(
                "endpoint client process cleanup was not confirmed"
            ) from error
