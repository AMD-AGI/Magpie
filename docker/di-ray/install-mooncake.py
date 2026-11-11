"""Replace only the pinned vendor image's unowned Mooncake files."""

import importlib.util
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

namespace = Path(importlib.util.find_spec("mooncake").origin).parent
expected = Path("/opt/venv/lib/python3.10/site-packages/mooncake")
if namespace != expected:
    raise RuntimeError(f"unexpected vendor Mooncake path: {namespace}")
if any(
    str(file).startswith("mooncake/")
    for distribution in metadata.distributions()
    for file in (distribution.files or [])
):
    raise RuntimeError(
        "vendor Mooncake has a package owner; review before replacing it"
    )
shutil.rmtree(namespace)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "/tmp/mooncake_transfer_engine_rocm-0.3.14-cp310-cp310-manylinux_2_35_x86_64.whl",
    ],
    check=True,
)
