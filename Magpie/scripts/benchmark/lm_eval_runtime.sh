#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Replace InferenceX's mutable evaluator setup with a caller-provided,
# hash-locked runtime. This helper performs only local reads and imports.

magpie_activate_lm_eval_runtime() {
    local runtime_root="${MAGPIE_LM_EVAL_RUNTIME_ROOT:-}"
    local expected_sha256="${MAGPIE_LM_EVAL_RUNTIME_SHA256:-}"
    local receipt_path="${MAGPIE_LM_EVAL_RUNTIME_RECEIPT:-}"
    local execution_mode="${MAGPIE_LM_EVAL_EXECUTION_MODE:-}"
    local require_readonly_mount="${MAGPIE_LM_EVAL_REQUIRE_READONLY_MOUNT:-}"

    if [[ -z "$runtime_root" || -z "$expected_sha256" || -z "$receipt_path" \
          || ( "$execution_mode" != "docker" && "$execution_mode" != "local" ) \
          || ( "$require_readonly_mount" != "0" && "$require_readonly_mount" != "1" ) ]]; then
        echo "ERROR: hash-locked lm-eval runtime environment is incomplete." >&2
        return 41
    fi

    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${runtime_root}/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

    python3 - "$runtime_root" "$expected_sha256" "$receipt_path" \
        "$execution_mode" "$require_readonly_mount" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
expected = sys.argv[2]
receipt = Path(sys.argv[3])
execution_mode = sys.argv[4]
require_readonly_mount = sys.argv[5] == "1"
manifest_path = root / "lm_eval_runtime_manifest.json"
sha256_re = re.compile(r"^[0-9a-f]{64}$")


def reject_constant(value):
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_readonly_directory(path, label):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    if info.st_mode & 0o222:
        raise ValueError(f"{label} must not have writable permission bits")


if not sha256_re.fullmatch(expected):
    raise SystemExit("ERROR: expected lm-eval runtime digest is invalid")
require_readonly_directory(root, "runtime root")
read_only_mount = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
if require_readonly_mount and not read_only_mount:
    raise SystemExit("ERROR: evaluator runtime bind mount is not read-only")
if {item.name for item in root.iterdir()} != {
    "lm_eval_runtime_manifest.json",
    "site-packages",
}:
    raise SystemExit("ERROR: runtime root contains unexpected entries")
require_readonly_directory(root / "site-packages", "site-packages")

manifest_info = manifest_path.lstat()
if (
    not stat.S_ISREG(manifest_info.st_mode)
    or manifest_info.st_nlink != 1
    or manifest_info.st_mode & 0o222
    or manifest_info.st_size <= 0
    or manifest_info.st_size > 64 * 1024 * 1024
):
    raise SystemExit("ERROR: runtime manifest is not a bounded read-only nlink=1 file")
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(
    manifest_bytes.decode("utf-8"),
    parse_constant=reject_constant,
    object_pairs_hook=reject_duplicates,
)
if not isinstance(manifest, dict) or set(manifest) != {
    "schema",
    "runtime_sha256",
    "site_packages",
    "identity",
    "files",
}:
    raise SystemExit("ERROR: runtime manifest keys do not match the v1 contract")
if manifest["schema"] != "apex.lm-eval-runtime/v1":
    raise SystemExit("ERROR: unsupported lm-eval runtime manifest schema")
if manifest["site_packages"] != "site-packages":
    raise SystemExit("ERROR: runtime manifest site_packages is invalid")
identity = manifest["identity"]
files = manifest["files"]
if not isinstance(identity, dict) or not isinstance(files, list) or not files:
    raise SystemExit("ERROR: runtime identity/files are invalid")

canonical = json.dumps(
    {"identity": identity, "files": files},
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
computed = hashlib.sha256(canonical).hexdigest()
if computed != expected or manifest["runtime_sha256"] != expected:
    raise SystemExit("ERROR: runtime manifest digest does not match expected digest")

site_packages = root / "site-packages"
actual_paths = []
for path in sorted(site_packages.rglob("*"), key=lambda item: item.as_posix()):
    info = path.lstat()
    relative = path.relative_to(site_packages).as_posix()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"ERROR: runtime symlink is forbidden: {relative}")
    if stat.S_ISDIR(info.st_mode):
        if info.st_mode & 0o222:
            raise SystemExit(f"ERROR: runtime directory is writable: {relative}")
        continue
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SystemExit(f"ERROR: runtime entry is not a regular nlink=1 file: {relative}")
    actual_paths.append(relative)

expected_paths = []
for item in files:
    if not isinstance(item, dict) or set(item) != {
        "path",
        "size_bytes",
        "mode",
        "sha256",
    }:
        raise SystemExit("ERROR: invalid runtime file record")
    raw_path = item["path"]
    pure_path = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
    if (
        pure_path is None
        or not raw_path
        or pure_path.is_absolute()
        or pure_path.as_posix() != raw_path
        or ".." in pure_path.parts
    ):
        raise SystemExit("ERROR: runtime file path is not canonical and relative")
    expected_paths.append(raw_path)
if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
    raise SystemExit("ERROR: runtime file records are not unique and sorted")
if actual_paths != expected_paths:
    raise SystemExit("ERROR: runtime files do not exactly match the manifest")

for item in files:
    path = site_packages / item["path"]
    info = path.lstat()
    if (
        not isinstance(item["size_bytes"], int)
        or isinstance(item["size_bytes"], bool)
        or item["size_bytes"] < 0
        or info.st_size != item["size_bytes"]
    ):
        raise SystemExit(f"ERROR: runtime file size mismatch: {item['path']}")
    if (
        not isinstance(item["mode"], int)
        or isinstance(item["mode"], bool)
        or item["mode"] & 0o222
        or stat.S_IMODE(info.st_mode) != item["mode"]
    ):
        raise SystemExit(f"ERROR: runtime file mode mismatch: {item['path']}")
    if not isinstance(item["sha256"], str) or not sha256_re.fullmatch(item["sha256"]):
        raise SystemExit(f"ERROR: runtime file digest is invalid: {item['path']}")
    if file_sha256(path) != item["sha256"]:
        raise SystemExit(f"ERROR: runtime file content mismatch: {item['path']}")

actual_abi = sys.implementation.cache_tag
expected_abi = identity.get("python_abi")
if actual_abi != expected_abi:
    raise SystemExit(f"ERROR: Python ABI mismatch: {actual_abi} != {expected_abi}")
actual_version = importlib.metadata.version("lm_eval")
expected_version = identity.get("lm_eval_version")
if actual_version != expected_version:
    raise SystemExit(f"ERROR: lm_eval version mismatch: {actual_version} != {expected_version}")

import lm_eval

module_path = Path(lm_eval.__file__).resolve(strict=True)
site_root = site_packages.resolve(strict=True)
try:
    module_relative = module_path.relative_to(site_root).as_posix()
except ValueError as exc:
    raise SystemExit("ERROR: lm_eval imported outside the supplied runtime") from exc
if not module_relative.startswith("lm_eval/"):
    raise SystemExit("ERROR: imported lm_eval module path is invalid")

payload = {
    "schema": "magpie.lm-eval-runtime-receipt/v1",
    "runtime_sha256": expected,
    "identity": identity,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "site_packages": "site-packages",
    "python_abi": actual_abi,
    "lm_eval_version": actual_version,
    "lm_eval_module": f"site-packages/{module_relative}",
    "execution_mode": execution_mode,
    "read_only_mount": read_only_mount,
    "verified": True,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
try:
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
finally:
    temporary.unlink(missing_ok=True)
PY
}

# InferenceX calls this hook immediately before importing/running lm_eval.
# Replacing it removes every mutable or network-backed dependency path.
_install_lm_eval_deps() {
    local status
    magpie_activate_lm_eval_runtime && return 0
    status=$?
    echo "ERROR: refusing to run lm-eval without a verified locked runtime." >&2
    exit "$status"
}

# Run lm-eval from a caller-bound evaluator policy instead of inheriting
# InferenceX's coupled context/output-budget heuristic.  The serving process
# continues to use MAX_MODEL_LEN; these two values govern only lm-eval's
# request admission and generated output budget.
magpie_run_lm_eval() {
    local port="${PORT:-8888}"
    local results_dir="${EVAL_RESULT_DIR:-${RESULT_DIR:-/workspace}/lm_eval}"
    local max_length="${MAGPIE_EVAL_MAX_LENGTH:-}"
    local max_gen_tokens="${MAGPIE_EVAL_MAX_GEN_TOKENS:-}"
    local tasks="${MAGPIE_EVAL_TASKS:-gsm8k}"
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-8}}"
    local batch_size="${MAGPIE_EVAL_BATCH_SIZE:-auto}"
    local python="${MAGPIE_EVAL_PYTHON:-python3}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) port="$2"; shift 2 ;;
            --results-dir) results_dir="$2"; shift 2 ;;
            *) echo "ERROR: unsupported Magpie lm-eval argument: $1" >&2; return 2 ;;
        esac
    done
    if [[ ! "$max_length" =~ ^[1-9][0-9]*$ ]] \
       || [[ ! "$max_gen_tokens" =~ ^[1-9][0-9]*$ ]] \
       || (( max_gen_tokens >= max_length )); then
        echo "ERROR: evaluator policy requires positive MAGPIE_EVAL_MAX_LENGTH and MAGPIE_EVAL_MAX_GEN_TOKENS with output < context." >&2
        return 42
    fi
    if [[ -z "${MAGPIE_EVAL_POLICY_ID:-}" || -z "${MAGPIE_EVAL_PRIMARY_METRIC:-}" ]]; then
        echo "ERROR: evaluator policy identity and primary metric are required." >&2
        return 42
    fi

    _install_lm_eval_deps || return $?
    mkdir -p "$results_dir" || return $?
    export EVAL_RESULT_DIR="$results_dir"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    local model_name="${MODEL_NAME:-${MODEL:-}}"
    local base_url="http://0.0.0.0:${port}/v1/chat/completions"
    local model_args="model=${model_name},base_url=${base_url},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${max_length}"
    local gen_kwargs="max_tokens=${max_gen_tokens},temperature=0,top_p=1"
    local -a command=(
        "$python" -m lm_eval
        --model local-chat-completions
        --apply_chat_template
        --tasks "$tasks"
        --output_path "$results_dir"
        --log_samples
        --batch_size "$batch_size"
        --model_args "$model_args"
        --gen_kwargs "$gen_kwargs"
    )
    if [[ -n "${MAGPIE_EVAL_LIMIT:-}" ]]; then
        command+=(--limit "$MAGPIE_EVAL_LIMIT")
    fi
    printf '[magpie] evaluator policy=%s primary=%s max_length=%s max_gen_tokens=%s\n' \
        "$MAGPIE_EVAL_POLICY_ID" "$MAGPIE_EVAL_PRIMARY_METRIC" \
        "$max_length" "$max_gen_tokens" >&2
    "${command[@]}"
}
