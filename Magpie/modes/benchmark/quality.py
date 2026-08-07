"""Parse persistent lm-eval artifacts into a bounded serving quality receipt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


PRIMARY_METRICS = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "exact_match",
    "acc_norm,none",
    "acc,none",
    "acc_norm",
    "acc",
    "pass@1,none",
    "pass@1",
)
MAX_REPORTED_ARTIFACTS = 256
MAX_REPORTED_TASKS = 256
MAX_METRICS_PER_TASK = 64


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_receipt(path: Path, workspace: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace)),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _numeric_metrics(data: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, value in data.items():
        lowered = str(name).lower()
        if "stderr" in lowered or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                metrics[str(name)] = number
    return metrics


def _primary(metrics: Mapping[str, float]) -> Tuple[Optional[str], Optional[float]]:
    for name in PRIMARY_METRICS:
        if name in metrics:
            return name, metrics[name]
    if metrics:
        name = sorted(metrics)[0]
        return name, metrics[name]
    return None, None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_lm_eval_quality(workspace: Path, *, requested: bool) -> Dict[str, Any]:
    """Return an auditable quality gate from ``workspace/lm_eval``.

    ``passed`` means the requested evaluation produced parseable task metrics; it
    is an evidence-completeness gate, not an absolute accuracy threshold. Apex can
    compare the exposed primary metrics between baseline and candidate runs.
    """

    workspace = Path(workspace)
    eval_dir = workspace / "lm_eval"
    result_files = (
        sorted(eval_dir.rglob("results*.json")) if eval_dir.is_dir() else []
    )
    artifact_files = (
        sorted(path for path in eval_dir.rglob("*") if path.is_file())
        if eval_dir.is_dir()
        else []
    )
    relative_artifacts = [
        str(path.relative_to(workspace))
        for path in artifact_files[:MAX_REPORTED_ARTIFACTS]
    ]
    artifact_receipts: List[Dict[str, Any]] = []
    result_receipts: List[Dict[str, Any]] = []
    sample_receipts: List[Dict[str, Any]] = []
    for path in artifact_files[:MAX_REPORTED_ARTIFACTS]:
        try:
            receipt = _artifact_receipt(path, workspace)
            artifact_receipts.append(receipt)
            if path.name.startswith("samples") and path.suffix == ".jsonl":
                sample_receipts.append(receipt)
        except OSError:
            pass
    for path in result_files[:MAX_REPORTED_ARTIFACTS]:
        try:
            result_receipts.append(_artifact_receipt(path, workspace))
        except OSError:
            pass
    sample_set_digest = (
        _canonical_digest(
            {
                "schema": "magpie.lm-eval-sample-set/v1",
                "artifacts": sample_receipts,
            }
        )
        if sample_receipts
        else None
    )

    if not result_files:
        status = "missing" if requested else "not_requested"
        missing_errors = (
            ["RUN_EVAL was requested but no lm_eval/results*.json artifact exists"]
            if requested
            else []
        )
        return {
            "kind": "lm_eval",
            "requested": requested,
            "status": status,
            "passed": False if requested else None,
            "evidence_present": False,
            "tasks": {},
            "artifacts": relative_artifacts,
            "artifact_count": len(artifact_files),
            "artifacts_truncated": len(artifact_files) > MAX_REPORTED_ARTIFACTS,
            "result_artifact_receipts": result_receipts,
            "artifact_receipts": artifact_receipts,
            "sample_artifact_receipts": sample_receipts,
            "sample_set_digest": sample_set_digest,
            "outcome_digest": None,
            "result_artifact_count": len(result_files),
            "result_artifacts_truncated": (
                len(result_files) > MAX_REPORTED_ARTIFACTS
            ),
            "errors": missing_errors,
            "error_count": len(missing_errors),
            "errors_truncated": False,
        }

    tasks: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for path in result_files:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        raw_results = data.get("results") if isinstance(data, Mapping) else None
        if not isinstance(raw_results, Mapping):
            errors.append(f"{path.name}: missing results object")
            continue
        for task_name, task_result in raw_results.items():
            if not isinstance(task_result, Mapping):
                errors.append(f"{path.name}:{task_name}: result is not an object")
                continue
            metrics = _numeric_metrics(task_result)
            primary_name, primary_value = _primary(metrics)
            if primary_name is None:
                errors.append(f"{path.name}:{task_name}: no numeric quality metric")
                continue
            reported_metrics = dict(
                sorted(metrics.items())[:MAX_METRICS_PER_TASK]
            )
            if primary_name not in reported_metrics:
                reported_metrics[primary_name] = primary_value
            tasks[str(task_name)] = {
                "primary_metric": primary_name,
                "value": primary_value,
                "metrics": reported_metrics,
                "metric_count": len(metrics),
                "metrics_truncated": len(metrics) > MAX_METRICS_PER_TASK,
                "source": str(path.relative_to(workspace)),
            }

    passed = bool(tasks) and not errors
    task_items = sorted(tasks.items())
    reported_tasks = dict(task_items[:MAX_REPORTED_TASKS])
    primary_outcomes = {
        task: {
            "metric": value["primary_metric"],
            "value": value["value"],
            "source": value["source"],
        }
        for task, value in reported_tasks.items()
    }
    outcome_digest = _canonical_digest(
        {
            "schema": "magpie.lm-eval-outcomes/v1",
            "primary_metric_policy": list(PRIMARY_METRICS),
            "outcomes": primary_outcomes,
            "result_artifacts": result_receipts,
            "sample_set_digest": sample_set_digest,
        }
    )
    return {
        "kind": "lm_eval",
        "requested": requested,
        "status": "passed" if passed else "invalid",
        "passed": passed,
        "evidence_present": bool(tasks),
        "tasks": reported_tasks,
        "primary_metric_policy": list(PRIMARY_METRICS),
        "primary_outcomes": primary_outcomes,
        "outcome_digest": outcome_digest,
        "sample_set_digest": sample_set_digest,
        "task_count": len(tasks),
        "tasks_truncated": len(tasks) > MAX_REPORTED_TASKS,
        "artifacts": relative_artifacts,
        "artifact_count": len(artifact_files),
        "artifacts_truncated": len(artifact_files) > MAX_REPORTED_ARTIFACTS,
        "result_artifact_receipts": result_receipts,
        "artifact_receipts": artifact_receipts,
        "sample_artifact_receipts": sample_receipts,
        "result_artifact_count": len(result_files),
        "result_artifacts_truncated": (
            len(result_files) > MAX_REPORTED_ARTIFACTS
        ),
        "errors": errors[:MAX_REPORTED_ARTIFACTS],
        "error_count": len(errors),
        "errors_truncated": len(errors) > MAX_REPORTED_ARTIFACTS,
    }
