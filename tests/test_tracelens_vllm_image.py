import hashlib
import json
import subprocess

from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.tracelens_runtime import (
    prepare_tracelens_runtime_image,
)
from Magpie.modes.benchmark.tracelens_vllm_image import (
    LABEL_WHEEL_MANIFEST,
    LABEL_WHEEL_MANIFEST_SHA256,
    VLLM_TRACELENS_FORBIDDEN,
    VLLM_TRACELENS_REQUIREMENTS,
    VllmTraceLensIdentity,
    _verification_script,
    _write_build_context,
    resolve_vllm_tracelens_identity,
    validate_vllm_tracelens_image,
)


def _identity():
    return VllmTraceLensIdentity(
        base_image="vllm/vllm-openai-rocm:v0.19.1",
        base_image_id="sha256:" + "1" * 64,
        base_image_locator=(
            "vllm/vllm-openai-rocm@sha256:" + "a" * 64
        ),
        vllm_version="0.19.1+rocm721",
        grpcio_version="1.78.0",
        source_commit="2" * 40,
        source_tree="3" * 40,
        patch_version="v19",
        patch_path=(
            "examples/custom_workflows/inference_analysis/vllm_patches/"
            "config_vllm_v0.19.0.patch"
        ),
        patch_sha256="4" * 64,
        patch_bytes=b"diff --git a/vllm/example.py b/vllm/example.py\n",
    )


def _wheel_manifest():
    records = [
        {
            "distribution": name,
            "filename": f"{name}-{version}-py3-none-any.whl",
            "version": version,
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name, version in VLLM_TRACELENS_REQUIREMENTS
    ]
    records.append(
        {
            "distribution": "tracelens",
            "filename": "tracelens-0.1.0-py3-none-any.whl",
            "version": "source-commit",
            "sha256": hashlib.sha256(b"tracelens").hexdigest(),
        }
    )
    return records


def _labels(identity):
    labels = identity.labels()
    manifest = json.dumps(_wheel_manifest(), sort_keys=True, separators=(",", ":"))
    labels[LABEL_WHEEL_MANIFEST] = manifest
    labels[LABEL_WHEEL_MANIFEST_SHA256] = hashlib.sha256(
        manifest.encode()
    ).hexdigest()
    return labels


def test_resolve_identity_uses_image_digest_and_committed_patch(monkeypatch, tmp_path):
    patch = b"committed patch bytes\n"
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image.docker_image_record",
        lambda _image: {
            "Id": "sha256:" + "1" * 64,
            "RepoDigests": ["registry/vllm@sha256:" + "a" * 64],
        },
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image._git_text",
        lambda _repo, *args: "2" * 40 if args[-1] == "HEAD" else "3" * 40,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image._git_bytes",
        lambda _repo, *_args: patch,
    )

    identity = resolve_vllm_tracelens_identity(
        base_image="registry/vllm:v0.19.1",
        vllm_version="0.19.1+rocm721",
        grpcio_version="1.78.0",
        tracelens_repo=tmp_path,
        patch_version="v19",
    )

    assert identity.base_image_locator == "registry/vllm@sha256:" + "a" * 64
    assert identity.source_commit == "2" * 40
    assert identity.source_tree == "3" * 40
    assert identity.patch_sha256 == hashlib.sha256(patch).hexdigest()
    labels = identity.labels()
    assert labels["io.magpie.tracelens.base-image-id"] == identity.base_image_id
    assert labels["io.magpie.tracelens.source-tree"] == identity.source_tree


def test_build_context_is_offline_minimal_and_identity_labeled(tmp_path):
    identity = _identity()
    labels = _write_build_context(tmp_path, identity, _wheel_manifest())

    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    verifier = (tmp_path / "verify.py").read_text(encoding="utf-8")
    identity_document = json.loads(
        (tmp_path / "identity.json").read_text(encoding="utf-8")
    )

    assert dockerfile.startswith(f"FROM {identity.base_image_locator}\n")
    assert "--no-index --no-deps" in dockerfile
    assert "git apply --check" in dockerfile
    assert "pip install --upgrade" not in dockerfile
    assert labels[LABEL_WHEEL_MANIFEST_SHA256]
    assert identity_document["tracelens_source_commit"] == identity.source_commit
    assert identity_document["tracelens_source_tree"] == identity.source_tree
    assert identity_document["tracelens_patch_sha256"] == identity.patch_sha256
    assert "metadata.version(\"grpcio\")" in verifier
    assert "metadata.version(\"vllm\")" in verifier
    for package in VLLM_TRACELENS_FORBIDDEN:
        assert package in verifier


def test_existing_image_validation_preserves_vllm_grpc_and_excludes_pollution(
    monkeypatch,
):
    identity = _identity()
    base_record = {
        "Id": identity.base_image_id,
        "RootFS": {"Layers": ["base-a", "base-b"]},
        "Config": {"Labels": {}},
    }
    image_record = {
        "Id": "sha256:" + "5" * 64,
        "RootFS": {"Layers": ["base-a", "base-b", "derived"]},
        "Config": {"Labels": _labels(identity)},
    }

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image.docker_image_record",
        lambda image: base_record if image == identity.base_image_id else image_record,
    )

    seen = {}

    def fake_probe(image):
        seen["image"] = image
        seen["script"] = _verification_script()
        return subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout=json.dumps(
                {
                    "grpcio_version": "1.78.0",
                    "platforms": ["MI300X", "MI325X"],
                    "tracelens_version": "0.1.0",
                    "vllm_version": "0.19.1+rocm721",
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image._runtime_probe",
        fake_probe,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image._patch_reverse_probe",
        lambda _image, _patch: subprocess.CompletedProcess(
            ["docker", "run"], 0, stdout=b"", stderr=b""
        ),
    )

    validation = validate_vllm_tracelens_image("derived:v19", identity)

    assert validation["valid"] is True
    assert validation["runtime_probe"]["grpcio_version"] == "1.78.0"
    assert validation["runtime_probe"]["vllm_version"] == "0.19.1+rocm721"
    for package in VLLM_TRACELENS_FORBIDDEN:
        assert package in seen["script"]


def test_existing_image_validation_rejects_stale_identity_without_probe(monkeypatch):
    identity = _identity()
    stale_labels = _labels(identity)
    stale_labels["io.magpie.tracelens.source-commit"] = "9" * 40
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image.docker_image_record",
        lambda _image: {
            "Id": "sha256:" + "5" * 64,
            "RootFS": {"Layers": ["base-a", "derived"]},
            "Config": {"Labels": stale_labels},
        },
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_vllm_image._runtime_probe",
        lambda _image: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    validation = validate_vllm_tracelens_image("derived:v19", identity)

    assert validation["valid"] is False
    assert validation["reason"] == "identity label mismatch"
    assert "io.magpie.tracelens.source-commit" in validation["label_mismatches"]


def test_prepare_rebuilds_stale_vllm_tag(monkeypatch, tmp_path):
    identity = _identity()
    workflow = tmp_path / "examples/custom_workflows/inference_analysis"
    patch_dir = workflow / "vllm_patches"
    patch_dir.mkdir(parents=True)
    (workflow / "build_docker_vllm.sh").write_text(
        "case ${VLLM_VERSION} in\n  v19) ;;\nesac\n",
        encoding="utf-8",
    )
    (patch_dir / "config_vllm_v0.19.0.patch").write_text(
        "patch",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tmp_path))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: True,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_package_version",
        lambda _image, package: {
            "vllm": identity.vllm_version,
            "grpcio": identity.grpcio_version,
        }.get(package),
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.resolve_vllm_tracelens_identity",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.validate_vllm_tracelens_image",
        lambda _image, _identity: {
            "valid": False,
            "reason": "identity label mismatch",
        },
    )
    builds = []

    def fake_build(**kwargs):
        builds.append(kwargs)
        return {
            "command": ["docker", "build", "<temporary-build-context>"],
            "source_wheel_command": ["docker", "run", "<staged-source>"],
            "requirements_download_command": ["docker", "run", "pip", "download"],
            "image_id": "sha256:" + "6" * 64,
            "image_labels": identity.labels(),
            "dependency_wheels": _wheel_manifest(),
            "dependency_wheel_manifest_sha256": "7" * 64,
            "validation": {"valid": True, "reason": "verified after rebuild"},
        }

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.build_vllm_tracelens_image",
        fake_build,
    )
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "docker_image": identity.base_image,
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = prepare_tracelens_runtime_image(cfg, identity.base_image, "mi355x")

    assert len(builds) == 1
    assert result["stale_image_rejected"] is True
    assert result["stale_image_rejection_reason"] == "identity label mismatch"
    assert result["built"] is True
    assert result["public_runtime_validation"]["valid"] is True
