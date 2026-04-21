import pytest
import yaml

from Magpie.config import KernelEvalConfig, KernelType
from Magpie.main import (
    _expand_env_vars,
    _parse_command_list,
    _parse_kernel_entry,
    load_kernel_config,
    parse_kernel_type,
)


def test_parse_kernel_type_supports_aliases():
    assert parse_kernel_type("hip") is KernelType.HIP
    assert parse_kernel_type("torch") is KernelType.PYTORCH
    assert parse_kernel_type("py") is KernelType.PYTORCH
    assert parse_kernel_type("triton") is KernelType.TRITON


def test_parse_kernel_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported kernel type 'unknown'"):
        parse_kernel_type("unknown")


def test_parse_command_list_handles_supported_shapes():
    assert _parse_command_list(None) is None
    assert _parse_command_list("") == []
    assert _parse_command_list("make build") == ["make", "build"]
    assert _parse_command_list(["pytest", "-q"]) == ["pytest", "-q"]
    assert _parse_command_list([["make", "clean"], ["make", "build"]]) == [
        ["make", "clean"],
        ["make", "build"],
    ]
    assert _parse_command_list([]) is None
    assert _parse_command_list(["pytest", 1]) is None


def test_expand_env_vars_recurses_through_nested_data(monkeypatch):
    monkeypatch.setenv("MAGPIE_TMP", "/tmp/magpie")

    value = {
        "path": "$MAGPIE_TMP/output",
        "commands": [["echo", "$MAGPIE_TMP"], "$MAGPIE_TMP/plain"],
        "unchanged": 42,
    }

    expanded = _expand_env_vars(value)

    assert expanded == {
        "path": "/tmp/magpie/output",
        "commands": [["echo", "/tmp/magpie"], "/tmp/magpie/plain"],
        "unchanged": 42,
    }


def test_parse_kernel_entry_parses_commands_and_env(monkeypatch):
    monkeypatch.setenv("SRC_DIR", "/tmp/src")

    cfg = _parse_kernel_entry(
        {
            "id": "vector_add",
            "type": "triton",
            "source_files": ["$SRC_DIR/kernel.py"],
            "working_dir": "$SRC_DIR",
            "env": {"PYTHONPATH": "$SRC_DIR/lib"},
            "testcase_command": "python run_test.py",
            "compile_command": [["python", "setup.py"], ["python", "build.py"]],
            "prof_command": ["python", "profile.py"],
        }
    )

    assert cfg is not None
    assert cfg.kernel_id == "vector_add"
    assert cfg.kernel_type is KernelType.TRITON
    assert cfg.source_file_path == ["/tmp/src/kernel.py"]
    assert cfg.working_dir == "/tmp/src"
    assert cfg.env == {"PYTHONPATH": "/tmp/src/lib"}
    assert cfg.testcase_command == ["python", "run_test.py"]
    assert cfg.compiling_command == [["python", "setup.py"], ["python", "build.py"]]
    assert cfg.prof_command == ["python", "profile.py"]


def test_load_kernel_config_collects_sections_and_expands_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGPIE_ROOT", str(tmp_path / "workspace"))

    config_path = tmp_path / "kernel_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kernel": {
                    "id": "single",
                    "type": "hip",
                    "source_files": ["$MAGPIE_ROOT/single.hip"],
                    "testcase_command": "./single_test",
                },
                "kernels": [
                    {
                        "id": "second",
                        "type": "triton",
                        "source_files": ["$MAGPIE_ROOT/second.py"],
                        "compile_command": [["python", "prepare.py"]],
                    }
                ],
                "performance": {"backend": "$MAGPIE_ROOT/perf"},
                "correctness": {"backend": "testcase"},
                "ray_config": {"cluster_address": "ray://127.0.0.1:10001"},
                "scheduler": {"max_workers": 2},
            }
        ),
        encoding="utf-8",
    )

    kernels, perf_overrides, corr_overrides, sched_overrides = load_kernel_config(
        config_path
    )

    assert [cfg.kernel_id for cfg in kernels] == ["single", "second"]
    assert kernels[0].source_file_path == [f"{tmp_path}/workspace/single.hip"]
    assert kernels[1].compiling_command == [["python", "prepare.py"]]
    assert perf_overrides == {"backend": f"{tmp_path}/workspace/perf"}
    assert corr_overrides == {"backend": "testcase"}
    assert sched_overrides["max_workers"] == 2
    assert sched_overrides["ray_config"] == {"cluster_address": "ray://127.0.0.1:10001"}
    assert sched_overrides["environment"] == "ray"


def test_kernel_eval_config_normalizes_single_and_multi_commands():
    cfg = KernelEvalConfig(
        kernel_id="test",
        kernel_type=KernelType.HIP,
        source_file_path="kernel.hip",
        compiling_command=["make", "build"],
        testcase_command=[["python", "setup.py"], ["python", "run.py"]],
        prof_command=["rocprof", "app"],
    )

    assert cfg.get_source_file_paths() == ["kernel.hip"]
    assert cfg.has_compile_command() is True
    assert cfg.has_testcase() is True
    assert cfg.has_prof_command() is True
    assert cfg.get_compile_commands() == [["make", "build"]]
    assert cfg.get_testcase_commands() == [["python", "setup.py"], ["python", "run.py"]]
    assert cfg.get_prof_commands() == [["rocprof", "app"]]


def test_kernel_eval_config_from_dict_accepts_name_and_value():
    by_name = KernelEvalConfig.from_dict({"kernel_type": "triton", "kernel_id": "k1"})
    by_value = KernelEvalConfig.from_dict(
        {"kernel_type": KernelType.CUDA.value, "kernel_id": "k2"}
    )

    assert by_name.kernel_type is KernelType.TRITON
    assert by_value.kernel_type is KernelType.CUDA


def test_load_kernel_config_rejects_unknown_kernel_type(tmp_path):
    config_path = tmp_path / "invalid_kernel_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kernel": {
                    "id": "broken",
                    "type": "mystery",
                    "source_files": ["broken.xxx"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported kernel type 'mystery'"):
        load_kernel_config(config_path)
