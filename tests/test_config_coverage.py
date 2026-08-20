from pathlib import Path

import pytest

from Magpie.config.correctness import (
    CorrectnessBackend,
    CorrectnessConfig,
    CorrectnessMode,
)
from Magpie.config.kernel import KernelEvalConfig
from Magpie.config.performance import (
    MetrixConfig,
    NcuConfig,
    PerfBackend,
    PerformanceConfig,
    RocprofComputeConfig,
)
from Magpie.config.pipeline import CompilingConfig, KernelType, PipelineConfig


def test_metrix_profile_args_include_all_optional_settings():
    cfg = MetrixConfig(
        profile="memory",
        metrics=["memory.l2_hit_rate", "compute.total_flops"],
        kernel_filter="gemm.*",
        num_replays=3,
        timeout_seconds=45,
        extra_args=["--verbose"],
    )

    assert cfg.get_profile_args("metrics.json") == [
        "--output",
        "metrics.json",
        "--num-replays",
        "3",
        "--timeout",
        "45",
        "--aggregate",
        "--profile",
        "memory",
        "--metrics",
        "memory.l2_hit_rate,compute.total_flops",
        "--kernel",
        "gemm.*",
        "--verbose",
    ]


def test_metrix_profile_args_work_with_defaults():
    assert MetrixConfig().get_profile_args("out.json") == [
        "--output",
        "out.json",
        "--num-replays",
        "1",
        "--timeout",
        "60",
        "--aggregate",
    ]


def test_rocprof_profile_and_analyze_args_cover_filters_and_expansion(tmp_path):
    cfg = RocprofComputeConfig(
        workload_dir=str(tmp_path / "workloads"),
        profile_args=["--no-roof", "--roof-only --quiet", 7],
        analyze_args=["--dispatch 2", "--verbose"],
        metric_blocks=["1", "10"],
        kernel_filter="gemm",
        dispatch_filter=[2, 4],
    )

    assert cfg.get_profile_args("case", "/override") == [
        "-n",
        "case",
        "-p",
        "/override",
        "-b",
        "1",
        "10",
        "-k",
        "gemm",
        "-d",
        "2",
        "4",
        "--no-roof",
        "--roof-only",
        "--quiet",
    ]
    assert cfg.get_analyze_args("/workload", "/csv") == [
        "-p",
        "/workload",
        "-b",
        "1",
        "10",
        "--output-format",
        "csv",
        "--output-name",
        "/csv",
        "--dispatch",
        "2",
        "--verbose",
    ]
    assert cfg.get_workload_path("case") == tmp_path / "workloads" / "case"
    assert cfg.get_workload_path("case", "/base") == Path("/base/case")


def test_rocprof_args_cover_empty_blocks_and_non_csv_output():
    cfg = RocprofComputeConfig(metric_blocks=[], output_format="json")

    assert cfg.get_profile_args("case") == ["-n", "case", "-p", "./workloads"]
    assert cfg.get_analyze_args("/workload", "/ignored") == ["-p", "/workload"]

    cfg.output_format = "csv"
    assert cfg.get_analyze_args("/workload") == [
        "-p",
        "/workload",
        "--output-format",
        "csv",
    ]


@pytest.mark.parametrize(
    ("kernel_type", "gpu_arch", "expected"),
    [
        (KernelType.HIP, None, PerfBackend.ROCPROF_COMPUTE),
        (KernelType.CUDA, None, PerfBackend.NCU),
        (KernelType.TRITON, "gfx942", PerfBackend.ROCPROF_COMPUTE),
        (KernelType.TRITON, "sm_90", PerfBackend.NCU),
        (KernelType.TRITON, "cpu", PerfBackend.NONE),
        (KernelType.TRITON, None, PerfBackend.NONE),
        (KernelType.PYTORCH, None, PerfBackend.NONE),
    ],
)
def test_performance_config_selects_backend(kernel_type, gpu_arch, expected):
    cfg = PerformanceConfig(kernel_type=kernel_type, gpu_arch=gpu_arch)

    assert cfg.get_backend() is expected
    assert isinstance(cfg.rocprof_config, RocprofComputeConfig)
    assert isinstance(cfg.ncu_config, NcuConfig)
    assert isinstance(cfg.metrix_config, MetrixConfig)


def test_performance_config_preserves_explicit_backend_and_configs():
    rocprof = RocprofComputeConfig(metric_blocks=[])
    ncu = NcuConfig(args=["--set", "full"])
    metrix = MetrixConfig(profile="quick")
    cfg = PerformanceConfig(
        backend=PerfBackend.METRIX,
        kernel_type=KernelType.CUDA,
        rocprof_config=rocprof,
        ncu_config=ncu,
        metrix_config=metrix,
    )

    assert cfg.get_backend() is PerfBackend.METRIX
    assert cfg.rocprof_config is rocprof
    assert cfg.ncu_config is ncu
    assert cfg.metrix_config is metrix
    assert PerformanceConfig().get_backend() is PerfBackend.NONE


def test_correctness_config_from_empty_dict_and_testcase_detection():
    cfg = CorrectnessConfig.from_dict({}, mode=CorrectnessMode.RESULT_COMPARISON)

    assert cfg.mode is CorrectnessMode.RESULT_COMPARISON
    assert cfg.backend is CorrectnessBackend.TESTCASE
    assert cfg.has_testcase() is False
    cfg.testcase_command = []
    assert cfg.has_testcase() is False
    cfg.testcase_command = ["pytest", "-q"]
    assert cfg.has_testcase() is True


def test_correctness_config_builds_complete_accordo_settings():
    cfg = CorrectnessConfig.from_dict(
        {
            "backend": "accordo",
            "workspace_path": "/workspace",
            "accordo": {
                "kernel_name": "gemm",
                "reference_binary": "./reference",
                "optimized_binary": "./optimized",
                "atol": 0.01,
                "rtol": 0.02,
                "equal_nan": True,
                "timeout_seconds": 12,
                "kernel_args": [["x", "ptr"]],
                "working_directory": "/work",
            },
        },
        mode=CorrectnessMode.RESULT_COMPARISON,
    )

    assert cfg.backend is CorrectnessBackend.ACCORDO
    assert cfg.mode is CorrectnessMode.RESULT_COMPARISON
    assert cfg.accordo_config.kernel_name == "gemm"
    assert cfg.accordo_config.reference_binary == "./reference"
    assert cfg.accordo_config.optimized_binary == "./optimized"
    assert cfg.accordo_config.atol == 0.01
    assert cfg.accordo_config.rtol == 0.02
    assert cfg.accordo_config.equal_nan is True
    assert cfg.accordo_config.timeout_seconds == 12
    assert cfg.accordo_config.kernel_args == [["x", "ptr"]]
    assert cfg.accordo_config.working_directory == "/work"
    assert cfg.accordo_config.workspace_path == "/workspace"


def test_correctness_config_allows_accordo_settings_with_testcase_backend():
    cfg = CorrectnessConfig.from_dict({"accordo": {"kernel_name": "gemm"}})

    assert cfg.backend is CorrectnessBackend.TESTCASE
    assert cfg.accordo_config.kernel_name == "gemm"


def test_kernel_eval_config_command_helpers_and_round_trip():
    cfg = KernelEvalConfig(
        kernel_id="gemm",
        kernel_type=KernelType.CUDA,
        source_file_path="kernel.cu",
        compiling_command=["make", "build"],
        testcase_command=[["setup"], ["run"]],
        prof_command=["ncu", "./run"],
        input_shapes=[(16, 16)],
        extra={"warmup": 2},
    )

    assert cfg.get_source_file_paths() == ["kernel.cu"]
    assert cfg.has_compile_command() is True
    assert cfg.has_testcase() is True
    assert cfg.has_prof_command() is True
    assert cfg.get_compile_commands() == [["make", "build"]]
    assert cfg.get_testcase_commands() == [["setup"], ["run"]]
    assert cfg.get_prof_commands() == [["ncu", "./run"]]

    restored = KernelEvalConfig.from_dict(cfg.to_dict())
    assert restored.kernel_id == "gemm"
    assert restored.kernel_type is KernelType.CUDA
    assert restored.input_shapes == [(16, 16)]
    assert restored.extra == {"warmup": 2}


def test_kernel_eval_config_empty_commands_and_enum_inputs():
    cfg = KernelEvalConfig(source_file_path=["a.hip", "b.hip"])

    assert cfg.get_source_file_paths() == ["a.hip", "b.hip"]
    assert cfg.has_compile_command() is False
    assert cfg.has_testcase() is False
    assert cfg.has_prof_command() is False
    assert cfg.get_compile_commands() == []
    assert cfg.get_testcase_commands() == []
    assert cfg.get_prof_commands() == []
    assert (
        KernelEvalConfig.from_dict({"kernel_type": KernelType.HIP}).kernel_type
        is KernelType.HIP
    )
    assert (
        KernelEvalConfig.from_dict({"kernel_type": KernelType.CUDA.value}).kernel_type
        is KernelType.CUDA
    )


def test_pipeline_config_uses_explicit_arch_and_supplied_configs():
    compiling = CompilingConfig(enable_default_compile=True)
    correctness = CorrectnessConfig()
    performance = PerformanceConfig(backend=PerfBackend.NCU)
    cfg = PipelineConfig(
        kernel_type=KernelType.TRITON,
        gpu_arch="sm_90",
        compiling_config=compiling,
        correctness_config=correctness,
        performance_config=performance,
    )

    assert cfg.compiling_config is compiling
    assert cfg.correctness_config is correctness
    assert cfg.performance_config is performance


def test_pipeline_config_detects_gpu_and_builds_defaults(monkeypatch):
    monkeypatch.setattr("Magpie.utils.detect_gpu", lambda: ("amd", "gfx942"))

    cfg = PipelineConfig(kernel_type=KernelType.TRITON)

    assert cfg.gpu_arch == "gfx942"
    assert isinstance(cfg.compiling_config, CompilingConfig)
    assert isinstance(cfg.correctness_config, CorrectnessConfig)
    assert cfg.performance_config.get_backend() is PerfBackend.ROCPROF_COMPUTE


@pytest.mark.parametrize(
    "detector",
    [lambda: ("amd", None), lambda: (_ for _ in ()).throw(OSError("missing tool"))],
)
def test_pipeline_config_reports_gpu_detection_failures(monkeypatch, detector):
    monkeypatch.setattr("Magpie.utils.detect_gpu", detector)

    with pytest.raises(RuntimeError, match="Please specify 'gpu_arch'"):
        PipelineConfig(kernel_type=KernelType.HIP)
