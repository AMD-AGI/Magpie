import json
import subprocess
from types import SimpleNamespace

import pytest

from Magpie.config import (
    AccordoConfig,
    CompilingConfig,
    CorrectnessBackend,
    CorrectnessConfig,
    EvalMode,
    KernelEvalConfig,
    KernelType,
    PerformanceConfig,
    PipelineConfig,
)
from Magpie.eval.compiling import Compiling
from Magpie.eval.correctness import Correctness


def pipeline(mode=EvalMode.ANALYZE, kernel_type=KernelType.HIP, correctness=None):
    return PipelineConfig(
        mode=mode,
        kernel_type=kernel_type,
        gpu_arch="gfx942" if kernel_type is not KernelType.CUDA else "sm_90",
        compiling_config=CompilingConfig(),
        correctness_config=correctness or CorrectnessConfig(),
        performance_config=PerformanceConfig(enabled=False),
    )


def test_compiling_skips_jit_and_disabled_default():
    compiler = Compiling(pipeline())
    assert compiler.run(KernelEvalConfig(kernel_type=KernelType.PYTORCH)) is None
    assert compiler.run(KernelEvalConfig(kernel_type=KernelType.TRITON)) is None
    assert compiler.run(KernelEvalConfig(kernel_type=KernelType.HIP)) is None


def test_compiling_custom_commands_success_failure_and_exception(monkeypatch):
    compiler = Compiling(pipeline())
    cfg = KernelEvalConfig(
        compiling_command=[["prepare"], ["build"]], working_dir="/work"
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("Magpie.eval.compiling.subprocess.run", run)
    assert compiler.run(cfg).success is True
    assert [call[0] for call in calls] == [["prepare"], ["build"]]
    monkeypatch.setattr(
        "Magpie.eval.compiling.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 2, stdout="failed"),
    )
    result = compiler.run(cfg)
    assert result.success is False
    assert "1/2 failed" in result.errors
    monkeypatch.setattr(
        "Magpie.eval.compiling.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("spawn")),
    )
    assert compiler.run(cfg).errors == "spawn"
    with pytest.raises(ValueError, match="No compiling"):
        compiler._compile_with_command(KernelEvalConfig())


def test_default_hip_compilation_paths(monkeypatch):
    cfg = pipeline()
    cfg.compiling_config.enable_default_compile = True
    compiler = Compiling(cfg)
    kernel = KernelEvalConfig(kernel_type=KernelType.HIP, source_file_path=["a.hip"])
    monkeypatch.setattr("Magpie.eval.compiling.shutil.which", lambda _: None)
    assert "hipcc not found" in compiler.run(kernel).errors
    monkeypatch.setattr("Magpie.eval.compiling.shutil.which", lambda _: "/bin/hipcc")
    monkeypatch.setattr(
        "Magpie.eval.compiling.compile_hip", lambda **kwargs: ("kernel.out", None, None)
    )
    result = compiler.run(kernel)
    assert result.success is True
    assert result.output_file_path == "kernel.out"
    monkeypatch.setattr(
        "Magpie.eval.compiling.compile_hip",
        lambda **kwargs: (None, None, "compile error"),
    )
    assert compiler.run(kernel).errors == "compile error"
    monkeypatch.setattr(
        "Magpie.eval.compiling.compile_hip",
        lambda **kwargs: (_ for _ in ()).throw(OSError("hip failure")),
    )
    assert compiler.run(kernel).errors == "hip failure"


def test_default_cuda_compilation_paths(monkeypatch):
    cfg = pipeline(kernel_type=KernelType.CUDA)
    cfg.compiling_config.enable_default_compile = True
    compiler = Compiling(cfg)
    kernel = KernelEvalConfig(kernel_type=KernelType.CUDA, source_file_path=["a.cu"])
    monkeypatch.setattr("Magpie.eval.compiling.shutil.which", lambda _: None)
    assert "nvcc not found" in compiler.run(kernel).errors
    monkeypatch.setattr("Magpie.eval.compiling.shutil.which", lambda _: "/bin/nvcc")
    monkeypatch.setattr(
        "Magpie.eval.compiling.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0),
    )
    result = compiler.run(kernel)
    assert result.success is True
    assert result.output_file_path.endswith("a.out")
    monkeypatch.setattr(
        "Magpie.eval.compiling.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stderr="nvcc bad"),
    )
    assert compiler.run(kernel).errors == "nvcc bad"
    monkeypatch.setattr(
        "Magpie.eval.compiling.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("cuda failure")),
    )
    assert compiler.run(kernel).errors == "cuda failure"


def test_correctness_analyze_and_testcase_paths(monkeypatch):
    corr_cfg = CorrectnessConfig(iteration_count=2)
    checker = Correctness(pipeline(correctness=corr_cfg))
    assert "requires testcase" in checker.run(None, KernelEvalConfig()).errors
    kernel = KernelEvalConfig(testcase_command=[["one"], ["two"]])
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="PASS"),
    )
    assert checker.run(None, kernel).success is True
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 2, stderr="bad"),
    )
    result = checker.run(None, kernel)
    assert result.success is False
    assert "1/2 failed" in result.errors
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("run failed")),
    )
    assert checker.run(None, kernel).errors == "run failed"


def test_correctness_compare_dispatch(monkeypatch):
    checker = Correctness(pipeline(mode=EvalMode.COMPARE))
    hip = KernelEvalConfig(kernel_type=KernelType.HIP)
    assert "requires testcase" in checker.run(None, hip).errors
    triton = KernelEvalConfig(kernel_type=KernelType.TRITON, source_file_path=["x.py"])
    monkeypatch.setattr(
        checker,
        "_run_triton_comparison",
        lambda *args: SimpleNamespace(success=True),
    )
    assert checker.run(None, triton).success is True
    pytorch = KernelEvalConfig(
        kernel_type=KernelType.PYTORCH, source_file_path=["x.py"]
    )
    monkeypatch.setattr(
        checker,
        "_run_pytorch_comparison",
        lambda *args: SimpleNamespace(success=True),
    )
    assert checker.run(None, pytorch).success is True
    checker.pipeline_cfg.mode = object()
    assert "Unknown evaluation mode" in checker.run(None, hip).errors
    monkeypatch.setattr(
        checker,
        "_run_analyze_mode",
        lambda *args: (_ for _ in ()).throw(ValueError("broken")),
    )
    checker.pipeline_cfg.mode = EvalMode.ANALYZE
    assert checker.run(None, hip).errors == "broken"


@pytest.mark.parametrize(
    ("completed", "expected_success", "error"),
    [
        (subprocess.CompletedProcess([], 0, stdout="PASS"), True, None),
        (
            subprocess.CompletedProcess([], 0, stdout="error only"),
            False,
            "reported failure",
        ),
        (subprocess.CompletedProcess([], 1, stderr="crash"), False, "kernel failed"),
    ],
)
def test_triton_comparison_results(
    tmp_path, monkeypatch, completed, expected_success, error
):
    source = tmp_path / "kernel.py"
    source.write_text("print('PASS')")
    checker = Correctness(
        pipeline(mode=EvalMode.COMPARE, kernel_type=KernelType.TRITON)
    )
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run", lambda *a, **k: completed
    )
    result = checker._run_triton_comparison(
        None,
        KernelEvalConfig(kernel_type=KernelType.TRITON, source_file_path=[str(source)]),
    )
    assert result.success is expected_success
    if error:
        assert error in result.errors.lower()
    assert (
        "No source"
        in checker._run_triton_comparison(
            None, KernelEvalConfig(kernel_type=KernelType.TRITON)
        ).errors
    )


def test_triton_comparison_timeout_and_exception(tmp_path, monkeypatch):
    source = tmp_path / "kernel.py"
    source.touch()
    checker = Correctness(
        pipeline(mode=EvalMode.COMPARE, kernel_type=KernelType.TRITON)
    )
    kernel = KernelEvalConfig(
        kernel_type=KernelType.TRITON, source_file_path=[str(source)]
    )
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("python", 120)),
    )
    assert "timed out" in checker._run_triton_comparison(None, kernel).errors
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("spawn")),
    )
    assert checker._run_triton_comparison(None, kernel).errors == "spawn"


def accordo_checker(config):
    corr = CorrectnessConfig(backend=CorrectnessBackend.ACCORDO, accordo_config=config)
    return Correctness(pipeline(correctness=corr))


def test_accordo_validation_config_and_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Magpie.eval.correctness.shutil.which", lambda _: "/bin/accordo"
    )
    checker = accordo_checker(
        AccordoConfig(
            kernel_name="gemm",
            reference_binary="ref",
            optimized_binary="opt",
            equal_nan=True,
            kernel_args=[("x", "ptr")],
            workspace_path=str(tmp_path),
        )
    )
    output = {"is_valid": True, "num_arrays_validated": 2, "matched_arrays": {"x": {}}}
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="log\n" + json.dumps(output))

    monkeypatch.setattr("Magpie.eval.correctness.subprocess.run", run)
    result = checker.run(None, KernelEvalConfig())
    assert result.success is True
    assert len(result.metrics) == 2
    assert "--equal-nan" in calls[0][0]
    assert "--kernel-args" in calls[0][0]


def test_accordo_validation_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("Magpie.eval.correctness.shutil.which", lambda _: None)
    assert (
        "not found"
        in accordo_checker(AccordoConfig()).run(None, KernelEvalConfig()).errors
    )
    monkeypatch.setattr(
        "Magpie.eval.correctness.shutil.which", lambda _: "/bin/accordo"
    )
    assert (
        "kernel_name"
        in accordo_checker(AccordoConfig()).run(None, KernelEvalConfig()).errors
    )
    cfg = AccordoConfig(kernel_name="gemm")
    assert "both" in accordo_checker(cfg).run(None, KernelEvalConfig()).errors
    cfg.reference_binary = "ref"
    cfg.optimized_binary = "opt"
    checker = accordo_checker(cfg)

    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 2, stdout='{"error":"invalid"}'
        ),
    )
    assert "invalid" in checker.run(None, KernelEvalConfig()).errors
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="not-json"),
    )
    assert "invalid JSON" in checker.run(None, KernelEvalConfig()).errors
    mismatch = {
        "is_valid": False,
        "num_mismatches": 1,
        "error_message": "different",
        "mismatches": [
            {"arg_name": "x", "dispatch_index": 2, "max_difference": 1.0},
            {"arg_name": "y", "max_difference": 2.0},
        ],
    }
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout=json.dumps(mismatch)),
    )
    result = checker.run(None, KernelEvalConfig())
    assert result.success is False
    assert len(result.metrics) == 3


def test_accordo_timeout_and_missing_binary(monkeypatch):
    monkeypatch.setattr(
        "Magpie.eval.correctness.shutil.which", lambda _: "/bin/accordo"
    )
    cfg = AccordoConfig(
        kernel_name="gemm", reference_binary="ref", optimized_binary="opt"
    )
    checker = accordo_checker(cfg)
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("accordo", 1)),
    )
    assert "timed out" in checker.run(None, KernelEvalConfig()).errors
    monkeypatch.setattr(
        "Magpie.eval.correctness.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert "not found" in checker.run(None, KernelEvalConfig()).errors


def test_dynamic_module_loading(tmp_path):
    checker = Correctness(pipeline())
    module_file = tmp_path / "demo.py"
    module_file.write_text("VALUE = 42\n")
    assert checker._load_module(str(module_file)).VALUE == 42
    with pytest.raises(FileNotFoundError):
        checker._load_module(str(tmp_path / "missing.py"))


class FakeTensor:
    def __init__(self, value=1):
        self.value = value
        self.on_cuda = False

    def cuda(self):
        self.on_cuda = True
        return self


class FakeBool:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeTorch:
    Tensor = FakeTensor

    def __init__(self, nan=False, inf=False, cuda=False):
        self.nan = nan
        self.inf = inf
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def no_grad(self):
        return NoGrad()

    def isnan(self, _value):
        return FakeBool(self.nan)

    def isinf(self, _value):
        return FakeBool(self.inf)


class FakeModel:
    def __init__(self, *init_inputs):
        self.init_inputs = init_inputs
        self.on_cuda = False

    def cuda(self):
        self.on_cuda = True
        return self

    def eval(self):
        return self

    def __call__(self, *_inputs):
        return FakeTensor()


def setup_fake_torch(monkeypatch, fake):
    monkeypatch.setattr("Magpie.eval.correctness.HAS_TORCH", True)
    monkeypatch.setattr("Magpie.eval.correctness.torch", fake)
    monkeypatch.setattr("Magpie.eval.correctness._ensure_torch", lambda: None)


def test_pytorch_comparison_success_nan_and_inf(monkeypatch):
    checker = Correctness(
        pipeline(mode=EvalMode.COMPARE, kernel_type=KernelType.PYTORCH)
    )
    kernel = KernelEvalConfig(
        kernel_type=KernelType.PYTORCH, source_file_path=["model.py"]
    )
    module = SimpleNamespace(
        get_inputs=lambda: FakeTensor(),
        get_init_inputs=lambda: [4],
        Model=FakeModel,
    )
    monkeypatch.setattr(checker, "_load_module", lambda _: module)
    fake = FakeTorch(cuda=True)
    setup_fake_torch(monkeypatch, fake)
    assert checker._run_pytorch_comparison(None, kernel).success is True
    assert fake.seeds

    fake.nan = True
    result = checker._run_pytorch_comparison(None, kernel)
    assert result.success is False
    assert "NaN" in result.errors
    fake.nan = False
    fake.inf = True
    result = checker._run_pytorch_comparison(None, kernel)
    assert result.success is False
    assert "Inf" in result.errors


def test_pytorch_comparison_validation_and_exception(monkeypatch):
    checker = Correctness(
        pipeline(mode=EvalMode.COMPARE, kernel_type=KernelType.PYTORCH)
    )
    kernel = KernelEvalConfig(kernel_type=KernelType.PYTORCH)
    monkeypatch.setattr("Magpie.eval.correctness.HAS_TORCH", False)
    monkeypatch.setattr("Magpie.eval.correctness.torch", None)
    monkeypatch.setattr("Magpie.eval.correctness._ensure_torch", lambda: None)
    assert "not available" in checker._run_pytorch_comparison(None, kernel).errors
    setup_fake_torch(monkeypatch, FakeTorch())
    assert "No source" in checker._run_pytorch_comparison(None, kernel).errors
    kernel.source_file_path = ["model.py"]
    monkeypatch.setattr(
        checker, "_load_module", lambda _: SimpleNamespace(Model=FakeModel)
    )
    assert "get_inputs" in checker._run_pytorch_comparison(None, kernel).errors
    monkeypatch.setattr(
        checker,
        "_load_module",
        lambda _: SimpleNamespace(get_inputs=lambda: [], Model=None),
    )
    assert "Model class" in checker._run_pytorch_comparison(None, kernel).errors
    monkeypatch.setattr(
        checker,
        "_load_module",
        lambda _: (_ for _ in ()).throw(OSError("load failed")),
    )
    assert checker._run_pytorch_comparison(None, kernel).errors == "load failed"
