from types import SimpleNamespace

import pytest

from Magpie.config import KernelEvalConfig, KernelType
from Magpie.core.task import ModeConfig, ModeType, Task


@pytest.fixture
def kernel_config():
    return KernelEvalConfig(
        kernel_id="unit-kernel",
        kernel_type=KernelType.HIP,
        source_file_path=["kernel.hip"],
        testcase_command=["./test"],
    )


@pytest.fixture
def task_factory(kernel_config):
    def make(mode=ModeType.ANALYZE, **mode_overrides):
        values = {"mode_type": mode, "gpu_arch": "gfx942"}
        values.update(mode_overrides)
        return Task(
            task_id=f"task-{mode.value}",
            kernel_configs=[] if mode is ModeType.BENCHMARK else [kernel_config],
            mode_config=ModeConfig(**values),
        )

    return make


@pytest.fixture
def ray_config(tmp_path):
    return SimpleNamespace(
        cluster_address="auto",
        install_magpie=False,
        magpie_install_path=None,
        pip_packages=[],
        hf_cache_dir=str(tmp_path / "hf"),
        multi_node=False,
        total_num_gpus=1,
        env_vars={},
        entrypoint_num_cpus=2,
        to_dict=lambda: {"cluster_address": "auto"},
    )
