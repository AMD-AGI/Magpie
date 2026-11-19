"""CPU-only contract tests for the separately installed SGLang plugin."""

import copy
import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PLUGIN = (Path(__file__).parents[1] / "Magpie/scripts/benchmark/"
          "expert_distribution_plugin/magpie_expert_totals.py")


class Tensor:
    host_copies = 0

    def __init__(self, rows, dtype="int32", device="test-device"):
        self.rows = copy.deepcopy(rows)
        self.shape = (len(rows), len(rows[0]))
        self.dtype = dtype
        self.device = device

    def add_(self, other):
        for row, values in zip(self.rows, other.rows):
            for index, value in enumerate(values):
                row[index] += value
                if self.dtype == "int32":
                    row[index] = (row[index] + 2**31) % 2**32 - 2**31
        return self

    def zero_(self):
        for row in self.rows:
            row[:] = [0] * len(row)
        return self

    def detach(self):
        return self

    def cpu(self):
        Tensor.host_copies += 1
        return Tensor(self.rows, self.dtype, "cpu")

    def clone(self):
        Tensor.host_copies += 1
        return Tensor(self.rows, self.dtype, self.device)

    def tolist(self):
        return copy.deepcopy(self.rows)


class FakeTorch:
    Tensor = Tensor
    int32, int64 = "int32", "int64"

    @staticmethod
    def zeros(shape, *, dtype, device):
        return Tensor([[0] * shape[1] for _ in range(shape[0])], dtype, device)

    @staticmethod
    def zeros_like(value, *, dtype):
        return FakeTorch.zeros(value.shape, dtype=dtype, device=value.device)


@pytest.fixture
def plugin(monkeypatch):
    spec = importlib.util.spec_from_file_location("test_magpie_expert_totals", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MAGPIE_RECORD_EXPERT_DISTRIBUTION", "1")
    Tensor.host_copies = 0
    return module


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(mode="stat", buffer_size=1, dumps=[], hooks=[], metrics_calls=0,
                            reset_calls=0, collectives=[])

    class Accumulator:
        def __init__(self, metadata, rank):
            self._expert_location_metadata, self._rank = metadata, rank

        @staticmethod
        def get_class():
            return NativeStat

    class Mixin(Accumulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._metric_history = deque(maxlen=10)

        def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
            state.metrics_calls += 1
            values = single_pass_data["global_physical_count"].tolist()[0]
            balancedness = sum(values) / len(values) / max(values) if max(values) else 0
            outputs["mean_balancedness"] = balancedness
            self._metric_history.append(balancedness)

        def reset(self):
            state.reset_calls += 1
            self._metric_history.clear()

    class NativeStat(Mixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._native_buffer = [[0] * self._expert_location_metadata.num_physical_experts
                                   for _ in range(self._expert_location_metadata.num_layers)]

        def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
            super().append(forward_pass_id, gatherer_key, single_pass_data, outputs)
            self._native_buffer = single_pass_data["global_physical_count"].tolist()

        def reset(self):
            super().reset()
            for row in self._native_buffer:
                row[:] = [0] * len(row)

        def dump(self, output_mode):
            # Stand in for native logical-count all_reduce/utilization broadcast.
            state.collectives.append((self._rank, output_mode, copy.deepcopy(self._native_buffer)))
            if output_mode == "object":
                return {"logical_count": self._native_buffer}

    module = ModuleType("sglang.srt.eplb.expert_distribution")
    module._Accumulator = Accumulator
    module._UtilizationRateAccumulatorMixin = Mixin
    module._StatAccumulator = NativeStat
    module._SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"
    module._dump_to_file = lambda name, data: state.dumps.append((name, data))
    module.get_exec = lambda: SimpleNamespace(moe=SimpleNamespace(
        expert_distribution_recorder_mode=state.mode,
        expert_distribution_recorder_buffer_size=state.buffer_size,
    ))
    module.torch = FakeTorch
    module.state = state
    module.NativeStat = NativeStat

    class HookRegistry:
        @staticmethod
        def register(target, callback, kind):
            state.hooks.append((target, callback, kind))

    hooks = ModuleType("sglang.srt.plugins.hook_registry")
    hooks.HookRegistry = HookRegistry
    hooks.HookType = SimpleNamespace(AROUND="around")
    for name in ("sglang", "sglang.srt", "sglang.srt.eplb", "sglang.srt.plugins"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    return module


def make_accumulator(plugin, native, rank=3):
    plugin.register()
    target, select, kind = native.state.hooks[0]
    assert target == "sglang.srt.eplb.expert_distribution._Accumulator.get_class"
    assert kind == "around"
    selected = select(native._Accumulator.get_class)
    metadata = SimpleNamespace(num_layers=2, num_physical_experts=3,
                               physical_to_logical_map=Tensor([[0, 1, 2], [0, 1, 2]]))
    return selected(metadata, rank)


def test_opt_out_does_not_import_sglang(plugin, monkeypatch):
    monkeypatch.delenv("MAGPIE_RECORD_EXPERT_DISTRIBUTION")
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: pytest.fail("Imported SGLang"))
    plugin.register()
    assert not plugin._registered


def test_register_is_idempotent_and_only_selects_stat(plugin, native):
    accumulator = make_accumulator(plugin, native)
    plugin.register()
    assert len(native.state.hooks) == 1
    assert isinstance(accumulator, native._UtilizationRateAccumulatorMixin)
    assert type(accumulator) is not native.NativeStat
    select = native.state.hooks[0][1]
    for mode in ("per_pass", "per_token", "stat_approx", None):
        native.state.mode = mode
        assert select(native._Accumulator.get_class) is native.NativeStat


def test_incompatible_private_api_does_not_register(plugin, native):
    native._Accumulator.get_class = lambda: native.NativeStat
    with pytest.raises(RuntimeError, match="Unsupported SGLang"):
        plugin.register()
    assert not native.state.hooks
    assert not plugin._registered


def test_capture_is_bounded_exact_int64_and_preserves_balancedness(plugin, native):
    accumulator = make_accumulator(plugin, native)
    counts = Tensor([[2**30, 2, 3], [4, 5, 6]])
    output = {}
    accumulator.append(900, "primary", {"global_physical_count": counts}, output)
    totals = accumulator._totals
    expected_balancedness = (2**30 + 5) / 3 / 2**30
    assert output["mean_balancedness"] == pytest.approx(expected_balancedness)
    for index in range(1, 1501):
        accumulator.append(900 + index, "primary", {"global_physical_count": counts}, {})
        assert accumulator._totals is totals
    assert Tensor.host_copies == 0
    assert totals.dtype == "int64"
    assert totals.device == counts.device
    assert totals.tolist() == [[1501 * 2**30, 3002, 4503], [6004, 7505, 9006]]
    assert accumulator._forward_pass_count == 1501
    assert native.state.metrics_calls == 1501
    assert len(accumulator._metric_history) == 10
    assert not hasattr(accumulator, "_records")
    assert accumulator._native_buffer == [[0]*3]*2
    # Later gatherer reuse cannot mutate previously accumulated counts.
    counts.zero_()
    assert totals.tolist()[0][0] == 1501 * 2**30


def test_dump_marker_empty_probe_capture_and_reset(plugin, native):
    accumulator = make_accumulator(plugin, native)
    accumulator.dump("file")
    filename, empty = native.state.dumps[-1]
    assert filename.startswith("expert_distribution_recorder_") and filename.endswith("_3.pt")
    assert empty["schema_version"] == 2
    assert empty["aggregation"] == "sum_over_capture"
    assert empty["forward_pass_count"] == 0
    assert empty["first_forward_pass_id"] is empty["last_forward_pass_id"] is None
    assert empty["global_physical_count"].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert empty["global_physical_count"].dtype == "int64"
    assert empty["global_physical_count"].device == "cpu"
    counts = Tensor([[1, 2, 3], [4, 5, 6]])
    accumulator.append(100, "primary", {"global_physical_count": counts}, {})
    accumulator.append(103, "primary", {"global_physical_count": counts}, {})
    accumulator.dump("file")
    captured = native.state.dumps[-1][1]
    assert captured["rank"] == 3 and captured["gatherer_key"] == "primary"
    assert captured["forward_pass_count"] == 2
    assert captured["first_forward_pass_id"] == 100
    assert captured["last_forward_pass_id"] == 103
    assert captured["global_physical_count"].tolist() == [[2, 4, 6], [8, 10, 12]]
    assert captured["last_physical_to_logical_map"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert native.state.collectives == [(3, "object", [[0]*3]*2)] * 2
    totals = accumulator._totals
    accumulator.reset()
    assert accumulator._totals is totals
    assert totals.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert accumulator._forward_pass_count == 0
    assert accumulator._first_forward_pass_id is accumulator._last_forward_pass_id is None
    assert not accumulator._metric_history and native.state.reset_calls == 1
    # A previously dumped CPU artifact is an independent snapshot.
    assert captured["global_physical_count"].tolist()[0] == [2, 4, 6]
    accumulator.append(3, "primary", {"global_physical_count": counts}, {})
    assert accumulator._first_forward_pass_id == 3


@pytest.mark.parametrize("pass_id", [-1, True, "1"])
def test_invalid_forward_ids_rejected(plugin, native, pass_id):
    accumulator = make_accumulator(plugin, native)
    with pytest.raises(ValueError, match="forward-pass IDs"):
        accumulator.append(pass_id, "primary", {"global_physical_count": Tensor([[1]*3]*2)}, {})


def test_duplicate_id_nonprimary_invalid_shape_dtype_device_fail_closed(plugin, native):
    accumulator = make_accumulator(plugin, native)
    data = {"global_physical_count": Tensor([[1, 2, 3], [4, 5, 6]])}
    with pytest.raises(ValueError, match="primary"):
        accumulator.append(1, "secondary", data, {})
    accumulator.append(10, "primary", data, {})
    for pass_id in (10, 9):
        with pytest.raises(ValueError, match="forward-pass IDs"):
            accumulator.append(pass_id, "primary", data, {})
    for tensor in (Tensor([[1, 2]]), Tensor([[1]*3]*2, dtype="float32")):
        with pytest.raises(ValueError, match="count tensor"):
            accumulator.append(11, "primary", {"global_physical_count": tensor}, {})
    with pytest.raises(ValueError, match="device changed"):
        accumulator.append(11, "primary", {"global_physical_count": Tensor([[1]*3]*2, device="other")}, {})
    assert accumulator._forward_pass_count == 1
    with pytest.raises(ValueError, match="file dumps"):
        accumulator.dump("object")


def test_worker_totals_are_never_pooled(plugin, native):
    first = make_accumulator(plugin, native, rank=1)
    second = make_accumulator(plugin, native, rank=2)
    first.append(1, "primary", {"global_physical_count": Tensor([[1]*3]*2)}, {})
    second.append(1, "primary", {"global_physical_count": Tensor([[2]*3]*2)}, {})
    first.dump("file")
    second.dump("file")
    assert native.state.dumps[-2][1]["global_physical_count"].tolist() == [[1]*3]*2
    assert native.state.dumps[-1][1]["global_physical_count"].tolist() == [[2]*3]*2


def test_native_fallback_and_plugin_share_dump_collective_protocol(plugin, native):
    active = make_accumulator(plugin, native, rank=1)
    fallback = native.NativeStat(active._expert_location_metadata, 2)
    active.append(1, "primary", {"global_physical_count": Tensor([[7]*3]*2)}, {})
    active.dump("file")
    fallback.dump("file")
    assert [item[:2] for item in native.state.collectives] == [(1, "object"), (2, "file")]
    assert native.state.collectives[0][2] == [[0]*3]*2
    assert native.state.dumps[-1][1]["global_physical_count"].tolist() == [[7]*3]*2


@pytest.mark.parametrize("size", [-1, 0, 2])
def test_nonbounded_native_buffer_configuration_rejected(plugin, native, size):
    native.state.buffer_size = size
    with pytest.raises(ValueError, match="buffer size 1"):
        make_accumulator(plugin, native)


def test_real_cpu_torch_roundtrip_when_available(plugin, native, tmp_path):
    torch = pytest.importorskip("torch")
    native.torch = torch
    plugin.register()
    cls = native.state.hooks[0][1](native._Accumulator.get_class)
    metadata = SimpleNamespace(num_layers=2, num_physical_experts=3,
                               physical_to_logical_map=torch.tensor([[0, 1, 2], [0, 1, 2]]))
    accumulator = cls(metadata, 0)
    native._dump_to_file = lambda name, data: torch.save(data, tmp_path / name)
    counts = torch.tensor([[2**30, 2, 3], [4, 5, 6]], dtype=torch.int32)
    for pass_id in range(3):
        accumulator.append(pass_id, "primary", {"global_physical_count": counts}, {})
    accumulator.dump("file")
    saved = torch.load(next(tmp_path.glob("*.pt")), weights_only=True, map_location="cpu")
    assert saved["global_physical_count"].dtype == torch.int64
    assert saved["global_physical_count"].tolist() == [[3 * 2**30, 6, 9], [12, 15, 18]]
