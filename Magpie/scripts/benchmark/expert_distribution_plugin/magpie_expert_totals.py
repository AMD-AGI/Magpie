"""SGLang plugin accumulating capture totals without retaining forward history.

Installed independently of Magpie in the serving Python environment. SGLang's
native utilization mixin still owns balancedness metrics; totals never combine
worker streams. The host verifies the dump marker before sending a workload.
"""

import importlib
import inspect
import logging
import os
import time

logger = logging.getLogger(__name__)
_registered = False
_MODULE = "sglang.srt.eplb.expert_distribution"


def _compatible(native):
    """Fail registration before touching hooks when the private seam changes."""
    base = native._Accumulator
    mixin, stat = native._UtilizationRateAccumulatorMixin, native._StatAccumulator
    if (not isinstance(base, type) or not isinstance(mixin, type)
            or not isinstance(stat, type) or not issubclass(stat, mixin)
            or not issubclass(mixin, base)
            or not isinstance(inspect.getattr_static(base, "get_class"), staticmethod)
            or native._SINGLE_PASS_GATHERER_KEY_PRIMARY != "primary"):
        raise RuntimeError("Unsupported SGLang expert accumulator API")
    inspect.signature(base.get_class).bind()
    inspect.signature(mixin.append).bind(None, 0, "primary", {}, {})
    inspect.signature(mixin.reset).bind(None)
    inspect.signature(stat.__init__).bind(None, None, 0)
    inspect.signature(stat.dump).bind(None, output_mode="object")
    inspect.signature(stat.reset).bind(None)
    inspect.signature(native._dump_to_file).bind("capture.pt", {})
    if not callable(native.get_exec):
        raise RuntimeError("Unsupported SGLang runtime accessor")
    for name in ("Tensor", "zeros", "zeros_like", "int32", "int64"):
        if not hasattr(native.torch, name):
            raise RuntimeError("Unsupported SGLang tensor API")


def _accumulator_class(native):
    torch = native.torch

    class _TotalsAccumulator(native._StatAccumulator):
        def __init__(self, *args, **kwargs):
            if native.get_exec().moe.expert_distribution_recorder_buffer_size != 1:
                raise ValueError("Magpie totals requires recorder buffer size 1")
            super().__init__(*args, **kwargs)
            metadata = self._expert_location_metadata
            self._shape = (metadata.num_layers, metadata.num_physical_experts)
            if any(type(size) is not int or size <= 0 for size in self._shape):
                raise ValueError("Invalid expert count dimensions")
            self._totals = None
            self._forward_pass_count = 0
            self._first_forward_pass_id = self._last_forward_pass_id = None

        def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
            if gatherer_key != "primary":
                raise ValueError("Magpie totals supports only the primary gatherer")
            if (type(forward_pass_id) is not int or forward_pass_id < 0
                    or (self._last_forward_pass_id is not None
                        and forward_pass_id <= self._last_forward_pass_id)):
                raise ValueError("Expected increasing nonnegative forward-pass IDs")
            counts = single_pass_data["global_physical_count"]
            if (not isinstance(counts, torch.Tensor) or tuple(counts.shape) != self._shape
                    or counts.dtype not in (torch.int32, torch.int64)):
                raise ValueError("Invalid physical-expert count tensor")
            if self._totals is None:
                self._totals = torch.zeros_like(counts, dtype=torch.int64)
            elif counts.device != self._totals.device:
                raise ValueError("Physical-expert count device changed during capture")
            self._totals.add_(counts)
            if self._first_forward_pass_id is None:
                self._first_forward_pass_id = forward_pass_id
            self._last_forward_pass_id = forward_pass_id
            self._forward_pass_count += 1
            # Preserve native balancedness reporting and its bounded metric history.
            # Bypass Stat.append: its one-slot ring stays zero, rather than
            # retaining actual counts which its dump would reduce over workers.
            return native._UtilizationRateAccumulatorMixin.append(
                self, forward_pass_id, gatherer_key, single_pass_data, outputs,
            )

        def reset(self):
            super().reset()
            if self._totals is not None:
                self._totals.zero_()
            self._forward_pass_count = 0
            self._first_forward_pass_id = self._last_forward_pass_id = None

        def dump(self, output_mode):
            if output_mode != "file":
                raise ValueError("Magpie totals supports file dumps only")
            # Match native Stat.dump collectives even if another worker failed
            # plugin activation. Only the unused zero ring participates; our
            # actual capture totals below are never reduced across workers.
            super().dump(output_mode="object")
            totals = self._totals
            if totals is None:
                # An empty dump proves plugin activation before the first request.
                totals = torch.zeros(self._shape, dtype=torch.int64, device="cpu")
            output = {
                "schema_version": 2, "aggregation": "sum_over_capture",
                "rank": self._rank, "gatherer_key": "primary",
                "forward_pass_count": self._forward_pass_count,
                "first_forward_pass_id": self._first_forward_pass_id,
                "last_forward_pass_id": self._last_forward_pass_id,
                "global_physical_count": totals.detach().cpu().clone(),
                "last_physical_to_logical_map": (
                    self._expert_location_metadata.physical_to_logical_map.detach().cpu().clone()
                ),
            }
            native._dump_to_file(
                f"expert_distribution_recorder_{time.time_ns()}_{os.getpid()}_{self._rank}.pt",
                output,
            )

    return _TotalsAccumulator


def register():
    """Entry point invoked by SGLang's general-plugin loader in every worker."""
    global _registered
    if os.environ.get("MAGPIE_RECORD_EXPERT_DISTRIBUTION") != "1" or _registered:
        return
    native = importlib.import_module(_MODULE)
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _compatible(native)
    totals_class = _accumulator_class(native)

    def select(original, *args, **kwargs):
        if native.get_exec().moe.expert_distribution_recorder_mode == "stat":
            return totals_class
        return original(*args, **kwargs)

    HookRegistry.register(f"{_MODULE}._Accumulator.get_class", select, HookType.AROUND)
    _registered = True
    logger.info("Registered Magpie bounded expert-count capture plugin")
