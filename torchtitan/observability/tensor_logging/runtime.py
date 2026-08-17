# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import cast, NamedTuple, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.activation_checkpoint import (
    _is_activation_checkpoint_recompute,
)

from .statistics import accumulate_tensor_statistics, StatisticBuffers


MetricSource: TypeAlias = nn.Module | nn.Parameter
ReducedBuffers: TypeAlias = tuple[torch.Tensor, torch.Tensor]

_REGISTERED_NAMES_ATTR = "_tensor_logging_registered_names"
_STATISTIC_BUFFER_SLOTS_ATTR = "_tensor_logging_statistic_buffer_slots"
_active_state: TensorLoggingState | None = None
_enabled = False
_include_tensor_logging_calls = False


class StatisticBufferSlot(NamedTuple):
    """Storage assigned to one public metric name during `init`."""

    sum_statistics: torch.Tensor
    maximum: torch.Tensor
    slot_index: torch.Tensor


def _registered_tensor_names(metric_source: MetricSource) -> list[str] | None:
    return cast(
        list[str] | None,
        metric_source.__dict__.get(_REGISTERED_NAMES_ATTR),
    )


def _statistic_buffer_slots(
    metric_source: MetricSource,
) -> dict[str, StatisticBufferSlot]:
    # Read this exact source without following `nn.Module` proxy lookup.
    slots = metric_source.__dict__.get(_STATISTIC_BUFFER_SLOTS_ATTR)
    if slots is None:
        raise KeyError(
            f"no initialized tensor metrics on {type(metric_source).__name__}"
        )
    return cast(dict[str, StatisticBufferSlot], slots)


def _slot(
    slots: dict[str, StatisticBufferSlot],
    registered_name: str,
) -> StatisticBufferSlot:
    try:
        return slots[registered_name]
    except KeyError:
        raise KeyError(f"unregistered tensor metric: {registered_name}") from None


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def register(
    metric_source: MetricSource,
    registered_names: Sequence[str],
) -> None:
    """Register source-local tensor names before logging is initialized.

    Args:
        metric_source: Module or parameter passed again at the logging callsite.
        registered_names: Local names accepted by `log_stats` for this source.

    Example:

        register(attention, ["xq"])
        log_stats(attention, xq=xq)
    """

    if _active_state is not None:
        raise RuntimeError("register tensor names before tensor_logging.init()")
    existing_names = _registered_tensor_names(metric_source)
    if existing_names is None:
        existing_names = []
        setattr(metric_source, _REGISTERED_NAMES_ATTR, existing_names)
    existing_names.extend(registered_names)


def register_fwd_bwd(
    metric_source: nn.Module,
    registered_names: Sequence[str],
) -> None:
    """Register paired `.x` and `.dx` statistics for each tensor name.

    Args:
        metric_source: Module passed again to `log_fwd_bwd_stats`.
        registered_names: Base names expanded to `<name>.x` and `<name>.dx`.

    Example:

        register_fwd_bwd(attention, ["xq", "head_out"])
    """

    register(metric_source, [f"{name}.x" for name in registered_names])
    register(metric_source, [f"{name}.dx" for name in registered_names])


@contextlib.contextmanager
def set_enabled(value: bool) -> Iterator[None]:
    """Choose whether `log_stats` calls record values in this scope.

    Example:

        with tensor_logging.set_enabled(is_tensor_log_step):
            loss = train_step(batch)
    """

    global _enabled
    previous = _enabled
    _enabled = value
    state = _active_state
    if state is not None:
        with spmd.no_typecheck():
            state.statistic_buffers.enabled.fill_(value)
    try:
        yield
    finally:
        _enabled = previous
        if state is not None:
            with spmd.no_typecheck():
                state.statistic_buffers.enabled.fill_(previous)


@contextlib.contextmanager
def _include_tensor_logging_calls_for_capture() -> Iterator[None]:
    """Include device-gated tensor-logging calls in CUDA-graph capture.

    CUDA graph warmup and capture can occur on a non-logging step. The captured
    graph still needs the recording ops; ``buffers.enabled`` decides at replay
    time whether they mutate statistics.
    """

    global _include_tensor_logging_calls
    previous = _include_tensor_logging_calls
    _include_tensor_logging_calls = True
    try:
        yield
    finally:
        _include_tensor_logging_calls = previous


def _wrap_to_include_tensor_logging_calls(
    forward_backward: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Keep device-gated tensor-logging calls in CUDA-graph capture."""

    @functools.wraps(forward_backward)
    def wrapped(*args, **kwargs):
        with _include_tensor_logging_calls_for_capture():
            return forward_backward(*args, **kwargs)

    return wrapped


def is_enabled() -> bool:
    """Return whether the current trainer scope records tensor statistics."""

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled


def should_compute_logged_values() -> bool:
    """Return whether callsites must compute the optional values they log.

    This is also true during compile and CUDA-graph capture on an unselected
    step so cadence changes reuse one graph; the device flag gates each replay.

    Example:

        if should_compute_logged_values():
            log_stats(router, entropy=compute_router_entropy(scores))
    """

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled or _include_tensor_logging_calls


def _is_installed() -> bool:
    """Return whether this run initialized tensor logging."""

    return _active_state is not None


def _infer_device(model_parts: Sequence[nn.Module]) -> torch.device:
    for model_part in model_parts:
        tensor = next(model_part.parameters(), None)
        if tensor is None:
            tensor = next(model_part.buffers(), None)
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def _public_prefix_by_metric_source(
    model_parts: Sequence[nn.Module],
    public_prefix_by_model_part: Mapping[nn.Module, str],
) -> dict[MetricSource, str]:
    """Map local modules and parameters to their public metric prefix.

    Example:

        public_prefix_by_model_part = {first_part: "", last_part: ""}
        # first_part.layers[0] -> "layers.0"
        # last_part.layers[7]  -> "layers.7"
    """

    names: dict[MetricSource, str] = {}
    default_prefixes = (
        [""]
        if len(model_parts) == 1
        else [f"model_parts.{i}" for i in range(len(model_parts))]
    )
    # A split model defaults to local prefixes unless PP supplies global overrides.
    for model_part, default_prefix in zip(model_parts, default_prefixes, strict=True):
        model_part_prefix = public_prefix_by_model_part.get(model_part, default_prefix)
        for module_name, module in model_part.named_modules():
            module_name = ".".join(
                part
                for part in module_name.split(".")
                if part != "_checkpoint_wrapped_module"
            )
            name = ".".join(part for part in (model_part_prefix, module_name) if part)
            names[module] = name
            for parameter_name, parameter in module.named_parameters(recurse=False):
                names[parameter] = ".".join(
                    part for part in (name, parameter_name) if part
                )
    return names


def _discover_registered_metrics(
    model_parts: Sequence[nn.Module],
    public_prefix_by_model_part: Mapping[nn.Module, str],
) -> list[tuple[MetricSource, str, str]]:
    """Resolve local registrations to public metric names.

    Each result is `(metric_source, registered_name, public_metric_name)`.
    Explicit PP roots may share a stage-conditional top-level boundary such as
    `input.x`; every other duplicate public name is rejected.

    Example:

        # model.layers[3].attention registered "xq.x"
        # -> (attention, "xq.x", "layers.3.attention.xq.x")

        # `{part: "" for part in parts}` keeps PP names as global model paths.
    """

    public_prefix_by_source = _public_prefix_by_metric_source(
        model_parts,
        public_prefix_by_model_part,
    )
    source_by_public_name: dict[str, MetricSource] = {}
    registered_metrics: list[tuple[MetricSource, str, str]] = []

    for metric_source, source_prefix in public_prefix_by_source.items():
        for registered_name in _registered_tensor_names(metric_source) or ():
            public_metric_name = ".".join(
                part for part in (source_prefix, registered_name) if part
            )
            previous_source = source_by_public_name.get(public_metric_name)
            if previous_source is not None:
                previous_prefix = public_prefix_by_model_part.get(previous_source)
                current_prefix = public_prefix_by_model_part.get(metric_source)
                same_explicit_pp_prefix = (
                    previous_prefix is not None
                    and current_prefix is not None
                    and previous_prefix == current_prefix
                )
                if previous_source is metric_source or not same_explicit_pp_prefix:
                    raise ValueError(
                        f"tensor metric registered twice: {public_metric_name}"
                    )
            else:
                source_by_public_name[public_metric_name] = metric_source
            registered_metrics.append(
                (metric_source, registered_name, public_metric_name)
            )
    return registered_metrics


def _gather_global_metric_names(local_names: set[str]) -> list[str]:
    """Give every PP rank the same sorted metric names.

    Example:

        rank 0: {"layers.0.attn.xq.x"}
        rank 1: {"layers.7.attn.xq.x"}
        result on both ranks:
            ["layers.0.attn.xq.x", "layers.7.attn.xq.x"]
    """

    if not dist.is_initialized():
        return sorted(local_names)
    names_by_rank: list[set[str]] = [set() for _ in range(dist.get_world_size())]
    dist.all_gather_object(names_by_rank, local_names)
    return sorted(set().union(*names_by_rank))


def _derive_metrics_from_statistics(
    public_metric_name: str,
    sum_statistics: Sequence[float],
    maximum: float,
) -> dict[str, int | float]:
    """Derive one metric's scalars from its reduced sufficient statistics.

    Example:

        # Recorded tensor: [0, 1, -2, 3]
        sum_statistics = [4, 0, 1, 1, 6, 14, 98]
        maximum = 3

        # Adds numel=4, zero_frac=0.25, abs_mean=1.5,
        # square_mean=3.5, kurtosis=-1, and abs_max=3.
    """

    metrics: dict[str, int | float] = {}
    numel = int(sum_statistics[0])
    nonfinite_count = int(sum_statistics[1])
    zero_count = int(sum_statistics[2])
    observation_count = int(sum_statistics[3])
    if observation_count == 0:
        return metrics
    finite_count = numel - nonfinite_count
    prefix = f"{public_metric_name}."
    metrics[prefix + "numel"] = numel
    metrics[prefix + "nonfinite_count"] = nonfinite_count
    metrics[prefix + "observation_count"] = observation_count
    if finite_count == 0:
        return metrics

    absolute_sum, square_sum, fourth_moment_sum = sum_statistics[4:]
    metrics[prefix + "zero_count"] = zero_count
    metrics[prefix + "zero_frac"] = zero_count / finite_count
    if math.isfinite(absolute_sum):
        metrics[prefix + "abs_sum"] = absolute_sum
        metrics[prefix + "abs_mean"] = absolute_sum / finite_count
    if math.isfinite(square_sum):
        square_mean = square_sum / finite_count
        metrics[prefix + "square_mean"] = square_mean
        metrics[prefix + "rms"] = square_mean**0.5
        if square_mean > 0 and math.isfinite(fourth_moment_sum):
            kurtosis = fourth_moment_sum / finite_count / square_mean**2 - 3
            if math.isfinite(kurtosis):
                metrics[prefix + "kurtosis"] = kurtosis
    if math.isfinite(maximum):
        metrics[prefix + "abs_max"] = maximum
    return metrics


class TensorLoggingState:
    """Fixed statistics buffers for one model's registered tensor names.

    Example:

        register(module, names)       construction
                  |
        init(model_parts)             fixed global slot order
                  |
        log_stats(module, values)     update assigned slots
                  |
        collect()                     WORLD SUM + MAX -> scalar dict
    """

    def __init__(
        self,
        model_parts: Sequence[nn.Module],
        *,
        device: torch.device | None = None,
        publish_filter_regex: str = "",
        public_prefix_by_model_part: Mapping[nn.Module, str] | None = None,
    ) -> None:
        self._metric_sources: list[MetricSource] = []
        self._publish_filter = (
            re.compile(publish_filter_regex) if publish_filter_regex else None
        )

        public_prefix_by_model_part = public_prefix_by_model_part or {}
        registered_metrics = _discover_registered_metrics(
            model_parts,
            public_prefix_by_model_part,
        )
        # PP stages own different modules but must reduce one shared slot order.
        self.public_metric_names = _gather_global_metric_names(
            {public_name for _, _, public_name in registered_metrics}
        )
        slot_index_by_public_name = {
            public_name: slot_index
            for slot_index, public_name in enumerate(self.public_metric_names)
        }

        self.statistic_buffers = StatisticBuffers(
            len(self.public_metric_names),
            device=device or _infer_device(model_parts),
        )
        # The backward custom op reads these constants with `.item()`; keeping
        # them on CPU avoids a device synchronization during capture.
        self._slot_indices = torch.arange(
            len(self.public_metric_names),
            dtype=torch.int64,
        )
        for metric_source, registered_name, public_name in registered_metrics:
            slots = metric_source.__dict__.get(_STATISTIC_BUFFER_SLOTS_ATTR)
            if slots is None:
                slots = {}
                setattr(metric_source, _STATISTIC_BUFFER_SLOTS_ATTR, slots)
                self._metric_sources.append(metric_source)
            slot_index = slot_index_by_public_name[public_name]
            slots[registered_name] = StatisticBufferSlot(
                sum_statistics=self.statistic_buffers.sum_statistics[slot_index],
                maximum=self.statistic_buffers.maxima[slot_index],
                slot_index=self._slot_indices[slot_index],
            )
        # Module state makes the slabs visible to compile and Graph Trainer.
        self._buffer_owner = model_parts[0]
        self._buffer_owner.add_module(
            "_tensor_logging_state",
            self.statistic_buffers,
        )
        self._closed = False

    def _accumulate(
        self,
        slot: StatisticBufferSlot,
        value: torch.Tensor,
    ) -> None:
        with spmd.no_typecheck():
            accumulate_tensor_statistics(
                _local_tensor(value),
                slot.sum_statistics,
                slot.maximum,
                self.statistic_buffers.enabled,
            )

    def _clear_buffers(self) -> None:
        self.statistic_buffers.clear()

    def snapshot_unreduced_statistics(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Clone unreduced slots for focused correctness tests."""

        return {
            metric_name: {
                "counts": self.statistic_buffers.sum_statistics[index, :4]
                .detach()
                .cpu()
                .clone(),
                "sums": self.statistic_buffers.sum_statistics[index, 4:]
                .detach()
                .cpu()
                .clone(),
                "maximum": self.statistic_buffers.maxima[index].detach().cpu().clone(),
            }
            for index, metric_name in enumerate(self.public_metric_names)
        }

    def _reduce_buffers(self) -> ReducedBuffers:
        """Clone and reduce every registered key in two packed WORLD slabs."""

        sum_statistics = self.statistic_buffers.sum_statistics.clone()
        maxima = self.statistic_buffers.maxima.clone()
        if dist.is_initialized():
            dist.all_reduce(sum_statistics, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        return sum_statistics, maxima

    def _buffers_to_metrics(
        self,
        reduced_buffers: ReducedBuffers,
    ) -> dict[str, int | float]:
        """Derive, aggregate, and filter values from reduced buffers."""

        # One device-to-host copy per buffer avoids synchronizing per metric.
        sum_statistics, maxima = (buffer.detach().cpu() for buffer in reduced_buffers)
        sum_statistics_by_slot = cast(list[list[float]], sum_statistics.tolist())
        maxima_by_slot = cast(list[float], maxima.tolist())
        metrics: dict[str, int | float] = {}
        for slot_index, public_metric_name in enumerate(self.public_metric_names):
            metrics.update(
                _derive_metrics_from_statistics(
                    public_metric_name,
                    sum_statistics_by_slot[slot_index],
                    maxima_by_slot[slot_index],
                )
            )
        # Filtering controls what gets logged, not what gets computed.
        if self._publish_filter is not None:
            metrics = {
                name: value
                for name, value in metrics.items()
                if self._publish_filter.search(name)
            }
        return metrics

    def collect(self) -> dict[str, int | float]:
        """Reduce, derive, and reset the statistics from the current window."""

        reduced_buffers = self._reduce_buffers()
        metrics = self._buffers_to_metrics(reduced_buffers)
        self._clear_buffers()
        return metrics

    def close(self) -> None:
        global _active_state
        if self._closed:
            return
        self._closed = True
        for metric_source in self._metric_sources:
            metric_source.__dict__.pop(_STATISTIC_BUFFER_SLOTS_ATTR, None)
        self._metric_sources.clear()
        self._buffer_owner._modules.pop("_tensor_logging_state")
        if _active_state is self:
            _active_state = None


def init(
    model_parts: nn.Module | Iterable[nn.Module],
    *,
    device: torch.device | None = None,
    publish_filter_regex: str = "",
    public_prefix_by_model_part: Mapping[nn.Module, str] | None = None,
) -> TensorLoggingState:
    """Assign registered tensor names fixed buffer slots and activate logging.

    Call once after model parallelization and optimizer construction. Later
    registration is rejected because compile and CUDA graphs require static storage.

    Args:
        model_parts: Model or PP model parts containing construction-time registrations.
        device: Device for fixed statistics buffers; inferred when omitted.
        publish_filter_regex: Allowlist over derived public metric names.
        public_prefix_by_model_part: Global metric path prefix for each local PP part.

    Example:

        register_fwd_bwd(model.layers[0], ["residual"])
        device = next(model.parameters()).device
        state = init(model, device=device)
    """

    global _active_state
    if _active_state is not None:
        raise RuntimeError("tensor logging already has active state")
    model_part_list = (
        [model_parts] if isinstance(model_parts, nn.Module) else list(model_parts)
    )
    state = TensorLoggingState(
        model_part_list,
        device=device,
        publish_filter_regex=publish_filter_regex,
        public_prefix_by_model_part=public_prefix_by_model_part,
    )
    _active_state = state
    return state


def _state() -> TensorLoggingState:
    if _active_state is None:
        raise RuntimeError("tensor logging is enabled before init()")
    return _active_state


@torch.library.custom_op(
    "torchtitan::record_tensor_statistics_cotangent",
    mutates_args=(),
)
def _record_tensor_statistics_cotangent(
    value: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    """Record one cotangent without exposing mutable buffers to autograd."""

    state = _state()
    index = int(slot_index.item())
    buffers = (
        state.statistic_buffers.sum_statistics[index],
        state.statistic_buffers.maxima[index],
        state.statistic_buffers.enabled,
    )
    accumulate_tensor_statistics(value, *buffers)


@_record_tensor_statistics_cotangent.register_fake
def _(
    value: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    return None


# Backward hooks can update the same slot. Preserve their program order under
# compile so this effectful, output-free call is not dropped or reordered.
_record_tensor_statistics_cotangent.register_effect(torch.library.EffectType.ORDERED)


def log_stats(
    metric_source: MetricSource,
    **named_tensors: torch.Tensor,
) -> None:
    """Accumulate current-pass statistics for registered named tensors.

    Args:
        metric_source: Module or parameter used during registration.
        **named_tensors: Registered names mapped to their current tensors.

    Example:

        log_stats(attention, xq=xq)
    """

    if _is_activation_checkpoint_recompute():
        return
    if not should_compute_logged_values():
        return
    state = _state()
    slots = _statistic_buffer_slots(metric_source)
    for registered_name, value in named_tensors.items():
        state._accumulate(_slot(slots, registered_name), value)


def log_fwd_bwd_stats(
    metric_source: nn.Module,
    **named_tensors: torch.Tensor,
) -> None:
    """Record one tensor now and its incoming cotangent during backward.

    Args:
        metric_source: Module used during `register_fwd_bwd`.
        **named_tensors: Registered base names mapped to differentiable tensors.

    Example:

        log_fwd_bwd_stats(attention, xq=xq)
    """

    if not torch.is_grad_enabled():
        return
    if _is_activation_checkpoint_recompute():
        return
    if not should_compute_logged_values():
        return

    state = _state()
    slots = _statistic_buffer_slots(metric_source)
    with spmd.no_typecheck():
        for registered_name, value in named_tensors.items():
            forward_slot = _slot(slots, f"{registered_name}.x")
            backward_slot = _slot(slots, f"{registered_name}.dx")

            # Observe forward now; the hook observes the incoming cotangent.
            state._accumulate(forward_slot, value)

            def record_cotangent(
                cotangent: torch.Tensor,
                slot_index=backward_slot.slot_index,
            ) -> torch.Tensor:
                # A cotangent is the gradient arriving at this tensor in backward.
                with spmd.no_typecheck():
                    _record_tensor_statistics_cotangent(
                        _local_tensor(cotangent),
                        slot_index,
                    )
                    return cotangent

            value.register_hook(record_cotangent)
