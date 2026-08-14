# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import cast, NamedTuple, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .statistics import accumulate_tensor_statistics, StatisticBuffers


Owner: TypeAlias = nn.Module | nn.Parameter
ReducedBuffers: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_REGISTERED_NAMES_ATTR = "_tensor_logging_registered_names"
_METRIC_ROWS_ATTR = "_tensor_logging_metric_rows"
_active_state: TensorLoggingState | None = None
_enabled = False
_recording_calls_included = False
_metric_side_effects_suppressed = False


class MetricRow(NamedTuple):
    """One registered metric's preallocated device storage."""

    counts: torch.Tensor
    sums: torch.Tensor
    maximum: torch.Tensor
    row_index: torch.Tensor


def _registered_metric_names(owner: Owner) -> list[str] | None:
    return cast(
        list[str] | None,
        owner.__dict__.get(_REGISTERED_NAMES_ATTR),
    )


def _metric_rows(owner: Owner) -> dict[str, MetricRow] | None:
    # Read this exact owner's metadata without following `nn.Module` proxy lookup.
    return cast(
        dict[str, MetricRow] | None,
        owner.__dict__.get(_METRIC_ROWS_ATTR),
    )


def _metric_row(owner: Owner, metric_name: str) -> MetricRow:
    rows = _metric_rows(owner)
    if rows is None or metric_name not in rows:
        raise KeyError(f"unregistered tensor metric: {metric_name}")
    return rows[metric_name]


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def register(owner: Owner, metric_names: Sequence[str]) -> None:
    """Declare the metric names an owner may pass to `log_stats`.

    Args:
        owner: Module or parameter that emits the tensors.
        metric_names: Keyword names accepted by `log_stats` for this owner.

    Example:

        register(attention, ["xq"])
        log_stats(attention, xq=xq)
    """

    registered_names = _registered_metric_names(owner)
    if registered_names is None:
        registered_names = []
        setattr(owner, _REGISTERED_NAMES_ATTR, registered_names)
    registered_names.extend(metric_names)


def register_fwd_bwd(owner: nn.Module, metric_names: Sequence[str]) -> None:
    """Register paired `.x` and `.dx` statistics for each tensor name.

    Args:
        owner: Module that owns the observed tensors.
        metric_names: Base names passed to `log_fwd_bwd_stats`.

    Example:

        register_fwd_bwd(attention, ["xq", "head_out"])
    """

    register(owner, [f"{name}.x" for name in metric_names])
    register(owner, [f"{name}.dx" for name in metric_names])


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
            state.buffers.enabled.fill_(value)
    try:
        yield
    finally:
        _enabled = previous
        if state is not None:
            with spmd.no_typecheck():
                state.buffers.enabled.fill_(previous)


@contextlib.contextmanager
def _include_recording_calls() -> Iterator[None]:
    """Emit device-gated recording ops even when the host cadence is disabled.

    CUDA graph warmup and capture can occur on a non-logging step. The captured
    graph still needs the recording ops; ``buffers.enabled`` decides at replay
    time whether they mutate statistics.
    """

    global _recording_calls_included
    previous = _recording_calls_included
    _recording_calls_included = True
    try:
        yield
    finally:
        _recording_calls_included = previous


def _metric_side_effects_are_suppressed() -> bool:
    return _metric_side_effects_suppressed


@contextlib.contextmanager
def _suppress_metric_side_effects() -> Iterator[None]:
    """Skip metric mutations while activation checkpointing recomputes a block."""

    global _metric_side_effects_suppressed
    previous = _metric_side_effects_suppressed
    _metric_side_effects_suppressed = True
    try:
        yield
    finally:
        _metric_side_effects_suppressed = previous


def is_enabled() -> bool:
    """Return whether the current trainer scope records tensor statistics."""

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled


def should_run_producers() -> bool:
    """Return whether callsites must produce values consumed by ``log_stats``.

    CUDA-graph capture may occur on an unselected step. Producers still belong
    in that graph; the device-side ``buffers.enabled`` gate controls replay.
    """

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled or _recording_calls_included


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


def _metric_prefix_by_owner(
    model_parts: Sequence[nn.Module],
    model_part_prefixes: Mapping[nn.Module, str],
) -> dict[Owner, str]:
    """Map local modules and parameters to their public metric prefix.

    Example:

        model_part_prefixes = {first_part: "", last_part: ""}
        # first_part.layers[0] -> "layers.0"
        # last_part.layers[7]  -> "layers.7"
    """

    names: dict[Owner, str] = {}
    default_prefixes = (
        [""]
        if len(model_parts) == 1
        else [f"model_parts.{i}" for i in range(len(model_parts))]
    )
    # A split model defaults to local prefixes unless PP supplies global overrides.
    for model_part, default_prefix in zip(model_parts, default_prefixes, strict=True):
        model_part_prefix = model_part_prefixes.get(model_part, default_prefix)
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


def _add_metric_values(
    metrics: dict[str, int | float],
    metric_name: str,
    counts: Sequence[int],
    sums: Sequence[float],
    maximum: float,
) -> None:
    """Convert one reduced buffer row into named scalar metrics.

    Example:

        # Recorded tensor: [0, 1, -2, 3]
        counts = [4, 0, 1, 1]
        sums = [6, 14, 98]
        maximum = 3

        # Adds numel=4, zero_frac=0.25, abs_mean=1.5,
        # square_mean=3.5, kurtosis=-1, and abs_max=3.
    """

    numel, nonfinite_count, zero_count, observation_count = counts
    if observation_count == 0:
        return
    finite_count = numel - nonfinite_count
    prefix = f"{metric_name}."
    metrics[prefix + "numel"] = numel
    metrics[prefix + "nonfinite_count"] = nonfinite_count
    metrics[prefix + "observation_count"] = observation_count
    if finite_count == 0:
        return

    absolute_sum, square_sum, fourth_moment_sum = sums
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


class TensorLoggingState:
    """Freeze metric names, bind them to rows, and collect step statistics.

    Example:

        register(owner, names)       construction
                  |
        init(model_parts)            one global row order
                  |
        log_stats(owner, values)     fixed device rows
                  |
        collect()                    packed WORLD reductions -> scalar dict
    """

    def __init__(
        self,
        model_parts: Sequence[nn.Module],
        *,
        device: torch.device | None = None,
        publish_filter_regex: str = "",
        model_part_prefixes: Mapping[nn.Module, str] | None = None,
    ) -> None:
        self._owners: list[Owner] = []
        self._publish_filter = (
            re.compile(publish_filter_regex) if publish_filter_regex else None
        )

        # Discover each local `(owner, metric_name)` and its public full name.
        model_part_prefixes = model_part_prefixes or {}
        owner_prefixes = _metric_prefix_by_owner(model_parts, model_part_prefixes)
        local_owner_by_full_name: dict[str, Owner] = {}
        local_bindings: list[tuple[Owner, str, str]] = []
        for owner, owner_prefix in owner_prefixes.items():
            metric_names = _registered_metric_names(owner)
            if metric_names is None:
                continue
            for metric_name in metric_names:
                full_name = ".".join(
                    part for part in (owner_prefix, metric_name) if part
                )
                previous_owner = local_owner_by_full_name.get(full_name)
                if previous_owner is not None:
                    same_owner = previous_owner is owner
                    previous_prefix = (
                        model_part_prefixes.get(previous_owner)
                        if isinstance(previous_owner, nn.Module)
                        else None
                    )
                    current_prefix = (
                        model_part_prefixes.get(owner)
                        if isinstance(owner, nn.Module)
                        else None
                    )
                    same_global_prefix = (
                        previous_prefix is not None
                        and current_prefix is not None
                        and previous_prefix == current_prefix
                    )
                    # Two PP model parts may share a global name; one owner may not.
                    if same_owner or not same_global_prefix:
                        raise ValueError(f"tensor metric registered twice: {full_name}")
                else:
                    local_owner_by_full_name[full_name] = owner
                local_bindings.append((owner, metric_name, full_name))

        # PP stages own different modules but must reduce one shared row order.
        self.metric_names = _gather_global_metric_names(set(local_owner_by_full_name))
        row_index_by_full_name = {
            name: row_index for row_index, name in enumerate(self.metric_names)
        }

        # Every rank allocates the same rows; each rank writes only local metrics.
        self.buffers = StatisticBuffers(
            len(self.metric_names),
            device=device or _infer_device(model_parts),
        )
        self._row_indices = torch.arange(len(self.metric_names), dtype=torch.int64)
        # Map each local metric to its row in the globally ordered buffers.
        for owner, metric_name, full_name in local_bindings:
            rows = _metric_rows(owner)
            if rows is None:
                rows = {}
                setattr(owner, _METRIC_ROWS_ATTR, rows)
                self._owners.append(owner)
            row_index = row_index_by_full_name[full_name]
            rows[metric_name] = MetricRow(
                counts=self.buffers.counts[row_index],
                sums=self.buffers.sums[row_index],
                maximum=self.buffers.maxima[row_index],
                row_index=self._row_indices[row_index],
            )
        # Module state makes the slabs visible to compile and Graph Trainer.
        self._state_owner = model_parts[0]
        self._state_owner.add_module("_tensor_logging_state", self.buffers)
        self._closed = False

    def _accumulate(self, owner: Owner, metric_name: str, value: torch.Tensor) -> None:
        row = _metric_row(owner, metric_name)
        with spmd.no_typecheck():
            accumulate_tensor_statistics(
                _local_tensor(value),
                row.counts,
                row.sums,
                row.maximum,
                self.buffers.enabled,
            )

    def _row_index(self, owner: Owner, metric_name: str) -> torch.Tensor:
        return _metric_row(owner, metric_name).row_index

    def _clear_buffers(self) -> None:
        self.buffers.clear()

    def snapshot_unreduced_statistics(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Clone unreduced rows for focused correctness tests."""

        return {
            metric_name: {
                "counts": self.buffers.counts[index].detach().cpu().clone(),
                "sums": self.buffers.sums[index].detach().cpu().clone(),
                "maximum": self.buffers.maxima[index].detach().cpu().clone(),
            }
            for index, metric_name in enumerate(self.metric_names)
        }

    def _reduce_buffers(self) -> ReducedBuffers:
        """Clone and reduce every registered key in three packed WORLD slabs."""

        counts = self.buffers.counts.clone()
        sums = self.buffers.sums.clone()
        maxima = self.buffers.maxima.clone()
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        return counts, sums, maxima

    def _buffers_to_metrics(
        self,
        reduced_buffers: ReducedBuffers,
    ) -> dict[str, int | float]:
        """Derive, aggregate, and filter values from reduced buffers."""

        # One device-to-host copy per buffer avoids synchronizing per metric.
        counts, sums, maxima = (buffer.detach().cpu() for buffer in reduced_buffers)
        count_rows = cast(list[list[int]], counts.tolist())
        sum_rows = cast(list[list[float]], sums.tolist())
        maximum_rows = cast(list[float], maxima.tolist())
        metrics: dict[str, int | float] = {}
        for index, metric_name in enumerate(self.metric_names):
            _add_metric_values(
                metrics,
                metric_name,
                count_rows[index],
                sum_rows[index],
                maximum_rows[index],
            )
        # Filtering controls sink volume, not GPU collection.
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
        for owner in self._owners:
            owner.__dict__.pop(_METRIC_ROWS_ATTR, None)
        self._owners.clear()
        self._state_owner._modules.pop("_tensor_logging_state")
        if _active_state is self:
            _active_state = None


def init(
    model_parts: nn.Module | Iterable[nn.Module],
    *,
    device: torch.device | None = None,
    publish_filter_regex: str = "",
    model_part_prefixes: Mapping[nn.Module, str] | None = None,
) -> TensorLoggingState:
    """Freeze registrations and install one active state for the model parts.

    Args:
        model_parts: Model or model parts whose registered owners should be active.
        device: Device for fixed statistic buffers; inferred when omitted.
        publish_filter_regex: Publication allowlist over dotted metric names.
        model_part_prefixes: Global prefixes for explicitly split model parts.

    Example:

        register_fwd_bwd(model.layers[0], ["residual"])
        state = init(model, device=torch.device("cuda"))
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
        model_part_prefixes=model_part_prefixes,
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
    row_index: torch.Tensor,
) -> None:
    """Record one cotangent without exposing mutable buffers to autograd."""

    state = _state()
    index = int(row_index.item())
    buffers = (
        state.buffers.counts[index],
        state.buffers.sums[index],
        state.buffers.maxima[index],
        state.buffers.enabled,
    )
    accumulate_tensor_statistics(value, *buffers)


@_record_tensor_statistics_cotangent.register_fake
def _(
    value: torch.Tensor,
    row_index: torch.Tensor,
) -> None:
    return None


_record_tensor_statistics_cotangent.register_effect(torch.library.EffectType.ORDERED)


def log_stats(owner: Owner, **named_tensors: torch.Tensor) -> None:
    """Accumulate current-pass statistics for registered named tensors.

    Args:
        owner: Module or parameter used during registration.
        **named_tensors: Registered names mapped to their current tensors.

    Example:

        log_stats(attention, xq=xq)
    """

    if _metric_side_effects_suppressed:
        return
    if not should_run_producers():
        return
    state = _state()
    for metric_name, value in named_tensors.items():
        state._accumulate(owner, metric_name, value)


def log_fwd_bwd_stats(
    owner: nn.Module,
    **named_tensors: torch.Tensor,
) -> None:
    """Record one tensor now and its incoming cotangent during backward.

    Args:
        owner: Module used during `register_fwd_bwd`.
        **named_tensors: Registered base names mapped to differentiable tensors.

    Example:

        log_fwd_bwd_stats(attention, xq=xq)
    """

    if not torch.is_grad_enabled():
        return
    if _metric_side_effects_suppressed:
        return
    if not should_run_producers():
        return

    state = _state()
    with spmd.no_typecheck():
        for metric_name, value in named_tensors.items():
            backward_row_index = state._row_index(owner, f"{metric_name}.dx")

            # Observe forward now; the hook observes the incoming cotangent.
            state._accumulate(owner, f"{metric_name}.x", value)

            def record_cotangent(
                cotangent: torch.Tensor,
                row_index=backward_row_index,
            ) -> torch.Tensor:
                # A cotangent is the gradient arriving at this tensor in backward.
                with spmd.no_typecheck():
                    _record_tensor_statistics_cotangent(
                        _local_tensor(cotangent),
                        row_index,
                    )
                    return cotangent

            value.register_hook(record_cotangent)
