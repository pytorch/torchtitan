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
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import cast, NamedTuple, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .statistics import (
    ABS_SUM_INDEX,
    accumulate_tensor_statistics,
    FOURTH_MOMENT_SUM_INDEX,
    NONFINITE_COUNT_INDEX,
    NUMEL_INDEX,
    OBSERVATION_COUNT_INDEX,
    SQUARE_SUM_INDEX,
    StatisticBuffers,
    ZERO_COUNT_INDEX,
)


MetricSource: TypeAlias = nn.Module | nn.Parameter
ReducedBuffers: TypeAlias = tuple[torch.Tensor, torch.Tensor]

# `register()` stores short names on the module or parameter itself.
# Example: attention -> ["xq.x", "xq.dx"].
_REGISTERED_METRIC_NAMES_ATTR = "_tensor_logging_registered_metric_names"

# `init()` records which shared-buffer row belongs to each short name.
# Example: attention["xq.x"] -> sum_statistics[17], maxima[17].
_STATISTIC_BUFFER_SLOTS_ATTR = "_tensor_logging_statistic_buffer_slots"

# One tensor-logging run can be active in this process. `_enabled` controls
# eager logging calls; `StatisticBuffers.enabled` controls device writes.
_active_state: TensorLoggingState | None = None
_enabled = False
_include_tensor_logging_calls = False


class StatisticBufferSlot(NamedTuple):
    """The shared-buffer row used by one registered metric name."""

    slot_index: torch.Tensor  # CPU scalar row used by the backward hook


def _get_registered_metric_names(metric_source: MetricSource) -> list[str] | None:
    """Return names registered directly on this module or parameter.

    Example:

        register_fwd_bwd(attention, ["xq"])
        _get_registered_metric_names(attention)
        # -> ["xq.x", "xq.dx"]
    """

    return cast(
        list[str] | None,
        metric_source.__dict__.get(_REGISTERED_METRIC_NAMES_ATTR),
    )


def _get_statistic_buffer_slots(
    metric_source: MetricSource,
) -> dict[str, StatisticBufferSlot]:
    """Return the shared-buffer row used by each name on this source.

    Example:

        _get_statistic_buffer_slots(attention)
        # -> {
        #     "xq.x": StatisticBufferSlot(
        #         slot_index=torch.tensor(17),
        #     ),
        #     "xq.dx": StatisticBufferSlot(
        #         slot_index=torch.tensor(18),
        #     ),
        # }
    """

    slots = metric_source.__dict__.get(_STATISTIC_BUFFER_SLOTS_ATTR)
    if slots is None:
        raise KeyError(
            f"no initialized tensor metrics on {type(metric_source).__name__}"
        )
    return cast(dict[str, StatisticBufferSlot], slots)


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
    existing_names = _get_registered_metric_names(metric_source)
    if existing_names is None:
        existing_names = []
        setattr(metric_source, _REGISTERED_METRIC_NAMES_ATTR, existing_names)
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
def set_enabled(value: bool) -> Generator[None, None, None]:
    """Choose whether this training step adds statistics to the buffers.

    In eager code, `False` skips logging calls. In compiled or CUDA-graph code,
    logging operations stay in the graph, but `enabled=0` makes them return
    without changing the buffers.

    Example:

        with set_enabled(step % tensor_logging_freq == 0):
            train_step()
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
        # Restore the setting from before this `with` block. Graph Trainer can
        # temporarily enable logging while tracing inside an off-cadence step.
        _enabled = previous
        if state is not None:
            with spmd.no_typecheck():
                state.statistic_buffers.enabled.fill_(previous)


def _wrap_fwd_bwd_for_tensor_logging_capture(
    forward_backward: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Include logging calls in CUDA-graph setup without changing the write flag."""

    @functools.wraps(forward_backward)
    def wrapped(*args, **kwargs):
        global _include_tensor_logging_calls
        previous = _include_tensor_logging_calls
        _include_tensor_logging_calls = True
        try:
            return forward_backward(*args, **kwargs)
        finally:
            _include_tensor_logging_calls = previous

    return wrapped


def is_enabled() -> bool:
    """Return whether the current trainer scope records tensor statistics."""

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled


def should_run_logging_calls() -> bool:
    """Return whether callsites must execute tensor-logging calls.

    Returns `False` on an eager non-logging step. Returns `True` while compiling
    or setting up a CUDA graph so the logging code is included in that graph.
    The separate `enabled` tensor still prevents buffer writes on non-logging
    steps.

    Example:

        if should_run_logging_calls():
            log_stats(router, entropy=compute_router_entropy(scores))
    """

    if torch.compiler.is_compiling():
        return _active_state is not None
    return _enabled or _include_tensor_logging_calls


def _is_installed() -> bool:
    """Return whether this run initialized tensor logging."""

    return _active_state is not None


def _discover_registered_metrics(
    model_parts: Sequence[nn.Module],
    *,
    pp_enabled: bool,
) -> list[tuple[MetricSource, str, str]]:
    """Build the full model name for every registered tensor metric.

    Example:

        # model.layers[3].attention registered "xq.x"
        # -> (attention, "xq.x", "layers.3.attention.xq.x")
    """
    model_part_roots = set(model_parts)
    source_by_full_name: dict[str, MetricSource] = {}
    registered_metrics: list[tuple[MetricSource, str, str]] = []

    # Visit each model chunk owned by this rank. Example: `(0, first_part)`
    # may hold embeddings and layers 0-3; `(1, second_part)` layers 6-7 + head.
    for model_part_index, model_part in enumerate(model_parts):
        # Independent non-PP parts need distinct prefixes. PP parts already keep
        # model paths such as `layers.3`, so they take no prefix.
        model_part_prefix = (
            ""
            if pp_enabled or len(model_parts) == 1
            else f"model_parts.{model_part_index}"
        )

        # Find the path of every module inside that chunk. Example:
        # `("layers.3.attention", attention_module)`.
        for module_name, module in model_part.named_modules():
            module_name = ".".join(
                part
                for part in module_name.split(".")
                if part != "_checkpoint_wrapped_module"
            )
            module_path = ".".join(
                part for part in (model_part_prefix, module_name) if part
            )

            metric_sources: list[tuple[MetricSource, str]] = [(module, module_path)]
            metric_sources.extend(
                (
                    parameter,
                    ".".join(part for part in (module_path, parameter_name) if part),
                )
                for parameter_name, parameter in module.named_parameters(recurse=False)
            )

            # A registration can belong to the module itself or one parameter.
            for metric_source, source_path in metric_sources:
                for registered_name in (
                    _get_registered_metric_names(metric_source) or ()
                ):
                    full_metric_name = ".".join(
                        part for part in (source_path, registered_name) if part
                    )
                    previous_source = source_by_full_name.get(full_metric_name)

                    # Copied PP Decoder roots can share `input.x`/`input.dx`.
                    # Reject every other duplicate full metric name.
                    copied_pp_root = (
                        pp_enabled
                        and previous_source in model_part_roots
                        and metric_source in model_part_roots
                        and previous_source is not metric_source
                    )
                    if previous_source is not None and not copied_pp_root:
                        raise ValueError(
                            f"tensor metric registered twice: {full_metric_name}"
                        )
                    source_by_full_name.setdefault(full_metric_name, metric_source)
                    registered_metrics.append(
                        (metric_source, registered_name, full_metric_name)
                    )
    return registered_metrics


def _gather_pipeline_metric_names(
    local_metric_names: set[str],
    *,
    pp_enabled: bool,
) -> list[str]:
    """Return the same sorted metric-name list on every pipeline rank.

    Example:

        rank 0: {"layers.0.attn.xq.x"}
        rank 1: {"layers.7.attn.xq.x"}
        result on both ranks:
            ["layers.0.attn.xq.x", "layers.7.attn.xq.x"]
    """

    if not pp_enabled or not dist.is_initialized():
        return sorted(local_metric_names)
    names_by_rank: list[set[str]] = [set() for _ in range(dist.get_world_size())]
    dist.all_gather_object(names_by_rank, local_metric_names)
    return sorted(set().union(*names_by_rank))


def _derive_metrics_from_statistics(
    full_metric_name: str,
    sum_statistics: Sequence[float],
    maximum: float,
) -> dict[str, int | float]:
    """Build one metric's scalar fields from its reduced buffer row.

    `sum_statistics` has this fixed field order:

        0: numel
        1: nonfinite_count
        2: zero_count
        3: observation_count
        4: abs_sum
        5: square_sum
        6: fourth_moment_sum

    Example:

        value = torch.tensor([0.0, 1.0, -2.0, 3.0])

        # Recording `value` once and reducing its row produces:
        sum_statistics = [4, 0, 1, 1, 6, 14, 98]
        maximum = 3

        _derive_metrics_from_statistics("scores", sum_statistics, maximum)
        # {
        #     "scores.numel": 4,
        #     "scores.nonfinite_count": 0,
        #     "scores.observation_count": 1,
        #     "scores.zero_count": 1,
        #     "scores.zero_frac": 0.25,
        #     "scores.abs_sum": 6,
        #     "scores.abs_mean": 1.5,
        #     "scores.square_mean": 3.5,
        #     "scores.rms": 3.5**0.5,
        #     "scores.kurtosis": -1.0,
        #     "scores.abs_max": 3,
        # }
    """

    metrics: dict[str, int | float] = {}
    numel = int(sum_statistics[NUMEL_INDEX])
    nonfinite_count = int(sum_statistics[NONFINITE_COUNT_INDEX])
    zero_count = int(sum_statistics[ZERO_COUNT_INDEX])
    observation_count = int(sum_statistics[OBSERVATION_COUNT_INDEX])
    if observation_count == 0:
        return metrics
    finite_count = numel - nonfinite_count
    prefix = f"{full_metric_name}."
    metrics[prefix + "numel"] = numel
    metrics[prefix + "nonfinite_count"] = nonfinite_count
    metrics[prefix + "observation_count"] = observation_count
    if finite_count == 0:
        return metrics

    absolute_sum = sum_statistics[ABS_SUM_INDEX]
    square_sum = sum_statistics[SQUARE_SUM_INDEX]
    fourth_moment_sum = sum_statistics[FOURTH_MOMENT_SUM_INDEX]
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
    """Hold the names and buffers for one active tensor-logging run.

    Example:

        register_fwd_bwd(attention, ["xq"])
        state = init(model, device=device)

        # If attention is `layers.3.attention`, init may assign:
        #   layers.3.attention.xq.x  -> row 17
        #   layers.3.attention.xq.dx -> row 18
        #
        # Shared buffer shapes:
        #   sum_statistics [number_of_metrics, 7]
        #   maxima         [number_of_metrics]
        with set_enabled(True):
            log_fwd_bwd_stats(attention, xq=xq)

        # Forward updates row 17. Backward updates row 18.
        metrics = state.collect()
        state.close()

    `collect()` reduces the two shared buffers, computes scalar metrics, applies
    the publication filter, and clears the buffers. `close()` removes the
    temporary lookup data added by `init()`.
    """

    def __init__(
        self,
        model_parts: Sequence[nn.Module],
        *,
        device: torch.device,
        publish_filter_regex: str = "",
        pp_enabled: bool = False,
    ) -> None:
        self._publish_filter = (
            re.compile(publish_filter_regex) if publish_filter_regex else None
        )

        # Find every registered short name and its full model path.
        registered_metrics = _discover_registered_metrics(
            model_parts,
            pp_enabled=pp_enabled,
        )

        # Give every PP rank the same row order for packed reductions.
        self.full_metric_names = _gather_pipeline_metric_names(
            {full_name for _, _, full_name in registered_metrics},
            pp_enabled=pp_enabled,
        )
        slot_index_by_full_name = {
            full_name: slot_index
            for slot_index, full_name in enumerate(self.full_metric_names)
        }

        # Allocate the reusable SUM/MAX buffers and CPU row-index tensors.
        self.statistic_buffers = StatisticBuffers(
            len(self.full_metric_names),
            device=device,
        )
        self._slot_indices = torch.arange(
            len(self.full_metric_names),
            dtype=torch.int64,
        )

        # Build each source's short-name -> shared-buffer-row mapping locally.
        slots_by_source: dict[MetricSource, dict[str, StatisticBufferSlot]] = {}
        for metric_source, registered_name, full_name in registered_metrics:
            source_slots = slots_by_source.setdefault(metric_source, {})
            row = slot_index_by_full_name[full_name]
            source_slots[registered_name] = StatisticBufferSlot(
                slot_index=self._slot_indices[row],
            )

        # Store each source's name-to-row lookup once.
        for metric_source, source_slots in slots_by_source.items():
            setattr(metric_source, _STATISTIC_BUFFER_SLOTS_ATTR, source_slots)
        self._metric_sources = list(slots_by_source)

        # Register the same shared object on one model part; this does not copy
        # the buffers or create one set per instrumented module.
        self._buffer_owner = model_parts[0]
        self._buffer_owner.add_module(
            "_tensor_logging_state",
            self.statistic_buffers,
        )
        self._closed = False

    def _accumulate_tensor_statistics(
        self,
        slot: StatisticBufferSlot,
        value: torch.Tensor,
    ) -> None:
        with spmd.no_typecheck():
            accumulate_tensor_statistics(
                _local_tensor(value),
                self.statistic_buffers.sum_statistics,
                self.statistic_buffers.maxima,
                self.statistic_buffers.enabled,
                slot.slot_index,
            )

    def _clear_buffers(self) -> None:
        self.statistic_buffers.clear()

    def snapshot_unreduced_statistics(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Clone unreduced slots for focused correctness tests."""

        return {
            metric_name: {
                "counts": self.statistic_buffers.sum_statistics[
                    index, : OBSERVATION_COUNT_INDEX + 1
                ]
                .detach()
                .cpu()
                .clone(),
                "sums": self.statistic_buffers.sum_statistics[index, ABS_SUM_INDEX:]
                .detach()
                .cpu()
                .clone(),
                "maximum": self.statistic_buffers.maxima[index].detach().cpu().clone(),
            }
            for index, metric_name in enumerate(self.full_metric_names)
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
        sum_statistics, maxima = reduced_buffers
        sum_statistics = sum_statistics.detach().cpu()
        maxima = maxima.detach().cpu()
        sum_statistics_by_slot = cast(list[list[float]], sum_statistics.tolist())
        maxima_by_slot = cast(list[float], maxima.tolist())
        metrics: dict[str, int | float] = {}
        for slot_index, full_metric_name in enumerate(self.full_metric_names):
            metrics.update(
                _derive_metrics_from_statistics(
                    full_metric_name,
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
        """Reduce, derive, and reset statistics from this training step."""

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
    device: torch.device,
    publish_filter_regex: str = "",
    pp_enabled: bool = False,
) -> TensorLoggingState:
    """Assign registered tensor names fixed buffer slots and activate logging.

    Call once after model parallelization and optimizer construction. Later
    registration is rejected because compile and CUDA graphs require static storage.

    Args:
        model_parts: Model or PP model parts containing construction-time registrations.
        device: Device for fixed statistics buffers.
        publish_filter_regex: Allowlist over derived public metric names.
        pp_enabled: Whether model parts are pipeline chunks with global layer paths.

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
        pp_enabled=pp_enabled,
    )
    _active_state = state
    return state


def _get_state() -> TensorLoggingState:
    if _active_state is None:
        raise RuntimeError("tensor logging is enabled before init()")
    return _active_state


# No explicit tensor argument is mutated. The ordered effect below represents
# the hidden update to the active state's shared statistic buffers.
@torch.library.custom_op(
    "torchtitan::record_tensor_statistics_cotangent",
    mutates_args=(),
)
def _record_tensor_statistics_cotangent(
    value: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    """Add one backward cotangent's statistics to its buffer row.

    `slot_index` is a CPU scalar containing the row number. For example, if
    `layers.3.attention.xq.dx` uses row 18, this call updates
    `sum_statistics[18]` and `maxima[18]`.

    The autograd hook passes only the cotangent and the row number. This custom
    op finds the active logging buffers, updates that row, and returns no tensor
    for autograd to differentiate.
    """

    state = _get_state()
    buffers = (
        state.statistic_buffers.sum_statistics,
        state.statistic_buffers.maxima,
        state.statistic_buffers.enabled,
        slot_index,
    )
    accumulate_tensor_statistics(value, *buffers)


# FakeTensor tracing needs only the input/output contract. The real op returns
# no tensor, so this fake version returns `None` without touching buffers.
@_record_tensor_statistics_cotangent.register_fake
def _(
    value: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    return None


# Backward hooks can update the same slot. Preserve their program order under
# compile so this effectful, output-free call is not dropped or reordered.
_record_tensor_statistics_cotangent.register_effect(torch.library.EffectType.ORDERED)


def _is_activation_checkpoint_recompute() -> bool:
    """Read the eager activation-checkpoint recompute state lazily."""

    from torchtitan.distributed.activation_checkpoint import (
        _is_activation_checkpoint_recompute as is_recompute,
    )

    return is_recompute()


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
    if not should_run_logging_calls():
        return
    state = _get_state()
    slots = _get_statistic_buffer_slots(metric_source)
    for registered_name, value in named_tensors.items():
        try:
            slot = slots[registered_name]
        except KeyError:
            raise KeyError(f"unregistered tensor metric: {registered_name}") from None
        state._accumulate_tensor_statistics(slot, value)


def log_fwd_bwd_stats(
    metric_source: nn.Module,
    **named_tensors: torch.Tensor,
) -> None:
    """Record one tensor now and its incoming cotangent during backward.

    Args:
        metric_source: Module used during `register_fwd_bwd`.
        **named_tensors: Registered base names mapped to tensors.

    Example:

        log_fwd_bwd_stats(attention, xq=xq)
    """

    if not torch.is_grad_enabled():
        return
    if _is_activation_checkpoint_recompute():
        return
    if not should_run_logging_calls():
        return

    state = _get_state()
    slots = _get_statistic_buffer_slots(metric_source)
    with spmd.no_typecheck():
        for registered_name, value in named_tensors.items():
            try:
                forward_slot = slots[f"{registered_name}.x"]
                backward_slot = slots[f"{registered_name}.dx"]
            except KeyError as error:
                raise KeyError(f"unregistered tensor metric: {error.args[0]}") from None

            # Observe forward now; the hook observes the incoming cotangent.
            state._accumulate_tensor_statistics(forward_slot, value)

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

            if value.requires_grad:
                value.register_hook(record_cotangent)
