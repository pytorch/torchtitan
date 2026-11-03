# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeGuard

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch_remat as remat
from torch import nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.config import Configurable

from .statistics import (
    ABS_SUM_INDEX,
    accumulate_tensor_statistics,
    FOURTH_MOMENT_SUM_INDEX,
    NONFINITE_COUNT_INDEX,
    NUMEL_INDEX,
    OBSERVATION_COUNT_INDEX,
    SQUARE_SUM_INDEX,
    SUM_STATISTIC_FIELD_COUNT,
    ZERO_COUNT_INDEX,
)

# The one TensorLogger of this process. Model code calls the module functions
# below instead of receiving a logger, like `structured_logger`.
_active_logger: "TensorLogger | None" = None

# Registered row names live on the module object, so the deepcopy that splits a
# model into pipeline stages carries them into each stage.
_ROW_NAMES_ATTR = "_tensor_logger_row_names"


def _register(module: nn.Module, row_names: list[str]) -> None:
    module.__dict__.setdefault(_ROW_NAMES_ATTR, []).extend(row_names)


def register_fwd(module: nn.Module, *names: str) -> None:
    """Register forward rows for ``log_fwd(module, name=tensor)``.

    Example: register_fwd(router, "scores") -> row "scores.fwd"
    """
    _register(module, [f"{name}.fwd" for name in names])


def register_fwd_bwd(module: nn.Module, *names: str) -> None:
    """Register a forward and a gradient row for ``log_fwd_bwd(module, name=tensor)``.

    Example: register_fwd_bwd(attention, "xq") -> rows "xq.fwd" and "xq.bwd"
    """
    register_fwd(module, *names)
    _register(module, [f"{name}.bwd" for name in names])


def _should_run_calls(logger: "TensorLogger | None") -> TypeGuard["TensorLogger"]:
    # TODO: let a run log a subset of registered tensors (e.g. by FQN) without
    # launching the other kernels.
    return (
        logger is not None
        # A captured accelerator graph replays what ran during capture, and
        # capture can happen on a step that does not record. So with graphs the
        # calls run on every step and the device flag `enabled` gates the writes.
        and (logger._recording or logger._run_calls_on_every_step)
        # torch_remat replays the block forward during backward; count it once.
        and not remat.is_recomputing()
    )


def log_fwd(module: nn.Module, **tensors: torch.Tensor) -> None:
    """Add statistics of forward tensors to the rows registered with ``register_fwd``.

    Example: log_fwd(router, scores=scores_TE)
    """
    logger = _active_logger
    if not _should_run_calls(logger):
        return
    for name, value in tensors.items():
        logger._accumulate(value, logger._row(module, f"{name}.fwd"))


def log_fwd_bwd(module: nn.Module, **tensors: torch.Tensor) -> None:
    """Add statistics of forward tensors and, during backward, of their gradients.

    Rows come from ``register_fwd_bwd``. Pass tensors made in this forward: each
    call adds a gradient hook to the tensor, and a parameter raises ValueError.

    Example: log_fwd_bwd(attention, xq=xq_THK)
    """
    logger = _active_logger
    # A no-grad forward (validation) has no gradient to pair with the forward row.
    if not _should_run_calls(logger) or not torch.is_grad_enabled():
        return
    for name, value in tensors.items():
        if value.requires_grad and value.is_leaf:
            raise ValueError(
                f"log_fwd_bwd({type(module).__name__}) got a leaf tensor for "
                f"{name!r} (e.g. a parameter); every call would add one more "
                "gradient hook. Log a tensor made in this forward."
            )
        logger._accumulate(value, logger._row(module, f"{name}.fwd"))
        if value.requires_grad:
            # The hook returns None, so the gradient is unchanged. SPMD type
            # checking also checks register_hook, which has no sharding rule.
            with spmd.no_typecheck():
                value.register_hook(
                    functools.partial(
                        logger._accumulate, row=logger._row(module, f"{name}.bwd")
                    )
                )


class TensorLogger(Configurable):
    """Per-row statistics of logged tensors, reduced over all ranks on log steps.

    Each rank adds its local observations into one row per registered name.
    ``collect()`` sums the rows over all ranks (max for ``abs_max``) and derives
    the published fields.

    Example:
        register_fwd_bwd(attention, "xq")          # attention.__init__
        log_fwd_bwd(attention, xq=xq_THK)          # attention.forward
        tensor_logger = config.build(model_parts=..., device=..., ...)
        with tensor_logger.maybe_record(step):     # records when step % log_freq == 0
            engine.forward_backward(...)
        tensor_logger.collect()
        # {"tensor_logger/layers.0.attention.xq.fwd.abs_mean": 0.81, ...}
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int
        """Record on steps that are multiples of ``log_freq``. It must be a multiple
        of ``metrics.log_freq``, so every recorded step is published."""

        def __post_init__(self) -> None:
            if self.log_freq <= 0:
                raise ValueError(
                    f"tensor_logger.log_freq must be positive, got {self.log_freq}."
                )

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        device: torch.device,
        accelerator_graph_enabled: bool,
        pp_schedule: _PipelineSchedule | None,
    ) -> None:
        global _active_logger
        if _active_logger is not None:
            raise RuntimeError("Only one TensorLogger can be active in a process.")
        self.config = config

        # "layers.3.attention.xq.fwd" -> (attention module, "xq.fwd"). Pipeline
        # stages keep global layer paths, so names match runs without PP.
        local_rows: dict[str, tuple[nn.Module, str]] = {}
        for model_part in model_parts:
            for path, module in model_part.named_modules():
                for row_name in module.__dict__.get(_ROW_NAMES_ATTR, ()):
                    full_name = f"{path}.{row_name}" if path else row_name
                    if full_name in local_rows:
                        raise ValueError(f"Tensor name registered twice: {full_name}")
                    local_rows[full_name] = (module, row_name)

        # collect() sums row i over ranks, so every rank must use row i for the
        # same name. Pipeline ranks own different layers: every rank allocates the
        # sorted union of all names. A rank that does not own a row adds the SUM
        # identity 0 and the MAX identity -inf.
        #   rank 0 owns rows 2, 3: [0,  0,  v1, v2]
        #   rank 1 owns rows 0, 1: [v3, v4, 0,  0 ]  -> SUM: [v3, v4, v1, v2]
        self._process_group = None
        names_by_rank: list[Any] = [set(local_rows)]
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._process_group = dist.group.WORLD
            names_by_rank = [None] * dist.get_world_size()
            dist.all_gather_object(names_by_rank, set(local_rows))
        self.row_names = sorted(set().union(*names_by_rank))
        if not self.row_names:
            raise ValueError(
                "tensor_logger is set, but no model module registered a tensor. "
                "See torchtitan/observability/tensor_logger/README.md."
            )

        row_index = {name: index for index, name in enumerate(self.row_names)}
        self._rows_by_module: dict[nn.Module, dict[str, int]] = {}
        for full_name, (module, row_name) in local_rows.items():
            self._rows_by_module.setdefault(module, {})[row_name] = row_index[full_name]

        num_rows = len(self.row_names)
        self._sum_statistics = torch.zeros(
            num_rows, SUM_STATISTIC_FIELD_COUNT, dtype=torch.float32, device=device
        )
        self._maxima = torch.full(
            (num_rows,), -torch.inf, dtype=torch.float32, device=device
        )
        # The kernel exits early when enabled == 0.
        self._enabled = torch.zeros((), dtype=torch.int32, device=device)
        self._run_calls_on_every_step = accelerator_graph_enabled
        self._recording = False
        self._has_unpublished_record = False

        self._pp_cleanup_handle = None
        if pp_schedule is not None:
            # Pipeline metadata inference runs real forwards and backwards before
            # the first microbatch, and again after each eval -> train switch.
            # Drop what they recorded.
            self._pp_cleanup_handle = (
                pp_schedule.register_post_metadata_inference_cleanup(self._clear)
            )
        _active_logger = self

    def _row(self, module: nn.Module, row_name: str) -> int:
        rows = self._rows_by_module.get(module, {})
        if row_name not in rows:
            raise ValueError(
                f"{type(module).__name__} logged {row_name!r}, which was not registered "
                "when the TensorLogger was built. Call register_fwd or "
                "register_fwd_bwd in the module's __init__."
            )
        return rows[row_name]

    # Statistics read local values, so the forward call and the gradient hook
    # both skip SPMD type checking.
    @spmd.no_typecheck
    def _accumulate(self, value: torch.Tensor, row: int) -> None:
        accumulate_tensor_statistics(
            value, self._sum_statistics, self._maxima, self._enabled, row
        )

    def _clear(self) -> None:
        self._sum_statistics.zero_()
        self._maxima.fill_(-torch.inf)

    @contextlib.contextmanager
    def maybe_record(self, step: int) -> Iterator[None]:
        """Record logged tensors inside this scope when ``step % log_freq == 0``."""
        self._recording = step % self.config.log_freq == 0
        if self._recording:
            self._clear()
            self._has_unpublished_record = True
        self._enabled.fill_(int(self._recording))
        try:
            yield
        finally:
            self._recording = False
            self._enabled.fill_(0)

    def collect(self) -> dict[str, float]:
        """Reduce the recorded rows over all ranks and derive the published fields.

        Returns {} when nothing was recorded since the last call. Every rank must
        call it on the same steps.
        """
        if not self._has_unpublished_record:
            return {}
        self._has_unpublished_record = False
        if self._process_group is not None:
            dist.all_reduce(
                self._sum_statistics, op=dist.ReduceOp.SUM, group=self._process_group
            )
            dist.all_reduce(
                self._maxima, op=dist.ReduceOp.MAX, group=self._process_group
            )
        # One device-to-host copy per buffer, on log steps only.
        sums_by_row = self._sum_statistics.tolist()
        maxima_by_row = self._maxima.tolist()
        metrics = {}
        for row_name, sums, abs_max in zip(
            self.row_names, sums_by_row, maxima_by_row, strict=True
        ):
            metrics.update(_derive_fields(f"tensor_logger/{row_name}", sums, abs_max))
        return metrics

    def close(self) -> None:
        global _active_logger
        if self._pp_cleanup_handle is not None:
            self._pp_cleanup_handle.remove()
        if _active_logger is self:
            _active_logger = None


def _derive_fields(prefix: str, sums: list[float], abs_max: float) -> dict[str, float]:
    """Turn one reduced row into published fields; {} for a row nobody observed.

    Example: one observation of [0., 1., -2., 3.] ->
        numel 4, observation_count 1, nonfinite_count 0, zero_frac 0.25,
        abs_mean 1.5, rms sqrt(3.5), kurtosis 98 / 4 / 3.5**2 - 3 = -1.0, abs_max 3
    """
    observation_count = sums[OBSERVATION_COUNT_INDEX]
    if observation_count == 0:
        return {}
    numel = sums[NUMEL_INDEX]
    nonfinite_count = sums[NONFINITE_COUNT_INDEX]
    fields = {
        "numel": numel,
        "observation_count": observation_count,
        "nonfinite_count": nonfinite_count,
        "abs_max": abs_max,
    }
    finite_count = numel - nonfinite_count
    if finite_count > 0:
        fields["zero_frac"] = sums[ZERO_COUNT_INDEX] / finite_count
        fields["abs_mean"] = sums[ABS_SUM_INDEX] / finite_count
        square_mean = sums[SQUARE_SUM_INDEX] / finite_count
        fields["rms"] = math.sqrt(square_mean)
        if square_mean > 0:
            fields["kurtosis"] = (
                sums[FOURTH_MOMENT_SUM_INDEX] / finite_count / square_mean**2 - 3
            )
    # abs_max is -inf when no value was finite, and an overflowing sum gives inf
    # or nan; leave such a field out instead of publishing a meaningless value.
    return {
        f"{prefix}.{name}": value
        for name, value in fields.items()
        if math.isfinite(value)
    }
