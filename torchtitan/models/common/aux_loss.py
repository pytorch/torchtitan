# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Auxiliary-loss gradient injection and distributed metric collection.

Auxiliary objectives inside the model (MoE load-balance, a future DSA indexer
KL, ...) cannot reach the trainer's loss under pipeline parallelism.
`LoggedAuxLoss` injects the gradient via an identity autograd function and
defers per-step metric readout. Contract:

- Normalization: `inject` scales by `1 / per_step_denominator`, set by the
  trainer from the resolved per-step token count (token:
  num_tokens_per_train_step; sequence: num_tokens_per_train_step //
  max_context_length; batch: 1 -- not microbatch/PP-additive, see PR #3000).
- Accumulation: in the autograd backward (once per microbatch, so AC recompute
  never double counts); `_zero_aux_losses` snapshots and clears per step
  (optimizer pre-hook).
- Reduction: mesh inferred from `aggregation_level` ("loss" for token, "batch"
  otherwise); PP reduced separately.
- Model specs register `register_aux_loss_zero_hook` after the optimizer.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Literal

import spmd_types as spmd
import torch
from torch import nn
from torch.distributed._functional_collectives import all_reduce

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type

__all__ = [
    "LoggedAuxLoss",
    "collect_aux_loss_metrics",
    "register_aux_loss_zero_hook",
]


class _AuxLossInjection(torch.autograd.Function):
    """Identity-forward autograd that injects an aux-loss gradient on backward.

    The metric also accumulates here: autograd runs each node's backward once
    per microbatch, so AC recompute never double counts. `spmd_typecheck`
    declares the in/out types (the body is opaque to the checker).

    TODO: with the move to torch_remat as the AC solution, the backward
    accumulation may no longer be needed as a double-counting workaround;
    revisit then (see review discussion).
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        carrier: torch.Tensor,
        aux_loss: torch.Tensor,
        acc_value: torch.Tensor,
        acc_sum: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(aux_loss, acc_value)
        ctx.acc_sum = acc_sum
        return carrier

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> None:
        spmd.assert_type(
            result,
            spmd.get_local_type(carrier),
            partition_spec=spmd.get_partition_spec(carrier),
        )

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_carrier: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        aux_loss, acc_value = ctx.saved_tensors
        ctx.acc_sum.add_(acc_value)
        return grad_carrier, torch.ones_like(aux_loss), None, None


class LoggedAuxLoss(Module):
    """Base class: subclasses call ``inject()`` each microbatch; the metric is
    snapshotted per step and reduced by ``collect_aux_loss_metrics``."""

    # Metric groups are identical on every pipeline stage: populated during
    # model build, before PP splitting, so each stage joins the collectives
    # with its own (zero) accumulators.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Per-step snapshots keyed by ``(reduce_mesh, metric_name)``, filled by
    # ``_zero_aux_losses`` at each optimizer step pre-hook.
    _step_snapshots: ClassVar[dict[tuple[str, str], torch.Tensor]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        coeff: float
        """Aux loss coefficient. Scales the gradient contribution."""
        aggregation_level: Literal["token", "sequence", "batch"]
        """Per-token, per-sequence, or per-global-batch value (see module docstring)."""

        per_step_denominator: int = -1
        """Per-step normalization denominator, set by the trainer after the
num-token tensor sizes are resolved (`-1` = unset; plain field so `build()`'s
`replace` keeps it)."""

    @property
    def reduce_mesh(self) -> str:
        """Mesh the metric is sum-reduced over: "loss" for token, "batch" otherwise."""

        return "loss" if self.aggregation_level == "token" else "batch"

    @property
    def metric_name(self) -> str:
        """Convert the class name from PascalCase to snake_case."""
        return re.sub(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            "_",
            type(self).__name__,
        ).lower()

    def __init__(self, config: Config):
        super().__init__()
        self.coeff = config.coeff
        self.aggregation_level = config.aggregation_level
        self.per_step_denominator = config.per_step_denominator
        # Running sum of scaled per-microbatch values, covering one step.
        self.register_buffer(
            "_acc_sum", torch.zeros((), dtype=torch.float32), persistent=False
        )
        LoggedAuxLoss._group_counts[(self.reduce_mesh, self.metric_name)] += 1

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            buffer_device = self._acc_sum.device
        with torch.device(buffer_device):
            self._acc_sum = torch.zeros((), dtype=torch.float32)

    def inject(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        """Inject the aux-loss gradient on `carrier`; accumulate the scaled metric."""
        if self.per_step_denominator <= 0:
            raise ValueError(
                "LoggedAuxLoss.per_step_denominator is not set: "
                "Decoder.update_from_config must fill it before the first "
                "forward."
            )
        scale = 1.0 / self.per_step_denominator
        acc_value = raw_sum.detach() * scale
        return _AuxLossInjection.apply(
            carrier, raw_sum * (self.coeff * scale), acc_value, self._acc_sum
        )


def _zero_aux_losses(model_parts) -> None:
    """Snapshot each accumulator (once per group) and clear it; step pre-hook."""
    LoggedAuxLoss._step_snapshots.clear()
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, LoggedAuxLoss):
                key = (module.reduce_mesh, module.metric_name)
                if key not in LoggedAuxLoss._step_snapshots:
                    LoggedAuxLoss._step_snapshots[key] = torch.zeros_like(
                        module._acc_sum
                    )
                LoggedAuxLoss._step_snapshots[key] += module._acc_sum
                module._acc_sum.zero_()


def collect_aux_loss_metrics(model_parts, parallel_dims) -> dict[str, float]:
    """Reduce the per-step snapshots over the inferred mesh and PP.

    Returns `{metric_name}/mean` per group, `{}` if none configured. All ranks
    call this at log time; value is the last completed step.

    TODO: replace with the GPU tensor logging effort.
    """

    if not LoggedAuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")
    local_sums = {
        key: LoggedAuxLoss._step_snapshots.get(
            key, torch.zeros((), dtype=torch.float32, device=device_type)
        )
        for key in LoggedAuxLoss._group_counts
    }
    metrics = {}
    for key, total in sorted(local_sums.items()):
        mesh_name, tag = key
        for mesh in (parallel_dims.get_optional_mesh(mesh_name), pp_mesh):
            if mesh is not None:
                total = all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / LoggedAuxLoss._group_counts[key]
    return metrics


def register_aux_loss_zero_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register the step pre-hook that snapshots and zeroes aux losses
    (same pattern as `register_moe_load_balancing_hook`)."""

    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _zero_aux_losses(model_parts)
    )
