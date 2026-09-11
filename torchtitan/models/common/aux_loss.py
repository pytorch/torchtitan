# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Auxiliary-loss gradient injection and distributed metric collection.

Normalization: every auxiliary loss is scaled by the step's global valid-token
count (``set_step_denominator``), the same denominator the main loss uses, so
the contributions stay comparable across parallelism degrees.  The per-step
metric is the mean over loss instances (layers) of that scaled value, summed
over data-parallel ranks and pipeline stages.

The metric accumulates in the model forward.  ``inject()`` wraps the
accumulation in a retained ``torch_remat`` region (``recompute=False``), so
activation checkpointing skips it during recomputation instead of re-running
the side effect, and call sites need no special handling.  The metric is
rolled into ``group_acc`` registers per step by an optimizer pre-hook and
reduced by ``collect_aux_loss_metrics``.

Known limitation: a ``torch_remat`` region only takes effect inside a
``torch_remat`` checkpoint (``RegionAC``).  Under the PyTorch-checkpoint based
policies (``FullAC``, ``SelectiveAC``) the enclosing block forward is replayed
during backward, so the accumulation runs once per replay and the logged
metric over-counts (2x under ``FullAC``).  The injected gradient is unaffected,
because the replayed forward rebuilds the graph the backward pass uses.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn
from torch.distributed._functional_collectives import all_reduce

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type

__all__ = [
    "AuxLoss",
    "collect_aux_loss_metrics",
    "register_aux_loss_zero_hook",
]


class _AuxLossInjection(torch.autograd.Function):
    """Identity-forward autograd that injects an aux-loss gradient on backward.

    ``spmd_typecheck`` declares the in/out types (the body is opaque to the
    checker).
    """

    @staticmethod
    def forward(ctx, carrier, aux_loss):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def spmd_typecheck(result, *, carrier):
        spmd.assert_type(
            result,
            spmd.get_local_type(carrier),
            partition_spec=spmd.get_partition_spec(carrier),
        )

    @staticmethod
    def backward(ctx, grad_carrier):  # pyrefly: ignore[bad-override]
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


class AuxLoss(Module):
    """Base class: subclasses call ``inject()`` each microbatch.  Each
    instance accumulates its scaled value in the ``instance_acc`` buffer;
    the optimizer pre-hook (like the aux-loss-free load balancing hook)
    rolls these into the ``group_acc`` registers, which
    ``collect_aux_loss_metrics`` reduces for logging.

    Normalization: ``denominator = global_valid_tokens`` for the step, set by
    the trainer via ``set_step_denominator`` before the first forward.
    Metric accumulation happens in the forward inside ``inject()``, which
    wraps it in a retained ``torch_remat`` region (``recompute=False``) so
    ``torch_remat``-based checkpointing never re-runs the accumulation.  Under
    the PyTorch-checkpoint based policies (``FullAC``, ``SelectiveAC``) the
    region is inert and the metric over-counts; see the module docstring.
    """

    # Metric groups are populated during model build, before PP splitting, so
    # every pipeline stage participates with its own (zero) accumulators and
    # every rank holds the same count.  That count is also the divisor in
    # ``collect_aux_loss_metrics``: the per-rank sums are summed over the
    # reduce mesh and the pipeline stages, then divided by it, giving the mean
    # over all layers of the model.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Global valid-token count of the current step, set by the trainer before
    # the first forward.  Shared by all instances: the framework normalizes
    # every auxiliary loss by the same per-step count, matching the main
    # loss, so the contributions are comparable across parallelism degrees.
    _step_denominator: ClassVar[torch.Tensor | None] = None

    # Per metric group (``(reduce_mesh, metric_name)``): this rank's total
    # value of the current step, rolled up from the per-instance
    # ``instance_acc`` buffers by ``_zero_aux_losses`` at each optimizer step
    # pre-hook and reduced by ``collect_aux_loss_metrics`` at log time.
    # Cleared at the next pre-hook.
    group_acc: ClassVar[dict[tuple[str, str], torch.Tensor]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        coeff: float
        """Aux loss coefficient.  Scales the gradient contribution."""
        reduce_mesh: str = "batch"
        """Mesh the per-step metric is summed over: ``"batch"`` (dp) for
        cp-identical losses like the microbatch-wise load-balance loss, ``"loss"``
        (dp+cp) for per-token-additive losses whose rank-local values add up
        across coordinates."""

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
        self.reduce_mesh = config.reduce_mesh
        # Per-instance accumulator: sum of this loss instance's scaled
        # per-microbatch values over the current training step.  Filled in
        # the forward; its value is rolled into the ``group_acc``
        # registers and it is zeroed by ``_zero_aux_losses`` at each
        # optimizer step pre-hook.
        self.register_buffer(
            "instance_acc", torch.zeros((), dtype=torch.float32), persistent=False
        )
        AuxLoss._group_counts[(self.reduce_mesh, self.metric_name)] += 1

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.instance_acc.device
        with torch.device(buffer_device):
            self.instance_acc = torch.zeros((), dtype=torch.float32)

    @classmethod
    def set_step_denominator(cls, denominator: torch.Tensor) -> None:
        """Set the current step's global valid-token count.

        The trainer calls this once per step with the same dp-summed token
        count the main loss normalizes by, so auxiliary losses stay on the
        same scale as the main loss and independent of parallelism degrees.
        """
        cls._step_denominator = denominator

    def inject(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        """Inject the aux-loss gradient on ``carrier``; accumulate the scaled metric.

        The accumulation is a forward side effect, so it runs inside a
        ``torch_remat`` region with ``recompute=False``: ``torch_remat``-based
        activation checkpointing retains the region instead of re-running it,
        and the metric is counted exactly once per microbatch.  Under the
        PyTorch-checkpoint based policies the region is inert and the metric
        over-counts; see the module docstring.  The region output is marked
        with ``recompute_needs_tensor`` because callers consume it with bare
        ops.  Subclasses only need to call this method.

        Args:
            raw_sum: Unnormalized per-microbatch loss value (differentiable).
            carrier: The tensor whose backward path carries the gradient.

        Returns:
            ``carrier`` unchanged (identity forward).
        """
        if AuxLoss._step_denominator is None:
            raise ValueError(
                "AuxLoss.set_step_denominator() must be called with the "
                "step's global valid-token count before the first forward."
            )
        out = remat.region(
            self._accumulate_and_inject,
            self.remat_region_name("aux_loss"),
            recompute=False,
        )(raw_sum, carrier=carrier)
        remat.recompute_needs_tensor(out)
        return out

    def _accumulate_and_inject(
        self, raw_sum: torch.Tensor, *, carrier: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate this microbatch's metric value and inject the gradient."""
        denominator = AuxLoss._step_denominator
        assert denominator is not None, "set_step_denominator() must be called"
        # Scaling the loss is local arithmetic on a per-step scalar.  The
        # denominator is set by the trainer outside the model forward, so it
        # carries no mesh annotation for the ambient one, and the loss's own
        # type varies with the layout (TP is Invariant with EP token sharding,
        # Replicate without), so no single restated type fits; the injection
        # itself still runs through the checker.
        with spmd.no_typecheck():
            scale = 1.0 / denominator
            injected = raw_sum * (self.coeff * scale)
        # Accumulate the metric in the forward.  The mask is the canonical
        # no_grad side-effect pattern (as for the MoE usage counters) and
        # keeps the buffer out of the autograd graph; the spmd unwrap marks
        # writing an untyped per-rank buffer as a non-SPMD op.
        with spmd.no_typecheck(), torch.no_grad():
            self.instance_acc.add_(raw_sum * scale)
        return _AuxLossInjection.apply(carrier, injected)


def _zero_aux_losses(model_parts) -> None:
    """Roll per-instance ``instance_acc`` values into the ``group_acc``
    registers (once per metric group) and clear them.

    Optimizer step pre-hook; mirrors ``register_moe_load_balancing_hook``.
    """
    AuxLoss.group_acc.clear()
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, AuxLoss):
                key = (module.reduce_mesh, module.metric_name)
                if key not in AuxLoss.group_acc:
                    AuxLoss.group_acc[key] = torch.zeros_like(module.instance_acc)
                AuxLoss.group_acc[key] += module.instance_acc
                module.instance_acc.zero_()


def collect_aux_loss_metrics(parallel_dims: ParallelDims) -> dict[str, float]:
    """Reduce the current step's ``group_acc`` registers for logging.

    Returns ``{metric_name}/mean`` per group, ``{}`` if none configured.  All
    ranks call this at log time.

    TODO(#4202): migrate aux-loss metric collection to the distributed tensor
    logging framework once it lands.
    """
    if not AuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")

    def _group_acc_or_zero(key: tuple[str, str]) -> torch.Tensor:
        group_acc_value = AuxLoss.group_acc.get(key)
        if group_acc_value is not None:
            return group_acc_value
        # Ranks that own no instance of this group still join the collectives
        # below with a zero contribution.
        return torch.zeros((), dtype=torch.float32, device=device_type)

    group_accs = {key: _group_acc_or_zero(key) for key in AuxLoss._group_counts}
    metrics = {}
    for key, total in sorted(group_accs.items()):
        mesh_name, tag = key
        reduce_mesh = parallel_dims.get_optional_mesh(mesh_name)
        for mesh in (reduce_mesh, pp_mesh):
            if mesh is None:
                continue
            # Sum: each coordinate contributes its own data, and every layer
            # lives on exactly one pipeline stage, so summing over the reduce
            # mesh and the stages counts every layer once.  Dividing by the
            # build-time instance count below (identical on every rank) then
            # gives the mean over all layers.
            total = all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / AuxLoss._group_counts[key]
    return metrics


def register_aux_loss_zero_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register the step pre-hook that rolls per-instance ``instance_acc``
    into the ``group_acc`` registers and zeroes the instances.

    Same pattern as ``register_moe_load_balancing_hook``
    (:func:`torchtitan.components.optimizer.optimizer.register_moe_load_balancing_hook`).
    """
    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: _zero_aux_losses(model_parts)
    )
