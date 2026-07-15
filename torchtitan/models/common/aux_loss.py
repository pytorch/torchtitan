# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-internal auxiliary losses: gradient injection and logging.

Auxiliary objectives computed inside the model (MoE load-balance, DSA KL, ...)
cannot reach the trainer's loss function under pipeline parallelism. The
gradient injection decouples the gradient from the scalar output, and the
logging framework accumulates per-step values with PP-safe collection.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

import spmd_types as spmd
import torch

from torch.distributed._functional_collectives import all_reduce as _all_reduce

from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type

__all__ = [
    "LoggedAuxLoss",
    "collect_aux_loss_metrics",
]


@spmd.register_autograd_function
class _AuxLossInjection(torch.autograd.Function):
    """Identity-forward autograd that injects an aux-loss gradient on backward."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, carrier: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def typecheck_forward(
        carrier: torch.Tensor, aux_loss: torch.Tensor
    ) -> torch.Tensor:
        return _AuxLossInjection.apply(carrier, aux_loss)

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_carrier: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


class LoggedAuxLoss(Module):
    """Accumulates per-microbatch aux losses via ``inject()`` with deferred readout.

    Subclasses call ``inject()`` each microbatch to inject the gradient and
    accumulate the scalar value. ``pop()`` reads and resets the accumulator.
    """

    # Populated during model build (before PP splitting) so counts and keys are
    # identical on every rank. PP stages without aux loss modules still join
    # collectives via pre-allocated zero accumulators in collect_aux_loss_metrics.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        coeff: float
        """Aux loss coefficient. Scales the gradient contribution."""
        tag: str = "aux_loss"
        """Metric name prefix for ``collect_aux_loss_metrics``."""
        reduce_mesh: str = "loss"
        """Mesh to average logged values over. PP is reduced separately."""
        global_batch_size: int | None = None
        """Per-token denominator, set by ``Decoder.update_from_config``."""
        ac_doubled: bool = False
        """AC double-count flag, set by ``Decoder.update_from_config``."""

    def __init__(self, config: Config):
        super().__init__()
        self.coeff = config.coeff
        self.tag = config.tag
        self.reduce_mesh = config.reduce_mesh
        self.global_batch_size = config.global_batch_size
        self.ac_doubled = config.ac_doubled
        self.register_buffer(
            "_acc", torch.zeros((), dtype=torch.float32), persistent=False
        )
        LoggedAuxLoss._group_counts[(config.reduce_mesh, config.tag)] += 1

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self._acc.device
        with torch.device(buffer_device):
            self._acc = torch.zeros((), dtype=torch.float32)

    def inject(self, carrier: torch.Tensor, raw_sum: torch.Tensor) -> torch.Tensor:
        """Inject aux loss gradient into the forward graph and accumulate for logging."""
        if self.training:
            self._acc.add_(raw_sum.detach().float() / self.global_batch_size)
        return _AuxLossInjection.apply(
            carrier, raw_sum * (self.coeff / self.global_batch_size)
        )

    def pop(self) -> torch.Tensor:
        """Pop and reset the accumulated value, correcting for AC double-write."""
        val = self._acc.detach().clone()
        if self.ac_doubled:
            val = val / 2
        self._acc.zero_()
        return val


def collect_aux_loss_metrics(model_parts, parallel_dims) -> dict[str, float]:
    """Collect and reduce aux loss metrics from all model parts for logging.

    Returns ``{tag}/mean`` per group; ``{}`` when no aux losses are configured.
    """
    if not LoggedAuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")
    local_sums = {
        key: torch.zeros((), dtype=torch.float32, device=device_type)
        for key in LoggedAuxLoss._group_counts
    }

    for part in model_parts:
        for block in getattr(part, "layers", {}).values():
            for module in block.modules():
                if isinstance(module, LoggedAuxLoss):
                    local_sums[(module.reduce_mesh, module.tag)] += module.pop()

    metrics = {}
    for key, total in sorted(local_sums.items()):
        mesh_name, tag = key
        for mesh in (parallel_dims.get_optional_mesh(mesh_name), pp_mesh):
            if mesh is not None:
                total = _all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / LoggedAuxLoss._group_counts[key]
    return metrics
