# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon policy and logical views over generic SPMD compute tensors."""

from collections.abc import MutableMapping
from contextlib import ExitStack

import spmd_types as spmd
import torch
from torch import Tensor
from torch.distributed.tensor import (
    DTensor,
    Partial as DtPartial,
    Placement as DtPlacement,
    Replicate as DtReplicate,
    Shard as DtShard,
)
from torch.optim._muon import muon


__all__ = ["SPMDMuon"]


class SPMDMuon(torch.optim.Muon):
    """Run Muon on typed local views while retaining persistent state identity.

    DTensor parameters and momentum remain the objects owned by the optimizer
    and checkpoint path. Only the tensors passed to Muon's functional update
    are plain, storage-sharing local views. Every view must prove that its final
    two matrix dimensions are complete on the current rank.
    """

    @staticmethod
    def _compute_placements(
        tensor: DTensor,
        matrix_shape: tuple[int, int] | None,
    ) -> tuple[DtPlacement, ...]:
        """Choose a physical DTensor layout containing complete Muon matrices."""
        compute_placements = []
        first_matrix_dim = tensor.ndim - 2
        for placement in tensor.placements:
            if isinstance(placement, DtPartial):
                raise spmd.SpmdTypeError(
                    "SPMDMuon requires gradients to be reduced before the "
                    "optimizer step; Partial storage is not a valid input"
                )
            if isinstance(placement, DtShard):
                shard_dim = placement.dim % tensor.ndim
                # A logical reshape makes every physical shard boundary
                # ambiguous. Native [..., M, N] tensors may retain shards only
                # on their leading matrix-batch dimensions.
                if (
                    matrix_shape is not None
                    or type(placement) is not DtShard
                    or shard_dim >= first_matrix_dim
                ):
                    placement = DtReplicate()
            compute_placements.append(placement)
        return tuple(compute_placements)

    def _compute_view(
        self,
        tensor: Tensor,
        source: Tensor | None = None,
        *,
        matrix_shape: tuple[int, int] | None,
        writeback: bool,
    ) -> Tensor:
        if isinstance(tensor, DTensor):
            stack = getattr(self, "_compute_view_stack", None)
            if stack is None:
                raise RuntimeError(
                    "SPMDMuon compute views must be opened inside step()"
                )
            return stack.enter_context(
                spmd.dtensor_compute_view(
                    tensor,
                    placements=self._compute_placements(tensor, matrix_shape),
                    writeback=writeback,
                )
            )
        if not spmd.has_local_type(tensor):
            if source is None:
                raise spmd.SpmdTypeError(
                    "SPMDMuon requires an SPMD-annotated plain parameter or a "
                    "DTensor; an untyped parameter cannot prove complete matrix "
                    "ownership"
                )
            spmd.assert_type_like(tensor, source)
        return tensor

    @staticmethod
    def _logical_matrix_view(
        tensor: Tensor, matrix_shape: tuple[int, int] | None
    ) -> Tensor:
        if matrix_shape is None:
            return tensor
        if (
            not isinstance(matrix_shape, tuple)
            or len(matrix_shape) != 2
            or not all(isinstance(dim, int) and dim > 0 for dim in matrix_shape)
        ):
            raise ValueError(
                "SPMDMuon matrix_shape must be a tuple of two positive integers, "
                f"got {matrix_shape!r}"
            )
        matrix_numel = matrix_shape[0] * matrix_shape[1]
        if tensor.numel() % matrix_numel != 0:
            raise ValueError(
                f"SPMDMuon cannot view shape {tuple(tensor.shape)} as a batch of "
                f"{matrix_shape}: {tensor.numel()} elements is not divisible by "
                f"{matrix_numel}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                "SPMDMuon matrix_shape requires a contiguous storage-sharing view"
            )
        batch_size = tensor.numel() // matrix_numel
        return tensor.view(batch_size, *matrix_shape)

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ) -> bool:
        for persistent_param in group["params"]:
            persistent_grad = persistent_param.grad
            if persistent_grad is None:
                continue

            if torch.is_complex(persistent_param):
                raise RuntimeError("Muon does not support complex parameters")
            if persistent_grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            matrix_shape = group.get("matrix_shape")
            param = self._compute_view(
                persistent_param,
                matrix_shape=matrix_shape,
                writeback=True,
            )
            grad = self._compute_view(
                persistent_grad,
                param,
                matrix_shape=matrix_shape,
                writeback=False,
            )
            spmd.assert_type_like(grad, param)
            for compute_tensor in (param, grad):
                spmd.assert_local_block(compute_tensor, trailing_dims=2)
            if param.shape != grad.shape:
                raise RuntimeError(
                    "SPMDMuon parameter and gradient local views must have the "
                    f"same shape, got {param.shape} and {grad.shape}"
                )

            storage_param = param
            param = self._logical_matrix_view(param, matrix_shape)
            grad = self._logical_matrix_view(grad, matrix_shape)

            # State remains keyed by the persistent parameter. Allocate it only
            # after the input layout has passed all safety checks.
            state = self.state[persistent_param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    persistent_grad, memory_format=torch.preserve_format
                )
            persistent_momentum = state["momentum_buffer"]
            momentum = self._compute_view(
                persistent_momentum,
                storage_param,
                matrix_shape=matrix_shape,
                writeback=True,
            )
            spmd.assert_type_like(momentum, storage_param)
            spmd.assert_local_block(momentum, trailing_dims=2)
            momentum = self._logical_matrix_view(momentum, matrix_shape)
            if momentum.shape != param.shape:
                raise RuntimeError(
                    "SPMDMuon momentum local view must match the parameter shape, "
                    f"got {momentum.shape} and {param.shape}"
                )

            params_with_grad.append(param)
            grads.append(grad)
            muon_momentum_bufs.append(momentum)

        return False

    @torch.no_grad()
    def step(self, closure=None):
        """Run each parameter group in its requested physical compute layout."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Scope gathered parameter/gradient/state buffers to one group so a
            # model-wide step does not retain every temporary compute layout.
            with ExitStack() as stack:
                self._compute_view_stack = stack
                try:
                    # Multi-leading-dimension flatten/unflatten is not yet
                    # representable by global PartitionSpec propagation. The
                    # entry assertion retains the global safety proof.
                    with spmd.local():
                        params_with_grad: list[Tensor] = []
                        grads: list[Tensor] = []
                        muon_momentum_bufs: list[Tensor] = []
                        has_complex = self._init_group(
                            group,
                            params_with_grad,
                            grads,
                            muon_momentum_bufs,
                        )
                        muon(
                            params_with_grad,
                            grads,
                            muon_momentum_bufs,
                            lr=group["lr"],
                            weight_decay=group["weight_decay"],
                            momentum=group["momentum"],
                            nesterov=group["nesterov"],
                            ns_coefficients=group["ns_coefficients"],
                            eps=group["eps"],
                            ns_steps=group["ns_steps"],
                            adjust_lr_fn=group["adjust_lr_fn"],
                            has_complex=has_complex,
                        )
                finally:
                    self._compute_view_stack = None
        return loss
