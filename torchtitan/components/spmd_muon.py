# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon adapter for temporary plain-tensor SPMD compute views."""

from collections.abc import MutableMapping

import spmd_types as spmd
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor


__all__ = ["SPMDMuon"]


class SPMDMuon(torch.optim.Muon):
    """Run Muon on typed local views while retaining persistent state identity.

    DTensor parameters and momentum remain the objects owned by the optimizer
    and checkpoint path. Only the tensors passed to Muon's functional update
    are plain, storage-sharing local views. Every view must prove that its final
    two matrix dimensions are complete on the current rank.
    """

    @staticmethod
    def _compute_view(tensor: Tensor, source: Tensor | None = None) -> Tensor:
        if isinstance(tensor, DTensor):
            return spmd.dtensor_to_local(tensor)
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
        partition_spec = spmd.get_partition_spec(tensor)
        if partition_spec is not None and any(
            entry is not None for entry in partition_spec
        ):
            raise spmd.SpmdTypeError(
                "SPMDMuon matrix_shape cannot reinterpret a physically sharded "
                "tensor because matrix-boundary alignment is not yet provable. "
                "Store the parameter as [..., M, N] and shard a leading dimension."
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

            param = self._compute_view(persistent_param)
            grad = self._compute_view(persistent_grad, param)
            spmd.assert_type_like(grad, param)
            for compute_tensor in (param, grad):
                spmd.assert_local_block(compute_tensor, trailing_dims=2)
            if param.shape != grad.shape:
                raise RuntimeError(
                    "SPMDMuon parameter and gradient local views must have the "
                    f"same shape, got {param.shape} and {grad.shape}"
                )

            matrix_shape = group.get("matrix_shape")
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
            momentum = self._compute_view(persistent_momentum, storage_param)
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

    def step(self, closure=None):
        # Multi-leading-dimension flatten/unflatten is not yet representable by
        # global PartitionSpec propagation. The entry assertion above retains
        # the global safety proof; the Muon body needs only local SPMD rules.
        with spmd.local():
            return super().step(closure)
