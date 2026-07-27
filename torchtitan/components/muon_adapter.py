# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt core Muon to persistent storage layouts and logical matrix views."""

from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack
from typing import Any, cast

import spmd_types as spmd
import torch
import torch.distributed.tensor.placement_types as placement_types
from torch import Tensor
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard
from torch.optim._muon import _compute_muon_update, muon
from torch.optim.optimizer import _to_scalar

from torchtitan.components.flex_shard import Owned


__all__ = ["MuonAdapter"]


def _is_shard_like(placement: Placement) -> bool:
    predicate = getattr(placement_types, "_is_shard_like", None)
    if predicate is not None:
        return predicate(placement)

    strided_shard_type = getattr(placement_types, "_StridedShard", None)
    return isinstance(placement, Shard) or (
        strided_shard_type is not None and isinstance(placement, strided_shard_type)
    )


def _has_local_type(tensor: Tensor) -> bool:
    return spmd.has_local_type(tensor)  # pyrefly: ignore [missing-attribute]


class MuonAdapter(torch.optim.Muon):
    """Run core Muon through optional storage and logical-view adaptation.

    DTensor parameters and momentum remain the objects owned by the optimizer
    and checkpoint path. Only the tensors passed to Muon's functional update
    are plain, storage-sharing local views. Every view must prove that its final
    two matrix dimensions are complete on the current rank. Ordinary untyped
    tensors retain the behavior of ``torch.optim.Muon``.
    """

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if param_group.get("fused") or param_group.get("foreach"):
            raise NotImplementedError(
                "MuonAdapter does not support fused or foreach implementations. "
                "Configure implementation='for-loop' or explicitly disable both "
                "options in each Muon parameter group."
            )
        super().add_param_group(param_group)

    @staticmethod
    def _compute_placements(
        tensor: DTensor,
        matrix_shape: tuple[int, int] | None,
    ) -> tuple[Placement, ...]:
        """Choose a physical DTensor layout containing complete Muon matrices."""
        compute_placements = []
        first_matrix_dim = tensor.ndim - 2
        for placement in tensor.placements:
            if isinstance(placement, Partial):
                raise spmd.SpmdTypeError(
                    "MuonAdapter requires gradients to be reduced before the "
                    "optimizer step; Partial storage is not a valid input"
                )
            if _is_shard_like(placement):
                shard_dim = cast(Any, placement).dim % tensor.ndim
                # A logical reshape makes every physical shard boundary
                # ambiguous. Native [..., M, N] tensors may retain shards only
                # on their leading matrix-batch dimensions.
                if (
                    matrix_shape is not None
                    or type(placement) is not Shard
                    or shard_dim >= first_matrix_dim
                ):
                    placement = Replicate()
            compute_placements.append(placement)
        return tuple(compute_placements)

    def _compute_view(
        self,
        tensor: Tensor,
        source: Tensor | None = None,
        *,
        compute_views: ExitStack,
        matrix_shape: tuple[int, int] | None,
        writeback: bool,
    ) -> Tensor:
        if isinstance(tensor, DTensor):
            return compute_views.enter_context(
                spmd.dtensor_compute_view(  # pyrefly: ignore [missing-attribute]
                    tensor,
                    placements=self._compute_placements(tensor, matrix_shape),
                    writeback=writeback,
                )
            )
        tensor_is_typed = _has_local_type(tensor)
        if source is not None:
            source_is_typed = _has_local_type(source)
            if source_is_typed:
                spmd.assert_type_like(tensor, source)
            elif tensor_is_typed:
                raise spmd.SpmdTypeError(
                    "MuonAdapter received a typed tensor whose parameter is untyped"
                )
        return tensor

    @staticmethod
    def _validate_matrix_shape(
        tensor: Tensor, matrix_shape: tuple[int, int] | None
    ) -> None:
        if matrix_shape is None:
            return
        if (
            not isinstance(matrix_shape, tuple)
            or len(matrix_shape) != 2
            or not all(isinstance(dim, int) and dim > 0 for dim in matrix_shape)
        ):
            raise ValueError(
                "MuonAdapter matrix_shape must be a tuple of two positive integers, "
                f"got {matrix_shape!r}"
            )
        matrix_numel = matrix_shape[0] * matrix_shape[1]
        if tensor.numel() % matrix_numel != 0:
            raise ValueError(
                f"MuonAdapter cannot view shape {tuple(tensor.shape)} as a batch of "
                f"{matrix_shape}: {tensor.numel()} elements is not divisible by "
                f"{matrix_numel}"
            )

    @classmethod
    def _logical_matrix_view(
        cls, tensor: Tensor, matrix_shape: tuple[int, int] | None
    ) -> Tensor:
        cls._validate_matrix_shape(tensor, matrix_shape)
        if matrix_shape is None:
            return tensor
        if not tensor.is_contiguous():
            raise ValueError(
                "MuonAdapter matrix_shape requires a contiguous storage-sharing view"
            )
        matrix_numel = matrix_shape[0] * matrix_shape[1]
        batch_size = tensor.numel() // matrix_numel
        return tensor.view(batch_size, *matrix_shape)

    def flex_shard_compute_requirement(
        self,
        param: Tensor,
        group: MutableMapping,
    ) -> Owned:
        if torch.is_complex(param):
            raise RuntimeError("Muon does not support complex parameters")
        matrix_shape = group.get("matrix_shape")
        self._validate_matrix_shape(param, matrix_shape)
        return Owned(trailing_dims=2)

    @staticmethod
    def flex_shard_validate_group(
        group_index: int,
        group: MutableMapping,
    ) -> None:
        ns_steps = group["ns_steps"]
        if (
            isinstance(ns_steps, bool)
            or not isinstance(ns_steps, int)
            or ns_steps < 0
            or ns_steps >= 100
        ):
            raise ValueError(
                f"group {group_index} ns_steps must be an integer in [0, 100), "
                f"got {ns_steps!r}"
            )
        coefficients = group["ns_coefficients"]
        if (
            not isinstance(coefficients, tuple)
            or len(coefficients) != 3
            or not all(isinstance(value, (int, float)) for value in coefficients)
        ):
            raise ValueError(
                f"group {group_index} must have exactly three numeric "
                "Newton-Schulz coefficients"
            )
        eps = group["eps"]
        if not isinstance(eps, (int, float)) or eps < 0:
            raise ValueError(
                f"group {group_index} eps must be non-negative, got {eps!r}"
            )
        for name in ("lr", "momentum", "weight_decay"):
            value = group[name]
            valid = (
                value.numel() == 1
                if isinstance(value, torch.Tensor)
                else isinstance(value, (int, float))
            )
            if not valid:
                raise ValueError(
                    f"group {group_index} {name} must be a number or scalar tensor, "
                    f"got {value!r}"
                )
            try:
                nonnegative = bool(value >= 0)
            except (RuntimeError, TypeError, ValueError):
                nonnegative = False
            if not nonnegative:
                raise ValueError(
                    f"group {group_index} {name} must be non-negative, got {value!r}"
                )
        if not isinstance(group["nesterov"], bool):
            raise ValueError(f"group {group_index} nesterov must be a bool")
        if group["adjust_lr_fn"] not in (
            None,
            "original",
            "match_rms_adamw",
            "spectral_unclamped",
        ):
            raise ValueError(
                f"group {group_index} has unsupported adjust_lr_fn "
                f"{group['adjust_lr_fn']!r}"
            )

    @staticmethod
    def flex_shard_group_signature(group: MutableMapping) -> object:
        def signature_value(value):
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return ("invalid_tensor", tuple(value.shape))
                return ("tensor", value.detach().item())
            return value

        return (
            signature_value(group["lr"]),
            signature_value(group["weight_decay"]),
            signature_value(group["momentum"]),
            group["nesterov"],
            group["ns_coefficients"],
            group["eps"],
            group["ns_steps"],
            group["adjust_lr_fn"],
            group.get("matrix_shape"),
        )

    def flex_shard_init_state(
        self,
        param: Tensor,
        grad: Tensor,
        group: MutableMapping,
    ) -> MutableMapping:
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                grad, memory_format=torch.preserve_format
            )
        return state

    @staticmethod
    def flex_shard_prepare(
        param: Tensor,
        grad: Tensor,
        state: Mapping[str, Tensor],
        group: MutableMapping,
        *,
        out: Tensor,
    ) -> None:
        if param.shape != grad.shape:
            raise ValueError("Muon parameter and gradient shapes must match")
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.lerp_(grad, 1 - group["momentum"])
        if group["nesterov"]:
            torch.lerp(grad, momentum_buffer, group["momentum"], out=out)
        else:
            out.copy_(momentum_buffer)

    def flex_shard_compute(
        self,
        compute_input: Tensor,
        group: MutableMapping,
    ) -> Tensor:
        logical_pre = self._logical_matrix_view(
            compute_input, group.get("matrix_shape")
        )
        if logical_pre.numel() == 0:
            return compute_input

        direction, adjusted_lr = _compute_muon_update(
            logical_pre,
            logical_pre.shape,
            lr=_to_scalar(group["lr"]),
            ns_coefficients=group["ns_coefficients"],
            eps=group["eps"],
            ns_steps=group["ns_steps"],
            adjust_lr_fn=group["adjust_lr_fn"],
        )
        # Preserve core Muon's add_(..., alpha=...) cast and scaling order while
        # reusing FlexShard's temporary input buffer for the signed update.
        logical_pre.zero_()
        logical_pre.add_(direction, alpha=-adjusted_lr)
        return compute_input

    @staticmethod
    def flex_shard_finalize(
        param: Tensor,
        update: Tensor,
        group: MutableMapping,
        *,
        out: Tensor,
    ) -> None:
        decay = 1 - group["lr"] * group["weight_decay"]
        if isinstance(decay, Tensor):
            torch.mul(param, decay, out=out)
            out.add_(update)
        else:
            torch.add(update, param, alpha=decay, out=out)

    def _validate_group(self, group: MutableMapping) -> None:
        """Reject deterministic input errors before opening mutable views."""
        matrix_shape = group.get("matrix_shape")
        for persistent_param in group["params"]:
            persistent_grad = persistent_param.grad
            if persistent_grad is None:
                continue

            if torch.is_complex(persistent_param):
                raise RuntimeError("Muon does not support complex parameters")
            if persistent_grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if persistent_param.shape != persistent_grad.shape:
                raise RuntimeError(
                    "MuonAdapter parameter and gradient must have the same shape, "
                    f"got {persistent_param.shape} and {persistent_grad.shape}"
                )

            self._validate_matrix_shape(persistent_param, matrix_shape)
            for tensor in (persistent_param, persistent_grad):
                if isinstance(tensor, DTensor):
                    self._compute_placements(tensor, matrix_shape)

            persistent_momentum = self.state.get(persistent_param, {}).get(
                "momentum_buffer"
            )
            if persistent_momentum is not None:
                if persistent_momentum.shape != persistent_param.shape:
                    raise RuntimeError(
                        "MuonAdapter momentum must match the parameter shape, "
                        f"got {persistent_momentum.shape} and "
                        f"{persistent_param.shape}"
                    )
                self._validate_matrix_shape(persistent_momentum, matrix_shape)
                if isinstance(persistent_momentum, DTensor):
                    self._compute_placements(persistent_momentum, matrix_shape)

    def _init_compute_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
        *,
        compute_views: ExitStack,
    ) -> bool:
        for persistent_param in group["params"]:
            persistent_grad = persistent_param.grad
            if persistent_grad is None:
                continue

            matrix_shape = group.get("matrix_shape")
            param = self._compute_view(
                persistent_param,
                compute_views=compute_views,
                matrix_shape=matrix_shape,
                writeback=True,
            )
            grad = self._compute_view(
                persistent_grad,
                param,
                compute_views=compute_views,
                matrix_shape=matrix_shape,
                writeback=False,
            )
            if _has_local_type(param):
                spmd.assert_type_like(grad, param)
                for compute_tensor in (param, grad):
                    spmd.assert_local_block(  # pyrefly: ignore [missing-attribute]
                        compute_tensor, trailing_dims=2
                    )
            if param.shape != grad.shape:
                raise RuntimeError(
                    "MuonAdapter parameter and gradient local views must have the "
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
                compute_views=compute_views,
                matrix_shape=matrix_shape,
                writeback=True,
            )
            if _has_local_type(storage_param):
                spmd.assert_type_like(momentum, storage_param)
                spmd.assert_local_block(  # pyrefly: ignore [missing-attribute]
                    momentum, trailing_dims=2
                )
            momentum = self._logical_matrix_view(momentum, matrix_shape)
            if momentum.shape != param.shape:
                raise RuntimeError(
                    "MuonAdapter momentum local view must match the parameter shape, "
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
            self._validate_group(group)

        for group in self.param_groups:
            # Scope gathered parameter/gradient/state buffers to one group so a
            # model-wide step does not retain every temporary compute layout.
            with ExitStack() as compute_views, spmd.local():
                # Multi-leading-dimension flatten/unflatten is not yet
                # representable by global PartitionSpec propagation. The
                # entry assertion retains the global safety proof.
                params_with_grad: list[Tensor] = []
                grads: list[Tensor] = []
                muon_momentum_bufs: list[Tensor] = []
                has_complex = self._init_compute_group(
                    group,
                    params_with_grad,
                    grads,
                    muon_momentum_bufs,
                    compute_views=compute_views,
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
        return loss
