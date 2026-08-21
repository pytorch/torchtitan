# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP-managed MXFP8 weight operands.

Tensor shape suffixes:
    N: output features
    K: input features
"""

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from .quantize import MXFP8WeightOperands, quantize_mxfp8_weight


aten = torch.ops.aten

# The wrapper carries the FSDP extension hooks, while its inner BF16 tensor
# carries the actual parameter storage. Parameter construction, checkpointing,
# device moves, and FSDP sharding apply these bookkeeping ops to the parameter;
# their outputs must keep the wrapper or subsequent FSDP hooks will be lost.
# This allowlist follows TorchAO's TrainingWeightWrapperBaseTensor. It is an
# audited list of parameter-lifecycle operations, not a list provided by FSDP.
#
# Keep this list narrow. Normal compute ops intentionally return plain tensors
# after operating on the inner BF16 value. Any new entry must be checked for
# aliasing and in-place semantics, especially if the op has multiple outputs.
_OPS_TO_PRESERVE_WEIGHT_WRAPPER = {
    aten.empty_like.default,
    aten.new_zeros.default,
    aten.slice.Tensor,
    aten.copy_.default,
    aten.view.default,
    aten.as_strided.default,
    aten._to_copy.default,
    aten._pin_memory.default,
    aten.split.Tensor,
    aten.clone.default,
    aten.transpose.int,
    aten.t.default,
    torch.ops.c10d.scatter_.default,
}


class MXFP8FSDPWeight(torch.Tensor):
    """Persistent high-precision parameter shard with MXFP8 FSDP hooks.

    This wrapper has tensor metadata but no parameter storage of its own.
    ``_data`` is the real high-precision local shard. The subclass exists so
    FSDP can discover ``fsdp_pre_all_gather`` and ``fsdp_post_all_gather`` on
    the parameter while optimizers and checkpoints continue to operate on the
    high-precision value.

    This is deliberately not a general-purpose propagating tensor subclass.
    Only audited parameter-lifecycle operations preserve the wrapper.
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            memory_format=suggest_memory_format(data),
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            pin_memory=data.is_pinned(),
            requires_grad=data.requires_grad,
        )

    def __init__(
        self,
        data: torch.Tensor,
    ) -> None:
        self._data = data

    # This wrapper has no storage of its own, so every tensor operation must be
    # redirected to the real BF16 shard in ``_data``. Use the ATen-level
    # dispatcher as the single interception point instead of maintaining a
    # second set of high-level __torch_function__ overrides. Operations in
    # _OPS_TO_PRESERVE_WEIGHT_WRAPPER are re-wrapped so the parameter retains
    # its FSDP hooks; normal compute returns plain tensors so the marker does
    # not propagate into activations.
    # pyrefly: ignore [bad-param-name-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(
        cls,
        func,
        types,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        del types
        kwargs = kwargs or {}

        def unwrap(tensor: MXFP8FSDPWeight) -> torch.Tensor:
            return tensor._data

        def wrap(tensor: torch.Tensor) -> MXFP8FSDPWeight:
            return cls(tensor)

        # Run the operation on the real parameter shards. tree_map_only also
        # handles wrappers nested inside lists, tuples, and keyword arguments.
        args, kwargs = pytree.tree_map_only(
            MXFP8FSDPWeight,
            unwrap,
            (args, kwargs),
        )
        if func == aten.detach.default:
            # nn.Parameter construction detaches its input. Preserve the marker
            # here or the parameter will lose the FSDP extension hooks early.
            return wrap(args[0])

        output = func(*args, **kwargs)
        if func not in _OPS_TO_PRESERVE_WEIGHT_WRAPPER:
            # Do not let the marker spread to arbitrary compute results. An op
            # omitted from the allowlist may therefore strip the wrapper; that
            # is intentional unless FSDP needs the result as a parameter shard.
            return output
        return pytree.tree_map_only(torch.Tensor, wrap, output)

    def __tensor_flatten__(self):
        # Let tensor-subclass serialization and tracing rebuild the wrapper
        # around the real inner shard instead of treating it as opaque state.
        return ["_data"], None

    def __repr__(self, *, tensor_contents: Any | None = None) -> str:
        del tensor_contents
        return (
            f"MXFP8FSDPWeight(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        metadata,
        outer_size,
        outer_stride,
    ):
        del metadata, outer_size, outer_stride
        return cls(inner_tensors["_data"])

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        # Requires https://github.com/pytorch/pytorch/pull/194114.
        # The post-all-gather hook builds an independent MXFP8 unsharded tensor.
        # Its raw high-precision inputs are temporary and do not need to share
        # the lifetime of the FSDP-managed MXFP8 tensors.
        return True

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        del mesh, outer_size, outer_stride, module
        param_dtype = mp_policy.param_dtype or self._data.dtype
        # This hook is required even without mixed precision: it tells FSDP to
        # all-gather the real inner shard instead of the storage-less wrapper,
        # and FSDP requires pre/post all-gather hooks to be defined as a pair.
        # The policy only selects the collective dtype; there is no side metadata.
        return (self._data.to(param_dtype),), ()

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        if metadata != ():
            raise AssertionError(f"Expected empty metadata, got {metadata}")
        (weight_NK,) = all_gather_outputs
        if weight_NK.dtype != param_dtype:
            raise AssertionError(
                f"Expected gathered weight dtype {param_dtype}, got {weight_NK.dtype}"
            )

        if out is None:
            # On the first unshard, return the logical unsharded tensor plus the
            # three independent inner tensors whose storage FSDP should manage.
            operands = quantize_mxfp8_weight(weight_NK)
            unsharded_tensor = MXFP8FSDPComputeWeight(
                operands,
                logical_shape=weight_NK.shape,
                logical_stride=weight_NK.stride(),
                logical_storage_offset=int(weight_NK.storage_offset()),
                orig_dtype=param_dtype,
            )
            return unsharded_tensor, unsharded_tensor.fsdp_managed_tensors()

        # On later unshards, FSDP reuses the same logical unsharded tensor and
        # passes it back through out=. Refill its inner tensors in-place so
        # module and autograd references remain valid across resharding.
        local_out = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(local_out, MXFP8FSDPComputeWeight):
            raise TypeError(
                "Expected an MXFP8FSDPComputeWeight or DTensor containing one, "
                f"got {type(out)}."
            )

        operands = quantize_mxfp8_weight(weight_NK)
        quantized_tensors = (
            operands.q_weight_fprop_KN,
            operands.s_weight_fprop_blocked,
            operands.s_weight_dgrad_blocked,
        )
        existing_tensors = local_out.fsdp_managed_tensors()
        # FSDP re-materializes the same logical unsharded parameter before
        # backward. Its released inner storage is repopulated in-place, just
        # like FSDP's own all-gather output buffers.
        with torch.autograd._unsafe_preserve_version_counter(existing_tensors):
            for existing_tensor, quantized_tensor in zip(
                existing_tensors,
                quantized_tensors,
                strict=True,
            ):
                existing_tensor.copy_(quantized_tensor)
        return None


class MXFP8FSDPComputeWeight(torch.Tensor):
    """Logical BF16 weight backed by 32x32 MXFP8 operands."""

    @staticmethod
    def __new__(
        cls,
        operands: MXFP8WeightOperands,
        *,
        logical_shape: torch.Size,
        logical_stride: tuple[int, ...],
        logical_storage_offset: int,
        orig_dtype: torch.dtype,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            logical_shape,
            strides=logical_stride,
            storage_offset=logical_storage_offset,
            dtype=orig_dtype,
            device=operands.q_weight_fprop_KN.device,
            layout=torch.strided,
        )

    def __init__(
        self,
        operands: MXFP8WeightOperands,
        *,
        logical_shape: torch.Size,
        logical_stride: tuple[int, ...],
        logical_storage_offset: int,
        orig_dtype: torch.dtype,
    ) -> None:
        self.q_weight_fprop_KN = operands.q_weight_fprop_KN
        self.s_weight_fprop_blocked = operands.s_weight_fprop_blocked
        self.q_weight_dgrad_NK = operands.q_weight_dgrad_NK
        self.s_weight_dgrad_blocked = operands.s_weight_dgrad_blocked
        self.logical_shape = logical_shape
        self.logical_stride = logical_stride
        self.logical_storage_offset = logical_storage_offset
        self.orig_dtype = orig_dtype

    # pyrefly: ignore [bad-param-name-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    def operands(self) -> MXFP8WeightOperands:
        return MXFP8WeightOperands(
            q_weight_fprop_KN=self.q_weight_fprop_KN,
            s_weight_fprop_blocked=self.s_weight_fprop_blocked,
            q_weight_dgrad_NK=self.q_weight_dgrad_NK,
            s_weight_dgrad_blocked=self.s_weight_dgrad_blocked,
        )

    def fsdp_managed_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the independent tensor allocations managed by FSDP.

        The unsharded tensor exposes four logical GEMM operands. DGRAD qdata is a
        transpose view of FPROP qdata, so FSDP manages only the FPROP qdata and
        the two blocked-scale tensors as independent allocations. FSDP keeps
        these objects stable and allocates, frees, or refills their storage
        across unshard and reshard transitions.
        """
        return (
            self.q_weight_fprop_KN,
            self.s_weight_fprop_blocked,
            self.s_weight_dgrad_blocked,
        )

    def __repr__(self, *, tensor_contents: Any | None = None) -> str:
        del tensor_contents
        return (
            "MXFP8FSDPComputeWeight("
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"
        )

    def new_view(
        self,
        shape: torch.Size,
        stride: tuple[int, ...],
        storage_offset: int,
    ) -> "MXFP8FSDPComputeWeight":
        return type(self)(
            self.operands(),
            logical_shape=shape,
            logical_stride=stride,
            logical_storage_offset=storage_offset,
            orig_dtype=self.orig_dtype,
        )

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(
        cls,
        func,
        types,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        del types
        kwargs = kwargs or {}
        weight = args[0]
        if func == aten.detach.default:
            return weight.new_view(
                weight.logical_shape,
                weight.logical_stride,
                weight.logical_storage_offset,
            )
        if func == aten.as_strided.default:
            shape = torch.Size(args[1])
            stride = tuple(args[2])
            storage_offset = (
                args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
            )
            return weight.new_view(shape, stride, storage_offset)
        raise NotImplementedError(f"{cls.__name__} does not implement {func}.")

    def __tensor_flatten__(self):
        metadata = (
            self.logical_shape,
            self.logical_stride,
            self.logical_storage_offset,
            self.orig_dtype,
        )
        return [
            "q_weight_fprop_KN",
            "s_weight_fprop_blocked",
            "s_weight_dgrad_blocked",
        ], metadata

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        metadata,
        outer_size,
        outer_stride,
    ):
        del outer_size, outer_stride
        (
            logical_shape,
            logical_stride,
            logical_storage_offset,
            orig_dtype,
        ) = metadata
        q_weight_fprop_KN = inner_tensors["q_weight_fprop_KN"]
        operands = MXFP8WeightOperands(
            q_weight_fprop_KN=q_weight_fprop_KN,
            s_weight_fprop_blocked=inner_tensors["s_weight_fprop_blocked"],
            q_weight_dgrad_NK=q_weight_fprop_KN.t(),
            s_weight_dgrad_blocked=inner_tensors["s_weight_dgrad_blocked"],
        )
        return cls(
            operands,
            logical_shape=logical_shape,
            logical_stride=logical_stride,
            logical_storage_offset=logical_storage_offset,
            orig_dtype=orig_dtype,
        )


__all__ = ["MXFP8FSDPComputeWeight", "MXFP8FSDPWeight"]
