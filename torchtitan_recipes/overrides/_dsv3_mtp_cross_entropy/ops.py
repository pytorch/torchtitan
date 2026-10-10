# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local BF16 CE with explicit autograd and compact normalization state."""

import spmd_types as spmd
import torch
import triton
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd.function import once_differentiable


# Keep opt-in integrations disabled until all three acceptance gates pass.
ACCEPTED = False
VOCAB_SIZE = 129280
TORCH_VERSION = "2.16.0.dev20261007+cu130"


@torch.compiler.assume_constant_result
def _is_negative_view(tensor: torch.Tensor) -> bool:
    # Dynamo guards the Negative dispatch key as part of TENSOR_MATCH, but
    # cannot represent Tensor.is_neg()'s boolean result in its FX graph.
    return tensor.is_neg()


def supports(logits: torch.Tensor, labels: torch.Tensor) -> bool:
    return (
        type(logits) in (torch.Tensor, FakeTensor)
        and type(labels) in (torch.Tensor, FakeTensor)
        and logits.is_cuda
        and labels.device == logits.device
        and logits.dtype == torch.bfloat16
        and labels.dtype == torch.int64
        and logits.ndim == 2
        and logits.shape == (4096, VOCAB_SIZE)
        and labels.shape == (4096,)
        and logits.is_contiguous()
        and labels.is_contiguous()
        and not _is_negative_view(logits)
        and not _is_negative_view(labels)
        and torch.__version__ == TORCH_VERSION
        and torch.version.cuda == "13.0"
        and triton.__version__ == "3.9.0"
        and (
            isinstance(logits, FakeTensor)
            or torch.cuda.get_device_capability(logits.device) == (10, 3)
        )
    )


def _check(logits, labels):
    if (
        logits.ndim != 2
        or logits.shape[1] != VOCAB_SIZE
        or not 0 < logits.shape[0] <= 4096
        or labels.shape != (logits.shape[0],)
        or logits.dtype != torch.bfloat16
        or labels.dtype != torch.int64
        or not logits.is_cuda
        or labels.device != logits.device
        or not logits.is_contiguous()
        or not labels.is_contiguous()
        or logits.is_neg()
        or labels.is_neg()
        or torch.__version__ != TORCH_VERSION
        or torch.version.cuda != "13.0"
        or triton.__version__ != "3.9.0"
        or not (
            isinstance(logits, FakeTensor)
            or torch.cuda.get_device_capability(logits.device) == (10, 3)
        )
    ):
        raise ValueError(
            "MTP CE requires contiguous CUDA BF16 logits[T,129280] and int64 "
            "labels[T], 1 <= T <= 4096, on the validated Torch/GB300 runtime"
        )


def _check_backward(logits, labels, stats, grad_loss):
    _check(logits, labels)
    if (
        stats.shape != (2, logits.shape[0])
        or stats.dtype != torch.float32
        or stats.device != logits.device
        or not stats.is_contiguous()
        or stats.is_neg()
        or grad_loss.shape != ()
        or grad_loss.dtype != torch.float32
        or grad_loss.device != logits.device
    ):
        raise ValueError(
            "MTP CE backward requires FP32 stats[2,T] and a scalar FP32 gradient "
            "on the logits device"
        )


@torch.library.custom_op(
    "torchtitan::dsv3_mtp_cross_entropy_forward", mutates_args=(), device_types="cuda"
)
def forward_op(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from .kernels import forward

    _check(logits, labels)
    return forward(logits, labels, ignore_index)


@forward_op.register_fake
def _forward_fake(logits, labels, ignore_index):
    _check(logits, labels)
    return (
        logits.new_empty((), dtype=torch.float32),
        logits.new_empty((2, logits.shape[0]), dtype=torch.float32),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_mtp_cross_entropy_backward", mutates_args=(), device_types="cuda"
)
def backward_op(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stats: torch.Tensor,
    grad_loss: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    from .kernels import backward

    _check_backward(logits, labels, stats, grad_loss)
    return backward(logits, labels, stats, grad_loss, ignore_index)


@backward_op.register_fake
def _backward_fake(logits, labels, stats, grad_loss, ignore_index):
    _check_backward(logits, labels, stats, grad_loss)
    return torch.empty_like(logits)


class MTPCrossEntropyFunction(torch.autograd.Function):
    """First-order CE; labels must be valid vocabulary IDs or ignore_index."""

    @staticmethod
    def spmd_typecheck(result, *, logits, labels, ignore_index):
        spmd.rules.einsum("tv,t->", logits, labels, out=result)

    @staticmethod
    def forward(ctx, logits, labels, ignore_index):  # pyrefly: ignore[bad-override]
        loss, stats = forward_op(logits, labels, ignore_index)
        ctx.save_for_backward(logits, labels, stats)
        ctx.ignore_index = ignore_index
        ctx.set_materialize_grads(False)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):  # pyrefly: ignore[bad-override]
        if grad_loss is None:
            return None, None, None
        logits, labels, stats = ctx.saved_tensors
        gradient = backward_op(logits, labels, stats, grad_loss, ctx.ignore_index)
        return gradient, None, None


def cross_entropy_sum(logits, labels, *, ignore_index=-100):
    """Experimental local-vocabulary CE for the validated GB300 runtime.

    This direct entry point runs the candidate for measurement. Model dispatch
    must also check ``ACCEPTED`` and ``supports`` before enabling it.
    """
    return MTPCrossEntropyFunction.apply(logits, labels, ignore_index)
