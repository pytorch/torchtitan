# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.cuda
import torch.distributed.tensor

import wandb
from torch import nn
from torch.autograd import Function

from torch.distributed.tensor import DTensor

from torch.distributed.tensor.experimental import register_sharding


IS_HISTC_SUPPORTED = None


def log_tensor_stats(tensor: torch.Tensor, name: str, num_bins: int = 64):  # noqa: C901
    global IS_HISTC_SUPPORTED
    """Add distribution statistics on a tensor's elements to the current History entry."""
    # TODO Handle the case of duplicate names.
    tensor = tensor.detach()
    flat = tensor.reshape(-1)

    if flat.is_cuda:
        if IS_HISTC_SUPPORTED is None:
            try:
                flat.histc(bins=num_bins)
            except RuntimeError:
                IS_HISTC_SUPPORTED = False
            else:
                IS_HISTC_SUPPORTED = True

        # As of torch 1.0.1.post2+nightly, float16 cuda summary ops are not supported (convert to float32)
        if not IS_HISTC_SUPPORTED:
            flat = flat.cpu()
        elif not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
            flat = flat.type(torch.cuda.FloatTensor)

    # Since we use histc, we need to make sure that torch supports the operation on CPU,
    # otherwise we'll get a runtime error. Hence, we need to upcast to float32.
    if not flat.is_cuda and not isinstance(
        flat, (torch.FloatTensor, torch.DoubleTensor)
    ):
        flat = flat.type(torch.FloatTensor)

    # Skip logging if all values are nan or inf or the tensor is empty.
    if flat.shape == torch.Size([0]) or (~torch.isfinite(flat)).all().item():
        return

    # Remove nans and infs if present. There's no good way to represent that in histograms.
    if not torch.isfinite(flat).all():
        flat = flat[torch.isfinite(flat)]

    tmin = flat.min().item()
    tmax = flat.max().item()
    # Anecdotally, this can somehow happen sometimes. Maybe a precision error
    # in min()/max() above. Swap here to prevent a runtime error.
    # If all values are equal, just return a single bin.
    if tmin > tmax:
        tmin, tmax = tmax, tmin
    if tmin == tmax:
        tensor = torch.Tensor([flat.numel()])
        tensor = tensor.cpu().clone().detach()
        bins = torch.Tensor([tmin, tmax])
    else:
        tensor = flat.histc(bins=num_bins, min=tmin, max=tmax)
        tensor = tensor.cpu().detach().clone()
        bins = torch.linspace(tmin, tmax, steps=num_bins + 1)

    wandb.run._log(
        {name: wandb.Histogram(np_histogram=(tensor.tolist(), bins.tolist()))},
        commit=False,
    )


def masked_mean(tensor, mask, per_seq=False):
    if per_seq:
        return ((tensor * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()
    else:
        return (tensor * mask).sum() / mask.sum()


def masked_sum(tensor, mask, per_seq=False):
    if per_seq:
        return (tensor * mask).sum(dim=-1)
    else:
        return (tensor * mask).sum()


@register_sharding(torch.ops.aten.amax.default)
def custom_amax_sharding(x, dim, keepdim):
    if isinstance(dim, list):
        if len(dim) == 1:
            dim = dim[0]
        else:
            raise ValueError(f"dim must be a single integer, got {dim}")
    amax_dim = dim if dim >= 0 else dim + x.ndim
    out_sharding = [torch.distributed.tensor.Partial(reduce_op="max"), None, None]
    in_sharding = [torch.distributed.tensor.Shard(amax_dim)]
    return [(out_sharding, in_sharding)]


@register_sharding(torch.ops.aten.amin.default)
def custom_amin_sharding(x, dim, keepdim):
    if isinstance(dim, list):
        if len(dim) == 1:
            dim = dim[0]
        else:
            raise ValueError(f"dim must be a single integer, got {dim}")
    amax_dim = dim if dim >= 0 else dim + x.ndim
    out_sharding = [torch.distributed.tensor.Partial(reduce_op="min"), None, None]
    in_sharding = [torch.distributed.tensor.Shard(amax_dim)]
    return [(out_sharding, in_sharding)]


def local_std(x: DTensor, dim: Optional[int] = None, keepdim: bool = False) -> DTensor:
    """
    Compute the local standard deviation of a tensor.
    """
    local_x = x.to_local()
    return DTensor.from_local(
        torch.std(local_x, dim=dim, keepdim=keepdim),
        device_mesh=x.device_mesh,
        placements=[torch.distributed.tensor.Partial(reduce_op="avg")],
    ).redistribute(
        device_mesh=x.device_mesh,
        placements=[torch.distributed.tensor.Replicate()],
    )


class VocabParallelEntropyFunction(Function):
    """
    Fused entropy loss computation with efficient backward pass.
    Saves only necessary tensors for gradient computation.
    """

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass computing entropy loss with mixed precision for stability.

        Args:
            logits: Local tensor [B*S, local_vocab]

        Returns:
            entropy_loss: Per-token entropy loss [B*S]
        """
        input_dtype = logits.dtype

        # Find global max for numerical stability
        if isinstance(logits, DTensor):
            logit_max = torch.amax(logits, dim=-1, keepdim=True).redistribute(
                device_mesh=logits.device_mesh,
                placements=[torch.distributed.tensor.Replicate()],
            )
        else:
            logit_max = torch.amax(logits, dim=-1, keepdim=True)

        # Compute stable softmax
        shifted_logits = logits - logit_max
        exp_logits = shifted_logits.exp()

        global_sum_exp = exp_logits.sum(dim=-1, keepdim=True)

        # Probabilities in original dtype
        probs = exp_logits / global_sum_exp

        # Cast to fp32 for log operations (more stable)
        # global_sum_exp_fp32 = global_sum_exp.float()
        # shifted_logits_fp32 = shifted_logits.float()
        # probs_fp32 = probs.float()

        # Compute log_probs in fp32
        log_probs = shifted_logits - global_sum_exp.log()

        # Entropy loss in fp32
        entropy_loss = torch.sum(probs * log_probs, dim=-1)

        # Save in original dtype to save memory
        ctx.save_for_backward(probs, log_probs)

        return entropy_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for entropy loss with mixed precision handling.
        """
        probs, log_probs = ctx.saved_tensors

        # # Cast to fp32 for stability if needed
        # if probs.dtype != torch.float32:
        #     probs_fp32 = probs.float()
        #     log_probs_fp32 = log_probs.float()

        #     # Compute entropy for each sample in fp32
        #     entropy_per_sample = torch.sum(probs_fp32 * log_probs_fp32, dim=-1, keepdim=True)

        #     # Gradient in fp32
        #     grad_logits = probs_fp32 * (log_probs_fp32 - entropy_per_sample)

        #     # Scale by incoming gradient and cast back
        #     grad_logits = grad_logits * grad_output.unsqueeze(-1).float()
        #     grad_logits = grad_logits.to(probs.dtype)
        # else:
        #     # Already fp32
        entropy_per_sample = torch.sum(probs * log_probs, dim=-1, keepdim=True)
        grad_logits = probs * (log_probs - entropy_per_sample)
        grad_logits = grad_logits * grad_output.unsqueeze(-1)

        return grad_logits


class VocabParallelEntropyLoss(nn.Module):
    """
    Entropy loss for vocabulary parallel outputs in TorchTitan.
    Handles distributed softmax computation with numerical stability.
    """

    def __init__(self, process_group=None):
        super().__init__()
        self.process_group = process_group

    def forward(self, logits: DTensor):
        """
        Compute entropy loss for vocabulary parallel logits.

        Uses https://arxiv.org/abs/1805.02867 method to calculate softmax locally

        Args:
            logits: Local logits tensor [batch_size, seq_len, local_vocab_size]
            target: Optional targets for masking (not used in pure entropy, but useful for masked LM)

        Returns:
            entropy_loss: Scalar entropy loss
        """
        # Get dimensions
        batch_size, seq_len, local_vocab_size = logits.shape

        # Step 1: Find max for numerical stability
        # gets local max, then all_reduces to get global max
        logit_max = torch.amax(logits, dim=-1, keepdim=True).redistribute(
            device_mesh=logits.device_mesh,
            placements=[torch.distributed.tensor.Replicate()],
        )  # [B, S, 1]

        # Step 3: Subtract global max from local logits
        shifted_logits = logits - logit_max

        # Step 4: Compute exp of shifted logits
        exp_logits = shifted_logits.exp()

        # Step 5: Sum exp to get global sum through DTensor mapping
        global_sum_exp = exp_logits.sum(dim=-1, keepdim=True)  # [B, S, 1]

        # Step 6: Compute local probabilities using global denominator
        probs = exp_logits / global_sum_exp  # [B, S, local_vocab]
        # log(p) = log(exp(shifted_logits) / sum(exp(shifted_logits)))
        #        = log(exp(shifted_logits)) - log(sum(exp(shifted_logits)))
        #        = shifted_logits - log(global_sum_exp)
        logp = shifted_logits - global_sum_exp.log()
        # Step 7: Compute entropy contribution: p * log(p)
        # Add small epsilon to avoid log(0)
        entropy = torch.sum(probs * logp, dim=-1)  # [B, S]
        return entropy
