# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.components.loss import ChunkedLossWrapper


class LocalMapChunkedLossWrapperWithParamGrads(ChunkedLossWrapper):
    """Experimental graph-trainer chunked loss using a local-map region.

    This prototype supports a bias-free linear lm_head under one-dimensional
    simple FSDP. The region receives one replicated weight, computes and
    accumulates all per-chunk gradients locally, and returns the weight gradient
    with Partial placement for one final reduce-scatter.
    """

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from torch.distributed._composable.fsdp import FSDPModule

        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
        assert not isinstance(lm_head, FSDPModule)
        assert isinstance(lm_head, nn.Linear) and lm_head.bias is None
        lm_head_params = tuple(lm_head.named_parameters())
        assert len(lm_head_params) == 1 and lm_head_params[0][0] == "weight"
        sharded_weight = lm_head_params[0][1]
        assert isinstance(sharded_weight, DTensor)
        assert sharded_weight.device_mesh.ndim == 1
        assert sharded_weight.placements == (Shard(0),)
        assert not isinstance(pred, DTensor)
        assert not isinstance(labels, DTensor)
        assert not isinstance(global_valid_tokens, DTensor)
        assert pred.requires_grad
        assert not loss_inputs

        mesh = sharded_weight.device_mesh
        replicated_placements = (Replicate(),)
        partial_placements = (Partial(reduce_op="sum"),)
        replicated_weight = sharded_weight.redistribute(
            placements=replicated_placements,
            forward_dtype=pred.dtype,
            backward_dtype=torch.float32,
        )

        def _local_chunked_loss(
            hidden_states: torch.Tensor,
            local_labels: torch.Tensor,
            weight: torch.Tensor,
            valid_tokens: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            seq_len = hidden_states.shape[1]
            torch._check(
                seq_len % self.num_chunks == 0,
                lambda: (
                    "LocalMapChunkedLossWrapperWithParamGrads sequence length "
                    "must be divisible by num_chunks"
                ),
            )
            chunk_len = seq_len // self.num_chunks
            hidden_chunks = tuple(
                chunk.detach().requires_grad_(True)
                for chunk in torch.split(
                    hidden_states,
                    [chunk_len] * self.num_chunks,
                    dim=1,
                )
            )
            label_chunks = torch.split(
                local_labels,
                [chunk_len] * self.num_chunks,
                dim=1,
            )
            local_weight = weight.detach().requires_grad_(True)
            hidden_grad = torch.zeros_like(hidden_states, dtype=torch.float32)
            weight_grad: torch.Tensor | None = None
            total_loss = hidden_states.new_zeros((), dtype=torch.float32)

            for chunk_idx, (hidden_chunk, label_chunk) in enumerate(
                zip(hidden_chunks, label_chunks, strict=True)
            ):
                logits = F.linear(hidden_chunk, local_weight)
                chunk_loss, chunk_metrics = self.loss_fn(
                    logits,
                    label_chunk,
                    valid_tokens,
                )
                assert not chunk_metrics
                total_loss = total_loss + chunk_loss.detach()
                chunk_hidden_grad, chunk_weight_grad = torch.autograd.grad(
                    chunk_loss,
                    (hidden_chunk, local_weight),
                )
                hidden_grad.narrow(1, chunk_idx * chunk_len, chunk_len).copy_(
                    chunk_hidden_grad
                )
                if weight_grad is None:
                    weight_grad = chunk_weight_grad.to(torch.float32)
                else:
                    weight_grad.add_(chunk_weight_grad)

            assert weight_grad is not None
            return (
                total_loss,
                hidden_grad.to(hidden_states.dtype),
                weight_grad,
            )

        local_region = local_map(
            _local_chunked_loss,
            out_placements=(
                partial_placements,
                replicated_placements,
                partial_placements,
            ),
            in_placements=(
                None,
                None,
                replicated_placements,
                None,
            ),
            device_mesh=mesh,
        )
        total_loss, accumulated_hidden_grad, partial_weight_grad = local_region(
            pred,
            labels,
            replicated_weight,
            global_valid_tokens,
        )
        sharded_weight_grad = partial_weight_grad.redistribute(
            placements=sharded_weight.placements
        )

        return (
            _PrecomputedLocalMapGrads.apply(
                pred,
                total_loss.to_local(),
                accumulated_hidden_grad.to_local(),
                sharded_weight,
                sharded_weight_grad,
            ),
            {},
        )


class _PrecomputedLocalMapGrads(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        total_loss: torch.Tensor,
        accumulated_hidden_grad: torch.Tensor,
        sharded_weight: DTensor,
        sharded_weight_grad: DTensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(accumulated_hidden_grad, sharded_weight_grad)
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        accumulated_hidden_grad, sharded_weight_grad = ctx.saved_tensors
        return (
            accumulated_hidden_grad,
            None,
            None,
            sharded_weight_grad,
            None,
        )
