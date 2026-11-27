# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch_remat as remat

from torchtitan.models.base.module import Module

from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.linear import Linear

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "DenseFeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """Base class for dense and mixture-of-experts feed-forward modules."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass


class DenseFeedForward(FeedForward):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the gate and up projections as ``[2, F, D]``. Config takes the
    **final** hidden_dim (no internal 2/3 scaling). Use
    compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        w13: Linear.Config
        w2: Linear.Config
        activation_fn: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)
        num_token_chunks: int = 1
        """Above 1, run over this many token chunks and save only the input.
        Backward recomputes one chunk at a time, so only one chunk's
        ``[T / num_token_chunks, 2, F]`` intermediates are alive, at the cost of
        one more forward of this module. For long context, where these
        intermediates dominate peak memory."""

        def __post_init__(self) -> None:
            FeedForward.Config.__post_init__(self)
            if self.num_token_chunks < 1:
                raise ValueError(
                    f"num_token_chunks must be positive, got {self.num_token_chunks}"
                )

    def __init__(self, config: Config):
        super().__init__()
        if config.w13.num_linears != 2:
            raise ValueError(
                "DenseFeedForward w13 requires num_linears=2 for gate/up projections"
            )
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()
        self.num_token_chunks = config.num_token_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_token_chunks == 1:
            return self._forward(x)
        return _TokenChunkedForward.apply(self, x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # w13 and w2 declare their own remat regions.
        gate_up_T2F = self.w13(x)
        remat.recompute_needs_tensor(gate_up_T2F)
        return self.w2(self.activation_fn(gate_up_T2F))


class _TokenChunkedForward(torch.autograd.Function):
    """``DenseFeedForward`` over token chunks, saving only the input.

    Forward runs each chunk without autograd. Backward reruns one chunk at a
    time with autograd and backpropagates it, so the parameters' ``.grad``
    accumulates chunk by chunk (in place, in their grad dtype, when ``Linear``
    uses in-place WGRAD accumulation).
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, module: DenseFeedForward, x: torch.Tensor
    ) -> torch.Tensor:
        ctx.module = module
        ctx.save_for_backward(x)
        out_chunks = []
        for index, x_chunk in enumerate(_token_chunks(module, x)):
            # Remat region names must be unique within a block's forward.
            with remat.name_scope(f"token_chunk_{index}"):
                out_chunks.append(module._forward(x_chunk))
        return torch.cat(out_chunks, dim=-2)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        (x,) = ctx.saved_tensors
        module = ctx.module
        grad_input = torch.empty_like(x)
        for x_chunk, grad_chunk, grad_input_chunk in zip(
            _token_chunks(module, x),
            _token_chunks(module, grad_output),
            _token_chunks(module, grad_input),
            strict=True,
        ):
            x_chunk = x_chunk.detach().requires_grad_()
            with torch.enable_grad():
                out_chunk = module._forward(x_chunk)
            torch.autograd.backward(out_chunk, grad_chunk)
            assert x_chunk.grad is not None
            grad_input_chunk.copy_(x_chunk.grad)
        return None, grad_input


def _token_chunks(
    module: DenseFeedForward, x: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return x.chunk(module.num_token_chunks, dim=-2)
