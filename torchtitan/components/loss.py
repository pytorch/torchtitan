# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeAlias

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def build_mse_loss(compile_config: CompileConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)
    return loss_fn


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Usage:
        accumulator = GradAccumulator(hidden_states, num_chunks=4, dtype=torch.float32)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.result()
    """

    def __init__(
        self,
        reference: torch.Tensor,
        num_chunks: int,
        *,
        seq_dim: int = 1,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the gradient accumulator.

        Args:
            reference: Reference tensor to get shape, device, and dtype from.
                The buffer will have the same shape as this tensor.
            num_chunks: Number of chunks that will be added.
            seq_dim: The sequence dimension along which chunks are accumulated.
            dtype: Optional dtype for the buffer. If None, uses the reference dtype.
        """
        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        buffer_dtype = dtype if dtype is not None else reference.dtype
        self._buffer = torch.zeros_like(reference, dtype=buffer_dtype)
        self._next_idx = 0

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        if self._next_idx >= self.num_chunks:
            raise ValueError(
                f"Already added {self.num_chunks} chunks, cannot add more"
            )

        if chunk_grad.dtype != self._buffer.dtype:
            chunk_grad = chunk_grad.to(self._buffer.dtype)

        chunk_seq_len = chunk_grad.shape[self.seq_dim]
        start = self._next_idx * chunk_seq_len
        end = start + chunk_seq_len

        slices = [slice(None)] * self._buffer.ndim
        slices[self.seq_dim] = slice(start, end)
        self._buffer[tuple(slices)] = chunk_grad

        self._next_idx += 1

    def result(self) -> torch.Tensor:
        """Return the accumulated gradient tensor."""
        return self._buffer


@contextmanager
def _no_reshard_after_backward(model: nn.Module):
    """Temporarily disable FSDP2 reshard_after_backward for the model.

    This avoids repeated all-gathers of lm_head weights when computing
    chunked loss, since each chunk triggers a backward through lm_head.

    In torchtitan, lm_head (``model.output``) and norm are typically grouped
    together in one FSDP unit via ``fully_shard([model.norm, model.output])``,
    and the top-level model is also wrapped with ``fully_shard(model)``.
    We disable reshard_after_backward on the top-level FSDPModule to prevent
    any resharding during the chunk loop. This is a slightly broader scope
    than strictly necessary (also keeps transformer layer params unsharded
    through the chunk loop), but the extra memory cost is negligible since
    transformer layer params are typically already managed by separate FSDP
    units that reshard after their own backward.

    Non-FSDP modules are silently ignored.
    """
    from torch.distributed._composable.fsdp import FSDPModule

    original_settings: list[bool] = []

    try:
        if isinstance(model, FSDPModule):
            state = model._get_fsdp_state()
            for pg in state._fsdp_param_groups:
                original_settings.append(pg.reshard_after_backward)
            model.set_reshard_after_backward(False)
        yield
    finally:
        if isinstance(model, FSDPModule):
            state = model._get_fsdp_state()
            for i, pg in enumerate(state._fsdp_param_groups):
                if i < len(original_settings):
                    pg.reshard_after_backward = original_settings[i]
            model.reshard()


class ChunkedCELoss:
    """Chunked cross-entropy loss that splits the sequence dimension to reduce peak memory.

    Instead of materializing the full [B, L, V] logits tensor at once, this splits
    the hidden states into N chunks along the sequence dimension and computes
    lm_head + cross_entropy_loss on each chunk sequentially. This reduces peak memory
    from O(B*L*V) to O(B*L/N*V).

    The flow:
    1. Model forward with skip_lm_head=True to get hidden states [B, L, D]
    2. Detach hidden states at the boundary
    3. Split detached hidden states into N chunks along seq dim
    4. For each chunk: lm_head(chunk) -> ce_loss(logits, labels_chunk) -> backward()
    5. Assemble chunk gradients into a full gradient [B, L, D] via GradAccumulator
    6. Backward through the decoder with the accumulated gradient

    Composability:
    - FSDP2: Temporarily disables reshard_after_backward on lm_head to avoid
      N all-gathers (one per chunk). Uses _no_reshard_after_backward context manager.
    - TP with Loss Parallel: If SP produces Shard placement on TP mesh, we
      redistribute to Replicate before chunking, then chunk the local tensor.
      Loss parallel still works within each chunk (N all-reduces total).
    - CP: Further chunks the local sequence dimension. Works out of the box.
    - Compile: ce_loss can be compiled independently; lm_head is not compiled
      (FSDP2 module would cause graph break).
    """

    def __init__(
        self,
        model: nn.Module,
        num_chunks: int,
        loss_fn: LossFunction,
    ):
        """Initialize ChunkedCELoss.

        Args:
            model: The decoder model. Must have an ``output`` attribute (the lm_head).
                The model reference is kept for FSDP2 reshard control.
            num_chunks: Number of chunks to split the sequence into.
            loss_fn: The base cross-entropy loss function (e.g. cross_entropy_loss
                or a compiled version of it).
        """
        from torchtitan.models.common.decoder import Decoder

        assert isinstance(model, Decoder), (
            f"ChunkedCELoss requires a Decoder model, got {type(model).__name__}"
        )
        assert model.output is not None, (
            "ChunkedCELoss requires the model to have an output (lm_head) layer"
        )

        self.model = model
        self.lm_head = model.output
        self.num_chunks = num_chunks
        self.loss_fn = loss_fn

    def __call__(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss with manual gradient assembly.

        This should be called after model forward with skip_lm_head=True.
        Handles detach, chunking, per-chunk lm_head + ce_loss + backward,
        gradient accumulation, and decoder backward internally.

        Args:
            hidden_states: Output of the decoder (before lm_head), shape [B, L, D].
            labels: Target labels, shape [B, L]. -100 for padding/ignored tokens.
            global_valid_tokens: Total valid token count across all DP ranks,
                used to scale each chunk's loss before backward.

        Returns:
            The scaled loss (loss_sum / global_valid_tokens) for logging.
        """
        num_chunks = self.num_chunks

        # Detach hidden states to stop gradient propagation at this boundary.
        # We'll manually backward through the decoder after all chunks.
        h_detached = hidden_states.detach().requires_grad_(True)

        # Split hidden states and labels into chunks along seq dim.
        # Use .contiguous() to break shared storage from torch.chunk(),
        # otherwise all chunks share the same underlying memory and the
        # autograd graph retains references, preventing memory from being freed.
        h_chunks = torch.chunk(h_detached, num_chunks, dim=1)
        h_chunks = [
            c.contiguous().detach().requires_grad_(True) for c in h_chunks
        ]
        label_chunks = torch.chunk(labels, num_chunks, dim=1)

        # Pre-allocate gradient accumulator (fp32 for numerical stability)
        grad_accumulator = GradAccumulator(
            h_detached, num_chunks=len(h_chunks), seq_dim=1, dtype=torch.float32
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        with _no_reshard_after_backward(self.model):
            for h_chunk, label_chunk in zip(h_chunks, label_chunks):
                # lm_head projection: [B, L/N, D] -> [B, L/N, V]
                logits = self.lm_head(h_chunk)

                # Cross-entropy loss on this chunk (sum reduction)
                chunk_loss = self.loss_fn(logits, label_chunk)
                total_loss = total_loss + chunk_loss.detach()

                # Scale loss before backward so gradients are properly normalized
                scaled_chunk_loss = chunk_loss / global_valid_tokens
                scaled_chunk_loss.backward()

                # Accumulate gradient for this chunk
                grad_accumulator.add(h_chunk.grad)

                # Free memory before next chunk
                h_chunk.grad = None
                del scaled_chunk_loss, chunk_loss, logits

        # Get the accumulated gradient and backward through the decoder
        accumulated_grad = grad_accumulator.result()
        assert accumulated_grad.dtype == torch.float32

        hidden_states.backward(accumulated_grad.to(hidden_states.dtype))

        return total_loss / global_valid_tokens


class ChunkedCELossFactory:
    """Factory for creating ChunkedCELoss after model construction.

    Since ChunkedCELoss needs the model's lm_head, and the model is not available
    at loss builder time, this factory is returned by build_chunked_cross_entropy_loss
    and called by the trainer with the model.
    """

    def __init__(self, num_chunks: int, loss_fn: LossFunction):
        self.num_chunks = num_chunks
        self.loss_fn = loss_fn

    def __call__(self, model: nn.Module) -> ChunkedCELoss:
        return ChunkedCELoss(model, num_chunks=self.num_chunks, loss_fn=self.loss_fn)


def build_chunked_cross_entropy_loss(
    compile_config: CompileConfig,
    *,
    num_chunks: int = 8,
    parallel_dims=None,
    **kwargs,
) -> ChunkedCELossFactory:
    """Build a factory for ChunkedCELoss.

    Since ChunkedCELoss needs the model's lm_head, and the model is not available
    at loss builder time, this returns a ``ChunkedCELossFactory`` that the trainer
    calls with the model to create the fully initialized ChunkedCELoss.
    """
    del parallel_dims, kwargs

    loss_fn = cross_entropy_loss
    if compile_config.enable and "loss" in compile_config.components:
        logger.info("Compiling the ce_loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=compile_config.backend)

    return ChunkedCELossFactory(num_chunks=num_chunks, loss_fn=loss_fn)
