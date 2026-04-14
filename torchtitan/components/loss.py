# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


class CrossEntropyLoss:
    """Cross-entropy loss with sum reduction for token-based normalization.

    This is the standard loss for autoregressive language model training.
    Predictions are flattened from [B, L, V] to [B*L, V] and cast to float32.
    Labels are flattened from [B, L] to [B*L]. Tokens with label == -100
    (IGNORE_INDEX) are excluded from the loss.
    """

    def __init__(self, compile_config: CompileConfig | None = None):
        self._fn: LossFunction = self._compute
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.info("Compiling the loss function with torch.compile")
            self._fn = torch.compile(self._fn, backend=compile_config.backend)

    @staticmethod
    def _compute(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(),
            labels.flatten(0, 1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._fn(pred, labels)


class MSELoss:
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    def __init__(self, compile_config: CompileConfig | None = None):
        self._fn: LossFunction = self._compute
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.info("Compiling the loss function with torch.compile")
            self._fn = torch.compile(self._fn, backend=compile_config.backend)

    @staticmethod
    def _compute(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            pred.float(), labels.float().detach(), reduction="sum"
        )

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._fn(pred, labels)


# Standalone functions for use in tests and experiments that don't need
# the compile-aware class wrapper.
cross_entropy_loss = CrossEntropyLoss._compute
mse_loss = MSELoss._compute


def build_cross_entropy_loss(compile_config: CompileConfig, **kwargs):
    """Build a CrossEntropyLoss instance."""
    del kwargs
    return CrossEntropyLoss(compile_config)


def build_mse_loss(compile_config: CompileConfig, **kwargs):
    """Build an MSELoss instance."""
    del kwargs
    return MSELoss(compile_config)


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Args:
        shape: Shape of the full gradient buffer (e.g. [B, L, D]).
        device: Device for the buffer.
        dtype: Dtype for the buffer.
        num_chunks: Number of chunks that will be added.
        seq_dim: The sequence dimension along which chunks are accumulated.

    Usage:
        accumulator = GradAccumulator((B, L, D), device="cuda", dtype=torch.float32, num_chunks=4)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.result()
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        num_chunks: int,
        seq_dim: int = 1,
    ):
        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._buffer = torch.zeros(shape, device=device, dtype=dtype)
        self._next_idx = 0

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        if self._next_idx >= self.num_chunks:
            raise ValueError(f"Already added {self.num_chunks} chunks, cannot add more")

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
    4. Disable FSDP reshard on lm_head to keep weight unsharded across chunks
    5. For each chunk: lm_head(chunk) -> ce_loss -> backward()
    6. Assemble chunk gradients into a full gradient [B, L, D] via GradAccumulator
    7. Backward through the decoder via hidden_states.backward(accumulated_grad)

    FSDP2 composability:
        The lm_head's FSDP reshard-after-forward and reshard-after-backward are
        temporarily disabled during the chunked loop so that the weight stays
        unsharded across all chunks (avoiding repeated all-gathers). Gradient
        sync is also disabled so that reduce-scatter is deferred until the
        decoder backward pass, where the lm_head's FSDP group participates
        in the normal backward hook chain.

    TP with Loss Parallel:
        Loss parallel still works within each chunk since the ``loss_parallel()``
        context wraps the entire chunked computation.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: ce_loss can be compiled independently; lm_head is not compiled.

    Args:
        model: The decoder model. Must have an ``output`` attribute (the lm_head).
        num_chunks: Number of chunks to split the sequence into.
        loss_fn: The base cross-entropy loss function (e.g. CrossEntropyLoss
            instance or a plain callable).
    """

    def __init__(
        self,
        model: nn.Module,
        num_chunks: int,
        loss_fn: LossFunction,
    ):
        from torchtitan.models.common.decoder import Decoder

        assert isinstance(
            model, Decoder
        ), f"ChunkedCELoss requires a Decoder model, got {type(model).__name__}"
        assert (
            model.output is not None
        ), "ChunkedCELoss requires the model to have an output (lm_head) layer"

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
                Must be connected to the model's autograd graph (not detached).
            labels: Target labels, shape [B, L]. -100 for padding/ignored tokens.
            global_valid_tokens: Total valid token count across all DP ranks,
                used to scale each chunk's loss before backward.

        Returns:
            The scaled loss (loss_sum / global_valid_tokens) for logging.
        """
        from torch.distributed._composable.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, Replicate

        num_chunks = self.num_chunks
        lm_head = self.lm_head
        is_fsdp = isinstance(lm_head, FSDPModule)

        # If SP is enabled, hidden states are Shard(1) on the TP mesh (each
        # rank has [B, L/tp, D]). Redistribute to Replicate so that each rank
        # has the full [B, L, D] before chunking. This all-gather ensures the
        # chunked lm_head receives Replicate input, producing correct Shard(-1)
        # output for loss parallel. The backward through redistribute converts
        # the Replicate gradient back to Shard(1) for the decoder backward.
        if isinstance(hidden_states, DTensor):
            placements = hidden_states.placements
            new_placements = tuple(
                Replicate() if not isinstance(p, Replicate) else p for p in placements
            )
            if new_placements != placements:
                hidden_states = hidden_states.redistribute(
                    hidden_states.device_mesh, new_placements
                )

        # Detach hidden states to stop gradient propagation at this boundary.
        # We'll manually backward through the decoder after all chunks.
        h_detached = hidden_states.detach().requires_grad_(True)

        # Split hidden states and labels into chunks along seq dim.
        # Use .contiguous() to break shared storage from torch.chunk(),
        # otherwise all chunks share the same underlying memory and the
        # autograd graph retains references, preventing memory from being freed.
        h_chunks = torch.chunk(h_detached, num_chunks, dim=1)
        h_chunks = [c.contiguous().detach().requires_grad_(True) for c in h_chunks]
        label_chunks = torch.chunk(labels, num_chunks, dim=1)

        # Pre-allocate gradient accumulator in fp32 for numerical stability.
        grad_accumulator = GradAccumulator(
            h_detached.shape,
            device=h_detached.device,
            dtype=torch.float32,
            num_chunks=len(h_chunks),
            seq_dim=1,
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        # Disable FSDP reshard on lm_head to keep weight unsharded across
        # all chunks, avoiding repeated all-gathers. Reduce-scatter fires
        # per-chunk, and FSDP2 accumulates the sharded gradients correctly.
        if is_fsdp:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)

        for h_chunk, label_chunk in zip(h_chunks, label_chunks):
            logits = lm_head(h_chunk)

            chunk_loss = self.loss_fn(logits, label_chunk)
            total_loss = total_loss + chunk_loss.detach()

            # Scale loss before backward so gradients are properly normalized
            scaled_chunk_loss = chunk_loss / global_valid_tokens
            scaled_chunk_loss.backward()

            # Collect this chunk's gradient and free it before the next
            # chunk to keep only one chunk's activations in memory.
            grad_accumulator.add(h_chunk.grad)
            h_chunk.grad = None
            del scaled_chunk_loss, chunk_loss, logits

        if is_fsdp:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.reshard()

        # Backward through the decoder with accumulated gradients.
        accumulated_grad = grad_accumulator.result()
        hidden_states.backward(accumulated_grad.to(hidden_states.dtype))

        return total_loss / global_valid_tokens
