# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig, Configurable
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


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


class BaseLoss(Configurable):
    """Base class for all loss functions.

    Provides compile support and a unified ``__call__`` signature:
    ``(pred, labels, global_valid_tokens) -> scaled_loss``.
    Subclasses set ``self.fn`` to their raw loss function in ``__init__``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def _maybe_compile(self, compile_config: CompileConfig | None) -> None:
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.info("Compiling the loss function with torch.compile")
            self.fn = torch.compile(self.fn, backend=compile_config.backend)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.fn(pred, labels) / global_valid_tokens


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with sum reduction for token-based normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)


class MSELoss(BaseLoss):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = mse_loss
        self._maybe_compile(compile_config)


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Args:
        reference: Reference tensor to derive shape, device, and DTensor metadata.
            If a DTensor, result() returns a DTensor with matching placements.
        num_chunks: Number of chunks that will be added.
        seq_dim: The sequence dimension along which chunks are accumulated.
        dtype: Dtype for the buffer.

    Usage:
        accumulator = GradAccumulator(hidden_states, num_chunks=4, dtype=torch.float32)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.result()
    """

    def __init__(
        self,
        reference: torch.Tensor,
        *,
        num_chunks: int,
        seq_dim: int = 1,
        dtype: torch.dtype,
    ):
        from torch.distributed.tensor import DTensor

        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0

        # Track DTensor metadata for transparent wrap-back in result()
        if isinstance(reference, DTensor):
            self._device_mesh = reference.device_mesh
            self._placements = reference.placements
            local = reference.to_local()
        else:
            self._device_mesh = None
            self._placements = None
            local = reference

        self._buffer = torch.zeros_like(local, dtype=dtype)

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        from torch.distributed.tensor import DTensor

        if self._next_idx >= self.num_chunks:
            raise ValueError(f"Already added {self.num_chunks} chunks, cannot add more")

        # Extract local tensor if DTensor
        if isinstance(chunk_grad, DTensor):
            chunk_grad = chunk_grad.to_local()

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
        """Return the accumulated gradient tensor, wrapped as DTensor if needed."""
        from torch.distributed.tensor import DTensor

        if self._device_mesh is not None:
            return DTensor.from_local(
                self._buffer,
                device_mesh=self._device_mesh,
                placements=self._placements,
            )
        return self._buffer


class ChunkedCELoss(BaseLoss):
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
        unsharded across all chunks (avoiding repeated all-gathers). Reduce-scatter
        fires per-chunk, and FSDP2 accumulates the sharded gradients correctly.

    TP / SP composability:
        Hidden states are redistributed to ``Replicate()`` on the TP mesh
        before chunking, so each chunk enters the lm_head as ``Replicate()``
        input regardless of whether SP is enabled. With SP, this is an
        all-gather from ``Shard(1)``; without SP, it's a no-op.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: ce_loss can be compiled independently; lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.num_chunks = config.num_chunks
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        ``pred`` should be hidden states from model forward with
        ``skip_lm_head=True``.

        Returns a differentiable loss. When ``.backward()`` is called on it
        (either by the trainer or the PP schedule), it triggers backward
        through the decoder via a custom autograd Function.
        """
        from torch.distributed._composable.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, Replicate

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        # If SP is enabled, hidden states are Shard(1) on the TP mesh dim.
        # Redistribute only the TP dim to Replicate before chunking so that
        # the lm_head receives Replicate input on TP. Other mesh dims (e.g.
        # CP) are left unchanged.
        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            if mesh.mesh_dim_names is not None and "tp" in mesh.mesh_dim_names:
                tp_dim = mesh.mesh_dim_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_dim], Replicate):
                    placements[tp_dim] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        # Detach hidden states to stop gradient propagation at this boundary.
        h_detached = hidden_states.detach().requires_grad_(True)

        # Split hidden states and labels into chunks along seq dim.
        # Use .contiguous() to break shared storage from torch.chunk(),
        # otherwise all chunks share the same underlying memory and the
        # autograd graph retains references, preventing memory from being freed.
        # TODO: When CP mesh is in DTensor, chunking along dim=1 won't work
        # directly with Shard(1) on CP. Need local_map to operate on local tensors
        h_chunks = torch.chunk(h_detached, num_chunks, dim=1)
        h_chunks = [c.contiguous().detach().requires_grad_(True) for c in h_chunks]
        label_chunks = torch.chunk(labels, num_chunks, dim=1)

        # Pre-allocate gradient accumulator in fp32 for numerical stability.
        # Pass h_detached as reference so GradAccumulator tracks DTensor
        # metadata and returns a DTensor from result() if needed.
        grad_accumulator = GradAccumulator(
            h_detached,
            num_chunks=num_chunks,
            dtype=torch.float32,
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        # Disable FSDP reshard on lm_head to keep weight unsharded across
        # all chunks, avoiding repeated all-gathers. Coalesce per-chunk
        # grad sync into a single reduce-scatter at the last chunk by
        # disabling gradient sync for chunks 0..N-2.
        if fsdp_enabled:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)

        last_idx = len(h_chunks) - 1
        for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
            if fsdp_enabled and i == last_idx:
                lm_head.set_requires_gradient_sync(True, recurse=False)

            logits = lm_head(h_chunk)

            chunk_loss_sum = self.fn(logits, label_chunk)
            chunk_loss = chunk_loss_sum / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()

            chunk_loss.backward()

            # Collect this chunk's gradient and free it before the next
            # chunk to keep only one chunk's activations in memory.
            grad_accumulator.add(h_chunk.grad)
            h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        # Return a differentiable loss via DecoderOutputGradientBackProp. When
        # .backward() is called (by the trainer or PP schedule), autograd
        # calls DecoderOutputGradientBackProp.backward which returns accumulated_grad
        # as the gradient for hidden_states, propagating through the decoder.
        return DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


class DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden_states (connected to decoder graph), the accumulated
    gradient from chunked lm_head backward, and the loss value. Returns the
    loss value as a differentiable tensor.

    Backward returns accumulated_grad as the gradient for hidden_states.
    Autograd then propagates this through the decoder layers automatically —
    no explicit hidden_states.backward() needed.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(accumulated_grad)
        # Return a tensor with the correct loss value. We clone to avoid
        # in-place issues, and the grad_fn comes from this Function.
        return loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (accumulated_grad,) = ctx.saved_tensors
        # Return accumulated_grad as the gradient for hidden_states.
        # Autograd then propagates this through hidden_states' existing
        # decoder graph — equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        return accumulated_grad, None, None
