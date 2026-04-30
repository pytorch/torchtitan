# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

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


class BaseLoss(ABC, Configurable):
    """Abstract base class for all loss functions.

    Provides compile support and a unified ``__call__`` signature:
    ``(pred, labels, global_valid_tokens) -> scaled_loss``.
    Subclasses must implement ``__init__`` and set ``self.fn``.
    """

    fn: LossFunction

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        ...

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
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss


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
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Placement

        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0
        self._device_mesh: DeviceMesh | None = None
        self._placements: tuple[Placement, ...] | None = None

        # Track DTensor metadata for transparent wrap-back in result()
        if isinstance(reference, DTensor):
            self._device_mesh = reference.device_mesh
            self._placements = reference.placements
            local = reference.to_local()
        else:
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
    1. Model forward with _skip_lm_head=True to get hidden states [B, L, D]
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

    def _redistribute_for_chunking(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Redistribute the TP axis of hidden_states from Shard(1) to Replicate.

        Required so each chunk enters lm_head with Replicate input on TP.
        With SP, this is an all-gather from Shard(1); without SP, it's a no-op.
        Plain tensors (non-DTensor) are returned unchanged.

        Shared by eager :meth:`__call__` (which then runs ``chunk_loss.backward()``)
        and :meth:`compute_traceable_grads` (which runs ``torch.autograd.grad``)
        so the redistribute happens once in one place.
        """
        from torch.distributed.tensor import DTensor, Replicate

        if not isinstance(hidden_states, DTensor):
            return hidden_states
        mesh = hidden_states.device_mesh
        if mesh.mesh_dim_names is None or "tp" not in mesh.mesh_dim_names:
            return hidden_states
        tp_dim = mesh.mesh_dim_names.index("tp")
        placements = list(hidden_states.placements)
        if isinstance(placements[tp_dim], Replicate):
            return hidden_states
        placements[tp_dim] = Replicate()
        return hidden_states.redistribute(mesh, tuple(placements))

    def _split_chunks(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        """Detach hidden_states + chunk hidden_states/labels along the seq axis.

        Returns ``(h_detached, h_chunks, label_chunks)`` where each ``h_chunk``
        is independently detached and ``requires_grad_`` is propagated from
        ``hidden_states`` (so validation/inference paths produce non-grad
        chunks while training paths produce grad-tracked chunks). ``.contiguous()``
        breaks the shared storage from ``torch.chunk()``.

        Shared by eager :meth:`__call__` and :meth:`compute_traceable_grads`.

        TODO: When CP mesh is in DTensor, chunking along dim=1 won't work
        directly with Shard(1) on CP. Need local_map to operate on local tensors.
        """
        requires_grad = hidden_states.requires_grad
        h_detached = hidden_states.detach().requires_grad_(requires_grad)
        h_chunks = [
            c.contiguous().detach().requires_grad_(requires_grad)
            for c in torch.chunk(h_detached, self.num_chunks, dim=1)
        ]
        label_chunks = torch.chunk(labels, self.num_chunks, dim=1)
        return h_detached, h_chunks, label_chunks

    def _compute_chunked_lm_head(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None,
        *,
        lm_head_params: list[torch.Tensor],
        mode: Literal["autograd_grad", "backward"],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None, torch.Tensor]:
        """Shared chunked forward + per-chunk lm_head backward.

        Returns ``(total_loss, accumulated_h_grad, lm_head_grads, hidden_states)``.

        - ``total_loss``: scaled scalar loss.
        - ``accumulated_h_grad``: per-chunk ``d(chunk_loss)/d(h_chunk)`` cat'd
          along the seq axis — the seed for the decoder backward.
        - ``lm_head_grads``: explicit grads for ``lm_head_params`` (for
          ``mode="autograd_grad"``); ``None`` for ``mode="backward"`` because
          they live on ``leaf.grad`` after the eager backward.
        - ``hidden_states``: post-TP-redistribute output with the autograd graph
          to the decoder still attached.

        FSDP coordination (``set_reshard_after_forward(False)`` +
        ``set_requires_gradient_sync(False)`` until last chunk) is applied only
        in ``mode="backward"`` when ``lm_head`` is an ``FSDPModule`` — that's
        the eager memory-efficient path. ``mode="autograd_grad"`` skips it
        because graph_trainer's graph passes handle bucketing/coalescing.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        assert self.lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        lm_head = self.lm_head
        fsdp_enabled = mode == "backward" and isinstance(lm_head, FSDPModule)

        hidden_states = self._redistribute_for_chunking(pred)
        _, h_chunks, label_chunks = self._split_chunks(hidden_states, labels)
        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        h_chunk_grads: list[torch.Tensor] = []

        accumulated_lm_head_grads: list[torch.Tensor] | None
        if mode == "autograd_grad":
            accumulated_lm_head_grads = [torch.zeros_like(p) for p in lm_head_params]
        else:
            accumulated_lm_head_grads = None
            for p in lm_head_params:
                p.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)

        last_idx = len(h_chunks) - 1
        for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
            if fsdp_enabled and i == last_idx:
                lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                    True, recurse=False
                )
            logits = lm_head(h_chunk)
            chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()
            if mode == "autograd_grad":
                chunk_grads = torch.autograd.grad(
                    chunk_loss, [h_chunk, *lm_head_params]
                )
                h_chunk_grads.append(chunk_grads[0])
                for j, g in enumerate(chunk_grads[1:]):
                    assert accumulated_lm_head_grads is not None
                    accumulated_lm_head_grads[j] = (
                        accumulated_lm_head_grads[j] + g
                    )
            else:
                chunk_loss.backward()
                assert h_chunk.grad is not None
                h_chunk_grads.append(h_chunk.grad)
                h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()

        accumulated_h_grad = torch.cat(h_chunk_grads, dim=1)
        if mode == "backward":
            # Cast back to hidden_states.dtype (parity with the original eager
            # path which used a fp32 ``GradAccumulator`` buffer + cast back).
            accumulated_h_grad = accumulated_h_grad.to(hidden_states.dtype)
            # Re-wrap with hidden_states' placement so the redistribute
            # backward downstream doesn't accidentally treat the seed as
            # ``Partial(sum)`` and all-reduce it (which would double under
            # TP). The TP-matmul backward populates ``h_chunk.grad`` with
            # ``Partial`` data; ``torch.cat`` preserves that placement, but
            # the eager ``__call__`` path writes through a fp32 buffer and
            # re-wraps via ``DTensor.from_local(..., placements=h_detached.placements)``
            # which is effectively a Replicate label on the same local data.
            # We mirror that explicitly here.
            from torch.distributed.tensor import DTensor as _DTensor
            if isinstance(accumulated_h_grad, _DTensor) and isinstance(
                hidden_states, _DTensor
            ):
                if accumulated_h_grad.placements != hidden_states.placements:
                    accumulated_h_grad = _DTensor.from_local(
                        accumulated_h_grad.to_local(),
                        device_mesh=hidden_states.device_mesh,
                        placements=hidden_states.placements,
                    )
        return total_loss, accumulated_h_grad, accumulated_lm_head_grads, hidden_states

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        ``pred`` should be hidden states from model forward with
        ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only — no per-chunk backward or gradient accumulation.

        Returns a differentiable loss. When ``.backward()`` is called on it
        (either by the trainer or the PP schedule), it triggers backward
        through the decoder via a custom autograd Function.
        """
        assert self.lm_head is not None, "Set lm_head before calling ChunkedCELoss"

        # Inference path: chunked forward only, no per-chunk backward.
        if not pred.requires_grad:
            hidden_states = self._redistribute_for_chunking(pred)
            _, h_chunks, label_chunks = self._split_chunks(hidden_states, labels)
            total_loss = hidden_states.new_zeros((), dtype=torch.float32)
            for h_chunk, label_chunk in zip(h_chunks, label_chunks):
                logits = self.lm_head(h_chunk)
                chunk_loss = self.fn(logits, label_chunk)
                if global_valid_tokens is not None:
                    chunk_loss = chunk_loss / global_valid_tokens
                total_loss = total_loss + chunk_loss.detach()
            return total_loss

        # Training path: share the per-chunk forward + per-chunk backward
        # with :meth:`compute_traceable_grads` via ``_compute_chunked_lm_head``
        # (mode="backward"). The decoder backward is deferred to the
        # trainer/PP schedule via ``_DecoderOutputGradientBackProp``: when
        # ``.backward()`` runs on the returned loss, the Function's backward
        # returns ``accumulated_grad`` as the gradient for ``hidden_states``,
        # which propagates through redistribute → pred → decoder.
        lm_head_params = [
            p for p in self.lm_head.parameters() if p.requires_grad
        ]
        total_loss, accumulated_grad, _, hidden_states = (
            self._compute_chunked_lm_head(
                pred,
                labels,
                global_valid_tokens,
                lm_head_params=lm_head_params,
                mode="backward",
            )
        )
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )

    def compute_traceable_grads(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        *,
        lm_head_params: list[torch.Tensor],
        decoder_params: list[torch.Tensor],
        mode: Literal["autograd_grad", "backward"] = "autograd_grad",
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Chunked CE loss + per-parameter gradients via a unified chunked loop.

        Shares the per-chunk forward + per-chunk backward with :meth:`__call__`
        through :meth:`_compute_chunked_lm_head`. ``mode`` selects how the
        decoder backward is driven:

        - ``"autograd_grad"`` (graph_trainer): one
          ``autograd.grad(hidden_states, decoder_params, grad_outputs=...)``
          with a ``/tp_size`` compensation on the seed (``autograd.grad``'s
          redistribute backward over-counts a Replicate ``grad_outputs`` under
          TP relative to ``loss.backward()`` through the same op).
        - ``"backward"`` (eager): ``hidden_states.backward(accumulated_h_grad)``
          drives the decoder bwd through the standard redistribute backward
          (per-rank slice — no ``/tp_size`` needed here).

        Returns ``(total_loss, lm_head_grads, decoder_grads)`` in both modes.
        """
        from torch.distributed.tensor import DTensor

        assert self.lm_head is not None, "Set lm_head before calling ChunkedCELoss"

        total_loss, accumulated_h_grad, lm_head_grads, hidden_states = (
            self._compute_chunked_lm_head(
                pred,
                labels,
                global_valid_tokens,
                lm_head_params=lm_head_params,
                mode=mode,
            )
        )

        if mode == "autograd_grad":
            if isinstance(hidden_states, DTensor):
                mesh = hidden_states.device_mesh
                if mesh.mesh_dim_names is not None and "tp" in mesh.mesh_dim_names:
                    tp_size = mesh["tp"].size()
                    if tp_size > 1:
                        accumulated_h_grad = accumulated_h_grad / tp_size
            decoder_grads = list(
                torch.autograd.grad(
                    hidden_states,
                    decoder_params,
                    grad_outputs=accumulated_h_grad,
                )
            )
            assert lm_head_grads is not None
            return total_loss, lm_head_grads, decoder_grads

        # mode == "backward": drive decoder via the same
        # ``_DecoderOutputGradientBackProp`` autograd Function that
        # ``__call__`` uses, then trigger ``.backward()`` on its scalar output.
        for p in decoder_params:
            p.grad = None
        diff_loss = _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_h_grad, total_loss
        )
        diff_loss.backward()
        decoder_grads = [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in decoder_params
        ]
        lm_head_grads = [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in lm_head_params
        ]
        return total_loss, lm_head_grads, decoder_grads


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden_states (connected to decoder graph), the accumulated
    gradient from chunked lm_head backward, and the loss value. Returns the
    loss value as a differentiable tensor.

    Backward returns accumulated_grad as the gradient for hidden_states.
    Autograd then propagates this through the decoder layers automatically —
    no explicit hidden_states.backward() needed.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
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
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (accumulated_grad,) = ctx.saved_tensors
        # Return accumulated_grad as the gradient for hidden_states.
        # Autograd then propagates this through hidden_states' existing
        # decoder graph — equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        return accumulated_grad, None, None
