# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
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
        support_autograd_grad: bool = False
        """If True, plumb the sharded lm_head param grads (already produced
        by the chunk loop's reduce-scatter) out as explicit autograd outputs
        so outer ``torch.autograd.grad(loss, [hidden_states,
        *lm_head.parameters()])`` returns them directly — no reliance on
        ``param.grad`` side effects. Required for FX tracing / replay where
        side-effecting ``.grad`` writes don't survive. Compatible with both
        outer ``loss.backward()`` and ``torch.autograd.grad`` consumers.
        TODO: we should use it as default then delete the config.
        """

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.num_chunks = config.num_chunks
        self.support_autograd_grad = config.support_autograd_grad
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

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
        from torch.distributed._composable.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, Replicate

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        # If SP is enabled, hidden states are Shard(1) on the TP mesh dim.
        # Redistribute only the TP dim to Replicate before chunking so that
        # the lm_head receives Replicate input on TP.
        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            if mesh.mesh_dim_names is not None and "tp" in mesh.mesh_dim_names:
                tp_dim = mesh.mesh_dim_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_dim], Replicate):
                    placements[tp_dim] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        # Check if it's training model or validation mode
        requires_grad = hidden_states.requires_grad

        # Split hidden states and labels into chunks along seq dim.
        # Use .contiguous() to break shared storage from torch.chunk().
        # TODO: When CP mesh is in DTensor, chunking along dim=1 won't work
        # directly with Shard(1) on CP. Need local_map to operate on local tensors
        h_detached = hidden_states.detach().requires_grad_(requires_grad)
        h_chunks = [
            c.contiguous().detach().requires_grad_(requires_grad)
            for c in torch.chunk(h_detached, num_chunks, dim=1)
        ]
        label_chunks = torch.chunk(labels, num_chunks, dim=1)

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
                lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                    True, recurse=False
                )

            logits = lm_head(h_chunk)

            chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()

            if requires_grad:
                chunk_loss.backward()
                assert h_chunk.grad is not None
                grad_accumulator.add(h_chunk.grad)
                h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()
        if not requires_grad:
            return total_loss

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        if self.support_autograd_grad:
            return _ChunkedLossWithParamGrads.apply(
                hidden_states,
                accumulated_grad,
                total_loss,
                lm_head,
                fsdp_enabled,
                *lm_head.parameters(),
            )

        # Return a differentiable loss via _DecoderOutputGradientBackProp. When
        # .backward() is called (by the trainer or PP schedule), autograd
        # calls _DecoderOutputGradientBackProp.backward which returns accumulated_grad
        # as the gradient for hidden_states, propagating through the decoder.
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


def _maybe_redistribute_multiply(g: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """Multiply ``g`` by ``grad_output`` for the autograd chain rule,
    redistributing ``grad_output``'s mesh to match ``g``'s first when both
    are DTensors on different meshes.

    Required because for FSDP+TP layouts ``grad_output`` (= ``ones_like(loss)``
    at the autograd start) lives on the loss output's mesh — typically the
    activation mesh ``(tp,)`` — while saved param grads live on the params'
    mesh ``(fsdp, tp)``. DTensor refuses cross-mesh ``aten.mul.Tensor`` so a
    naive ``g * grad_output`` would crash. For the only ``grad_output`` shape
    we actually encounter (a ``Replicate()`` scalar, which is what
    ``torch.autograd.grad`` constructs for a scalar loss output), the
    redistribute is a no-op at runtime — no collective fires.
    """
    from torch.distributed.tensor import DTensor, Replicate

    if (
        isinstance(g, DTensor)
        and isinstance(grad_output, DTensor)
        and g.device_mesh != grad_output.device_mesh
    ):
        grad_output = grad_output.redistribute(
            g.device_mesh, [Replicate()] * g.device_mesh.ndim
        )
    return g * grad_output


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
        return _maybe_redistribute_multiply(accumulated_grad, grad_output), None, None


class _ChunkedLossWithParamGrads(torch.autograd.Function):
    """Like ``_DecoderOutputGradientBackProp`` but also plumbs sharded grads
    for the lm_head parameters out as explicit autograd outputs, so outer
    ``torch.autograd.grad(loss, [hidden_states, *lm_head.parameters()])``
    returns correct grads instead of relying on ``param.grad`` side effects.

    Forward is invoked *after* the chunked ``chunk_loss.backward()`` loop has
    populated each lm_head param's sharded ``.grad`` (via FSDP's last-chunk
    reduce-scatter). Forward captures those grads, clears ``.grad``, and
    disables grad sync — so that outer ``loss.backward()`` consumers, whose
    AccumulateGrad would otherwise (a) double-add onto ``.grad`` and (b)
    re-fire FSDP's reduce-scatter on already-sharded data, get clean
    behavior. Backward queues a callback to restore grad sync after the
    engine drains the rest of the backward graph.

    Outer ``torch.autograd.grad`` consumers bypass AccumulateGrad entirely
    and just receive the saved sharded grads directly.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
        *lm_params: torch.Tensor,
    ) -> torch.Tensor:
        # The chunk loop above already populated each lm_head param's
        # ``.grad`` with the correctly sharded value via the FSDP last-chunk
        # post-accumulate-grad hook (reduce-scatter). Capture those grads
        # into saved_tensors so backward ca route them as autograd outputs
        # for the lm_head param inputs of this Function. Additionally, we need
        # following changes:
        # 1. We need to clear ``.grad`` so a subsequent outer ``loss.backward()`` doesn't
        # double-add when AccumulateGrad fires on those params with our returned grads.
        # 2. We need to disable FSDP grad sync on lm_head: outer .backward() would
        # otherwise re-fire the post-accumulate-grad hook on already-sharded
        # data. The restore is queued in backward() below.
        sharded_param_grads = [p.grad.detach() for p in lm_params]
        for p in lm_params:
            p.grad = None
        if fsdp_enabled:
            lm_head.set_requires_gradient_sync(False, recurse=False)
        ctx.save_for_backward(accumulated_h_grad, *sharded_param_grads)
        ctx.lm_head = lm_head
        ctx.fsdp_enabled = fsdp_enabled
        return total_loss.detach().clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        accumulated_h_grad = saved[0]
        param_grads = saved[1:]
        if ctx.fsdp_enabled:
            # Restore FSDP grad sync that forward() disabled. Use
            # queue_callback to defer the restore until the engine drains
            # the rest of the backward graph — including each lm_head
            # param's AccumulateGrad firing on the grads we return below.
            # If we restored here (synchronously, before returning), the
            # first AccumulateGrad would see sync=True and try to
            # reduce-scatter our already-sharded grad → wrong result.
            lm_head = ctx.lm_head
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: lm_head.set_requires_gradient_sync(True, recurse=False)
            )
        return (
            _maybe_redistribute_multiply(accumulated_h_grad, grad_output),
            None,
            None,
            None,
            None,
            *(_maybe_redistribute_multiply(g, grad_output) for g in param_grads),
        )
