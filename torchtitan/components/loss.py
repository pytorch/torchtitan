# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from torchtitan.config import CompileConfig, Configurable
from torchtitan.distributed.spmd_state import (
    current_mesh,
    is_spmd_active,
)
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


@spmd.register_autograd_function
class _LossParallelCrossEntropy(torch.autograd.Function):
    """Vocab-parallel cross entropy with a fused sharded-logits backward.

    Inputs are ``logits: [B, L, V/tp]`` and TP-invariant ``labels: [B, L]``.
    The output is a scalar sum-reduced loss that is invariant on TP ranks.
    """

    @staticmethod
    def typecheck_forward(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        current_mesh().get_group("tp")
        result = _LossParallelCrossEntropy.apply(logits, labels)

        out_type = dict(spmd.get_local_type(logits))
        out_type["tp"] = spmd.I
        spmd.assert_type(result, out_type)
        return result

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        tp_pg = current_mesh().get_group("tp")
        tp_rank = dist.get_rank(tp_pg)

        logits_shape = logits.shape
        vocab_chunk_size = logits_shape[-1]
        logits_2d = logits.flatten(0, -2).float()
        labels_1d = labels.flatten()

        local_max = logits_2d.max(dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_pg)

        shifted = logits_2d - global_max.unsqueeze(-1)
        exp_shifted = shifted.exp()
        global_exp_sum = exp_shifted.sum(dim=-1)
        dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=tp_pg)

        vocab_start = tp_rank * vocab_chunk_size
        valid = labels_1d != IGNORE_INDEX
        in_range = (
            valid
            & (labels_1d >= vocab_start)
            & (labels_1d < vocab_start + vocab_chunk_size)
        )
        local_idx = (labels_1d - vocab_start).clamp(min=0, max=vocab_chunk_size - 1)
        local_target_logit = logits_2d[
            torch.arange(labels_1d.shape[0], device=logits.device), local_idx
        ]
        local_target_logit = local_target_logit * in_range.to(logits_2d.dtype)
        target_logit = local_target_logit.clone()
        dist.all_reduce(target_logit, op=dist.ReduceOp.SUM, group=tp_pg)

        log_global_exp_sum = global_exp_sum.log()
        per_token_loss = -target_logit + log_global_exp_sum + global_max
        loss = (per_token_loss * valid.to(per_token_loss.dtype)).sum()

        log_probs = shifted - log_global_exp_sum.unsqueeze(-1)
        ctx.save_for_backward(log_probs, labels_1d)
        ctx.logits_shape = logits_shape
        ctx.logits_dtype = logits.dtype
        ctx.vocab_start = vocab_start
        ctx.vocab_chunk_size = vocab_chunk_size
        return loss

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        log_probs, labels_1d = ctx.saved_tensors
        grad_logits = torch.exp(log_probs)

        valid = labels_1d != IGNORE_INDEX
        in_range = (
            valid
            & (labels_1d >= ctx.vocab_start)
            & (labels_1d < ctx.vocab_start + ctx.vocab_chunk_size)
        )
        if in_range.any():
            row_idx = torch.arange(labels_1d.shape[0], device=labels_1d.device)[
                in_range
            ]
            local_idx = labels_1d[in_range] - ctx.vocab_start
            grad_logits[row_idx, local_idx] -= 1.0

        grad_logits = grad_logits * valid.to(grad_logits.dtype).unsqueeze(-1)
        grad_logits = grad_logits * grad_output.to(grad_logits.dtype)
        grad_logits = grad_logits.reshape(ctx.logits_shape).to(ctx.logits_dtype)
        return grad_logits, None


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
    def __init__(
        self, config: Config, *, compile_config: CompileConfig | None = None
    ): ...

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
        Hidden states are redistributed to ``R`` on the TP mesh before
        chunking, so each chunk enters the lm_head as ``R`` input regardless
        of whether SP is enabled. With SP, this is an all-gather from
        ``S(1)``; without SP, it's a no-op.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    SPMD composability:
        The chunk loop body runs inside a local SPMD boundary that strips all
        SPMD types — chunking, flatten, and CE operate on plain local tensors.
        The boundary re-annotates outputs on exit.

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
        self.enable_sp: bool = False
        self.loss_parallel: bool = False

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def chunked_loss_and_grad(
        self,
        h_detached: torch.Tensor,
        labels: torch.Tensor,
        *,
        global_valid_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk loop body: chunk → lm_head → CE → backward → accumulate.

        Operates on plain local tensors when called via local SPMD.

        Returns (total_loss, accumulated_grad_buffer).
        """
        from torch.distributed._composable.fsdp import FSDPModule

        num_chunks = self.num_chunks
        lm_head = self.lm_head
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        requires_grad = h_detached.requires_grad

        # Split hidden states and labels into chunks along seq dim.
        # Use .contiguous() to break shared storage from torch.chunk().
        # TODO: With CP enabled, chunking along dim=1 should happen inside
        # local SPMD so it operates on local tensors.
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

        total_loss = h_detached.new_zeros((), dtype=torch.float32)
        if spmd.is_type_checking():
            mesh_axis_names = spmd.current_mesh_names() or {}
            for axis_name in ("dp", "cp"):
                if axis_name not in mesh_axis_names:
                    continue
                total_loss = spmd.reinterpret(
                    total_loss,
                    current_mesh().get_group(axis_name),
                    src=spmd.R,
                    dst=spmd.P,
                    expert_mode=True,
                )
            if "tp" in mesh_axis_names:
                total_loss = spmd.convert(
                    total_loss,
                    current_mesh().get_group("tp"),
                    src=spmd.R,
                    dst=spmd.I,
                    expert_mode=True,
                )

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
            if self.loss_parallel:
                chunk_loss = _LossParallelCrossEntropy.apply(logits, label_chunk)
            else:
                chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                loss_scale = global_valid_tokens
                if spmd.is_type_checking() and isinstance(loss_scale, torch.Tensor):
                    mesh_axis_names = current_mesh().mesh_dim_names
                    assert mesh_axis_names is not None
                    if "cp" in mesh_axis_names:
                        loss_scale = spmd.convert(
                            loss_scale,
                            current_mesh().get_group("cp"),
                            src=spmd.I,
                            dst=spmd.R,
                            expert_mode=True,
                        )
                chunk_loss = chunk_loss / loss_scale

            if spmd.is_type_checking():
                mesh_axis_names = current_mesh().mesh_dim_names
                assert mesh_axis_names is not None
                for axis_name in ("dp", "cp"):
                    if axis_name not in mesh_axis_names:
                        continue
                    chunk_loss = spmd.reinterpret(
                        chunk_loss,
                        current_mesh().get_group(axis_name),
                        src=spmd.V,
                        dst=spmd.P,
                        expert_mode=True,
                    )

            total_loss = total_loss + chunk_loss.detach()

            if requires_grad:
                with spmd.no_typecheck():
                    chunk_loss.backward()
                    assert h_chunk.grad is not None
                    grad_accumulator.add(h_chunk.grad)
                    h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()

        return total_loss, grad_accumulator.result()

    def _spmd_out_types(self, h_detached: torch.Tensor) -> tuple:
        """Build expected output types for (total_loss, accumulated_grad).

        total_loss: P on all DP axes (including CP) — each rank has a partial
            loss sum from its local batch/seq tokens. I@tp because TP loss
            computation is duplicated after the vocab-parallel reductions.
        accumulated_grad: same types as h_detached (the input).
        """
        loss_type: dict = {}
        mesh_axis_names = spmd.current_mesh_names() or {}
        for axis_name in ("dp", "cp"):
            if axis_name in mesh_axis_names:
                loss_type[axis_name] = spmd.P
        if "tp" in mesh_axis_names:
            loss_type["tp"] = spmd.I

        grad_types = dict(spmd.get_local_type(h_detached))
        grad_spec = spmd.get_partition_spec(h_detached)
        grad_leaf = (grad_types, grad_spec) if grad_spec else grad_types

        return (loss_type, grad_leaf)

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
        hidden_states = pred
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"

        # SPMD path: all-gather S(1)@tp -> I@tp before chunking; loss compute
        # is duplicated on TP after the sequence gather.
        if is_spmd_active() and self.enable_sp:
            bwd = {"op_dtype": hidden_states.dtype}
            hidden_states = spmd.redistribute(
                hidden_states,
                current_mesh().get_group("tp"),
                src=spmd.S(1),
                dst=spmd.I,
                backward_options=bwd,
            )

        requires_grad = hidden_states.requires_grad
        h_detached = hidden_states.detach().requires_grad_(requires_grad)

        typechecking = is_spmd_active() and spmd.is_type_checking()
        if typechecking:
            loss_out_type, grad_out_leaf = self._spmd_out_types(h_detached)

        with spmd.typecheck(local=True) if typechecking else contextlib.nullcontext():
            total_loss, grad_buffer = self.chunked_loss_and_grad(
                h_detached,
                labels,
                global_valid_tokens=global_valid_tokens,
            )

        if typechecking:
            spmd.assert_type(total_loss, loss_out_type)
            if isinstance(grad_out_leaf, tuple):
                spmd.assert_type(
                    grad_buffer, grad_out_leaf[0], partition_spec=grad_out_leaf[1]
                )
            else:
                spmd.assert_type(grad_buffer, grad_out_leaf)

        if not requires_grad:
            return total_loss

        accumulated_grad = grad_buffer.to(hidden_states.dtype)

        # Return a differentiable loss via _DecoderOutputGradientBackProp. When
        # .backward() is called (by the trainer or PP schedule), autograd
        # calls _DecoderOutputGradientBackProp.backward which returns accumulated_grad
        # as the gradient for hidden_states, propagating through the decoder.
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


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
    def typecheck_forward(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Output has the same type as ``loss`` (3rd arg)."""
        result = _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, loss
        )
        if spmd.has_local_type(loss):
            spmd.assert_type(result, spmd.get_local_type(loss))
        return result

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


spmd.register_autograd_function(_LossParallelCrossEntropy)
spmd.register_autograd_function(_DecoderOutputGradientBackProp)
