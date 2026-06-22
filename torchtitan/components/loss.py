# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import CompileConfig, Configurable
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    if isinstance(pred, DTensor) and isinstance(labels, DTensor):
        return _cross_entropy_via_local_map(pred, labels)

    if isinstance(pred, DTensor):
        assert get_spmd_backend() == "default"
        if pred.placements == (Shard(pred.ndim - 1),):
            return _LossParallelCrossEntropy.apply(
                pred.to_local().flatten(0, 1).float(),
                labels.flatten(0, 1),
                pred.device_mesh.get_group("tp"),
                pred.shape[-1],
            )

    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


@spmd.register_autograd_function
class _LossParallelCrossEntropy(torch.autograd.Function):
    """
    Vocab-parallel cross-entropy on plain (non-DTensor) local tensors.

    Replaces ``torch.distributed.tensor.parallel.loss_parallel()`` with an
    explicit autograd Function so that SPMD code can operate on local tensors
    and process groups directly, without the DTensor-based context manager.

    Supports uneven vocab sharding (last TP rank may hold fewer classes) and
    ``IGNORE_INDEX`` labels.  Forward uses three TP all-reduces (max, sumexp,
    gather) to aggregate intermediate results in distributed softmax;
    backward is fused (NLL + log-softmax) with zero collectives.

    All inputs and outputs are plain ``torch.Tensor`` (not DTensor).
    """

    @staticmethod
    def typecheck_forward(
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
    ) -> torch.Tensor:
        """
        SPMD type: logits S(-1)@TP, labels I@TP → loss I@TP.
        Non-TP axes are passed through from logits to the output.
        """
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        result = _LossParallelCrossEntropy.apply(
            logits,
            labels,
            tp_group,
            global_vocab_size,
        )
        output_type = dict(spmd.get_local_type(logits))
        output_type[tp_group] = spmd.I
        spmd.assert_type(result, output_type)
        return result

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
    ) -> torch.Tensor:
        """Compute exact CE loss from local vocab shards via TP all-reduces."""
        logits_shape = logits.shape
        logits_2d = logits.flatten(0, -2).float()
        labels_1d = labels.flatten()

        # Compute this rank's vocab shard bounds for the local logits.
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
        vocab_start = min(global_vocab_size, chunk_size * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk_size)
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits_2d.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits_2d.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "_LossParallelCrossEntropy does not support empty vocab shards."
            )

        # All-reduce max for numerically stable distributed log-softmax.
        local_max = torch.amax(logits_2d, dim=-1, keepdim=True)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

        # All-reduce sum over shifted logits for the global softmax denominator.
        shifted = logits_2d - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        dist.all_reduce(shifted_sumexp, op=dist.ReduceOp.SUM, group=tp_group)
        log_probs = shifted - torch.log(shifted_sumexp)

        # Mask labels outside this vocab shard; the TP all-reduce below selects
        # the owner rank's log probability for each target token.
        safe_labels = torch.where(labels_1d != IGNORE_INDEX, labels_1d, 0)
        out_of_range = (safe_labels < vocab_start) | (
            safe_labels >= vocab_start + local_vocab_size
        )
        local_labels = safe_labels - vocab_start
        local_labels[out_of_range] = 0

        local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        local_result[out_of_range.unsqueeze(-1)] = 0
        dist.all_reduce(local_result, op=dist.ReduceOp.SUM, group=tp_group)

        # Compute summed NLL loss, dropping ignored labels.
        result = -local_result.squeeze(-1)
        result = torch.where(labels_1d != IGNORE_INDEX, result, 0)
        loss = result.sum()

        # Save local-shard log probabilities for the fused CE backward.
        ctx.save_for_backward(log_probs, labels_1d)
        ctx.logits_shape = logits_shape
        ctx.logits_dtype = logits.dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        return loss

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        log_probs, labels_1d = ctx.saved_tensors
        safe_labels = torch.where(labels_1d != IGNORE_INDEX, labels_1d, 0)
        out_of_range = (safe_labels < ctx.vocab_start) | (
            safe_labels >= ctx.vocab_start + ctx.local_vocab_size
        )
        local_labels = safe_labels - ctx.vocab_start
        local_labels[out_of_range] = 0

        grad_input = torch.zeros_like(log_probs)
        row_idx = torch.arange(local_labels.shape[0], device=local_labels.device)
        grad_update = out_of_range.to(grad_input.dtype) - 1.0
        grad_input[row_idx, local_labels] = grad_update

        grad_output = torch.where(
            (labels_1d != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        grad_logits = grad_logits.reshape(ctx.logits_shape).to(ctx.logits_dtype)
        return grad_logits, None, None, None


def _cross_entropy_via_local_map(
    pred: DTensor,
    labels: DTensor,
) -> torch.Tensor:
    mesh = pred.device_mesh
    # Labels don't have a vocab dim.
    expected_labels_placements = tuple(
        Replicate() if isinstance(p, Shard) and p.dim == 2 else p
        for p in pred.placements
    )
    if labels.placements != expected_labels_placements:
        raise ValueError(
            f"cross_entropy_loss: expected labels placements {expected_labels_placements}, "
            f"got {labels.placements}"
        )

    # After local flatten(0, 1), tensor dims are [batch*seq, vocab].
    # Per-axis placement:
    #   Shard on batch/seq -> Shard(0) (valid because reduction is sum)
    #   Shard on vocab -> Shard(1)
    vocab_sharded = any(isinstance(p, Shard) and p.dim == 2 for p in pred.placements)

    # Per-axis output placement for sum reduction:
    #   Shard on non-vocab-dim -> Partial
    #   Shard on vocab-dim -> Replicate
    out_placements = [
        Partial() if isinstance(p, Shard) and p.dim != 2 else Replicate()
        for p in pred.placements
    ]

    @local_map(
        out_placements=out_placements,
        in_placements=(pred.placements, labels.placements),
        in_grad_placements=(pred.placements, labels.placements),
        device_mesh=mesh,
    )
    def _local_cross_entropy(
        pred_local: torch.Tensor, labels_local: torch.Tensor
    ) -> torch.Tensor:
        flat_pred = pred_local.flatten(0, 1).float()
        flat_labels = labels_local.flatten(0, 1)
        if not vocab_sharded:
            return torch.nn.functional.cross_entropy(
                flat_pred,
                flat_labels,
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )
        return _LossParallelCrossEntropy.apply(
            flat_pred,
            flat_labels,
            mesh.get_group("tp"),
            pred.shape[-1],
        )

    return _local_cross_entropy(pred, labels)


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
        global_valid_tokens: float | None = None,
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

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
    ) -> torch.Tensor:
        loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss


class LoopedEntropyLoss(BaseLoss):
    """Full-materialized entropy-regularized loss for looped decoder training."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        beta: float = 0.05
        """Entropy coefficient."""

        eps: float = 1e-8
        """Numerical epsilon for log probabilities."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        if config.beta < 0:
            raise ValueError("LoopedEntropyLoss beta must be non-negative")
        if config.eps <= 0:
            raise ValueError("LoopedEntropyLoss eps must be positive")
        self.fn: LossFunction = cross_entropy_loss
        self.beta = config.beta
        self.eps = config.eps
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def compute_exit_probabilities(
        gate_logits_by_step: torch.Tensor | None,
        *,
        steps: int,
        batch_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a [T, B, S] exit-step distribution.

        If no gate logits are provided, all steps receive uniform mass. With
        gate logits, steps 1..T-1 receive lambda_t * survival_{t-1}, and the
        final step receives all remaining survival mass.
        """
        if gate_logits_by_step is None:
            return torch.full(
                (steps, *batch_shape),
                1.0 / steps,
                device=device,
                dtype=torch.float32,
            )

        if gate_logits_by_step.ndim != 3:
            raise ValueError(
                "gate_logits_by_step must have shape [steps, batch, seq], "
                f"got {tuple(gate_logits_by_step.shape)}"
            )
        if gate_logits_by_step.shape[0] != steps:
            raise ValueError(
                f"Expected {steps} gate-logit steps, got {gate_logits_by_step.shape[0]}"
            )
        if tuple(gate_logits_by_step.shape[1:]) != batch_shape:
            raise ValueError(
                "gate logits batch/seq shape must match logits, got "
                f"{tuple(gate_logits_by_step.shape[1:])} vs {batch_shape}"
            )

        if steps == 1:
            return torch.ones(
                (1, *batch_shape),
                device=device,
                dtype=torch.float32,
            )

        gates = torch.sigmoid(gate_logits_by_step.float())
        survival = torch.ones(batch_shape, device=device, dtype=torch.float32)
        probs = []
        for step_idx in range(steps - 1):
            exit_prob = gates[step_idx] * survival
            probs.append(exit_prob)
            survival = survival * (1.0 - gates[step_idx])
        probs.append(survival)
        return torch.stack(probs, dim=0)

    def __call__(
        self,
        logits_by_step: torch.Tensor,
        gate_logits_by_step: torch.Tensor | None,
        labels: torch.Tensor,
        global_valid_tokens: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits_by_step.ndim != 4:
            raise ValueError(
                "logits_by_step must have shape [steps, batch, seq, vocab], "
                f"got {tuple(logits_by_step.shape)}"
            )

        steps, batch, seq, vocab = logits_by_step.shape
        if labels.shape != (batch, seq):
            raise ValueError(
                f"labels shape must be {(batch, seq)}, got {tuple(labels.shape)}"
            )

        expanded_labels = labels.unsqueeze(0).expand(steps, -1, -1)
        per_step_ce = torch.nn.functional.cross_entropy(
            logits_by_step.reshape(-1, vocab).float(),
            expanded_labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(steps, batch, seq)

        exit_probs = self.compute_exit_probabilities(
            gate_logits_by_step,
            steps=steps,
            batch_shape=(batch, seq),
            device=logits_by_step.device,
            dtype=logits_by_step.dtype,
        )

        valid_mask = labels != IGNORE_INDEX
        expected_task_loss = (exit_probs * per_step_ce).sum(dim=0)
        entropy = -(exit_probs * exit_probs.clamp_min(self.eps).log()).sum(dim=0)

        task_loss_sum = expected_task_loss[valid_mask].sum()
        entropy_sum = entropy[valid_mask].sum()
        loss = task_loss_sum - self.beta * entropy_sum

        metrics_denominator = valid_mask.sum().clamp_min(1).to(loss.dtype)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens

        self._record_metrics(
            task_loss_sum=task_loss_sum,
            entropy_sum=entropy_sum,
            exit_probs=exit_probs,
            valid_mask=valid_mask,
            denominator=metrics_denominator,
        )
        return loss

    @torch.no_grad()
    def _record_metrics(
        self,
        *,
        task_loss_sum: torch.Tensor,
        entropy_sum: torch.Tensor,
        exit_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        denominator: torch.Tensor,
    ) -> None:
        denom = denominator.clamp_min(1)
        metrics = {
            "looped_loss/task_loss": float((task_loss_sum.detach() / denom).item()),
            "looped_loss/entropy": float((entropy_sum.detach() / denom).item()),
        }

        if bool(valid_mask.any().item()):
            steps = exit_probs.shape[0]
            step_ids = torch.arange(
                1,
                steps + 1,
                device=exit_probs.device,
                dtype=exit_probs.dtype,
            ).view(steps, 1, 1)
            avg_exit_step = (exit_probs * step_ids).sum(dim=0)[valid_mask].mean()
            metrics["looped_loss/avg_exit_step"] = float(avg_exit_step.item())
            exit_mass = exit_probs[:, valid_mask].mean(dim=1)
            for idx, value in enumerate(exit_mass, start=1):
                metrics[f"looped_loss/exit_mass_step_{idx}"] = float(value.item())

        self.last_metrics = metrics


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
        reference: Reference tensor for shape, device, and DTensor-ness. If a
            DTensor, only its device mesh is reused; the placement of the
            returned DTensor is taken from the first added chunk (see add()),
            not from this reference, so the buffer is labeled with the actual
            gradient placement (e.g. Partial(sum) on the TP axis when the
            forward used a Replicate input with a Shard(0) weight, as in
            ColwiseParallel lm_head) rather than the activation placement.
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
        # Captured from the first added chunk; see __init__ docstring.
        self._placements: tuple[Placement, ...] | None = None

        if isinstance(reference, DTensor):
            self._device_mesh = reference.device_mesh
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

        if isinstance(chunk_grad, DTensor):
            if self._placements is None:
                self._placements = chunk_grad.placements
            elif chunk_grad.placements != self._placements:
                # All chunks come from the same op chain and must share a
                # placement. Otherwise the buffer mixes frames and result()
                # would mislabel them.
                raise ValueError(
                    f"chunk_grad placement {chunk_grad.placements} does not "
                    f"match first chunk's placement {self._placements}"
                )
            chunk_grad = chunk_grad.to_local()
        elif self._placements is not None:
            # Earlier chunks were DTensor but this one is a plain tensor;
            # mixing the two would silently drop the implied reduction.
            raise ValueError(
                "chunk_grad is a plain tensor but earlier chunks were "
                f"DTensor with placement {self._placements}"
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
        """Return the accumulated gradient tensor, wrapped as DTensor if needed.

        When the chunks were Partial(sum), the returned DTensor is also
        Partial(sum); autograd performs the implied reduction once when this
        gradient lands on the decoder-side leaf.
        """
        from torch.distributed.tensor import DTensor

        if self._device_mesh is not None:
            if self._placements is None:
                raise ValueError(
                    "No DTensor chunk was added; cannot wrap the buffer as "
                    "DTensor without a known placement. Either pass DTensor "
                    "chunks to add(), or use a plain reference tensor."
                )
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
        The root decoder norm emits hidden states that are replicated on the
        TP axis before chunking, so each chunk enters the lm_head as
        ``Replicate()`` input regardless of whether SP is enabled.

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
        global_valid_tokens: float | None = None,
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

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedCELoss"
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        # Check if it's training model or validation mode
        requires_grad = hidden_states.requires_grad

        # Chunking always operates on the *local* view: when ``t`` is a
        # Shard(1) DTensor, chunking the global view would distribute whole
        # chunks across ranks (e.g. size=2, num_chunks=8: chunks 0-3 on
        # rank 0, 4-7 on rank 1), leaving half the per-chunk DTensors with
        # local seq=0 and breaking GradAccumulator's slice writes.
        # ``local_map`` runs the chunking body on plain tensors; under the
        # non-DTensor (eager) path we call ``_chunk_local`` directly.
        # ``.contiguous()`` breaks shared storage from ``torch.chunk``.
        def _chunk_local(t):
            return tuple(c.contiguous() for c in torch.chunk(t, num_chunks, dim=1))

        def _chunk(t):
            if not isinstance(t, DTensor):
                return _chunk_local(t)
            p = t.placements
            wrapped = local_map(
                _chunk_local,
                out_placements=(p,) * num_chunks,
                in_placements=(p,),
                device_mesh=t.device_mesh,
            )
            return wrapped(t)

        with spmd.local():
            # ``detach`` + ``requires_grad_`` makes each chunk a leaf so it
            # accumulates ``.grad`` for ``GradAccumulator``.
            h_chunks = [
                c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
            ]
            label_chunks = list(_chunk(labels))

            grad_accumulator = None
            if requires_grad:
                grad_accumulator = GradAccumulator(
                    hidden_states,
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

                chunk_loss = self.fn(
                    logits,
                    label_chunk,
                )
                if global_valid_tokens is not None:
                    chunk_loss = chunk_loss / global_valid_tokens
                total_loss = total_loss + chunk_loss.detach()

                if requires_grad:
                    chunk_loss.backward()
                    assert h_chunk.grad is not None
                    assert grad_accumulator is not None
                    grad_accumulator.add(h_chunk.grad)
                    h_chunk.grad = None

            if fsdp_enabled:
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.set_requires_gradient_sync(True, recurse=False)
                lm_head.reshard()
            if not requires_grad:
                return total_loss

            assert grad_accumulator is not None
            accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        return self._gradient_backprop(
            hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled
        )

    @staticmethod
    def _gradient_backprop(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        """Return a differentiable loss via _DecoderOutputGradientBackProp.
        When ``.backward()`` is called (by the trainer or PP schedule),
        autograd calls ``_DecoderOutputGradientBackProp.backward`` which
        returns ``accumulated_grad`` as the gradient for ``hidden_states``,
        propagating through the decoder. Subclasses override to swap in a
        different autograd Function.
        """
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden_states (connected to decoder graph), the accumulated
    gradient from chunked lm_head backward, and the loss value. Returns a
    detached loss with this Function as its grad_fn.

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
        return loss.detach()

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
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we need
        # to properly handle. The complicated part is that grad_output might not be
        # on the same device mesh as accumlated_grad.
        return accumulated_grad, None, None
