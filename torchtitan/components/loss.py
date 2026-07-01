# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import CompileConfig, Configurable
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
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
                "sum",
            )
    elif get_spmd_backend() == "spmd_types" and spmd_mesh_size("tp") > 1:
        return _LossParallelCrossEntropy.apply(
            pred.flatten(0, 1).float(),
            labels.flatten(0, 1),
            current_spmd_mesh().get_group("tp"),  # pyrefly: ignore[missing-attribute]
            global_vocab_size,
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
        reduction: str = "sum",
    ) -> torch.Tensor:
        """
        SPMD type: logits S(-1)@TP, labels I@TP -> loss I@TP.
        Non-TP axes are passed through from logits to the output.
        """
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        result = _LossParallelCrossEntropy.apply(
            logits,
            labels,
            tp_group,
            global_vocab_size,
            reduction,
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
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute exact CE from local vocab shards via TP all-reduces.

        ``reduction="sum"`` returns the scalar summed loss (SFT/CE).
        ``reduction="none"`` returns the per-token NLL ``[N]``, which GRPO
        negates to get per-token logprobs without all-gathering the vocab.
        """
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
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # All-reduce sum over shifted logits for the global softmax denominator.
        shifted = logits_2d - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(
            shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
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
        local_result = funcol.all_reduce(
            local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )

        # Per-token NLL, dropping ignored labels (logprob 0 for ignored).
        result = -local_result.squeeze(-1)
        result = torch.where(labels_1d != IGNORE_INDEX, result, 0)

        # Save local-shard log probabilities for the fused CE backward.
        ctx.save_for_backward(log_probs, labels_1d)
        ctx.logits_shape = logits_shape
        ctx.logits_dtype = logits.dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        ctx.reduction = reduction
        if reduction == "none":
            return result
        return result.sum()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
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

        # reduction="none" gives a per-token ``[N]`` upstream grad; reshape to
        # ``[N, 1]`` to broadcast over the local vocab. "sum" gives the scalar
        # loss grad, which broadcasts as-is.
        if ctx.reduction == "none":
            grad_output = grad_output.reshape(-1, 1)
        grad_output = torch.where(
            (labels_1d != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        grad_logits = grad_logits.reshape(ctx.logits_shape).to(ctx.logits_dtype)
        return grad_logits, None, None, None, None


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
            "sum",
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
    ``(pred, labels, global_valid_tokens) -> (scaled_loss, metrics)``.
    Subclasses must implement ``__init__``. Leaf losses set ``self.fn`` and
    reuse the default ``__call__``.
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the scaled loss and any metrics computed by the loss."""
        loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            # TODO(pianpwk): Teach spmd_types that P / scalar preserves P.
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
        return loss, {}


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with sum reduction for token-based normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        global_vocab_size: int | None = None
        """Full vocabulary size, needed for spmd_types loss-parallel CE."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self.fn(pred, labels, global_vocab_size=self.global_vocab_size)
        if global_valid_tokens is not None:
            # TODO(pianpwk): Teach spmd_types that P / scalar preserves P.
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
        return loss, {}


class MSELoss(BaseLoss):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = mse_loss
        self._maybe_compile(compile_config)


def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position logprobs from logits and labels.

    Output shape matches input: ``[batch, seq_len]``. Any DTensor placement
    handling is centralized here so RL losses that call ``compute_logprobs`` do
    not need to duplicate the vocab-gather logic.
    """
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        # Gather vocab-sharded TP logits before computing per-token logprobs.
        placements = tuple(
            Replicate()
            if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1)
            else p
            for p in logits.placements
        )
        logits = logits.redistribute(placements=placements).to_local()
    elif get_spmd_backend() == "spmd_types" and spmd_mesh_size("tp") > 1:
        # spmd_types returns a plain local vocab shard. Labels are global token
        # ids, so cross_entropy needs full-vocab logits.
        mesh = current_spmd_mesh()
        assert mesh is not None
        # dst=I, not R: the vocab all-gather's grad is the replicated upstream
        # grad sliced back to this rank's vocab shard (I's backward), not an
        # all-reduce (R's backward). The latter over-counts by tp_degree and
        # diverges from the DTensor path above, whose redistribute grad slices.
        logits = spmd.redistribute(
            logits,
            mesh.get_group("tp"),
            src=spmd.S(-1),
            dst=spmd.I,
        )

    B, L, V = logits.shape
    return -F.cross_entropy(
        logits.float().reshape(B * L, V),
        labels.reshape(B * L),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).reshape(B, L)


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


class ChunkedLossWrapper(BaseLoss):
    """Chunked loss wrapper that splits the sequence dimension to reduce peak memory.

    Instead of materializing the full [B, L, V] logits tensor at once, this splits
    the hidden states into N chunks along the sequence dimension and computes
    lm_head + loss on each chunk sequentially. This reduces peak memory
    from O(B*L*V) to O(B*L/N*V).

    The inner ``loss_fn`` defaults to ``CrossEntropyLoss`` and is called once per
    chunk on logits from that chunk. Additional per-token ``loss_inputs`` are
    chunked along the same sequence dimension and forwarded to the inner loss.

    The flow:
    1. Model forward with _skip_lm_head=True to get hidden states [B, L, D]
    2. Detach hidden states at the boundary
    3. Split detached hidden states into N chunks along seq dim
    4. Disable FSDP reshard on lm_head to keep weight unsharded across chunks
    5. For each chunk: lm_head(chunk) -> loss_fn(logits, labels, gvt) -> backward()
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

    Compile: the inner ``loss_fn`` can be compiled independently; lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

        loss_fn: BaseLoss.Config = field(default_factory=CrossEntropyLoss.Config)
        """Loss applied to each chunk's logits."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.num_chunks = config.num_chunks
        self.loss_fn: BaseLoss = config.loss_fn.build(compile_config=compile_config)
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute chunked loss.

        ``pred`` should come from model forward with ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only -- no per-chunk backward or gradient accumulation.

        Returns a differentiable loss and metrics. When ``.backward()`` is called
        on the loss, it triggers backward through the decoder via a custom
        autograd Function.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
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
        # Equal chunk sizes also match GradAccumulator's sequential slice
        # writes, which use one chunk length for each write offset.
        def _chunk_local(t):
            seq_len = t.shape[1]
            torch._check(
                seq_len % num_chunks == 0,
                lambda: "ChunkedLossWrapper sequence length must be divisible by num_chunks",
            )
            chunk_len = seq_len // num_chunks
            return tuple(
                c.contiguous() for c in torch.split(t, [chunk_len] * num_chunks, dim=1)
            )

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
            input_chunks = {
                key: _chunk(value) if isinstance(value, torch.Tensor) else value
                for key, value in loss_inputs.items()
            }

            grad_accumulator = None
            if requires_grad:
                grad_accumulator = GradAccumulator(
                    hidden_states,
                    num_chunks=num_chunks,
                    dtype=torch.float32,
                )

            total_loss = hidden_states.new_zeros((), dtype=torch.float32)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                # TODO(pianpwk): would be nice if mutate_type accepted multiple axes.
                for axis_name, dst in {
                    "dp": spmd.P,
                    "cp": spmd.P,
                    "tp": spmd.I,
                }.items():
                    total_loss = spmd.mutate_type(
                        total_loss, axis_name, src=spmd.R, dst=dst
                    )
            metrics: dict[str, torch.Tensor] = {}

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

                chunk_inputs = {
                    key: chunks[i] if isinstance(chunks, tuple) else chunks
                    for key, chunks in input_chunks.items()
                }
                chunk_loss, chunk_metrics = self.loss_fn(
                    logits, label_chunk, global_valid_tokens, **chunk_inputs
                )
                metrics = self._combine_chunk_metrics(metrics, chunk_metrics)
                if get_spmd_backend() == "spmd_types":
                    # V -> P reinterpret, after exiting local region.
                    spmd.assert_type(chunk_loss, {"dp": spmd.P, "cp": spmd.P})
                total_loss = total_loss + chunk_loss.detach()

                if requires_grad:
                    with spmd.no_typecheck():
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
                return total_loss, metrics

            assert grad_accumulator is not None
            accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        with spmd.no_typecheck():
            loss = self._gradient_backprop(
                hidden_states,
                accumulated_grad,
                total_loss,
                lm_head,
                fsdp_enabled,
            )
        return loss, metrics

    @staticmethod
    def _combine_chunk_metrics(
        current: dict[str, torch.Tensor],
        values: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Combine metrics from one sequence chunk into the local accumulator.

        Mean/fraction metrics are expected to already be normalized by the
        global valid-token count, so summing chunk contributions gives the
        global mean for this rank's microbatch contribution. The trainer still
        performs the cross-rank loss-mesh reduction on the returned metrics.
        """
        for key, value in values.items():
            previous = current.get(key)
            if previous is None:
                current[key] = value
            elif key.endswith(("/mean", "/frac", "_mean", "_frac")):
                current[key] = previous + value
            elif key.endswith("/max"):
                current[key] = torch.maximum(previous, value)
            elif key.endswith("/min"):
                current[key] = torch.minimum(previous, value)
            else:
                raise ValueError(
                    f"Do not know how to reduce metric '{key}'. "
                    "Use a /mean, /frac, _mean, _frac, /max, or /min suffix."
                )
        return current

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
    Autograd then propagates this through the decoder layers automatically --
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
        # decoder graph -- equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we need
        # to properly handle. The complicated part is that grad_output might not be
        # on the same device mesh as accumlated_grad.
        return accumulated_grad, None, None
