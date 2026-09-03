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
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import CompileConfig, Configurable
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


@dataclass(frozen=True, slots=True)
class LossTerm:
    """One prediction and its aligned inputs for a chunked objective.

    Args:
        pred: Hidden states consumed by the language-model head.
        labels: Labels aligned with ``pred`` along the sequence axis.
        inputs: Additional loss inputs aligned with the same sequence axis.
        weight: Static multiplier applied to both the reported loss and the
            gradient for this term.
    """

    pred: torch.Tensor
    labels: torch.Tensor
    inputs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """Cross-entropy over ``pred[T, V]`` and ``labels[T]`` with sum reduction."""
    if isinstance(pred, DTensor):
        assert get_spmd_backend() == "partial_dtensor"
        if pred.placements == (Shard(1),):
            return _LossParallelCrossEntropy.apply(
                pred.to_local().float(),
                labels,
                pred.device_mesh.get_group("tp"),
                pred.shape[-1],
                "sum",
            )
    elif get_spmd_backend() == "spmd_types" and spmd_mesh_size("tp") > 1:
        return _LossParallelCrossEntropy.apply(
            pred.float(),
            labels,
            current_spmd_mesh().get_group("tp"),  # pyrefly: ignore[missing-attribute]
            global_vocab_size,
        )

    return torch.nn.functional.cross_entropy(
        pred.float(),
        labels,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


class _LossParallelCrossEntropy(torch.autograd.Function):
    """
    Vocab-parallel cross-entropy on local ``[T, V_local]`` logits.

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
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> None:
        """
        SPMD type: logits S(-1)@TP, labels I@TP -> loss I@TP.
        Non-TP axes are passed through from logits to the output.
        """
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        spmd.assert_local_type_like(
            result,
            logits,
            {tp_group: spmd.I},  # pyrefly: ignore [bad-argument-type]
        )

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
        ``reduction="none"`` returns the per-token NLL ``[T]``, which GRPO
        negates to get per-token logprobs without all-gathering the vocab.
        """
        logits_dtype = logits.dtype
        logits = logits.float()

        # Compute this rank's vocab shard bounds for the local logits.
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
        vocab_start = min(global_vocab_size, chunk_size * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk_size)
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "_LossParallelCrossEntropy does not support empty vocab shards."
            )

        torch._assert_async(
            torch.all(
                (labels == IGNORE_INDEX)
                | ((labels >= 0) & (labels < global_vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {global_vocab_size})",
        )

        # All-reduce max for numerically stable distributed log-softmax.
        local_max = torch.amax(logits, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # All-reduce sum over shifted logits for the global softmax denominator.
        shifted = logits - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(
            shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        log_probs = shifted - torch.log(shifted_sumexp)

        # Mask labels outside this vocab shard; the TP all-reduce below selects
        # the owner rank's log probability for each target token.
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
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
        result = torch.where(labels != IGNORE_INDEX, result, 0)

        # Save local-shard log probabilities for the fused CE backward.
        ctx.save_for_backward(log_probs, labels)
        ctx.logits_dtype = logits_dtype
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
        log_probs, labels = ctx.saved_tensors
        safe_labels = torch.where(labels != IGNORE_INDEX, labels, 0)
        out_of_range = (safe_labels < ctx.vocab_start) | (
            safe_labels >= ctx.vocab_start + ctx.local_vocab_size
        )
        local_labels = safe_labels - ctx.vocab_start
        local_labels[out_of_range] = 0

        grad_input = torch.zeros_like(log_probs)
        row_idx = torch.arange(local_labels.shape[0], device=local_labels.device)
        grad_update = out_of_range.to(grad_input.dtype) - 1.0
        grad_input[row_idx, local_labels] = grad_update

        # reduction="none" gives a per-token ``[T]`` upstream grad; unsqueeze to
        # ``[T, 1]`` to broadcast over the local vocab. "sum" gives the scalar
        # loss grad, which broadcasts as-is.
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where(
            (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        grad_logits = grad_logits.to(ctx.logits_dtype)
        return grad_logits, None, None, None, None


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

    def _build_loss_terms(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        **loss_inputs: Any,
    ) -> tuple[LossTerm, ...]:
        """Align model outputs and targets into loss terms."""
        if not isinstance(pred, torch.Tensor):
            raise ValueError(
                f"{type(self).__name__} expects one prediction tensor, "
                f"got {type(pred).__name__}."
            )
        return (LossTerm(pred, labels, loss_inputs),)

    def _compute_loss_term(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute one unnormalized loss term."""
        del loss_inputs
        return self.fn(pred, labels), {}

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the scaled loss and any metrics computed by the loss."""
        del kwargs
        loss = self.fn(pred, labels)
        # loss: V->P, annotate global_valid_tokens
        if get_spmd_backend() == "spmd_types" and current_spmd_mesh() is not None:
            spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})
            if global_valid_tokens is not None:
                spmd.assert_type(
                    global_valid_tokens,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_valid_tokens is not None:
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
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        loss = self.fn(pred, labels, global_vocab_size=self.global_vocab_size)
        # loss: V->P, annotate global_valid_tokens
        if get_spmd_backend() == "spmd_types" and current_spmd_mesh() is not None:
            spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})
            if global_valid_tokens is not None:
                spmd.assert_type(
                    global_valid_tokens,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss, {}

    def _compute_loss_term(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute unnormalized cross entropy for one loss term."""
        del loss_inputs
        return (
            self.fn(
                pred,
                labels,
                global_vocab_size=self.global_vocab_size,
            ),
            {},
        )


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
    *,
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs from ``logits[T, V]`` and ``labels[T]``.

    Any DTensor placement handling is centralized here so RL losses that call
    ``compute_logprobs`` do not need to duplicate the vocab-gather logic.

    When ``return_entropy`` is set, also returns per-token Shannon entropy
    ``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, with shape
    ``[T]``. Both share the single vocab gather + fp32 upcast.
    Entropy is a metric only, so it is computed under ``no_grad``: it never
    contributes gradient and must not build an autograd graph over the logits
    softmax.

    Returns ``logprobs`` when ``return_entropy`` is False, else
    ``(logprobs, entropy)``.
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
        # dst=I, not R: the vocab all-gather's grad is the replicated upstream
        # grad sliced back to this rank's vocab shard (I's backward), not an
        # all-reduce (R's backward). The latter over-counts by tp_degree and
        # diverges from the DTensor path above, whose redistribute grad slices.
        logits = spmd.redistribute(
            logits,
            "tp",
            src=spmd.S(-1),
            dst=spmd.I,
        )

    # Single bf16->fp32 upcast, reused by both logprobs and (optionally) entropy.
    logits = logits.float()
    logprobs = -F.cross_entropy(
        logits,
        labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    if not return_entropy:
        return logprobs
    with torch.no_grad():
        entropy = torch.logsumexp(logits, dim=-1) - (
            torch.softmax(logits, dim=-1) * logits
        ).sum(dim=-1)
    return logprobs, entropy


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
        seq_dim: int = 0,
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

    Instead of materializing the full [T, V] logits tensor at once, this splits
    the hidden states into N chunks along the token dimension and computes
    lm_head + loss on each chunk sequentially. This reduces peak memory
    from O(T*V) to O(T/N*V).

    The inner ``loss_fn`` defaults to ``CrossEntropyLoss``. It builds one or
    more weighted ``LossTerm`` objects. Each term pairs a model output with its
    aligned labels and auxiliary inputs. Every term is evaluated one chunk at
    a time through the shared language-model head.

    The flow:
    1. Build aligned ``LossTerm`` objects before chunking.
    2. Split each term's model output, labels, and aligned tensor inputs into
       ``num_chunks`` equal chunks along the sequence axis.
    3. Detach each model-output chunk at the language-model-head boundary and
       make it a leaf for gradient collection.
    4. Unshard the FSDP language-model head once, keep it unsharded while
       processing every term and chunk, synchronize its accumulated gradients
       on the final backward, and then reshard it once.
    5. For each chunk, run ``lm_head(chunk) -> loss_fn(logits, labels)``,
       scale the chunk's summed loss by the shared ``global_valid_tokens``,
       and call ``backward()``.
    6. Use ``GradAccumulator`` to assemble the chunk gradients into one full
       hidden-state gradient of shape ``[T, D]`` for each loss term.
    7. Backpropagate through the decoder once using all accumulated gradients,
       equivalent to ``torch.autograd.backward(model_outputs,
       accumulated_grads)``.

    FSDP2 composability:
        The lm_head is unsharded once, and its reshard-after-forward and
        reshard-after-backward are temporarily disabled so that its parameters
        stay unsharded across all loss terms and chunks, avoiding repeated
        all-gathers. Gradient synchronization remains disabled until the final
        loss term's final chunk, when one reduce-scatter processes the
        accumulated lm_head parameter gradients before the lm_head is resharded.

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
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute chunked loss.

        Every prediction represented by ``pred`` must come from model forward
        with ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only -- no per-chunk backward or gradient accumulation.

        Returns a differentiable loss and metrics. When ``.backward()`` is called
        on the loss, it triggers backward through the decoder via a custom
        autograd Function.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
        loss_terms = self.loss_fn._build_loss_terms(pred, labels, **loss_inputs)
        assert loss_terms, (
            f"{type(self.loss_fn).__name__}._build_loss_terms() "
            "must return at least one loss term"
        )
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        requires_grad = loss_terms[0].pred.requires_grad
        if any(
            loss_term.pred.requires_grad != requires_grad
            for loss_term in loss_terms[1:]
        ):
            raise ValueError(
                "All chunked-loss predictions must agree on whether gradients "
                "are required."
            )

        # Chunking always operates on the *local* view: when ``t`` is a
        # Shard(0) DTensor, chunking the global view would distribute whole
        # chunks across ranks (e.g. size=2, num_chunks=8: chunks 0-3 on
        # rank 0, 4-7 on rank 1), leaving half the per-chunk DTensors with
        # local seq=0 and breaking GradAccumulator's slice writes.
        # ``local_map`` runs the chunking body on plain tensors; under the
        # non-DTensor (eager) path we call ``_chunk_local`` directly.
        # Equal chunk sizes also match GradAccumulator's sequential slice
        # writes, which use one chunk length for each write offset.
        def _chunk_local(t):
            seq_len = t.shape[0]
            torch._check(
                seq_len % num_chunks == 0,
                lambda: "ChunkedLossWrapper sequence length must be divisible by num_chunks",
            )
            chunk_len = seq_len // num_chunks
            return tuple(
                c.contiguous() for c in torch.split(t, [chunk_len] * num_chunks, dim=0)
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
            total_loss = loss_terms[0].pred.new_zeros((), dtype=torch.float32)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                total_loss = spmd.mutate_type(
                    total_loss,
                    src=spmd.R,
                    dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
                )
            metrics: dict[str, torch.Tensor] = {}
            hidden_states: list[torch.Tensor] = []
            accumulated_grads: list[torch.Tensor] = []

            # Keep lm_head unsharded across all loss terms and chunks to avoid
            # repeated all-gathers. Disable gradient sync until the final loss
            # term's final chunk, coalescing all lm_head parameter gradients
            # into one reduce-scatter.
            if fsdp_enabled:
                lm_head.set_reshard_after_forward(False)
                lm_head.set_reshard_after_backward(False)
                lm_head.set_requires_gradient_sync(False, recurse=False)
                # An implicit unshard stores an all-gather event in FSDP's shared
                # all_gather_state for the next FSDP module to consume. Since
                # lm_head is the final FSDP forward in this loop, eager warmup
                # leaves that state uncleared, and CUDA graph capture cannot wait
                # on its eager event. Explicitly unshard while FSDP is idle to
                # avoid populating the shared state.
                with spmd.no_typecheck():
                    lm_head.unshard()

            gradient_sync_restored = False
            for loss_term_index, loss_term in enumerate(loss_terms):
                term_pred = loss_term.pred
                h_chunks = tuple(
                    chunk.detach().requires_grad_(requires_grad)
                    for chunk in _chunk(term_pred)
                )
                label_chunks = _chunk(loss_term.labels)
                input_chunks = {
                    key: _chunk(value) if isinstance(value, torch.Tensor) else value
                    for key, value in loss_term.inputs.items()
                }
                grad_accumulator = (
                    GradAccumulator(
                        term_pred,
                        num_chunks=num_chunks,
                        dtype=torch.float32,
                    )
                    if requires_grad
                    else None
                )

                for chunk_index, (h_chunk, label_chunk) in enumerate(
                    zip(h_chunks, label_chunks, strict=True)
                ):
                    is_last_work = (
                        loss_term_index == len(loss_terms) - 1
                        and chunk_index == len(h_chunks) - 1
                    )
                    if fsdp_enabled and is_last_work:
                        lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                            True, recurse=False
                        )
                        gradient_sync_restored = True

                    chunk_inputs = {
                        key: chunks[chunk_index]
                        if isinstance(chunks, tuple)
                        else chunks
                        for key, chunks in input_chunks.items()
                    }
                    chunk_loss, chunk_metrics = self.loss_fn._compute_loss_term(
                        lm_head(h_chunk),
                        label_chunk,
                        **chunk_inputs,
                    )
                    if global_valid_tokens is not None:
                        with spmd.no_typecheck():
                            chunk_loss = chunk_loss / global_valid_tokens
                    metrics = self._combine_chunk_metrics(metrics, chunk_metrics)
                    weighted_chunk_loss = chunk_loss * loss_term.weight
                    total_loss = total_loss + weighted_chunk_loss.detach()

                    if requires_grad:
                        with spmd.no_typecheck():
                            weighted_chunk_loss.backward()
                            assert h_chunk.grad is not None
                            assert grad_accumulator is not None
                            grad_accumulator.add(h_chunk.grad)
                            h_chunk.grad = None

                hidden_states.append(term_pred)
                if grad_accumulator is not None:
                    accumulated_grads.append(
                        grad_accumulator.result().to(term_pred.dtype)
                    )

            if fsdp_enabled:
                assert gradient_sync_restored
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.reshard()
            if not requires_grad:
                return total_loss, metrics

        with spmd.no_typecheck():
            loss = self._gradient_backprop(
                tuple(hidden_states),
                tuple(accumulated_grads),
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
        hidden_states: tuple[torch.Tensor, ...],
        accumulated_grads: tuple[torch.Tensor, ...],
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        """Bridge chunked lm-head gradients back to the decoder.

        Each loss term's chunk gradients are assembled into one full-sequence
        gradient for its model output. Backward applies ``accumulated_grads``
        to the corresponding ``hidden_states``, equivalent to
        ``torch.autograd.backward(hidden_states, accumulated_grads)``.
        """
        del lm_head, fsdp_enabled
        return _DecoderOutputGradientBackProp.apply(
            len(hidden_states),
            *hidden_states,
            *accumulated_grads,
            total_loss,
        )


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Route precomputed chunked-loss gradients to multiple model outputs."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, num_predictions: int, *args: torch.Tensor) -> torch.Tensor:
        # args packs N model outputs, N accumulated gradients, and one loss.
        if len(args) != 2 * num_predictions + 1:
            raise ValueError(
                "Chunked-loss autograd bridge expected "
                f"{2 * num_predictions + 1} tensor arguments for "
                f"{num_predictions} predictions, got {len(args)}."
            )
        ctx.num_predictions = num_predictions
        ctx.save_for_backward(*args[num_predictions : 2 * num_predictions])
        return args[-1].detach()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # Return each accumulated gradient for its corresponding hidden state.
        # Autograd then propagates them through the existing decoder graph,
        # equivalent to torch.autograd.backward(hidden_states, accumulated_grads),
        # but expressed as return values so autograd traverses the graph once.
        # This assumes callers backpropagate the returned loss directly, so
        # grad_output is 1. If callers transform or scale the loss first, these
        # gradients must also be scaled by grad_output after aligning its DTensor
        # mesh and placements with each accumulated gradient.
        del grad_output
        accumulated_grads = ctx.saved_tensors
        return (
            None,
            *accumulated_grads,
            *(None for _ in range(ctx.num_predictions)),
            None,
        )
