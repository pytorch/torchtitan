# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map

from torchtitan.config import CompileConfig, Configurable
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    if isinstance(pred, DTensor) and isinstance(labels, DTensor):
        return _cross_entropy_via_local_map(pred, labels)

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


def _cross_entropy_via_local_map(pred: DTensor, labels: DTensor) -> torch.Tensor:
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
    def _flatten_placement(p):
        if isinstance(p, Shard):
            return Shard(0 if p.dim == 0 else p.dim - 1)
        return p

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

        # vocab_sharded == True => loss parallel case
        # TODO: rewrite the entire loss parallel using megatron style.
        flat_pred_placements = tuple(_flatten_placement(p) for p in pred.placements)
        flat_labels_placements = tuple(_flatten_placement(p) for p in labels.placements)
        pred_dtensor = DTensor.from_local(
            flat_pred, mesh, flat_pred_placements, run_check=False
        )
        labels_dtensor = DTensor.from_local(
            flat_labels, mesh, flat_labels_placements, run_check=False
        )
        loss_dtensor = torch.nn.functional.cross_entropy(
            pred_dtensor,
            labels_dtensor,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        assert isinstance(loss_dtensor, DTensor)
        return loss_dtensor.to_local()

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
    Every loss returns the same ``(loss, metrics)`` pair so callers never have
    to branch on the return type; ``metrics`` is an empty dict for losses that
    don't emit any (e.g. ``CrossEntropyLoss``, ``MSELoss``).
    Subclasses must implement ``__init__``. Leaf losses set ``self.fn`` and
    reuse the default ``__call__``; composite losses (e.g. ``ChunkedLoss``) may
    override ``__call__`` and delegate to an inner loss instead.
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self.fn(pred, labels)
        # Normalization is owned by the loss: leaf losses (CrossEntropyLoss,
        # MSELoss) inherit this division, and ChunkedLoss delegates to its
        # inner loss so each chunk is normalized the same way.
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        # Leaf losses emit no metrics; the empty dict keeps the
        # ``(loss, metrics)`` contract uniform across all losses.
        return loss, {}


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


# Clamp |log(pi_theta / pi_old)| before exp() so a large generator/trainer
# logprob mismatch -- notably the NaNs vLLM can emit under cudagraph -- cannot
# overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


class GRPOLoss(BaseLoss):
    """Token-wise GRPO/DAPO clipped surrogate (PPO-style), computed from logits.

    Like the leaf losses (``CrossEntropyLoss``), this is a ``BaseLoss`` whose
    ``pred`` is *logits* ``[B, S, V]``. It reduces them to per-token logprobs
    internally (via ``logits_to_logprobs``) and applies the clipped surrogate::

        ratio_t   = exp(clamp(policy_logprob_t - generator_logprob_t))  # pi_theta / pi_old
        clipped_t = clamp(ratio_t, 1 - clip_low, 1 + clip_high)
        loss_t    = -min(ratio_t * A_t, clipped_t * A_t)

    The lower and upper clip bounds are independent: equal bounds give standard
    (symmetric) GRPO, while a larger upper bound is DAPO "clip-higher"
    (https://arxiv.org/abs/2503.14476), keeping more probability mass on
    up-weighted tokens to counter entropy collapse. A token whose generator
    logprob is non-finite (e.g. vLLM under cudagraph) has no valid old-policy
    reference, so it is dropped from both the loss and the denominator rather
    than trained as if it were on-policy.

    The scalar loss sums over response tokens (``loss_mask`` nonzero) and divides
    by the global ``global_valid_tokens``, so gradient accumulation across chunks
    and DP ranks matches a single large-batch step. Unlike the leaf losses it
    overrides ``__call__`` to (a) take the per-token ``generator_logprobs``,
    ``advantages``, and ``loss_mask`` as keyword arguments and (b) return a
    metrics dict (ratio statistics) alongside the loss for DP all-reduce.

    When wrapped by ``ChunkedLoss``, the per-chunk logits are reduced to logprobs
    one chunk at a time, so the full ``[B, S, V]`` logits are never materialized.
    All operations are per-token, so the chunked result is bitwise-identical to
    the un-chunked one.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        ratio_clip_low: float = 0.2
        """Lower PPO clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper PPO clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Equal to
        ``ratio_clip_low`` for symmetric GRPO; set larger for DAPO "clip-higher"
        (e.g. 0.28)."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high
        # GRPO overrides ``__call__`` and has no single ``self.fn`` to compile,
        # so loss compile does not apply. Warn instead of silently ignoring it.
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.warning(
                "Loss compile is enabled, but GRPOLoss does not support "
                "torch.compile; it will run uncompiled."
            )

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-token GRPO/DAPO clipped surrogate loss from logits.

        Args:
            pred: ``[B, S, V]`` logits from ``lm_head`` (a chunk when wrapped by
                ``ChunkedLoss``, otherwise the full sequence).
            labels: ``[B, S]`` pre-shifted target token ids; ``IGNORE_INDEX``
                positions contribute logprob 0 and gradient 0.
            global_valid_tokens: total response tokens across all microbatches
                and DP ranks; the loss denominator so gradient accumulation is
                equivalent to a single large-batch step.
            generator_logprobs: ``[B, S]`` log pi_old(a_t | s_t) from the
                sampling policy.
            advantages: ``[B, S]`` per-token advantages (0 for prompt/padding).
            loss_mask: ``[B, S]`` mask; nonzero for response tokens.

        Returns:
            ``(loss, metrics)`` where ``loss`` is a scalar tensor and ``metrics``
            is a dict of scalar tensors pre-normalized for SUM reduction across
            DP ranks.
        """
        trainer_logprobs = logits_to_logprobs(pred, labels)

        # A non-finite generator logprob (notably vLLM under cudagraph) has no
        # valid old-policy reference, so drop that token from the loss and the
        # denominator (cleaner than nan->0, which would train it as on-policy).
        # ``response_mask`` keeps the original tokens for the nan-frac metric.
        response_mask = loss_mask
        raw_log_ratio = trainer_logprobs - generator_logprobs
        loss_mask = loss_mask & torch.isfinite(raw_log_ratio)

        # Clamp the log-ratio before exp() so a large mismatch can't overflow to
        # inf/NaN, then take the per-token importance ratio pi_theta / pi_old.
        log_ratio = torch.clamp(
            torch.nan_to_num(raw_log_ratio), -_MAX_LOG_RATIO, _MAX_LOG_RATIO
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1 - self.ratio_clip_low, 1 + self.ratio_clip_high
        )
        token_pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        loss_denominator = max(global_valid_tokens, 1)
        loss = (token_pg_loss * loss_mask).sum() / loss_denominator

        with torch.no_grad():
            masked_ratio = ratio * loss_mask
            metrics = {
                "loss/ratio_mean": masked_ratio.sum() / loss_denominator,
                "loss/ratio_clipped_frac": (
                    (torch.abs(ratio - clipped_ratio) > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
                # Fraction of response tokens with a non-finite generator logprob
                # (dropped above); tracked against the original response_mask.
                "loss/generator_logprob_nan_frac": (
                    (~torch.isfinite(generator_logprobs)).float() * response_mask
                ).sum()
                / loss_denominator,
            }
        return loss, metrics


def logits_to_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``[B, S, V]`` logits -> ``[B, S]`` logprob of each label token.

    Localizes DTensor logits first: this path runs only for RL losses, which
    disable loss parallel, so the vocab is replicated (not sharded) and
    ``to_local`` yields the full vocab. (SFT/CE never takes this path — it uses
    the fused, loss-parallel-aware cross-entropy instead.) Ignored positions
    (``IGNORE_INDEX``) get logprob 0 and gradient 0.
    """
    if isinstance(logits, DTensor):
        # RL disables loss parallel, so logits are Replicate on the TP axis here.
        logits = logits.to_local()
    batch_size, seq_len, vocab_size = logits.shape
    nll = torch.nn.functional.cross_entropy(
        logits.float().reshape(batch_size * seq_len, vocab_size),
        labels.reshape(batch_size * seq_len),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    return -nll.reshape(batch_size, seq_len)


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


class ChunkedLoss(BaseLoss):
    """Memory-saving executor that wraps an inner logits-domain loss.

    Unlike the leaf losses (``CrossEntropyLoss``, ``MSELoss``), this is not a new
    loss function: it owns the ``lm_head`` and runs a configurable inner
    ``loss_fn`` over the model's hidden states in chunks. Consequently its
    ``__call__`` ``pred`` is *hidden states* ``[B, L, D]`` (model forward with
    ``_skip_lm_head=True``), whereas a leaf loss's ``pred`` is *logits*
    ``[B, L, V]``. The trainer wires this up explicitly (``set_lm_head`` +
    ``_skip_lm_head``), so the wrapper and the leaf losses are intentionally
    not substitutable despite sharing ``BaseLoss``.

    Instead of materializing the full [B, L, V] logits tensor at once, this splits
    the hidden states into N chunks along the sequence dimension and computes
    lm_head + loss on each chunk sequentially. This reduces peak memory
    from O(B*L*V) to O(B*L/N*V).

    The inner ``loss_fn`` is a ``BaseLoss`` and defaults to ``CrossEntropyLoss``.
    Per chunk, the chunk logits are passed to
    ``loss_fn(logits, labels, global_valid_tokens, **loss_inputs)``; the inner
    loss owns its normalization by the global ``global_valid_tokens``, so the
    per-chunk losses are additive and sum to the full-sequence loss. With the
    default ``CrossEntropyLoss`` this is bitwise-identical to the legacy chunked
    cross-entropy.

    RL losses (e.g. ``GRPOLoss``) are also ``BaseLoss`` subclasses: they reduce
    the chunk logits to per-token logprobs internally, take extra per-token
    ``loss_inputs`` (``generator_logprobs``, ``advantages``, ``loss_mask``), and
    return a ``(loss, metrics)`` pair. Because the inner loss consumes logits,
    the full ``[B, L, V]`` is still never materialized across all chunks.

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
        Hidden states are redistributed to ``Replicate()`` on the TP mesh
        before chunking, so each chunk enters the lm_head as ``Replicate()``
        input regardless of whether SP is enabled. With SP, this is an
        all-gather from ``Shard(1)``; without SP, it's a no-op.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: the inner ``loss_fn`` can be compiled independently (via
    ``compile_config``); lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

        loss_fn: BaseLoss.Config = field(default_factory=CrossEntropyLoss.Config)
        """Loss applied to each chunk's logits. Defaults to ``CrossEntropyLoss``
        (the legacy behavior). The inner loss owns its normalization by
        ``global_valid_tokens``. RL losses (e.g. ``GRPOLoss``) additionally
        consume per-token ``loss_inputs`` and return a metrics dict."""

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
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the chunked loss by running the inner ``loss_fn`` per chunk.

        ``pred`` should be hidden states from model forward with
        ``_skip_lm_head=True`` (not logits, unlike a leaf loss). Each chunk's
        logits are passed to the inner ``loss_fn``.

        ``loss_inputs`` are extra per-token tensors (e.g. ``generator_logprobs``,
        ``advantages``, ``loss_mask``) chunked along the sequence dimension in
        lockstep with ``labels`` and forwarded to the inner loss.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only — no per-chunk backward or gradient accumulation.

        Returns ``(loss, metrics)`` like every other loss; ``metrics`` is empty
        when the inner loss emits none (e.g. ``CrossEntropyLoss``). When
        ``.backward()`` is called on the returned loss (by the trainer or the PP
        schedule), it triggers backward through the decoder via a custom autograd
        Function.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLoss"
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

        # ``detach`` + ``requires_grad_`` makes each chunk a leaf so it
        # accumulates ``.grad`` for ``GradAccumulator``.
        h_chunks = [
            c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
        ]
        label_chunks = list(_chunk(labels))
        # Chunk extra per-token loss inputs (e.g. generator_logprobs, advantages,
        # loss_mask) along the sequence dim in lockstep with labels.
        input_chunks = {key: _chunk(value) for key, value in loss_inputs.items()}

        grad_accumulator = None
        if requires_grad:
            grad_accumulator = GradAccumulator(
                hidden_states,
                num_chunks=num_chunks,
                dtype=torch.float32,
            )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)
        # Populated only by losses that emit metrics (e.g. GRPO); summed across
        # chunks. Empty for losses that return a bare tensor (e.g. CrossEntropyLoss).
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

            # Delegate to the inner loss on the chunk's logits. ``loss_inputs``
            # (e.g. generator_logprobs/advantages/loss_mask for GRPO) are sliced
            # to this chunk; CrossEntropyLoss / MSELoss receive none. The inner
            # loss owns its normalization by the global ``global_valid_tokens``,
            # so summing the per-chunk losses (and metrics) equals the
            # full-sequence result. With the default ``CrossEntropyLoss`` this is
            # bitwise-identical to the legacy chunked cross-entropy.
            chunk_inputs = {key: chunks[i] for key, chunks in input_chunks.items()}
            chunk_loss, chunk_metrics = self.loss_fn(
                logits, label_chunk, global_valid_tokens, **chunk_inputs
            )
            for key, value in chunk_metrics.items():
                metrics[key] = metrics.get(key, 0) + value

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
        if metrics:
            metrics["loss/mean"] = total_loss.detach()
        if not requires_grad:
            return total_loss, metrics

        assert grad_accumulator is not None
        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        return self._gradient_backprop(
            hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled
        ), metrics

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
