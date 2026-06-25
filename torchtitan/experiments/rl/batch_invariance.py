# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariance helpers for bitwise trainer/generator numerics parity.

Groups the pieces that make the vLLM generator match the trainer op-for-op under
batch-invariant mode: a model-config converter that pins FlexAttention kernel
options, plus two runtime patches (bmm and the v2 logprob kernel).
"""

import logging
from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import FlexAttention
from torchtitan.protocols.model import ModelConfigConverter

logger = logging.getLogger(__name__)


class BatchInvariantFlexConverter(ModelConfigConverter):
    """Pin flex attention kernel options for batch-invariant mode.

    Sets fixed BLOCK_M/BLOCK_N=16 and BACKEND=TRITON on all
    FlexAttention layers.

    BACKEND=TRITON is to avoid flex_decode kernel.
    """

    # the triton BLOCK_N tile size needs to be pinned for stable numerics and
    # needs to match vLLM's for identical results, today vLLM default is 16
    # TODO: run some experiments to determine impact of small vs large tile sizes
    _BLOCK_M = 16
    _BLOCK_N = 16

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config):
        for layer_cfg in model_config.layers:
            inner = layer_cfg.attention.inner_attention
            if isinstance(inner, FlexAttention.Config):
                inner.kernel_options["BACKEND"] = "TRITON"
                inner.kernel_options["BLOCK_M"] = self._BLOCK_M
                inner.kernel_options["BLOCK_N"] = self._BLOCK_N
        return model_config


_batch_invariant_bmm_lib: torch.library.Library | None = None


def patch_bmm_for_batch_invariance() -> None:
    """Override ``aten::bmm`` with vLLM's batch-invariant bmm kernel.

    torchtitan's batch-invariant mode (``batch_invariant_ops``, applied by
    ``set_batch_invariance``) overrides ``mm``/``addmm``/``_log_softmax``/
    ``mean.dim`` but not ``bmm``. The MoE router gate (3-D activation @ 2-D
    weight) lowers to ``aten::bmm`` in the generator but ``aten::mm`` in the
    trainer, so without this the generator's gate scores drift from the
    trainer's and flip top-k expert routing, breaking on-policy logprob parity.

    TODO: Investigate how to drop bmm batch invariant patch in generator.
    """
    global _batch_invariant_bmm_lib
    if _batch_invariant_bmm_lib is not None:
        return
    from vllm.model_executor.layers.batch_invariant import bmm_batch_invariant

    _batch_invariant_bmm_lib = torch.library.Library("aten", "IMPL")
    _batch_invariant_bmm_lib.impl("bmm", bmm_batch_invariant, "CUDA")
    # pyrefly: ignore[bad-assignment]
    torch.bmm = bmm_batch_invariant


def force_logprobs_fn_for_batch_invariance() -> None:
    """Make vLLM's v2 logprob path dispatch.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops.

    Swapping the kernel to routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit.
    """
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.experiments.rl.actors.trainer import compute_logprobs

    def generator_compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token logprobs for vLLM's v2 sampler (replaces its fused kernel).

        Args:
            logits: ``[N, V]`` next-token logits for N sampled positions
                (V = vocab_size).
            token_ids: ``[N, K]`` the K token ids to score per position (vLLM
                passes the sampled token's logprob plus any top-k logprobs it requested).

        Returns:
            ``[N, K]`` logprob of each of the K token ids at each position.
        """
        # vLLM gives token_ids [N, K]. SamplingParams(logprobs=0) makes real
        # requests K=1, but we can't assert that: vLLM's kernel warmup probes
        # this patched fn with K>1 (e.g. K=6), so we must handle any K. Map the
        # N positions to one sequence (B=1, S=N) and reuse the trainer's
        # compute_logprobs (one token per position) once per column.
        #
        # NOTE: each element of token_ids is scored independently, after the
        # whole generated sequence is marterialized. We iterate column-by-column
        # purely because torchtitan's compute_logprobs takes one token id per
        # position; it is not a cross-column/cross-position dependency.
        logits = logits.unsqueeze(0)  # [1, N, V]  (B=1, S=N)
        token_ids = token_ids.to(torch.int64)
        per_column = [
            compute_logprobs(logits, token_ids[:, k].unsqueeze(0))  # [1, N]
            for k in range(token_ids.shape[1])
        ]
        return torch.stack(per_column, dim=-1).squeeze(0)  # [N, K]

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )
