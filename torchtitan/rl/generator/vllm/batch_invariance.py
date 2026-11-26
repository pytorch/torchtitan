# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging

import torch

logger = logging.getLogger(__name__)


def force_trainer_logprobs_for_batch_invariance() -> None:
    """Make vLLM's v2 logprob path dispatch.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops.

    Swapping the kernel routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit. The wrapper supplies
    full logits to vLLM, so the wrapper explicitly selects the replicated path.
    """
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.components.loss import compute_logprobs, compute_topk_logprobs

    def generator_compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token logprobs for vLLM's v2 sampler (replaces its fused kernel).

        Args:
            logits: Full-vocabulary ``[N, V]`` next-token logits for N sampled
                positions, already gathered by the vLLM model wrapper and divided by
                the temperature (``processed_logprobs``).
            token_ids: ``[N, K]`` the K token ids to score per position: the sampled
                token, then the top-k a request asked for (score centering).

        Returns:
            ``[N, K]`` logprob of each of the K token ids at each position.
        """
        # The trainer's functions for each column: compute_logprobs for the sampled
        # token, compute_topk_logprobs for the top-k. vLLM's kernel warmup probes this
        # function with K > 1 too, so handle any K.
        token_ids = token_ids.to(torch.int64)
        sampled = compute_logprobs(logits, token_ids[:, 0], vocab_parallel_group=None)
        if token_ids.shape[1] == 1:
            return sampled.unsqueeze(-1)
        topk = compute_topk_logprobs(
            logits,
            token_ids[:, 1:],
            labels=token_ids[:, 0],
            label_logprobs=sampled,
            vocab_parallel_group=None,
            global_vocab_size=None,
            # vLLM already divided the logits; dividing by 1 is exact.
            temperature=torch.ones(logits.shape[0], device=logits.device),
        )
        return torch.cat([sampled.unsqueeze(-1), topk], dim=-1)  # [N, K]

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )
