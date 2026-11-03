# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging

import torch

logger = logging.getLogger(__name__)


def force_logprobs_fn_for_batch_invariance() -> None:
    """Make vLLM's v2 logprob path dispatch.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops.

    Swapping the kernel routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit. The wrapper supplies
    full logits to vLLM, so the wrapper explicitly selects the replicated path.
    """
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.components.loss import compute_topk_logprobs

    def generator_compute_token_logprobs(
        logits: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-token logprobs for vLLM's v2 sampler (replaces its fused kernel).

        Args:
            logits: Full-vocabulary ``[N, V]`` next-token logits for N sampled
                positions, already gathered by the vLLM model wrapper.
            token_ids: ``[N, K]`` the K token ids to score per position (vLLM
                passes the sampled token's logprob plus any top-k logprobs it requested).

        Returns:
            ``[N, K]`` logprob of each of the K token ids at each position.
        """
        # token_ids is [N, K]: column 0 is the sampled token, then
        # SamplingConfig.num_topk_logprobs top-k ids (kernel warmup also probes K=6).
        # The trainer scores column 0 with compute_logprobs: -cross_entropy, which is
        # log_softmax + nll_loss, and nll_loss only picks one entry. So it matches this
        # log_softmax + gather bitwise.
        return compute_topk_logprobs(
            logits,
            token_ids,
            vocab_parallel_group=None,
            global_vocab_size=None,
        )

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )
