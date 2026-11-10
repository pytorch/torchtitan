# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-side patches that make SGLang's logprobs match the trainer's bitwise."""

import functools
import logging

import torch

logger = logging.getLogger(__name__)


@functools.cache
def force_trainer_logprobs_for_batch_invariance() -> None:
    """Return the sampled tokens' logprobs from the trainer's ``compute_logprobs``.

    SGLang's sampler returns ``log(probs)`` from its fused temperature-softmax kernel
    or, in its deterministic mode (on with ``DebugConfig.seed``), ``log_softmax`` of
    ``logits / T`` with its own ops. Either differs from the trainer's
    ``compute_logprobs(logits / T)`` in the last bits. This registers a sampler, for
    every built-in sampling backend, that recomputes the returned logprobs with the
    trainer's operation on the same full-vocabulary logits. Sampling itself is
    unchanged. Process-wide; runs once.
    """
    from sglang.srt.arg_groups.choices import BUILT_IN_SAMPLING_BACKENDS
    from sglang.srt.layers.sampler import register_sampler_backend, Sampler

    from torchtitan.components.loss import compute_logprobs

    class TrainerLogprobsSampler(Sampler):
        def forward(
            self,
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
            **kwargs,
        ) -> torch.Tensor:
            args = (
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )
            if not return_logprob or sampling_info.is_all_greedy:
                return super().forward(*args, **kwargs)
            # A copy: the sampler overwrites the logits with probabilities in place.
            logits = logits_output.next_token_logits / sampling_info.temperatures
            next_token_ids = super().forward(*args, **kwargs)
            logits_output.next_token_logprobs = compute_logprobs(
                logits, next_token_ids.long(), vocab_parallel_group=None
            )
            return next_token_ids

    for backend in BUILT_IN_SAMPLING_BACKENDS:
        register_sampler_backend(backend, TrainerLogprobsSampler)
    logger.info(
        "Registered a sampler that returns the trainer's compute_logprobs, so "
        "generator and trainer share one logprob code path"
    )
