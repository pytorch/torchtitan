# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-side patches that make SGLang's logprobs match the trainer's bitwise."""

import logging

import torch

logger = logging.getLogger(__name__)


def force_trainer_logprobs_for_batch_invariance() -> None:
    """Compute the sampled tokens' logprobs with the trainer's ``compute_logprobs``.

    SGLang's sampler fuses temperature and softmax into one kernel and returns
    ``log(probs)``, which differs from the trainer's ``compute_logprobs(logits / T)``
    in the last bits. This wraps ``Sampler.forward`` to recompute the returned logprobs
    from ``logits / T``, the trainer's operation on the same full-vocabulary logits.
    Sampling itself is unchanged. Idempotent; process-wide.
    """
    from sglang.srt.layers.sampler import Sampler

    from torchtitan.components.loss import compute_logprobs

    sample = Sampler.forward
    if getattr(sample, "_trainer_logprobs", False):
        return

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
        if not return_logprob or sampling_info.is_all_greedy:
            return sample(
                self,
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
                **kwargs,
            )
        # A copy: the sampler overwrites the logits with probabilities in place.
        logits = logits_output.next_token_logits / sampling_info.temperatures
        next_token_ids = sample(
            self,
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
            **kwargs,
        )
        logits_output.next_token_logprobs = compute_logprobs(
            logits, next_token_ids.long(), vocab_parallel_group=None
        )
        return next_token_ids

    forward._trainer_logprobs = True
    # TODO(https://github.com/sgl-project/sglang/issues/48475): SGLang has no hook for
    # the logprobs its sampler returns; patch its forward.
    Sampler.forward = forward
    logger.info(
        "Patched SGLang's Sampler.forward to return the trainer's compute_logprobs, "
        "so generator and trainer share one logprob code path"
    )
