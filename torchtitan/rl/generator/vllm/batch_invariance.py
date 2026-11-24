# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


# TODO: delete once vLLM's batch-invariant mode runs the V2 sampler's logprob and
# temperature kernels as PyTorch ops.
def force_trainer_logprobs_for_batch_invariance() -> None:
    """Replace two Triton kernels of vLLM's V2 sampler with PyTorch ops, so generator
    logprobs match the trainer's bit for bit.

        vLLM function            replaced by
        compute_token_logprobs   the trainer's ``compute_logprobs``, once per token id column
        apply_temperature        ``logits.div_(temperature)``, the trainer's division

    Batch-invariant mode overrides aten ops only, and these Triton kernels never call aten.
    """
    import vllm.v1.worker.gpu.sample.gumbel as vllm_gumbel
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob

    from torchtitan.components.loss import compute_logprobs

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
        # vLLM gives token_ids [N, K]. SamplingParams(logprobs=0) makes real
        # requests K=1, but we can't assert that: vLLM's kernel warmup probes
        # this patched fn with K>1 (e.g. K=6), so we must handle any K. Reuse
        # the trainer's compute_logprobs once per column.
        #
        # NOTE: each element of token_ids is scored independently, after the
        # whole generated sequence is marterialized. We iterate column-by-column
        # purely because torchtitan's compute_logprobs takes one token id per
        # position; it is not a cross-column/cross-position dependency.
        token_ids = token_ids.to(torch.int64)
        per_column = [
            compute_logprobs(
                logits,
                token_ids[:, k],
                vocab_parallel_group=None,
            )
            for k in range(token_ids.shape[1])
        ]
        return torch.stack(per_column, dim=-1)  # [N, K]

    # vLLM's Triton kernel rounds fp32 division less precisely than PyTorch: at
    # temperature 0.8, 15% of logits differ by up to 1 ulp (GB300), so generator
    # logprobs would not match the trainer's.
    def generator_apply_temperature(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        """In place, divide each row of ``[N, V]`` fp32 logits by its request's temperature.

        Example:

            temperature = [0.0, 0.8, 0.7]     # one per request; 0.0 = greedy
            expanded_idx_mapping = [1, 0, 2]  # row -> request
            # rows are divided by [0.8, 1.0, 0.7]: greedy rows stay unchanged, as in vLLM's kernel
        """
        row_temperature = temperature[expanded_idx_mapping]
        row_temperature = torch.where(row_temperature == 0.0, 1.0, row_temperature)
        logits.div_(row_temperature.unsqueeze(-1))

    # vLLM modules import these functions by name (e.g. prompt_logprob.py, states.py),
    # so rebind the name in every loaded vLLM module that holds vLLM's function.
    # Modules loaded later import the patched function.
    for name, defining_module, replacement in (
        ("compute_token_logprobs", vllm_logprob, generator_compute_token_logprobs),
        ("apply_temperature", vllm_gumbel, generator_apply_temperature),
    ):
        original = getattr(defining_module, name)
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("vllm") and (
                vars(module).get(name) is original
            ):
                setattr(module, name, replacement)
    logger.info(
        "Patched vLLM compute_token_logprobs and apply_temperature with PyTorch ops "
        "so generator and trainer share one logprob code path"
    )
