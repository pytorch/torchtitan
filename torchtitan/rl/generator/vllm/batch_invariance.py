# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging

import torch

logger = logging.getLogger(__name__)


def patch_vllm_logprob_and_temperature_kernels() -> None:
    """Replace the Triton logprob and temperature kernels of vLLM's v2 sampler with PyTorch ops.

    The v2 GPU sampler computes per-token logprobs with a fused Triton kernel
    (``compute_token_logprobs`` -> ``_topk_log_softmax_kernel``) that inlines
    ``log(softmax(logits))`` and never calls PyTorch ops. It also divides logits
    by the temperature in a Triton kernel (``apply_temperature``).

    Swapping both kernels routes the generator and the trainer through
    the same set of ops, so logprobs match bit-for-bit. The wrapper supplies
    full logits to vLLM, so the wrapper explicitly selects the replicated path.
    """
    import vllm.v1.worker.gpu.sample.gumbel as vllm_gumbel
    import vllm.v1.worker.gpu.sample.logprob as vllm_logprob
    import vllm.v1.worker.gpu.sample.prompt_logprob as vllm_prompt_logprob
    import vllm.v1.worker.gpu.sample.states as vllm_sampling_states

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

    # vLLM's Triton kernel divides less precisely than PyTorch: at temperature 0.8,
    # 15% of fp32 logits differ by up to 1 ulp (GB300), so the trainer's logprobs
    # differ too.
    def generator_apply_temperature(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        """In place, divide each row of ``[N, V]`` fp32 logits by its request's temperature.

        Greedy rows (temperature 0) stay unchanged, as in vLLM's kernel.
        """
        row_temperature = temperature[expanded_idx_mapping]
        row_temperature = torch.where(row_temperature == 0.0, 1.0, row_temperature)
        logits.div_(row_temperature.unsqueeze(-1))

    # prompt_logprob.py and states.py import these functions by name, so their
    # bindings are patched too. If vLLM moved a function, patching the old binding
    # would leave its Triton kernel running with no error, so check first.
    for importer, name, defining_module in (
        (vllm_prompt_logprob, "compute_token_logprobs", vllm_logprob),
        (vllm_sampling_states, "apply_temperature", vllm_gumbel),
    ):
        if getattr(importer, name, None) is not getattr(defining_module, name):
            raise RuntimeError(
                f"{importer.__name__}.{name} is not {defining_module.__name__}.{name}: "
                "vLLM moved it, so update this patch"
            )
    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    vllm_prompt_logprob.compute_token_logprobs = generator_compute_token_logprobs
    vllm_gumbel.apply_temperature = generator_apply_temperature
    vllm_sampling_states.apply_temperature = generator_apply_temperature
    logger.info(
        "Patched vLLM compute_token_logprobs and apply_temperature with PyTorch ops "
        "so generator and trainer share one logprob code path"
    )
