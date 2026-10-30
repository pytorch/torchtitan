# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime helper for bitwise trainer/generator numerics parity."""

import logging

import torch

logger = logging.getLogger(__name__)

MAX_FLEX_BATCH_INVARIANT_REQUESTS = 64


def validate_flex_page_mapping_request_limit(num_requests: int) -> None:
    if num_requests > MAX_FLEX_BATCH_INVARIANT_REQUESTS:
        raise ValueError(
            "ROCm FlexAttention batch invariance supports at most "
            f"{MAX_FLEX_BATCH_INVARIANT_REQUESTS} concurrent requests, "
            f"got {num_requests}"
        )


def enable_rocm_batch_invariant_page_mapping() -> None:
    """Use a request-independent page lookup for ROCm FlexAttention.

    Each physical page has one logical index. A separate ownership bitmask
    preserves request isolation and shared prefix-cache pages.
    """
    from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

    if getattr(
        FlexAttentionMetadata,
        "_torchtitan_rocm_batch_invariant_page_mapping",
        False,
    ):
        return

    def get_paged_mask_mod(self):
        validate_flex_page_mapping_request_limit(self.num_reqs)

        active_mapping = self.physical_to_logical[: self.num_reqs]
        valid_mapping = active_mapping >= 0
        # A physical page is either exclusive to one request or shared as the
        # same logical prefix page, so all valid entries in a column agree.
        physical_to_logical = active_mapping.amax(dim=0)
        request_indices = torch.arange(
            self.num_reqs,
            device=active_mapping.device,
            dtype=torch.int64,
        )[:, None]
        request_bits = 1 << request_indices
        physical_request_mask = torch.where(valid_mapping, request_bits, 0).sum(dim=0)
        doc_ids = self.doc_ids
        query_start_loc = self.query_start_loc
        decode_offset = self.decode_offset
        seq_lens = self.seq_lens
        logical_mask_mod = self.logical_mask_mod
        block_size = self.block_size

        def global_page_mask_mod(
            batch_index,
            head_index,
            query_index,
            physical_kv_index,
        ):
            query_request = doc_ids[query_index]
            query_request_i64 = query_request.to(torch.int64)
            physical_kv_block = physical_kv_index // block_size
            physical_kv_offset = physical_kv_index % block_size
            logical_block = physical_to_logical[physical_kv_block]
            logical_kv_index = logical_block * block_size + physical_kv_offset
            logical_query_index = (
                query_index
                - query_start_loc[query_request]
                + decode_offset[query_request]
            )
            request_bit = torch.ones_like(query_request_i64) << query_request_i64
            request_owns_page = (
                physical_request_mask[physical_kv_block] & request_bit
            ) != 0
            return (
                (logical_block >= 0)
                & (logical_kv_index < seq_lens[query_request])
                & request_owns_page
                & logical_mask_mod(
                    batch_index,
                    head_index,
                    logical_query_index,
                    logical_kv_index,
                )
            )

        return global_page_mask_mod

    FlexAttentionMetadata.get_paged_mask_mod = get_paged_mask_mod
    FlexAttentionMetadata._torchtitan_rocm_batch_invariant_page_mapping = True
    logger.info("Enabled ROCm batch-invariant page mapping for vLLM FlexAttention")


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

    from torchtitan.components.loss import compute_logprobs

    if getattr(vllm_logprob, "_torchtitan_batch_invariant", False):
        return

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
        # whole generated sequence is materialized. We iterate column-by-column
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

    vllm_logprob.compute_token_logprobs = generator_compute_token_logprobs
    vllm_logprob._torchtitan_batch_invariant = True
    logger.info(
        "Patched vLLM compute_token_logprobs with trainer's implementation "
        "so generator and trainer share one logprob code path"
    )
