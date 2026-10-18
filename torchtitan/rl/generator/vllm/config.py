# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM-specific generator configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode, PassConfig

logger = logging.getLogger(__name__)


# vLLM's default max_num_batched_tokens (vllm's per-step budget:
# prefill + decode tokens summed over the batch). Used as the CUDA graph capture
# cap for "FULL" (which graphs prefill / mixed batches, so capture sizes must
# reach the per-step budget or those batches fall back to eager) when
# ``Config.max_num_batched_tokens`` is unset; when that field is set, its value
# is used instead (and also drives the vLLM engine).
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048


@dataclass(kw_only=True, slots=True)
class VLLMCudaGraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    Local compile regions come from the model config's ``local_compile_regions``.
    Only CUDA graph capture, which is vLLM-specific, is controlled here.

    ``mode`` selects which vLLM CUDA graph mode to capture; see that field and
    ``get_vllm_compilation_config`` for the per-mode trade-offs. The default,
    ``FULL``, graphs the whole forward (prefill included).
    """

    mode: Literal["NONE", "FULL_DECODE_ONLY", "FULL"] = "FULL"
    """Which vLLM CUDA graph mode to capture:

    - ``"NONE"``: disable compilation and CUDA graph capture.
    - ``"FULL_DECODE_ONLY"``: graph pure-decode batches; prefill / mixed
      batches run eager. Cheap (no inductor compile).
    - ``"FULL"`` (default): graph the whole forward, prefill included, attention
      captured too.
    """

    capture_sizes: list[int] | None = None
    """Explicit CUDA graph capture batch sizes. When ``None`` (default), sizes are
    auto-derived by ``_default_capture_sizes``. When set, these sizes are deduped and
    sorted (``FULL`` also drops sizes above the model's ``max_context_length``). When
    expert sequence parallelism is enabled, capture sizes that are not multiples of
    its degree are removed."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        expert_sequence_parallel_size: int,
        enable_sequence_parallel: bool,
        max_num_batched_tokens: int | None = None,
    ) -> CompilationConfig:
        """Build a vLLM ``CompilationConfig`` for the generator.

        When ``capture_sizes`` is set, those exact sizes are captured. Otherwise
        sizes come from ``_default_capture_sizes``, whose largest size is the cap.
        The cap is ``max_num_seqs`` for ``FULL_DECODE_ONLY`` (decode batch ==
        num_seqs). ``FULL`` also graphs prefill, whose per-step token count is
        bounded by ``max_num_batched_tokens`` (the configured value, else
        ``_DEFAULT_MAX_NUM_BATCHED_TOKENS``), so the cap extends to it -- otherwise
        prefill chunks larger than the cap fall back to eager. ``FULL`` also drops sizes
        above ``max_model_len``: vLLM captures a size with dummy requests that long, and
        one past ``max_model_len`` overruns its KV block table.

            FULL, max_model_len=1000, auto sizes                 -> [1, 2, ..., 992, 1000]
            FULL, max_model_len=1024, capture_sizes=[512, 2048]  -> [512]
            FULL_DECODE_ONLY (one-token dummies)                 -> unchanged

        ``expert_sequence_parallel_size`` is the TP-axis shard count used by the
        internally sequence-sharded MoE path. A value greater than one removes
        capture sizes that are not multiples of this count, preventing CUDA graph
        padding from producing a token count that cannot be evenly TP-sharded.

        ``enable_sequence_parallel`` is forwarded to vLLM's sequence parallelism
        pass. vLLM filters dense-SP CUDA graph sizes using its own TP size.

        All modes capture with ``mode=CompilationMode.NONE`` to avoid nesting
        vLLM's Inductor compile with TorchTitan local compile.
        """
        if self.mode == "NONE":
            return CompilationConfig(
                cudagraph_mode=CUDAGraphMode.NONE,
                mode=CompilationMode.NONE,
                pass_config=PassConfig(
                    enable_sp=enable_sequence_parallel,
                    sp_min_token_num=1 if enable_sequence_parallel else None,
                ),
            )
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        if expert_sequence_parallel_size <= 0:
            raise ValueError(
                "expert_sequence_parallel_size must be positive, got "
                f"{expert_sequence_parallel_size}"
            )
        if max_num_batched_tokens is not None:
            _max_cuda_graph_capture_size = max_num_batched_tokens
        else:
            _max_cuda_graph_capture_size = _DEFAULT_MAX_NUM_BATCHED_TOKENS
        cap = max_num_seqs
        if self.mode == "FULL":
            cap = min(max(cap, _max_cuda_graph_capture_size), max_model_len)
        if self.capture_sizes is not None:
            if not self.capture_sizes or any(s <= 0 for s in self.capture_sizes):
                raise ValueError(
                    "cuda_graph.capture_sizes must be a non-empty list of positive "
                    f"ints, got {self.capture_sizes}"
                )
            sizes = sorted(set(self.capture_sizes))
        else:
            sizes = _default_capture_sizes(cap, max_num_seqs)

        if self.mode == "FULL":
            removed_sizes = [size for size in sizes if size > max_model_len]
            if removed_sizes:
                logger.warning(
                    "CUDA graph capture sizes %s are removed because they exceed "
                    "max_model_len %d",
                    removed_sizes,
                    max_model_len,
                )
            sizes = [size for size in sizes if size <= max_model_len]
            if not sizes:
                raise ValueError(
                    f"No CUDA graph capture sizes are <= max_model_len {max_model_len}; "
                    f"got capture_sizes {self.capture_sizes}"
                )

        if expert_sequence_parallel_size > 1:
            removed_sizes = [
                size for size in sizes if size % expert_sequence_parallel_size != 0
            ]
            if removed_sizes:
                logger.warning(
                    "CUDA graph capture sizes %s are removed because they are not "
                    "multiples of expert_sequence_parallel_size %d",
                    removed_sizes,
                    expert_sequence_parallel_size,
                )
            sizes = [
                size for size in sizes if size % expert_sequence_parallel_size == 0
            ]
            if not sizes:
                raise ValueError(
                    "No CUDA graph capture sizes are divisible by "
                    f"expert_sequence_parallel_size {expert_sequence_parallel_size}"
                )

        return CompilationConfig(
            cudagraph_mode=self.mode,
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=sizes,
            pass_config=PassConfig(
                enable_sp=enable_sequence_parallel,
                sp_min_token_num=1 if enable_sequence_parallel else None,
            ),
        )


def _default_capture_sizes(cap: int, max_num_seqs: int) -> list[int]:
    """CUDA graph sizes up to ``cap``. vLLM pads each step up to the next size.

    1 2 4
    8 16 24 ... 248     +8      vLLM's spacing: 34 sequences pad to 40, not 64
    256 272 ... 1024    +16     goes on to max_num_seqs when it is above 1024
    1536 ... 4096       +512    prefill chunks: only FULL's cap reaches these sizes
    6144 ... cap        +2048
    max_num_seqs, cap   exact sizes: a full decode batch, the largest chunk
    """
    dense_limit = max(1024, max_num_seqs)
    sizes = {
        *[1, 2, 4],
        *range(8, 256, 8),
        *range(256, dense_limit + 1, 16),
        *range(1536, 4097, 512),
        *range(6144, cap + 1, 2048),
        # Linear-attention (GDN, KDA) decode graphs stop at max_num_seqs. Without this size,
        # at max_num_seqs=230 a batch of 225-230 pads to 232 and runs without a CUDA graph.
        max_num_seqs,
        cap,
    }
    return sorted(size for size in sizes if size <= cap)
