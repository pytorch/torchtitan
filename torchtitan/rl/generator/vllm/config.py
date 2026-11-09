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

from vllm import EngineArgs
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode, PassConfig
from vllm.usage.usage_lib import UsageContext

from torchtitan.tools.utils import has_cuda_capability

logger = logging.getLogger(__name__)


def resolve_max_num_batched_tokens(
    max_num_batched_tokens: int | None,
    *,
    max_num_seqs: int,
    max_model_len: int,
) -> int:
    """Return ``max_num_batched_tokens``, or what ``vllm serve`` picks for this GPU when it is None.

    ``vllm serve`` reads vLLM's table (``EngineArgs.get_batch_defaults``), keyed on total device
    memory plus an A100 name check, and caps the value at ``max_num_seqs * max_model_len``. The
    table as of vLLM 0.32:
        >= 160 GiB (B200, GB300)     -> 16384
        >= 70 GiB, not A100 (H100)   -> 8192
        other GPUs (A100, A10G)      -> 2048
    GPUs below compute capability 9.0 (FA2) and ROCm keep 2048.

    ``LLMEngine.from_engine_args`` uses ``UsageContext.ENGINE_CONTEXT``, which has no entry in
    that table, so vLLM itself would pick 2048 on every GPU. The FULL CUDA graph sizes are built
    before the engine, so the engine and the FULL CUDA graph cap both take the value returned here.
    """
    if max_num_batched_tokens is not None:
        return max_num_batched_tokens
    if has_cuda_capability(9, 0):
        # world_size scales the table only on vLLM's CPU backend, which the generator never uses.
        default_max_num_batched_tokens, _ = EngineArgs.get_batch_defaults(world_size=1)
        max_num_batched_tokens = default_max_num_batched_tokens[
            UsageContext.OPENAI_API_SERVER
        ]
    else:
        # TODO(https://github.com/Dao-AILab/flash-attention/issues/3117): drop once fixed. FA2 writes
        # out of bounds past 64 tokens per profile request, and max_num_seqs_per_generator_dp_rank
        # floors max_num_seqs for a 2048 budget. vLLM's table gives an A800 80 GB 8192.
        max_num_batched_tokens = 2048
    max_num_batched_tokens = min(max_num_batched_tokens, max_num_seqs * max_model_len)
    logger.info(
        "max_num_batched_tokens is unset: using %d for this GPU", max_num_batched_tokens
    )
    return max_num_batched_tokens


@dataclass(kw_only=True, slots=True)
class VLLMCudaGraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    Local compile regions come from the model config's ``local_compile.regions``.
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
      captured too. vLLM switches to ``FULL_DECODE_ONLY``, with a warning, when an
      attention backend graphs only uniform decode batches (the MLA backend does).
    """

    capture_sizes: list[int] | None = None
    """Explicit CUDA graph capture batch sizes. When ``None`` (default), sizes are
    auto-derived by ``_default_capture_sizes``. When set, these sizes are deduped and
    sorted (``FULL`` rejects sizes no step reaches: above ``max_num_seqs`` x the
    model's ``max_context_length``). When expert sequence parallelism is enabled,
    capture sizes that are not multiples of its degree are removed."""

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
        max_num_batched_tokens: int,
        expert_sequence_parallel_size: int,
        enable_sequence_parallel: bool,
    ) -> CompilationConfig:
        """Build a vLLM ``CompilationConfig`` for the generator.

        When ``capture_sizes`` is set, those exact sizes are captured. Otherwise
        sizes come from ``_default_capture_sizes``, whose largest size is the cap.
        The cap is ``max_num_seqs`` for ``FULL_DECODE_ONLY`` (decode batch ==
        num_seqs). ``FULL`` also graphs prefill, whose per-step token count is
        bounded by ``max_num_batched_tokens``, so the cap extends to it -- otherwise
        prefill chunks larger than the cap fall back to eager. A step holds at most
        ``max_num_seqs`` requests of at most ``max_model_len`` tokens, so ``FULL`` also
        stops at ``max_num_seqs * max_model_len``. ``TorchTitanGPUModelRunner._dummy_run``
        keeps each capture request within ``max_model_len``.

        With ``max_num_batched_tokens=2048``:

            FULL, max_num_seqs=8, max_model_len=1000, auto sizes          -> [1, 2, ..., 1536, 2048]
            FULL, max_num_seqs=1, max_model_len=1000, auto sizes          -> [1, 2, ..., 992, 1000]
            FULL, max_num_seqs=1, max_model_len=1000, capture_sizes=[2048] -> ValueError
            FULL_DECODE_ONLY                                              -> unchanged

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
        cap = max_num_seqs
        max_step_tokens = max_num_seqs * max_model_len
        if self.mode == "FULL":
            cap = min(max(cap, max_num_batched_tokens), max_step_tokens)
        if self.capture_sizes is not None:
            if not self.capture_sizes or any(s <= 0 for s in self.capture_sizes):
                raise ValueError(
                    "cuda_graph.capture_sizes must be a non-empty list of positive "
                    f"ints, got {self.capture_sizes}"
                )
            if self.mode == "FULL" and max(self.capture_sizes) > max_step_tokens:
                raise ValueError(
                    f"cuda_graph.capture_sizes {self.capture_sizes} has sizes above "
                    f"max_num_seqs x max_model_len = {max_step_tokens}, which no step reaches"
                )
            sizes = sorted(set(self.capture_sizes))
        else:
            sizes = _default_capture_sizes(cap, max_num_seqs)

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

    Up to 1024 this copies vLLM's default ladder (``VllmConfig._set_cudagraph_sizes`` in
    ``vllm/config/vllm.py``). That code runs inside the engine and takes either our list
    or its own, so we can't call it and still add the prefill sizes.

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
        # FULL decode graphs (GDN/KDA and FA4 decode kernels) stop at max_num_seqs. Without this
        # size, at max_num_seqs=230 a batch of 225-230 pads to 232 and replays the mixed graph.
        max_num_seqs,
        cap,
    }
    return sorted(size for size in sizes if size <= cap)
