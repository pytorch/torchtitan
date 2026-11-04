# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The generator's attention inside SGLang: TorchTitan's attention layers hand q/k/v to
SGLang's ``RadixAttention``, which owns the paged KV cache, and SGLang's FlashAttention
backend runs the trainer's varlen kernel on it.

Legend: T = tokens of the step, H = query heads, K = query/key head dim, V = value head
dim, P = KV pages, S = page size, G = KV heads (all rank-local under TP).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_parallel
from torch.nn.attention import current_flash_attention_impl

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import InnerAttention
from torchtitan.rl.generator.paged_attention import paged_varlen_attention_out
from torchtitan.tools.utils import activate_cuda_flash_attention_impl

# Name SGLang's attention backend registry selects ``TorchTitanVarlenAttnBackend`` by.
TORCHTITAN_VARLEN_ATTENTION_BACKEND = "torchtitan_varlen"


class TorchTitanVarlenAttnBackend(FlashAttentionBackend):
    """SGLang's FlashAttention backend with the trainer's kernel.

    SGLang's backend builds the paged-KV metadata (page table, sequence lengths, CUDA
    graph buffers) and writes the KV cache; every kernel call goes through the two
    functions ``_init_flash_attention_functions`` sets, which this subclass points at
    PyTorch's varlen FlashAttention, the kernel the trainer's ``VarlenInnerAttention``
    runs. It supports causal attention over the paged KV cache only; any other call
    (no-cache prefill, cascade, soft-capping, FP8 KV) raises.

    Args:
        model_runner: SGLang's model runner.
    """

    def __init__(self, model_runner) -> None:
        activate_cuda_flash_attention_impl()
        fa_impl = current_flash_attention_impl()
        if fa_impl not in ("FA3", "FA4"):
            raise ValueError(
                f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND} needs FA3 or FA4 (SM90+), "
                f"got {fa_impl}"
            )
        # The FA version selects SGLang's metadata layout for that kernel.
        super().__init__(model_runner, fa_impl_ver=4 if fa_impl == "FA4" else 3)

    def _init_flash_attention_functions(self, device_capability) -> None:
        # TODO(https://github.com/sgl-project/sglang/issues/48475): overrides a private
        # hook of SGLang's FlashAttentionBackend; SGLang has no public way to plug a
        # kernel into its FA metadata.
        self._get_scheduler_metadata = None
        self._get_fa_runtime_policy = None
        self.flash_attn_varlen_func = _no_cache_attention_unsupported
        self.flash_attn_with_kvcache = self._paged_attention

    def _paged_attention(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        return_softmax_lse: bool = False,
        out: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        num_splits: int = 0,
        max_seqlen_k: int | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """SGLang's ``flash_attn_with_kvcache`` call, run as the trainer's varlen kernel.

        ``q`` is ``(T, H, K)``, ``k_cache`` / ``v_cache`` are ``(P, S, G, K)`` /
        ``(P, S, G, V)``. ``cu_seqlens_k_new`` and ``num_splits`` are ignored: SGLang
        stored the new KV already, and ``paged_varlen_attention_out`` picks the splits.
        """
        del cu_seqlens_k_new, num_splits
        unsupported = {key for key, value in kwargs.items() if value is not None}
        if unsupported:
            raise NotImplementedError(
                f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND} does not support "
                f"{sorted(unsupported)}"
            )
        if not causal:
            raise NotImplementedError(
                f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND} supports causal attention only"
            )
        if softcap != 0.0:
            raise NotImplementedError(
                f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND} does not support soft-capping"
            )
        num_tokens, num_heads, _ = q.shape
        if out is None:
            out = q.new_empty(num_tokens, num_heads, v_cache.shape[-1])
        # Causal attention is the window (left, 0) in the varlen API; SGLang passes a
        # right edge of -1 for unbounded, which causal=True caps at the diagonal.
        left, _ = window_size
        return paged_varlen_attention_out(
            out,
            q,
            k_cache,
            v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            # An upper bound is enough: the kernel reads each sequence's length from
            # seqused_k, and a fixed bound keeps CUDA-graph replays valid.
            max_seqlen_k=self.max_context_len if max_seqlen_k is None else max_seqlen_k,
            seqused_k=cache_seqlens,
            block_table=page_table,
            scale=softmax_scale,
            window_size=(left, 0),
            enable_gqa=num_heads != k_cache.shape[2],
            return_lse=return_softmax_lse,
        )


def _no_cache_attention_unsupported(**kwargs) -> torch.Tensor:
    raise NotImplementedError(
        f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND} runs attention over the paged KV cache "
        "only; SGLang's no-cache prefill path is not supported"
    )


class _SGLangRadixAttention(RadixAttention, Module):
    """SGLang's ``RadixAttention`` as a TorchTitan ``Module``; it holds no state."""


class SGLangInnerAttention(InnerAttention):
    """Adapter from TorchTitan's attention to SGLang's ``RadixAttention``.

    ``RadixAttention`` stores the step's K/V into SGLang's paged KV cache and runs the
    attention backend over it, reading the step's batch metadata from the forward batch
    ``SGLangModelWrapper.forward`` publishes. Used as ``inner_attention`` of a full
    attention layer; the q/k/v it receives are rank-local under TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        layer_id: int
        """Index of the layer's KV cache in SGLang's KV pool."""
        num_heads: int
        num_kv_heads: int
        head_dim: int
        scale: float | None = None

        @property
        def is_full_attention(self) -> bool:
            # The engine replaces full softmax attention only.
            return True

    def __init__(self, config: Config) -> None:
        super().__init__()
        tp_size = get_parallel().tp_size
        for name in ("num_heads", "num_kv_heads"):
            if getattr(config, name) % tp_size != 0:
                raise ValueError(
                    f"{name} ({getattr(config, name)}) must be divisible by "
                    f"tensor_parallel_degree ({tp_size})"
                )
        self.head_dim = config.head_dim
        self.scale = (
            config.scale if config.scale is not None else config.head_dim**-0.5
        )
        self.radix_attention = _SGLangRadixAttention(
            num_heads=config.num_heads // tp_size,
            head_dim=config.head_dim,
            scaling=self.scale,
            num_kv_heads=config.num_kv_heads // tp_size,
            layer_id=config.layer_id,
            prefix=f"model.layers.{config.layer_id}.attention.inner_attention",
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run SGLang paged attention on rank-local tensors.

        Returns:
            ``(T, H, V)``.
        """
        if attention_metadata is not None:
            raise ValueError(
                "SGLangInnerAttention does not take attention_metadata; SGLang "
                "manages causal masking and the KV cache."
            )
        if scale is not None and scale != self.scale:
            raise ValueError(
                f"Model attention scale {scale} != SGLang scale {self.scale}"
            )
        # SGLang runs GQA from the layer's head counts; enable_gqa needs nothing more.
        del enable_gqa
        unsupported = sorted(
            name for name, value in kwargs.items() if value is not None
        )
        if unsupported:
            raise ValueError(f"SGLangInnerAttention does not support {unsupported}")
        num_tokens = q_THK.shape[0]
        # RadixAttention returns (T, H * V).
        out = self.radix_attention(q_THK, k_THK, v_THV, get_forward_batch())
        return out.view(num_tokens, -1, self.head_dim)
