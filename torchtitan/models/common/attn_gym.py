# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym kernels with placeholders when the package is missing."""

# TODO: Revisit Attention Gym imports and dependency handling to remove this
# placeholder workaround.

from importlib import import_module
from types import ModuleType
from typing import Any, NoReturn, TYPE_CHECKING

if TYPE_CHECKING:
    from attn_gym.linear import (
        chunk_gdn,
        KernelOptions,
        paged_chunk_gdn,
        recurrent_gdn,
        recurrent_gdn_decode,
    )
    from attn_gym.linear._delta_rule.gate import _FusedGate, gate_transform
    from attn_gym.linear.context_parallel import (
        _ContextParallelChunk,
        context_parallel_conv_history,
        ContextParallelRouting,
    )
    from attn_gym.linear.gdn.context_parallel import context_parallel_gdn
    from attn_gym.linear.gdn.impl.cudnn import ChunkGdnCudnnPacked
    from attn_gym.linear.gdn.ops import _ChunkGDN
    from attn_gym.linear.kda import (
        bound_gate,
        chunk_kda,
        paged_chunk_kda,
        recurrent_kda_decode,
    )
    from attn_gym.linear.kda.context_parallel import context_parallel_kda
    from attn_gym.linear.kda.fwd.triton.l2norm_fwd import _L2Norm, l2norm
    from attn_gym.linear.kda.impl.cudnn import ChunkKdaCudnn
    from attn_gym.linear.kda.impl.fused import _ChunkKDA
    from attn_gym.linear.kda.masking import _MaskRows
    from attn_gym.linear.short_conv import (
        causal_conv1d,
        causal_conv1d_decode,
        paged_causal_conv1d,
    )
    from attn_gym.linear.short_conv.cute import _ConfiguredShortConv, _ShortConv
    from attn_gym.sparse import lightning_indexer
    from attn_gym.sparse.gather_attn import gather_attn

__all__ = [
    "_ChunkGDN",
    "_ChunkKDA",
    "_ConfiguredShortConv",
    "_ContextParallelChunk",
    "_FusedGate",
    "_L2Norm",
    "_MaskRows",
    "_ShortConv",
    "bound_gate",
    "causal_conv1d",
    "causal_conv1d_decode",
    "ChunkGdnCudnnPacked",
    "ChunkKdaCudnn",
    "chunk_gdn",
    "chunk_kda",
    "context_parallel_conv_history",
    "context_parallel_gdn",
    "context_parallel_kda",
    "ContextParallelRouting",
    "gate_transform",
    "gather_attn",
    "KernelOptions",
    "l2norm",
    "lightning_indexer",
    "paged_causal_conv1d",
    "paged_chunk_gdn",
    "paged_chunk_kda",
    "recurrent_gdn",
    "recurrent_gdn_decode",
    "recurrent_kda_decode",
    "require_attn_gym",
]

_EXPORTS = {
    "_ChunkGDN": "attn_gym.linear.gdn.ops",
    "_ChunkKDA": "attn_gym.linear.kda.impl.fused",
    "_ConfiguredShortConv": "attn_gym.linear.short_conv.cute",
    "_ContextParallelChunk": "attn_gym.linear.context_parallel",
    "_FusedGate": "attn_gym.linear._delta_rule.gate",
    "_L2Norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "_MaskRows": "attn_gym.linear.kda.masking",
    "_ShortConv": "attn_gym.linear.short_conv.cute",
    "bound_gate": "attn_gym.linear.kda",
    "causal_conv1d": "attn_gym.linear.short_conv",
    "causal_conv1d_decode": "attn_gym.linear.short_conv",
    "ChunkGdnCudnnPacked": "attn_gym.linear.gdn.impl.cudnn",
    "ChunkKdaCudnn": "attn_gym.linear.kda.impl.cudnn",
    "chunk_gdn": "attn_gym.linear",
    "chunk_kda": "attn_gym.linear.kda",
    "context_parallel_conv_history": "attn_gym.linear.context_parallel",
    "context_parallel_gdn": "attn_gym.linear.gdn.context_parallel",
    "context_parallel_kda": "attn_gym.linear.kda.context_parallel",
    "ContextParallelRouting": "attn_gym.linear.context_parallel",
    "gate_transform": "attn_gym.linear._delta_rule.gate",
    "gather_attn": "attn_gym.sparse.gather_attn",
    "KernelOptions": "attn_gym.linear",
    "l2norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "lightning_indexer": "attn_gym.sparse",
    "paged_causal_conv1d": "attn_gym.linear.short_conv",
    "paged_chunk_gdn": "attn_gym.linear",
    "paged_chunk_kda": "attn_gym.linear.kda",
    "recurrent_gdn": "attn_gym.linear",
    "recurrent_gdn_decode": "attn_gym.linear",
    "recurrent_kda_decode": "attn_gym.linear.kda",
}


def _raise_missing(*args: Any, **kwargs: Any) -> NoReturn:
    raise ModuleNotFoundError(
        "Attention Gym kernels require attn_gym. "
        "Install it with: python -m pip install 'attn-gym[linear,cudnn]==0.0.16'.",
        name="attn_gym",
    )


class _MissingKernel:
    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        _raise_missing()

    apply = staticmethod(_raise_missing)
    from_fragments = staticmethod(_raise_missing)


def _load_module(name: str) -> ModuleType | None:
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != "attn_gym":
            raise
        return None


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _load_module(_EXPORTS[name])
    value = _MissingKernel if module is None else getattr(module, name)
    globals()[name] = value
    return value


def require_attn_gym() -> None:
    if _load_module("attn_gym") is None:
        _raise_missing()
