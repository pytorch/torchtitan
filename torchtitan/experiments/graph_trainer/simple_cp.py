# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""All-to-all context parallel lowering for GraphTrainer.

The model constructs its global attention mask, slices sequence inputs, and
calls ordinary attention. ``simple_cp_flex`` and ``simple_cp_sdpa`` mark calls;
``SimpleCPTransform`` replaces them during tracing with sequence-to-head
all-to-alls, attention, and a head-to-sequence all-to-all.
"""

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast, overload

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import AuxOutput, AuxRequest, BlockMask

from torchtitan.experiments.graph_trainer.make_fx_tracer import TraceTimeTransform
from torchtitan.experiments.graph_trainer.subgraph_regions import subgraph


def _launch_sequence_to_head(
    x: torch.Tensor, *, group_name: str, num_cp_ranks: int
) -> torch.Tensor:
    # [B, H_global, S_local, D] -> [B, H_local, S_global, D]
    batch, num_heads, local_seq, head_dim = x.shape
    if num_heads % num_cp_ranks:
        raise ValueError(
            f"All-to-all context parallel requires {num_heads=} divisible by "
            f"{num_cp_ranks=}"
        )

    # Split H_global into one H_local shard per destination CP rank. Every
    # destination receives different heads over this rank's same S_local shard.
    x = x.reshape(batch, num_cp_ranks, num_heads // num_cp_ranks, local_seq, head_dim)
    x = x.movedim(1, 0)
    # all_to_all_single requires a contiguous input buffer.
    x = x.contiguous()
    # The leading dimension contains one equal-sized chunk per destination rank.
    return torch.ops._c10d_functional.all_to_all_single(
        x, [1] * num_cp_ranks, [1] * num_cp_ranks, group_name
    )


def _wait_sequence_to_head(x: torch.Tensor) -> torch.Tensor:
    num_cp_ranks, batch, num_local_heads, local_seq, head_dim = x.shape
    x = torch.ops._c10d_functional.wait_tensor(x)
    # The received leading dimension now enumerates source sequence shards.
    # Joining [CP, S_local] reconstructs S_global for this rank's H_local.
    return x.permute(1, 2, 0, 3, 4).reshape(
        batch, num_local_heads, local_seq * num_cp_ranks, head_dim
    )


def _launch_head_to_sequence(
    x: torch.Tensor, *, group_name: str, num_cp_ranks: int
) -> torch.Tensor:
    # [B, H_local, S_global, D] -> [B, H_global, S_local, D]
    batch, num_local_heads, global_seq, head_dim = x.shape
    if global_seq % num_cp_ranks:
        raise ValueError(
            f"All-to-all context parallel requires {global_seq=} divisible by "
            f"{num_cp_ranks=}"
        )

    local_seq = global_seq // num_cp_ranks
    # Split S_global back into one S_local shard per destination CP rank. Every
    # destination receives the same sequence range from each source head rank.
    x = x.reshape(batch, num_local_heads, num_cp_ranks, local_seq, head_dim)
    x = x.movedim(2, 0)
    # all_to_all_single requires a contiguous input buffer.
    x = x.contiguous()
    # The leading dimension contains one equal-sized chunk per destination rank.
    return torch.ops._c10d_functional.all_to_all_single(
        x, [1] * num_cp_ranks, [1] * num_cp_ranks, group_name
    )


def _wait_head_to_sequence(x: torch.Tensor) -> torch.Tensor:
    num_cp_ranks, batch, num_local_heads, local_seq, head_dim = x.shape
    x = torch.ops._c10d_functional.wait_tensor(x)
    # The received leading dimension now enumerates source head shards.
    # Joining [CP, H_local] reconstructs H_global over this rank's S_local.
    return x.permute(1, 0, 2, 3, 4).reshape(
        batch, num_local_heads * num_cp_ranks, local_seq, head_dim
    )


# Project score modification onto rank r's local heads:
# f_local(b, h, q, k) = f_global(b, r * H_local + h, q, k).
def _global_to_local_score_mod(
    global_score_mod: Callable[..., Any] | None, global_head_start: int
) -> Callable[..., Any] | None:
    if global_score_mod is None:
        return None

    def local_score_mod(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        return global_score_mod(
            score, batch, head + global_head_start, query_idx, key_idx
        )

    return local_score_mod


# Project the global BlockMask onto rank r's local heads:
# M_local[:, h, ...] = M_global[:, r * H_local + h, ...].
def _global_to_local_block_mask(
    global_block_mask: BlockMask | None,
    *,
    num_cp_ranks: int,
    cp_rank: int,
    num_local_query_heads: int,
) -> BlockMask | None:
    if global_block_mask is None or global_block_mask.kv_num_blocks.shape[1] == 1:
        return global_block_mask

    num_global_query_heads = num_local_query_heads * num_cp_ranks
    if global_block_mask.kv_num_blocks.shape[1] != num_global_query_heads:
        raise ValueError(
            "Expected the BlockMask head dimension to be 1 or the global query "
            f"head count ({num_global_query_heads}), got "
            f"{global_block_mask.kv_num_blocks.shape[1]}"
        )

    global_head_start = cp_rank * num_local_query_heads
    global_head_end = global_head_start + num_local_query_heads
    local_full_kv_num_blocks = local_full_kv_indices = None
    if global_block_mask.full_kv_num_blocks is not None:
        assert global_block_mask.full_kv_indices is not None
        local_full_kv_num_blocks = global_block_mask.full_kv_num_blocks[
            :, global_head_start:global_head_end
        ]
        local_full_kv_indices = global_block_mask.full_kv_indices[
            :, global_head_start:global_head_end
        ]

    global_mask_mod = global_block_mask.mask_mod

    def local_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_idx: torch.Tensor,
        key_idx: torch.Tensor,
    ) -> torch.Tensor:
        return global_mask_mod(batch, head + global_head_start, query_idx, key_idx)

    # Sequence block indices remain global; only the head axis is sliced.
    return BlockMask.from_kv_blocks(
        global_block_mask.kv_num_blocks[:, global_head_start:global_head_end],
        global_block_mask.kv_indices[:, global_head_start:global_head_end],
        local_full_kv_num_blocks,
        local_full_kv_indices,
        BLOCK_SIZE=global_block_mask.BLOCK_SIZE,
        mask_mod=local_mask_mod,
        seq_lengths=global_block_mask.seq_lengths,
    )


# Project the global SDPA mask onto rank r's local heads:
# A_local[..., h, :, :] = A_global[..., r * H_local + h, :, :].
def _global_to_local_sdpa_mask(
    global_attn_mask: torch.Tensor | None,
    *,
    num_cp_ranks: int,
    cp_rank: int,
    num_local_query_heads: int,
) -> torch.Tensor | None:
    if global_attn_mask is None or global_attn_mask.ndim < 3:
        return global_attn_mask

    num_mask_heads = global_attn_mask.shape[-3]
    if num_mask_heads == 1:
        return global_attn_mask

    num_global_query_heads = num_local_query_heads * num_cp_ranks
    if num_mask_heads != num_global_query_heads:
        raise ValueError(
            "Expected the SDPA mask head dimension to be 1 or the global query "
            f"head count ({num_global_query_heads}), got {num_mask_heads}"
        )

    global_head_start = cp_rank * num_local_query_heads
    return global_attn_mask.narrow(-3, global_head_start, num_local_query_heads)


class _AllToAllContextParallel:
    def __init__(self, mesh: DeviceMesh) -> None:
        self.group_name: str = dist._get_process_group_name(mesh.get_group())
        self.num_cp_ranks: int = mesh.size()
        self.cp_rank: int = mesh.get_local_rank()

    @overload
    def _restore_stat(self, stat: None) -> None:
        ...

    @overload
    def _restore_stat(self, stat: torch.Tensor) -> torch.Tensor:
        ...

    def _local_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return _wait_head_to_sequence(
            _launch_head_to_sequence(
                x,
                group_name=self.group_name,
                num_cp_ranks=self.num_cp_ranks,
            )
        )

    def _restore_stat(self, stat: torch.Tensor | None) -> torch.Tensor | None:
        if stat is None:
            return None
        return self._local_sequence(stat.unsqueeze(-1)).squeeze(-1)

    def _global_sequence_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "All-to-all context parallel expects query, key, and value in "
                "[batch, heads, sequence, head_dim] layout"
            )
        if query.shape[-2] != key.shape[-2] or query.shape[-2] != value.shape[-2]:
            raise ValueError("All-to-all context parallel only supports self-attention")

        # Start all QKV transfers before the first wait so communication can
        # overlap packing on the compute stream.
        query = _launch_sequence_to_head(
            query, group_name=self.group_name, num_cp_ranks=self.num_cp_ranks
        )
        key = _launch_sequence_to_head(
            key, group_name=self.group_name, num_cp_ranks=self.num_cp_ranks
        )
        value = _launch_sequence_to_head(
            value, group_name=self.group_name, num_cp_ranks=self.num_cp_ranks
        )
        return (
            _wait_sequence_to_head(query),
            _wait_sequence_to_head(key),
            _wait_sequence_to_head(value),
        )

    def flex_attention(
        self,
        attention: Callable[..., Any],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable[..., Any] | None = None,
        block_mask: BlockMask | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        kernel_options: dict[str, Any] | None = None,
        *,
        return_aux: AuxRequest | None = None,
    ) -> Any:
        query, key, value = self._global_sequence_qkv(query, key, value)
        global_head_start = self.cp_rank * query.shape[1]
        result = attention(
            query,
            key,
            value,
            score_mod=_global_to_local_score_mod(
                global_score_mod=score_mod,
                global_head_start=global_head_start,
            ),
            block_mask=_global_to_local_block_mask(
                global_block_mask=block_mask,
                num_cp_ranks=self.num_cp_ranks,
                cp_rank=self.cp_rank,
                num_local_query_heads=query.shape[1],
            ),
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
            kernel_options=kernel_options,
            return_aux=return_aux,
        )

        if return_aux is not None:
            output, aux = cast(tuple[torch.Tensor, AuxOutput], result)
            return self._local_sequence(output), AuxOutput(
                lse=self._restore_stat(aux.lse),
                max_scores=self._restore_stat(aux.max_scores),
            )
        if return_lse:
            output, lse = cast(tuple[torch.Tensor, torch.Tensor], result)
            return self._local_sequence(output), self._restore_stat(lse)
        return self._local_sequence(cast(torch.Tensor, result))

    def sdpa(
        self,
        attention: Callable[..., Any],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        query, key, value = self._global_sequence_qkv(query, key, value)
        output = attention(
            query,
            key,
            value,
            attn_mask=_global_to_local_sdpa_mask(
                global_attn_mask=attn_mask,
                num_cp_ranks=self.num_cp_ranks,
                cp_rank=self.cp_rank,
                num_local_query_heads=query.shape[1],
            ),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return self._local_sequence(output)


# ContextVar is threadlocal
_active_simple_cp: ContextVar[_AllToAllContextParallel | None] = ContextVar(
    "graph_trainer_simple_cp", default=None
)


def _simple_cp(
    attention: Callable[..., Any], lowering: Callable[..., Any]
) -> Callable[..., Any]:
    @functools.wraps(attention)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        transform = _active_simple_cp.get()
        if transform is None:
            return attention(*args, **kwargs)
        with subgraph("simple_cp", role="context_parallel", preserve_order=True):
            return lowering(transform, attention, *args, **kwargs)

    return wrapped


def simple_cp_flex(attention: Callable[..., Any]) -> Callable[..., Any]:
    """Mark FlexAttention for simple CP lowering during GraphTrainer tracing."""

    return _simple_cp(attention, _AllToAllContextParallel.flex_attention)


def simple_cp_sdpa(attention: Callable[..., Any]) -> Callable[..., Any]:
    """Mark SDPA for simple CP lowering during GraphTrainer tracing."""

    return _simple_cp(attention, _AllToAllContextParallel.sdpa)


@dataclass(frozen=True)
class SimpleCPTransform(TraceTimeTransform):
    """Lower marked attention calls to all-to-all CP during tracing."""

    cp_mesh: DeviceMesh | None

    @contextmanager
    def activate(self) -> Iterator[None]:
        cp_mesh = self.cp_mesh
        if cp_mesh is None:
            yield
            return
        if cp_mesh.ndim != 1:
            raise ValueError(
                "SimpleCPTransform requires a 1D context-parallel mesh, "
                f"got {cp_mesh.ndim}D"
            )
        if cp_mesh.size() == 1:
            yield
            return

        token = _active_simple_cp.set(_AllToAllContextParallel(cp_mesh))
        try:
            yield
        finally:
            _active_simple_cp.reset(token)
