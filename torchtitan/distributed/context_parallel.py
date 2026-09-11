# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel partitioning and load-balancing APIs."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import spmd_types as spmd
import torch
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types

__all__ = [
    "ContextParallelLoadBalancer",
    "HeadTailLoadBalancer",
    "PTRRLoadBalancer",
    "ContextParallelPartitioner",
]


def _cp_shard_dims(input_sharding: dict[str, SpmdType]) -> dict[str, int]:
    """Derive ``{name: seq_dim}`` for inputs whose CP mesh axis is a Shard.

    Inputs whose CP axis is Replicate/Partial (e.g. an image stream that is
    not sequence-sharded) are omitted and thus left untouched by CP.
    """
    dims: dict[str, int] = {}
    for name, layout in input_sharding.items():
        axis_type = _per_axis_types(layout).get(MeshAxisName.CP)
        if isinstance(axis_type, spmd.Shard):
            dims[name] = axis_type.dim
    return dims


class ContextParallelLoadBalancer(Configurable, ABC):
    """Build a PyTorch context-parallel load-balancing policy."""

    _load_balancer_impl: _LoadBalancer

    @abstractmethod
    def token_partition(self, num_tokens: int) -> list[list[tuple[int, int]]]:
        """Return each CP rank's global token ranges in local tensor order."""


class HeadTailLoadBalancer(ContextParallelLoadBalancer):
    """Balance CP tokens using PyTorch's head-tail strategy."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for head-tail context-parallel load balancing."""

    def __init__(
        self,
        config: Config,
        *,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType] | None,
        cp_mesh: DeviceMesh,
    ) -> None:
        shard_dims = (
            _cp_shard_dims(input_shardings)
            if input_shardings is not None
            else {"input": 0, "labels": 0, "positions": 0}
        )
        seq_lens: dict[str, int] = {}
        for name, seq_dim in shard_dims.items():
            value = input_dict.get(name)
            if isinstance(value, torch.Tensor):
                seq_lens[name] = value.size(seq_dim)
        if not seq_lens:
            raise ValueError(
                "HeadTailLoadBalancer requires at least one CP-sharded tensor."
            )
        if len(set(seq_lens.values())) != 1:
            raise ValueError(
                "HeadTailLoadBalancer requires all CP-sharded tensors to have "
                f"the same sequence length, but got {seq_lens}."
            )
        seq_len = next(iter(seq_lens.values()))
        self._world_size = cp_mesh.size(0)
        self._load_balancer_impl = _HeadTailLoadBalancer(
            seq_len, self._world_size, cp_mesh.device_type
        )

    def token_partition(self, num_tokens: int) -> list[list[tuple[int, int]]]:
        """Return the head-tail token partition."""
        world_size = self._world_size
        num_blocks = 2 * world_size
        if num_tokens % num_blocks:
            raise ValueError(
                "Head-tail context parallelism requires the token count "
                f"({num_tokens}) to be divisible by {num_blocks}."
            )
        block_size = num_tokens // num_blocks
        return [
            [
                (rank * block_size, (rank + 1) * block_size),
                (
                    (num_blocks - rank - 1) * block_size,
                    (num_blocks - rank) * block_size,
                ),
            ]
            for rank in range(world_size)
        ]

class PTRRLoadBalancer(ContextParallelLoadBalancer):
    """Balance context-parallel tokens with PTRR.

    The current implementation derives its partition from FlexAttention
    ``BlockMask`` metadata, but it can be extended to support variable-length
    attention metadata.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for PTRR context-parallel load balancing."""

        mask_key: str | None = None
        """Mask used to derive the partition when attention masks are a mapping."""

    def __init__(
        self,
        config: Config,
        *,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType] | None,
        cp_mesh: DeviceMesh,
    ) -> None:
        cp_world_size = cp_mesh.size(0)
        mask_key = config.mask_key
        attention_masks = input_dict.get("attention_masks")

        # The current implementation builds _PTRRLoadBalancer from one
        # BlockMask. When attention_masks is a mapping, mask_key selects the
        # mask that defines the shared partition. The resulting load balancer
        # shards the named inputs and every BlockMask leaf.
        if attention_masks is None:
            raise ValueError(
                "PTRRLoadBalancer requires attention_masks to be a BlockMask "
                "or Mapping[str, BlockMask], but got None."
            )
        if isinstance(attention_masks, Mapping):
            if mask_key is None:
                raise ValueError(
                    "PTRRLoadBalancer received a Mapping[str, BlockMask] but no "
                    "mask key was specified. Set PTRRLoadBalancer.Config(mask_key=...) "
                    "to one of: "
                    f"{sorted(attention_masks.keys())}"
                )
            if mask_key not in attention_masks:
                raise ValueError(
                    f"PTRR mask key '{mask_key}' is not a "
                    f"key in attention_masks. Available keys: "
                    f"{sorted(attention_masks.keys())}"
                )
            ptrr_mask = attention_masks[mask_key]
        else:
            ptrr_mask = attention_masks
        if not isinstance(ptrr_mask, BlockMask):
            raise ValueError(
                "PTRRLoadBalancer requires the selected mask to be a BlockMask, "
                f"but got {type(ptrr_mask).__name__}."
            )
        self._load_balancer_impl = _PTRRLoadBalancer(ptrr_mask, cp_world_size)

    def token_partition(self, num_tokens: int) -> list[list[tuple[int, int]]]:
        del num_tokens
        raise ValueError(
            f"{type(self).__name__} does not expose contiguous token fragments."
        )

class ContextParallelPartitioner:
    """Shard one batch across CP ranks using an optional load balancer.

    Without a load balancer, tokens are partitioned contiguously. When an
    attention backend shards metadata, it uses this partitioner to keep the
    metadata aligned with model inputs.

    Args:
        input_dict: Current model inputs used to build a data-dependent load
            balancer.
        input_shardings: Per-input SPMD layout; the CP sequence dimension for
            each input is derived via ``_cp_shard_dims``. Inputs whose CP mesh
            axis is replicated or partial are omitted and left untouched. When
            ``None``, defaults to sharding
            ``{"input": 0, "labels": 0, "positions": 0}``, the standard
            decoder inputs for callers without a per-input layout.
        cp_mesh: Device mesh for the context-parallel mesh axis.
        load_balancer_config: Optional load-balancer configuration. ``None``
            selects contiguous partitioning.
    """

    def __init__(
        self,
        *,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType] | None,
        cp_mesh: DeviceMesh,
        load_balancer_config: ContextParallelLoadBalancer.Config | None,
    ) -> None:
        self._cp_mesh = cp_mesh
        self._shard_dims = (
            _cp_shard_dims(input_shardings)
            if input_shardings is not None
            else {"input": 0, "labels": 0, "positions": 0}
        )
        self._load_balancer: ContextParallelLoadBalancer | None = None
        self._load_balancer_impl: _LoadBalancer | None = None
        if load_balancer_config is not None:
            self._load_balancer = load_balancer_config.build(
                input_dict=input_dict,
                input_shardings=input_shardings,
                cp_mesh=cp_mesh,
            )
            assert isinstance(self._load_balancer, ContextParallelLoadBalancer)
            self._load_balancer_impl = self._load_balancer._load_balancer_impl

    def shard_buffers(
        self,
        buffers: list[Any] | tuple[Any, ...],
        seq_dims: tuple[int, ...],
    ) -> tuple[Any, ...]:
        """Shard buffers with this batch's context-parallel partition.

        Callers select the buffers and specify the sequence dimension of each
        one. ``shard_inputs`` delegates named model inputs to this method;
        attention backends that shard metadata call it for metadata they own.
        Both paths therefore use the same per-batch CP partition.

        Args:
            buffers: Tensors or attention metadata to shard.
            seq_dims: Sequence dimension for each corresponding buffer.

        Returns:
            Sharded buffers in the same order as ``buffers``.
        """
        return tuple(
            _context_parallel_shard(
                mesh=self._cp_mesh,
                buffers=buffers,
                seq_dims=seq_dims,
                load_balancer=self._load_balancer_impl,
            )
        )

    @property
    def cp_mesh(self) -> DeviceMesh:
        """Device mesh whose CP axis owns this partition."""
        return self._cp_mesh

    def token_partition(self, num_tokens: int) -> list[list[tuple[int, int]]]:
        """Return each CP rank's global token ranges in local tensor order."""
        if self._load_balancer is not None:
            return self._load_balancer.token_partition(num_tokens)

        world_size = self._cp_mesh.size(0)
        if num_tokens % world_size:
            raise ValueError(
                f"Context parallelism requires the token count ({num_tokens}) "
                f"to be divisible by the CP degree ({world_size})."
            )
        block_size = num_tokens // world_size
        return [
            [(rank * block_size, (rank + 1) * block_size)]
            for rank in range(world_size)
        ]

    def shard_inputs(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Shard named tensors for context parallelism.

        Each tensor named in the resolved input-sharding map and present in
        ``input_dict`` is sharded along its declared sequence dimension using
        this batch's shared partition. The sharded tensors are written back to
        ``input_dict``.

        Attention metadata is left unchanged for the owning attention backend
        to shard separately. Position resolution (per-document vs. sequential)
        is handled upstream by the model's ``preprocess_inputs`` or the trainer.

        Args:
            input_dict: Model-forward inputs keyed by name, containing
                ``input``, ``labels``, and any extra keyword arguments. Tensor
                entries selected by the resolved input-sharding map, such as
                ``input``, ``labels``, and ``positions``, are sharded and
                written back.

        Returns:
            The same ``input_dict`` object, mutated in place with its named
            tensor entries sharded. When no named tensor is present to shard,
            it is returned unchanged.
        """
        shard_names = tuple(
            name
            for name in self._shard_dims
            if isinstance(input_dict.get(name), torch.Tensor)
        )
        if not shard_names:
            return input_dict
        input_tensors = cast(
            tuple[torch.Tensor, ...],
            tuple(input_dict[name] for name in shard_names),
        )
        seq_dims = tuple(self._shard_dims[name] for name in shard_names)
        sharded_inputs = cast(
            tuple[torch.Tensor, ...], self.shard_buffers(input_tensors, seq_dims)
        )
        for name, input_tensor in zip(shard_names, sharded_inputs):
            input_dict[name] = input_tensor
        return input_dict
