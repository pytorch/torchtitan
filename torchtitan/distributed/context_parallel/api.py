# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast, ClassVar

import spmd_types as spmd
import torch
import tyro
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configs import ContextParallelLoadBalancerConfig
from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types


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


class ContextParallelLoadBalancer(Configurable):
    """Build and apply one batch's shared context-parallel partition.

    The base implementation shards contiguous sequence fragments. Subclasses
    provide alternative per-batch partitions.

    A new instance is built for every batch so data-dependent strategies such
    as PTRR can derive their partition from the current attention mask. Named
    tensors and backend-owned attention metadata use this same instance so all
    buffers follow one CP mesh and load-balancer partition.

    ``input_shardings`` determines the sequence dimension for each named
    tensor via ``_cp_shard_dims``. Inputs whose CP mesh axis is replicated or
    partial are omitted and left untouched. When no layout is provided, the
    standard decoder inputs ``input``, ``labels``, and ``positions`` are
    sharded along dimension 0.

    Args:
        config: Load-balancer configuration.
        input_dict: Model-forward inputs keyed by name, containing ``input``,
            ``labels``, and any extra keyword arguments. Tensor entries named
            in the resolved input-sharding map, such as ``input``, ``labels``,
            and ``positions``, are sharded by ``shard_inputs`` and written back.
            The ``attention_masks`` entry, if present, is used to construct the
            PTRR load balancer. It is not modified here and is later handled by
            the attention backend as required.
        input_shardings: Per-input SPMD layout; the CP sequence dimension for
            each input is derived via ``_cp_shard_dims``. Inputs whose CP mesh
            axis is replicated or partial are omitted and left untouched. When
            ``None``, defaults to sharding
            ``{"input": 0, "labels": 0, "positions": 0}``, the standard
            decoder inputs for callers without a per-input layout.
        cp_mesh: Device mesh for the context-parallel mesh axis.
    """

    Config: ClassVar[
        type[ContextParallelLoadBalancerConfig]
    ] = ContextParallelLoadBalancerConfig

    def __init__(
        self,
        config: Config,
        *,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType] | None,
        cp_mesh: DeviceMesh,
    ) -> None:
        self.config = config
        self._cp_mesh = cp_mesh
        self._shard_dims = (
            _cp_shard_dims(input_shardings)
            if input_shardings is not None
            else {"input": 0, "labels": 0, "positions": 0}
        )
        self._shard_names = tuple(
            name
            for name in self._shard_dims
            if isinstance(input_dict.get(name), torch.Tensor)
        )
        self._load_balancer_impl: _LoadBalancer | None = None
        if not self._shard_names:
            return
        first_name = self._shard_names[0]
        seq_len = cast(torch.Tensor, input_dict[first_name]).size(
            self._shard_dims[first_name]
        )
        self._load_balancer_impl = self._build_load_balancer_impl(
            input_dict=input_dict,
            seq_len=seq_len,
        )

    @property
    def cp_mesh(self) -> DeviceMesh:
        """Device mesh whose CP axis owns this partition."""
        return self._cp_mesh

    def _build_load_balancer_impl(
        self,
        *,
        input_dict: dict[str, Any],
        seq_len: int,
    ) -> _LoadBalancer | None:
        del input_dict, seq_len
        return None

    def shard(
        self,
        buffers: list[Any] | tuple[Any, ...],
        seq_dims: tuple[int, ...],
    ) -> tuple[Any, ...]:
        """Shard buffers with this batch's context-parallel partition.

        Callers select the buffers and specify the sequence dimension of each
        one. ``shard_inputs`` delegates named model inputs to this method;
        attention backends call it directly for metadata they own. Both paths
        therefore use the same CP mesh and per-batch load-balancer partition.

        Args:
            buffers: Tensors or attention metadata to shard.
            seq_dims: Sequence dimension for each corresponding buffer.

        Returns:
            Sharded buffers in the same order as ``buffers``.
        """
        return cast(
            tuple[Any, ...],
            _context_parallel_shard(
                mesh=self._cp_mesh,
                buffers=buffers,
                seq_dims=seq_dims,
                load_balancer=self._load_balancer_impl,
            ),
        )

    def shard_inputs(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Shard named tensors for context parallelism.

        Each tensor named in the resolved input-sharding map and present in
        ``input_dict`` is sharded along its declared sequence dimension using
        this batch's shared load balancer. The sharded tensors are written back
        to ``input_dict``.

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
        if not self._shard_names:
            return input_dict
        buffers = cast(
            tuple[torch.Tensor, ...],
            tuple(input_dict[name] for name in self._shard_names),
        )
        seq_dims = tuple(self._shard_dims[name] for name in self._shard_names)
        sharded_buffers = cast(tuple[torch.Tensor, ...], self.shard(buffers, seq_dims))
        for name, buffer in zip(self._shard_names, sharded_buffers):
            input_dict[name] = buffer
        return input_dict


class HeadTailLoadBalancer(ContextParallelLoadBalancer):
    """Shard tokens with PyTorch's head-tail load balancer."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancerConfig):
        """Configuration for head-tail context-parallel load balancing."""

    def _build_load_balancer_impl(
        self,
        *,
        input_dict: dict[str, Any],
        seq_len: int,
    ) -> _LoadBalancer:
        del input_dict
        return _HeadTailLoadBalancer(
            seq_len, self._cp_mesh.size(0), self._cp_mesh.device_type
        )


class PTRRLoadBalancer(ContextParallelLoadBalancer):
    """Shard tokens using a partition derived from a FlexAttention mask."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancerConfig):
        """Configuration for PTRR context-parallel load balancing."""

        mask_key: Annotated[str | None, tyro.conf.Suppress] = None
        """Mask used to derive the partition when attention masks are a mapping."""

    def _build_load_balancer_impl(
        self,
        *,
        input_dict: dict[str, Any],
        seq_len: int,
    ) -> _LoadBalancer:
        del seq_len
        cp_world_size = self._cp_mesh.size(0)
        mask_key = cast(PTRRLoadBalancer.Config, self.config).mask_key
        attention_masks = input_dict.get("attention_masks")

        # FlexInnerAttention uses _PTRRLoadBalancer, which is built from one
        # BlockMask. When attention_masks is a mapping, mask_key selects
        # the mask that defines the shared partition. The resulting load
        # balancer shards the named inputs and every BlockMask leaf.
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
        return _PTRRLoadBalancer(ptrr_mask, cp_world_size)
