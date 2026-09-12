# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
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

from torchtitan.config.configs import ContextParallelLoadBalancerConfig
from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import AttentionMasksType


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
        config: Load-balancer configuration. A ``load_balancer_type`` of
            ``None`` selects contiguous sharding; ``headtail`` and ``ptrr``
            select their respective load-balancing strategies.
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
        ptrr_mask_key: When ``config.load_balancer_type`` is ``"ptrr"`` and
            the attention masks are a ``dict[str, BlockMask]``, selects which
            mask the ``PTRRLoadBalancer`` is built from. Ignored otherwise.
    """

    Config = ContextParallelLoadBalancerConfig

    def __init__(
        self,
        config: Config,
        *,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType] | None,
        cp_mesh: DeviceMesh,
        ptrr_mask_key: str | None = None,
    ) -> None:
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
            config.load_balancer_type,
            seq_len,
            input_dict.get("attention_masks"),
            ptrr_mask_key,
        )

    def _build_load_balancer_impl(
        self,
        load_balancer_type: str | None,
        seq_len: int,
        attention_masks: AttentionMasksType | None,
        ptrr_mask_key: str | None,
    ) -> _LoadBalancer | None:
        cp_world_size = self._cp_mesh.size(0)
        if load_balancer_type is None:
            return None
        if load_balancer_type == "headtail":
            return _HeadTailLoadBalancer(
                seq_len, cp_world_size, self._cp_mesh.device_type
            )

        assert load_balancer_type == "ptrr"
        # FlexInnerAttention uses _PTRRLoadBalancer, which is built from one
        # BlockMask. When attention_masks is a mapping, ptrr_mask_key selects
        # the mask that defines the shared partition. The resulting load
        # balancer shards the named inputs and every BlockMask leaf.
        if attention_masks is None:
            raise ValueError(
                "PTRRLoadBalancer requires attention_masks to be a BlockMask "
                "or Mapping[str, BlockMask], but got None."
            )
        if isinstance(attention_masks, Mapping):
            if ptrr_mask_key is None:
                raise ValueError(
                    "PTRRLoadBalancer received a Mapping[str, BlockMask] but no "
                    "mask key was specified. Set "
                    "--parallelism.context_parallel_ptrr_mask_key to one of: "
                    f"{sorted(attention_masks.keys())}"
                )
            if ptrr_mask_key not in attention_masks:
                raise ValueError(
                    f"context_parallel_ptrr_mask_key '{ptrr_mask_key}' is not a "
                    f"key in attention_masks. Available keys: "
                    f"{sorted(attention_masks.keys())}"
                )
            ptrr_mask = attention_masks[ptrr_mask_key]
        else:
            ptrr_mask = attention_masks
        if not isinstance(ptrr_mask, BlockMask):
            raise ValueError(
                "PTRRLoadBalancer requires the selected mask to be a BlockMask, "
                f"but got {type(ptrr_mask).__name__}."
            )
        return _PTRRLoadBalancer(ptrr_mask, cp_world_size)

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


def prepare_context_parallel_batch(
    input_dict: dict[str, Any],
    *,
    input_shardings: dict[str, SpmdType],
    cp_mesh: DeviceMesh,
    load_balancer_config: ContextParallelLoadBalancerConfig,
    ptrr_mask_key: str | None = None,
) -> tuple[dict[str, Any], ContextParallelLoadBalancer]:
    """Build one batch's CP partition and shard its named inputs."""
    load_balancer = load_balancer_config.build(
        input_dict=input_dict,
        input_shardings=input_shardings,
        cp_mesh=cp_mesh,
        ptrr_mask_key=ptrr_mask_key,
    )
    assert isinstance(load_balancer, ContextParallelLoadBalancer)
    return load_balancer.shard_inputs(input_dict), load_balancer
