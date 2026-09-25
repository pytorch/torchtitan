# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel partitioning and load-balancing APIs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
from spmd_types import SpmdType

# TODO(acisseJZhong): Move these implementations into TorchTitan to avoid
# depending on private PyTorch context-parallel APIs.
from torch.distributed.tensor.experimental._attention import (
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import (
    _per_axis_types,
    current_spmd_mesh,
    spmd_mesh_group,
)

if TYPE_CHECKING:
    from torchtitan.models.common.attention import (
        AttentionMasksType,
        FlexAttentionMetadata,
    )

__all__ = [
    "ContextParallelLoadBalancer",
    "HeadTailCPLoadBalancer",
    "PTRRFlexAttentionCPLoadBalancer",
    "get_cp_input_seq_len",
    "shard_tensors",
]


class ContextParallelLoadBalancer(Configurable, ABC):
    """Generate a context-parallel token permutation for one batch."""

    @abstractmethod
    def generate_permutation(self) -> torch.Tensor:
        """Generate the global token permutation for this batch."""
        raise NotImplementedError


class HeadTailCPLoadBalancer(ContextParallelLoadBalancer):
    """Balance CP tokens using PyTorch's head-tail strategy."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for head-tail context-parallel load balancing."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        attention_metadata: AttentionMasksType | None,
    ) -> None:
        del config, attention_metadata
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        spmd_mesh = current_spmd_mesh()
        if cp_group is None or spmd_mesh is None:
            raise RuntimeError(
                "CP load balancing requires an active multi-rank CP mesh axis."
            )
        self.seq_len = seq_len
        self.cp_size = cp_group.size()
        self.device = spmd_mesh.device_type

    def generate_permutation(self) -> torch.Tensor:
        """Generate a head-tail token permutation."""
        permutation = _HeadTailLoadBalancer(
            self.seq_len, self.cp_size, self.device
        )._generate_indices()
        return permutation


# TODO(acisseJZhong): Maintain an explicit mapping from each CP load balancer
# to the CP attention backends it supports, and validate configured pairs.
class PTRRFlexAttentionCPLoadBalancer(ContextParallelLoadBalancer):
    """Balance FlexAttention tokens with PTRR.

    The current cost model assumes a K/V all-gather CP backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for PTRR context-parallel load balancing."""

        mask_key: str | None = None
        """Mask used to derive the partition when context metadata is a mapping."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        attention_metadata: FlexAttentionMetadata,
    ) -> None:
        del seq_len
        mask_key = config.mask_key

        if attention_metadata is None:
            raise ValueError(
                "PTRR load balancing requires context metadata to be a BlockMask "
                "or Mapping[str, BlockMask], but got None."
            )
        if isinstance(attention_metadata, Mapping):
            if mask_key is None:
                raise ValueError(
                    "PTRR load balancing received a Mapping[str, BlockMask] but no "
                    "mask key was specified. Set "
                    "PTRRFlexAttentionCPLoadBalancer.Config(mask_key=...) "
                    "to one of: "
                    f"{sorted(attention_metadata.keys())}"
                )
            if mask_key not in attention_metadata:
                raise ValueError(
                    f"PTRR mask key '{mask_key}' is not a key in context metadata. "
                    f"Available keys: {sorted(attention_metadata.keys())}"
                )
            block_mask = attention_metadata[mask_key]
        else:
            block_mask = attention_metadata
        if not isinstance(block_mask, BlockMask):
            raise ValueError(
                "PTRR load balancing requires the selected metadata to be a "
                f"BlockMask, but got {type(block_mask).__name__}."
            )
        self.block_mask = block_mask
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "CP load balancing requires an active multi-rank CP mesh axis."
            )
        self.cp_size = cp_group.size()

    def generate_permutation(self) -> torch.Tensor:
        """Generate a PTRR token permutation from the selected BlockMask."""
        permutation = _PTRRLoadBalancer(
            self.block_mask, self.cp_size
        )._generate_indices()
        return permutation


def _cp_shard_dims(input_shardings: dict[str, SpmdType]) -> dict[str, int]:
    """Derive ``{name: seq_dim}`` for inputs whose CP mesh axis is a Shard.

    Inputs whose CP axis is Replicate/Partial (e.g. an image stream that is
    not sequence-sharded) are omitted and thus left untouched by CP.
    """
    dims: dict[str, int] = {}
    for name, layout in input_shardings.items():
        axis_type = _per_axis_types(layout).get(MeshAxisName.CP)
        if isinstance(axis_type, spmd.Shard):
            dims[name] = axis_type.dim
    return dims


def get_cp_input_seq_len(
    input_dict: Mapping[str, Any],
    *,
    input_shardings: dict[str, SpmdType],
) -> int:
    """Return the pre-sharding sequence length shared by the CP inputs."""
    seq_lens: dict[str, int] = {}
    for name, seq_dim in _cp_shard_dims(input_shardings).items():
        value = input_dict.get(name)
        if isinstance(value, torch.Tensor):
            seq_lens[name] = value.size(seq_dim)

    if not seq_lens:
        raise ValueError("CP requires at least one declared sequence tensor.")
    if len(set(seq_lens.values())) != 1:
        raise ValueError(
            "All CP-sharded tensors must have the same sequence length, "
            f"but got {seq_lens}."
        )
    return next(iter(seq_lens.values()))


def _permute_tensor(
    tensor: torch.Tensor,
    *,
    seq_dim: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Apply a CP token permutation using PyTorch CP indexing semantics."""
    if permutation.ndim != 2:
        raise ValueError(
            "CP permutation must have shape [1, seq_len] or [batch, seq_len], "
            f"but got {tuple(permutation.shape)}."
        )
    if permutation.shape[1] != tensor.shape[seq_dim]:
        raise ValueError(
            f"CP permutation length ({permutation.shape[1]}) must match tensor "
            f"sequence length ({tensor.shape[seq_dim]})."
        )
    permutation_batch_size = permutation.shape[0]
    tensor_batch_size = tensor.shape[0] if seq_dim > 0 else 1
    if permutation_batch_size not in (1, tensor_batch_size):
        raise ValueError(
            "CP permutation batch size must be 1 or match the tensor batch "
            f"size, but got {permutation_batch_size} and {tensor_batch_size}."
        )

    if seq_dim == 0:
        # tensor has shape [seq_len] or [seq_len, ...]
        # Just use the first (and only) batch of indices
        tensor = torch.index_select(tensor, dim=0, index=permutation[0])
    else:
        indices = permutation
        if permutation_batch_size == 1:
            indices = indices.expand(tensor_batch_size, -1)

        # permutation has shape [B, seq_len] where:
        #   - dim 0 corresponds to tensor dim 0 (batch)
        #   - dim 1 corresponds to tensor dim seq_dim
        # Need to insert dimensions for all dims between 0 and seq_dim,
        # and all dims after seq_dim.

        # Insert dimensions between batch (dim 0) and seq_dim
        for dim in range(1, seq_dim):
            indices = indices.unsqueeze(dim)

        # Insert dimensions after seq_dim
        for _ in range(seq_dim + 1, tensor.ndim):
            indices = indices.unsqueeze(-1)

        # Expand to match tensor's shape
        indices = indices.expand(tensor.shape)
        tensor = torch.gather(tensor, dim=seq_dim, index=indices)

    return tensor


def shard_tensors(
    input_dict: dict[str, Any],
    *,
    input_shardings: dict[str, SpmdType],
    permutation: torch.Tensor | None,
) -> dict[str, Any]:
    """Permute and shard declared tensor inputs over the active CP mesh axis."""
    shard_dims = _cp_shard_dims(input_shardings)
    shard_names = tuple(
        name for name in shard_dims if isinstance(input_dict.get(name), torch.Tensor)
    )
    if not shard_names:
        return input_dict

    tensors = cast(
        tuple[torch.Tensor, ...], tuple(input_dict[name] for name in shard_names)
    )
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError("All CP-sharded inputs must be on the same device.")
    if permutation is not None and permutation.device != device:
        raise ValueError("The CP permutation and sharded inputs must share a device.")

    cp_group = spmd_mesh_group(MeshAxisName.CP)
    if cp_group is None:
        raise RuntimeError("CP sharding requires an active multi-rank CP mesh axis.")

    for name, tensor in zip(shard_names, tensors, strict=True):
        seq_dim = shard_dims[name]
        if permutation is not None:
            tensor = _permute_tensor(
                tensor,
                seq_dim=seq_dim,
                permutation=permutation,
            )
        input_dict[name] = spmd.shard(
            tensor,
            cp_group,
            src=spmd.R,
            dst=spmd.S(seq_dim),
        )
    return input_dict
