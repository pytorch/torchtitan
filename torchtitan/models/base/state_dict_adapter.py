# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import json

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.placement_types import _StridedShard, Placement, Shard

from .model import BaseModel


logger = logging.getLogger(__name__)


def stack_or_view(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    """``torch.stack(tensors, dim)``, as a view when the tensors already are its slices.

    ``to_hf`` exposes a stacked parameter as views of it (``unbind``, per-expert
    slicing) and an HF load fills those views in place, so on the load path the
    tensors ``from_hf`` stacks are slices of the destination parameter. Stacking
    them would allocate a second copy of that parameter (every routed expert on
    the rank for MoE models). When all tensors share one storage, dtype, shape
    and strides and each starts a fixed offset after the previous one, return
    that strided view of the storage instead; otherwise stack. DTensors are
    checked by their local tensors and keep their mesh; their placements gain
    the new dimension.
    """
    first = tensors[0]
    if isinstance(first, DTensor):
        if any(
            t.device_mesh != first.device_mesh or t.placements != first.placements
            for t in tensors
        ):
            return torch.stack(tensors, dim=dim)
        local = stack_or_view([t._local_tensor for t in tensors], dim)
        placements = []
        for p in first.placements:
            # _StridedShard is not a Shard subclass in every PyTorch version.
            if isinstance(p, _StridedShard) and p.dim >= dim:
                p = _StridedShard(p.dim + 1, split_factor=p.split_factor)
            elif isinstance(p, Shard) and p.dim >= dim:
                p = Shard(p.dim + 1)
            placements.append(p)
        shape = (*first.shape[:dim], len(tensors), *first.shape[dim:])
        return DTensor.from_local(
            local,
            first.device_mesh,
            placements,
            run_check=False,
            shape=torch.Size(shape),
            stride=torch.empty(shape, device="meta").stride(),
        )
    if len(tensors) == 1:
        return first.unsqueeze(dim)
    step = tensors[1].storage_offset() - first.storage_offset()
    storage = first.untyped_storage().data_ptr()
    is_view = step > 0 and all(
        t.untyped_storage().data_ptr() == storage
        and t.dtype == first.dtype
        and t.shape == first.shape
        and t.stride() == first.stride()
        and t.storage_offset() == first.storage_offset() + i * step
        for i, t in enumerate(tensors)
    )
    if not is_view:
        return torch.stack(tensors, dim=dim)
    return first.as_strided(
        (*first.shape[:dim], len(tensors), *first.shape[dim:]),
        (*first.stride()[:dim], step, *first.stride()[dim:]),
        first.storage_offset(),
    )


def dtensor_safe(fn):
    """Run a row-reshaping permute that is invalid on a tensor sharded along the
    permuted (row) dim.

    In the live save/load path ``to_hf`` / ``from_hf`` receive DTensors that
    FSDP shards along dim 0 (the q/k output rows). The head-splitting
    ``view(n_heads, ...)`` cannot unflatten an unevenly-sharded dim and raises
    ``Cannot unflatten unevenly sharded tensor``. Redistribute to Replicate,
    permute the full local tensor, then restore the original placements. Plain
    (non-DTensor) tensors take the fast path unchanged.
    """

    @functools.wraps(fn)
    def wrapper(self, w, *args, **kwargs):
        if isinstance(w, DTensor):
            placements = w.placements
            mesh = w.device_mesh
            replicated = w.redistribute(
                device_mesh=mesh, placements=[Replicate()] * mesh.ndim
            )
            local = fn(self, replicated.to_local(), *args, **kwargs)
            out = DTensor.from_local(
                local, mesh, [Replicate()] * mesh.ndim, run_check=False
            )
            return out.redistribute(device_mesh=mesh, placements=placements)
        return fn(self, w, *args, **kwargs)

    return wrapper


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    Args:
        model_config: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    fqn_to_index_mapping: dict[Any, int] | None
    hf_assets_path: str | None

    @abstractmethod
    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        pass

    @abstractmethod
    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict
            to_quantized_layout: Use the key names of a quantized HF checkpoint,
                for an HF load with ``from_quantized``. Only the names change:
                the values keep their dequantized shapes and are not quantized.
                Adapters whose quantized and unquantized checkpoints share
                names can ignore this.

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict
            is_quantized_layout: Whether ``hf_state_dict`` uses the key names of
                a quantized HF checkpoint. Only the names differ: the storage
                reader has already dequantized the values. In an HF load,
                ``hf_state_dict`` is the dict ``to_hf`` returned, so this must
                match the ``to_quantized_layout`` value ``to_hf`` was called
                with.

        Returns:
            The converted native model state dict
        """
        pass

    @abstractmethod
    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint

        Args:
            path: the path to read HF checkpoint
            from_quantized: whether the checkpoint uses a quantized format
            thread_count: the reader thread count, or the reader default if unset

        Returns:
            The HuggingFace storage reader to read from HF checkpoint

        """
        pass


class StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter base class which provides convenient default behavior to build fqn_to_index_mapping"""

    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        # HF load calls to_hf on the destination state dict before from_hf.
        # Remember each native QKV layout so the rebuilt tensor fits that target.
        self._qkv_linear_sharding: dict[
            str, tuple[DeviceMesh, tuple[Placement, ...]]
        ] = {}
        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. \
                    Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
                )
                hf_safetensors_indx = None

            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {}
                for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                    # pyrefly: ignore [missing-attribute]
                    indx = re.search(r"\d+", raw_indx).group(0)
                    self.fqn_to_index_mapping[hf_key] = int(indx)
            else:
                self.fqn_to_index_mapping = None
        else:
            self.fqn_to_index_mapping = None

    def _validate_hf_rope_config(
        self,
        expected_rope_cls: type,
    ) -> None:
        for layer in self.model_config.layers:  # pyrefly: ignore [missing-attribute]
            rope = layer.attention.rope
            # NoPE layers carry no rope config, so there is nothing to validate.
            if rope is None:
                continue
            if not isinstance(rope, expected_rope_cls):
                expected_name = expected_rope_cls.__qualname__
                raise ValueError(
                    f"HF checkpoint conversion assumes {expected_name}; "
                    f"got {type(rope).__qualname__}."
                )

    @staticmethod
    def _split_stacked_linear(
        state_dict: dict[str, Any],
        *,
        fused_key: str,
        logical_keys: tuple[str, ...],
        dim: int,
    ) -> None:
        """Expose a native stacked parameter as logical projection views."""
        if fused_key not in state_dict:
            return
        projections = state_dict.pop(fused_key).unbind(dim)
        assert len(projections) == len(logical_keys)
        state_dict.update(zip(logical_keys, projections, strict=True))

    @staticmethod
    def _stack_logical_linears(
        state_dict: dict[str, Any],
        *,
        fused_key: str,
        logical_keys: tuple[str, ...],
        dim: int,
    ) -> None:
        """Stack logical projection parameters into one native parameter."""
        if not all(key in state_dict for key in logical_keys):
            return
        state_dict[fused_key] = stack_or_view(
            [state_dict.pop(key) for key in logical_keys], dim=dim
        )

    def _split_qkv_linear(
        self,
        state_dict: dict[str, Any],
        *,
        prefix: str,
        head_dim: int,
        heads_per_kv: int,
    ) -> None:
        """Expose a native packed QKV parameter as logical Q/K/V projections."""
        qkv_group_size = heads_per_kv + 2
        for param_name in ("weight", "bias"):
            fused_key = f"{prefix}wqkv.{param_name}"
            if fused_key not in state_dict:
                continue
            tensor = state_dict.pop(fused_key)
            # The packed groups cross a dim-0 shard boundary when the FSDP
            # degree does not divide the KV-head count, so reshape a replicated
            # DTensor and keep the result distributed for HF checkpoint I/O.
            if isinstance(tensor, DTensor):
                self._qkv_linear_sharding[fused_key] = (
                    tensor.device_mesh,
                    tensor.placements,
                )
                tensor = tensor.redistribute(
                    tensor.device_mesh, [Replicate()] * tensor.device_mesh.ndim
                )
            num_kv_heads = tensor.shape[0] // (qkv_group_size * head_dim)
            tail = tensor.shape[1:]
            packed = tensor.reshape(
                num_kv_heads,
                qkv_group_size,
                head_dim,
                *tail,
            )
            state_dict[f"{prefix}wq.{param_name}"] = (
                packed[:, :heads_per_kv].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wk.{param_name}"] = (
                packed[:, heads_per_kv].reshape(-1, *tail).contiguous()
            )
            state_dict[f"{prefix}wv.{param_name}"] = (
                packed[:, heads_per_kv + 1].reshape(-1, *tail).contiguous()
            )

    def _merge_qkv_linear(
        self,
        state_dict: dict[str, Any],
        *,
        prefix: str,
        head_dim: int,
        heads_per_kv: int,
    ) -> None:
        """Pack logical Q/K/V projections into one native QKV parameter."""
        for param_name in ("weight", "bias"):
            logical_keys = tuple(
                f"{prefix}{projection}.{param_name}"
                for projection in ("wq", "wk", "wv")
            )
            if not all(key in state_dict for key in logical_keys):
                continue
            wq, wk, wv = (state_dict.pop(key) for key in logical_keys)
            # Loading may provide dim-0-sharded DTensors whose local shapes
            # cannot express whole QKV groups. Rebuild from replicated inputs,
            # then restore the native placement captured before HF loading.
            if isinstance(wq, DTensor):
                wq, wk, wv = (
                    tensor.redistribute(
                        tensor.device_mesh,
                        [Replicate()] * tensor.device_mesh.ndim,
                    )
                    for tensor in (wq, wk, wv)
                )
            num_kv_heads = wk.shape[0] // head_dim
            tail = wq.shape[1:]
            q = wq.reshape(num_kv_heads, heads_per_kv, head_dim, *tail)
            k = wk.reshape(num_kv_heads, 1, head_dim, *tail)
            v = wv.reshape(num_kv_heads, 1, head_dim, *tail)
            fused_key = f"{prefix}wqkv.{param_name}"
            fused = torch.cat([q, k, v], dim=1).reshape(-1, *tail)
            if fused_key in self._qkv_linear_sharding:
                assert isinstance(fused, DTensor)
                mesh, placements = self._qkv_linear_sharding[fused_key]
                fused = fused.redistribute(mesh, placements)
            state_dict[fused_key] = fused

    def _native_fused_linears_to_hf(
        self,
        state_dict: dict[str, Any],
        *,
        split_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """Convert native fused linear parameters to logical HF-facing keys.

        Model-specific adapters subsequently rename the logical keys to their
        corresponding HF keys.

        Args:
            state_dict: Native model state.
            split_routed_experts: Split routed W13 weights into logical W1/W3
                tensors for HF schemas that store experts independently.
        """
        from torchtitan.models.common.attention import QKVLinear
        from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
        from torchtitan.models.common.feed_forward import DenseFeedForward
        from torchtitan.models.common.moe import RoutedExperts

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(
            DenseFeedForward.Config
        ):
            prefix = f"{fqn}." if fqn else ""
            for name in ("weight", "bias"):
                self._split_stacked_linear(
                    result,
                    fused_key=f"{prefix}w13.{name}",
                    logical_keys=(
                        f"{prefix}w1.{name}",
                        f"{prefix}w3.{name}",
                    ),
                    dim=0,
                )

        for fqn, config, _parent, _ in self.model_config.traverse(QKVLinear.Config):
            prefix = f"{fqn}." if fqn else ""
            self._split_qkv_linear(
                result,
                prefix=prefix,
                head_dim=config.head_dim,
                heads_per_kv=config.n_heads // config.n_kv_heads,
            )

        if split_routed_experts:
            expert_configs = [
                *self.model_config.traverse(RoutedExperts.Config),
                *self.model_config.traverse(DistMoeRoutedExperts.Config),
            ]
            for fqn, _config, _parent, _ in expert_configs:
                prefix = f"{fqn}." if fqn else ""
                self._split_stacked_linear(
                    result,
                    fused_key=f"{prefix}w13.weight",
                    logical_keys=(
                        f"{prefix}w1_EFD",
                        f"{prefix}w3_EFD",
                    ),
                    dim=1,
                )

        return result

    def _native_fused_linears_from_hf(
        self,
        state_dict: dict[str, Any],
        *,
        fuse_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """Convert logical HF-facing keys to native fused linear parameters.

        Args:
            state_dict: Logical HF-facing model state.
            fuse_routed_experts: Fuse logical routed W1/W3 tensors into W13 for
                HF schemas that store experts independently.
        """
        from torchtitan.models.common.attention import QKVLinear
        from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
        from torchtitan.models.common.feed_forward import DenseFeedForward
        from torchtitan.models.common.moe import RoutedExperts

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(
            DenseFeedForward.Config
        ):
            prefix = f"{fqn}." if fqn else ""
            for name in ("weight", "bias"):
                self._stack_logical_linears(
                    result,
                    fused_key=f"{prefix}w13.{name}",
                    logical_keys=(
                        f"{prefix}w1.{name}",
                        f"{prefix}w3.{name}",
                    ),
                    dim=0,
                )

        for fqn, config, _parent, _ in self.model_config.traverse(QKVLinear.Config):
            prefix = f"{fqn}." if fqn else ""
            self._merge_qkv_linear(
                result,
                prefix=prefix,
                head_dim=config.head_dim,
                heads_per_kv=config.n_heads // config.n_kv_heads,
            )

        if fuse_routed_experts:
            expert_configs = [
                *self.model_config.traverse(RoutedExperts.Config),
                *self.model_config.traverse(DistMoeRoutedExperts.Config),
            ]
            for fqn, _config, _parent, _ in expert_configs:
                prefix = f"{fqn}." if fqn else ""
                self._stack_logical_linears(
                    result,
                    fused_key=f"{prefix}w13.weight",
                    logical_keys=(
                        f"{prefix}w1_EFD",
                        f"{prefix}w3_EFD",
                    ),
                    dim=1,
                )

        return result

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        if thread_count is None:
            # Two workers overlap shard reads without heavy storage pressure.
            thread_count = 2
        return HuggingFaceStorageReader(path, thread_count=thread_count)
