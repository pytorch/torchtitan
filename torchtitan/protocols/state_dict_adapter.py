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
from torch.distributed.tensor.placement_types import Placement

from .model import BaseModel


logger = logging.getLogger(__name__)


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
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        pass

    @abstractmethod
    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint

        Args:
            path: the path to read HF checkpoint

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
        state_dict[fused_key] = torch.stack(
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

    def _native_fused_linears_to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert native fused linear parameters to logical HF-facing keys.

        Model-specific adapters subsequently rename the logical keys to their
        corresponding HF keys.
        """
        from torchtitan.models.common.attention import QKVLinear
        from torchtitan.models.common.feed_forward import FeedForward

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(FeedForward.Config):
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

        return result

    def _native_fused_linears_from_hf(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert logical HF-facing keys to native fused linear parameters."""
        from torchtitan.models.common.attention import QKVLinear
        from torchtitan.models.common.feed_forward import FeedForward

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(FeedForward.Config):
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

        return result

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)
