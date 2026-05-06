# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.tools.logging import logger

from .model import BaseModel


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
        self.hf_assets_path = hf_assets_path
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

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)

    @staticmethod
    def fused_to_separate_qkv(
        fused_weight: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused wqkv weight [n_kv * R * hd, dim] into separate Q, K, V."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        dim = fused_weight.shape[1]
        # [n_kv_heads, R, head_dim, dim]
        w = fused_weight.view(n_kv_heads, r_dim, head_dim, dim)
        wq = w[:, :heads_per_kv, :, :].reshape(n_heads * head_dim, dim)
        wk = w[:, heads_per_kv, :, :].reshape(n_kv_heads * head_dim, dim)
        wv = w[:, heads_per_kv + 1, :, :].reshape(n_kv_heads * head_dim, dim)
        return wq, wk, wv

    @staticmethod
    def separate_to_fused_qkv(
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Combine separate Q, K, V weights into fused wqkv layout."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        dim = wq.shape[1]
        # Reshape to per-KV-group: [n_kv_heads, heads_per_kv, head_dim, dim]
        q = wq.view(n_kv_heads, heads_per_kv, head_dim, dim)
        k = wk.view(n_kv_heads, 1, head_dim, dim)
        v = wv.view(n_kv_heads, 1, head_dim, dim)
        # [n_kv_heads, R, head_dim, dim]
        fused = torch.cat([q, k, v], dim=1)
        return fused.reshape(n_kv_heads * r_dim * head_dim, dim)
