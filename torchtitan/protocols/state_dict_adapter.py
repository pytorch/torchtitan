# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map_only

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import plain_tensor_to_dtensor_state_dict
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

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        return self.convert_save_state_dict(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        return self.convert_load_state_dict(hf_state_dict)

    @abstractmethod
    def convert_save_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def convert_load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint

        Args:
            path: the path to read HF checkpoint

        Returns:
            The HuggingFace storage reader to read from HF checkpoint

        """
        raise NotImplementedError


class StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter base class which provides convenient default behavior to build fqn_to_index_mapping"""

    def __init__(
        self,
        model_config: BaseModel.Config,
        hf_assets_path: str | None,
    ):
        self.model_config = model_config
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

    def convert_save_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return self.to_hf(state_dict)

    def convert_load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return self.from_hf(state_dict)

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    def _validate_hf_rope_config(
        self,
        expected_rope_cls: type,
    ) -> None:
        for layer in self.model_config.layers:  # pyrefly: ignore [missing-attribute]
            rope = layer.attention.rope
            if not isinstance(rope, expected_rope_cls):
                expected_name = expected_rope_cls.__qualname__
                raise ValueError(
                    f"HF checkpoint conversion assumes {expected_name}; "
                    f"got {type(rope).__name__}."
                )

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)


class PlainToDTensorStateDictAdapter(BaseStateDictAdapter):
    def __init__(
        self,
        state_dict_layouts: Mapping[str, spmd.SpmdType] | None = None,
        parallel_dims: ParallelDims | None = None,
    ) -> None:
        self.state_dict_layouts = state_dict_layouts
        self.parallel_dims = parallel_dims
        self.optimizers = None

    @staticmethod
    def _optimizer_layouts(optimizers, state_dict):
        params = {
            fqn: param
            for optimizer in optimizers.optimizers
            for group in optimizer.param_groups
            for fqn, param in zip(group["param_names"], group["params"], strict=True)
        }
        return {
            key: spmd.SpmdType(
                dict(spmd.get_local_type(param)), spmd.get_partition_spec(param)
            )
            for key, value in state_dict.items()
            for fqn, param in params.items()
            if key.startswith(f"state.{fqn}.")
            and isinstance(value, torch.Tensor)
            and value.shape == param.shape
            and spmd.has_local_type(param)
        }

    def convert_save_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        converted = plain_tensor_to_dtensor_state_dict(
            state_dict,
            state_dict_layouts=self.state_dict_layouts,
            parallel_dims=self.parallel_dims,
        )
        self.optimizers = state_dict.get("optimizer")
        if self.optimizers is not None:
            optimizer_state = self.optimizers.state_dict()
            layouts = self._optimizer_layouts(self.optimizers, optimizer_state)
            converted_optimizer = dict(optimizer_state)
            converted_optimizer.update(
                plain_tensor_to_dtensor_state_dict(
                    {key: optimizer_state[key] for key in layouts},
                    state_dict_layouts=layouts,
                )
            )
            converted["optimizer"] = converted_optimizer
        return converted

    def convert_load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        converted = tree_map_only(DTensor, lambda value: value.to_local(), state_dict)
        if self.optimizers is not None and "optimizer" in converted:
            # Optimizer state may not exist before loading, so preserve its
            # normal load_state_dict path after DCP fills the target shards.
            self.optimizers.load_state_dict(converted["optimizer"])
            converted["optimizer"] = self.optimizers
        return converted
