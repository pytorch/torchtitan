# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from torch.distributed.checkpoint import HuggingFaceStorageReader

from .model import BaseModel


logger = logging.getLogger(__name__)


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

    def _linear_state_dict_to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove the physical singleton axis from ordinary Linear parameters."""
        from torchtitan.models.common.linear import canonical_linear_fqn, Linear

        result = dict(state_dict)
        for fqn, config, parent, _ in self.model_config.traverse(Linear.Config):
            assert isinstance(config, Linear.Config)
            if config.num_linears != 1:
                continue
            fqn = canonical_linear_fqn(fqn, parent)
            for name, physical_ndim in (("weight", 3), ("bias", 2)):
                key = f"{fqn}.{name}"
                value = result.get(key)
                if value is not None and value.ndim == physical_ndim:
                    result[key] = value.squeeze(0)
        if getattr(self.model_config, "enable_weight_tying", False):
            value = result.get("tok_embeddings.weight")
            if value is not None and value.ndim == 3:
                result["tok_embeddings.weight"] = value.squeeze(0)
        return result

    def _linear_state_dict_from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Restore the physical singleton axis on ordinary Linear parameters."""
        from torchtitan.models.common.linear import canonical_linear_fqn, Linear

        result = dict(state_dict)
        for fqn, config, parent, _ in self.model_config.traverse(Linear.Config):
            assert isinstance(config, Linear.Config)
            if config.num_linears != 1:
                continue
            fqn = canonical_linear_fqn(fqn, parent)
            for name, hf_ndim in (("weight", 2), ("bias", 1)):
                key = f"{fqn}.{name}"
                value = result.get(key)
                if value is not None and value.ndim == hf_ndim:
                    result[key] = value.unsqueeze(0)
        if getattr(self.model_config, "enable_weight_tying", False):
            value = result.get("tok_embeddings.weight")
            if value is not None and value.ndim == 2:
                result["tok_embeddings.weight"] = value.unsqueeze(0)
        return result

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)
