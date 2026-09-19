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

import torch
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
        """Split native fused feed-forward parameters into their HF layout."""
        from torchtitan.models.common.feed_forward import FeedForward

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(FeedForward.Config):
            prefix = f"{fqn}." if fqn else ""
            for name in ("weight", "bias"):
                fused_key = f"{prefix}w13.{name}"
                if fused_key not in result:
                    continue
                gate_up = result.pop(fused_key)
                result[f"{prefix}w1.{name}"] = gate_up[0]
                result[f"{prefix}w3.{name}"] = gate_up[1]

        return result

    def _linear_state_dict_from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Stack HF feed-forward parameters into their native fused layout."""
        from torchtitan.models.common.feed_forward import FeedForward

        result = dict(state_dict)
        for fqn, _config, _parent, _ in self.model_config.traverse(FeedForward.Config):
            prefix = f"{fqn}." if fqn else ""
            for name in ("weight", "bias"):
                gate_key = f"{prefix}w1.{name}"
                up_key = f"{prefix}w3.{name}"
                if gate_key not in result or up_key not in result:
                    continue
                result[f"{prefix}w13.{name}"] = torch.stack(
                    [result.pop(gate_key), result.pop(up_key)], dim=0
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
