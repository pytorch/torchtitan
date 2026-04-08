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

from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.tools.logging import logger


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats (e.g. HuggingFace).

    Subclasses implement ``to_hf`` and ``from_hf`` as explicit procedural
    conversions — each adapter reads one function and sees every key rename
    and value transform for that model.

    The base class constructor handles ``model_config``,
    ``hf_assets_path``, and ``fqn_to_index_mapping``.

    Args:
        model_config: model-specific configuration object.
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    fqn_to_index_mapping: dict[Any, int] | None
    hf_assets_path: str | None

    def __init__(self, model_config: Any, hf_assets_path: str | None = None):
        self.hf_assets_path = hf_assets_path
        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. "
                    "Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
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

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint.

        Args:
            path: the path to read HF checkpoint
            from_quantized: whether loading from quantized checkpoint format

        Returns:
            The HuggingFace storage reader to read from HF checkpoint
        """
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)
