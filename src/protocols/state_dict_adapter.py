import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.distributed.checkpoint import HuggingFaceStorageReader  # type: ignore

from src.tools.logging import logger

from .model import BaseModelArgs


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    Args:
        model_args: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    fqn_to_index_mapping: Dict[Any, int] | None

    @abstractmethod
    def __init__(
        self,
        model_args: BaseModelArgs,
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
    def get_hf_storage_reader(self, path: str) -> HuggingFaceStorageReader:
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
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
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
                    indx = re.search(r"\d+", raw_indx).group(0)  # type: ignore
                    self.fqn_to_index_mapping[hf_key] = int(indx)
            else:
                self.fqn_to_index_mapping = None

    def get_hf_storage_reader(self, path: str) -> HuggingFaceStorageReader:
        return HuggingFaceStorageReader(path)
