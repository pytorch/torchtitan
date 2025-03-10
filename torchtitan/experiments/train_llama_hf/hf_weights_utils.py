# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
from collections import defaultdict
from pathlib import Path

import torch.nn as nn
from huggingface_hub import repo_exists, snapshot_download
from safetensors import safe_open
from torch.distributed.tensor import distribute_tensor, DTensor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from torchtitan.tools.logging import logger

INDEX_NAME_MAPPING = {
    "safetensors": SAFE_WEIGHTS_INDEX_NAME,
}

PATTERNS_TO_REMOVE = [
    "._orig_mod",  # Some optimizers add suffixes
    "._fsdp_wrapped_module",  # FSDP wrapper
    "._checkpoint_wrapped_module",  # checkpoint wrapper
    ".module",  # DataParallel/DistributedDataParallel
    "_module.",  # Some wrappers add prefix
]


def normalize_state_dict_key(
    key: str, patterns_to_remove: list[str] = PATTERNS_TO_REMOVE
) -> str:
    """
    Normalize the state dict key, remove the prefix or suffix added by various wrappers.
    Args:
        key: The original state dict key
    Returns:
        The normalized key
    """
    normalized_key = key
    for pattern in patterns_to_remove:
        normalized_key = normalized_key.replace(pattern, "")

    return normalized_key


def get_weight_map(pretrained_model_path: Path) -> dict[str, str]:
    """
    Get the weight map from the pretrained model.
    Args:
        pretrained_model_path: The path to the pretrained model.
    Returns:
        weight_map: A dictionary mapping from the path to the weight map to the list of state dict keys.
    """
    index_file = pretrained_model_path / INDEX_NAME_MAPPING["safetensors"]
    if not index_file.exists():
        return None
    with open(index_file, "r") as f:
        metadata = json.load(f)
    return metadata["weight_map"]


def group_state_dict_keys_and_st_partition_paths(
    pretrained_model_path: Path,
    state_dict_keys,
    weight_map,
    state_dict_map: dict[str, str] = None,
):
    """
    Group state dict keys and save them to a file.
    Args:
        pretrained_model_path: The path to the pretrained model.
        state_dict_keys: The state dict keys to group.
        weight_map: The weight map.
        state_dict_map: A dictionary mapping from the state dict key to the weight path.
    Returns:
        st_partition_map: A dictionary mapping from the weight path to the list of state dict keys.
    """
    st_partition_map = defaultdict(list)
    for state_dict_key in state_dict_keys:
        ckpt_state_dict_key = (
            state_dict_map[state_dict_key]
            if state_dict_map is not None
            else state_dict_key
        )
        if weight_map is None:
            partition_path = pretrained_model_path / "model.safetensors"
        else:
            partition_path = pretrained_model_path / weight_map[ckpt_state_dict_key]
        st_partition_map[partition_path].append(state_dict_key)
    return st_partition_map


def load_sharded_state_dict_for_model_from_path(
    pretrained_model_path: Path,
    model: nn.Module,
    mapping_dict: dict[str, str] = None,
    **kwargs,
):
    """
    Load the state dict sharded (depends on DTensor) from the pretrained model path. It only load the weights for current rank.
    Args:
        pretrained_model_path: The path to the pretrained model, it could be a local path or an s3 path.
        model: The model to load the state dict into.
        **kwargs: other arguments for torch.nn.Module.load_state_dict
    """
    # check exceptions
    if not pretrained_model_path.exists():
        raise ValueError(
            f"The pretrained model path {pretrained_model_path} does not exist."
        )
    if not pretrained_model_path.is_dir():
        raise ValueError(
            f"The pretrained model path {pretrained_model_path} is not a directory."
        )
    # get the weight map
    weight_map = get_weight_map(pretrained_model_path)
    model_state_dict = model.state_dict()
    model_state_dict_keys = list(model_state_dict.keys())

    # create a mapping_dict between the original state_dict_key and the weight_map_key if not provided
    mapping_dict = (
        mapping_dict
        if mapping_dict is not None
        else {key: normalize_state_dict_key(key) for key in model_state_dict_keys}
    )
    st_partition_map = group_state_dict_keys_and_st_partition_paths(
        pretrained_model_path, model_state_dict_keys, weight_map, mapping_dict
    )

    # get the sharded state dict
    state_dict = {}
    for safetensor_partition_path, state_dict_keys in st_partition_map.items():
        with safe_open(safetensor_partition_path, framework="pt", device="cpu") as f:
            for state_dict_key in state_dict_keys:
                model_tensor = model_state_dict[state_dict_key]
                ckpt_state_dict_key = mapping_dict[state_dict_key]
                if isinstance(model_tensor, DTensor):
                    local_tensor = f.get_tensor(ckpt_state_dict_key)
                    state_dict[state_dict_key] = distribute_tensor(
                        local_tensor,
                        model_tensor.device_mesh,
                        model_tensor.placements,
                    )
                else:
                    state_dict[state_dict_key] = f.get_tensor(ckpt_state_dict_key)
    model.load_state_dict(state_dict, **kwargs)
    del state_dict
    gc.collect()


def load_sharded_state_dict_for_model_from_hf(
    pretrained_model_id_or_path: str,
    model: nn.Module,
    **kwargs,
):
    """
    Load the state dict sharded (depends on DTensor) from the pretrained model path. It only load the weights for current rank.
    Args:
        pretrained_model_id_or_path: The id or path to the pretrained model, it could be a repo id in huggingface,
        or a local path
        model: The model to load the state dict into.
        **kwargs: other arguments for torch.nn.Module.load_state_dict
    """
    logger.info(f"Loading the state dict from {pretrained_model_id_or_path}")
    pretrained_model_id_or_path = Path(pretrained_model_id_or_path)
    if not pretrained_model_id_or_path.exists():
        if not repo_exists(str(pretrained_model_id_or_path)):
            raise ValueError(
                f"The pretrained model {pretrained_model_id_or_path} does not exist"
            )
        logger.info(
            f"Try to download the model from huggingface: {pretrained_model_id_or_path}"
        )
        pretrained_model_path = Path(
            snapshot_download(str(pretrained_model_id_or_path))
        )
    elif not pretrained_model_id_or_path.is_dir():
        raise ValueError(
            f"The pretrained model path {pretrained_model_id_or_path} is not a directory."
        )
    else:
        pretrained_model_path = pretrained_model_id_or_path

    load_sharded_state_dict_for_model_from_path(pretrained_model_path, model, **kwargs)
