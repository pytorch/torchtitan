# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Dict, Optional, Set, Tuple

import torch
from safetensors import safe_open

from transformers.utils import cached_file


logger = logging.getLogger(__name__)

_DEFAULT_SAFETENSOR_FILE_NAME = "model.safetensors.index.json"


def read_weights_from_json(file_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        if "weight_map" in data and isinstance(data["weight_map"], dict):
            return data["weight_map"]
        else:
            logger.error("No 'weight_map' dictionary found in the JSON file.")
            return None
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"An error occurred while reading the JSON file: {str(e)}")
        return None


def get_hf_weight_map_and_path(
    model_id: str,
) -> Tuple[Dict[str, str], str]:
    """Get the weight map for a given HF model id and also the cache path for loading the weights"""
    try:
        index_file = cached_file(model_id, _DEFAULT_SAFETENSOR_FILE_NAME)
    except Exception as e:
        logger.error(
            f"Model `{model_id}` not found in HF cache. "
            f"You can download the model using `python download.py {model_id}"
        )
        raise e

    weight_map = read_weights_from_json(index_file)
    weight_path = os.path.dirname(index_file)
    logger.info(f"Loading weights from: {weight_path}")
    return weight_map, weight_path


def get_needed_files(
    state_dict: Dict[str, torch.Tensor], weight_map: Dict[str, str]
) -> Set[str]:
    needed_files = set()
    for param in state_dict.keys():
        file = weight_map.get(param)
        if file:
            needed_files.add(file)
        elif param.endswith("weight"):
            raise ValueError(
                f"Parameter {param} not found in weight map, please check..."
            )
    logger.debug(f"Needed files: {needed_files}")
    return needed_files


def load_safetensor_file(
    full_path: str, device: torch.device
) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(full_path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    logger.debug(f"Loaded {len(tensors)} tensors from {full_path}")
    return tensors


def load_safetensor_weights(
    model: torch.nn.Module,
    weight_map: Dict[str, str],
    file_location: str,
    device: torch.device,
):
    """
    Load safetensor weights into a `nn.Module`.

    Args:
        model (Module): The PyTorch module to load weights into. It may be a
        model chunk or a full model.
        weight_map (Dict[str, str]): Mapping of model parameters to file names.
        file_location (str): Directory containing the weight files.
        device (torch.device): The device to load tensors onto.
    """
    model_state_dict = model.state_dict()
    needed_files = get_needed_files(model_state_dict, weight_map)
    updated_states: Set[str] = set()

    for file in needed_files:
        full_path = os.path.join(file_location, file)
        try:
            checkpoint = load_safetensor_file(full_path, "cpu")
        except FileNotFoundError:
            logger.error(f"File not found: {full_path}")
        except Exception as e:
            logger.error(f"Error during checkpoint processing of {full_path}: {str(e)}")

        matched_keys = set(checkpoint.keys()) & set(model_state_dict.keys())
        for key in matched_keys:
            # Check shape
            if model_state_dict[key].shape != checkpoint[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"model needs {model_state_dict[key].shape}, but "
                    f"checkpoint has {checkpoint[key].shape}"
                )
            model_state_dict[key] = checkpoint[key].to(device)

        updated_states.update(matched_keys)

    missing_keys = set(model_state_dict.keys()) - updated_states
    if missing_keys:
        raise RuntimeError(
            f"Partially updated state dict. Missing parameters: {missing_keys}"
        )

    model.load_state_dict(model_state_dict, strict=False, assign=True)
    logger.debug(f"Successfully loaded {len(updated_states)} weights into model")


def load_weights_from_hf(
    model: torch.nn.Module,
    distribution: str,
    device: torch.device,
):
    """
    Load the weights from Hugging Face format (index file + multiple safetensor
    files), and fill into `model`.  Model config is needed b/c we permute
    wq and wk weights based on attn heads.
    """

    weight_map, weight_path = get_hf_weight_map_and_path(distribution)

    load_safetensor_weights(
        model,
        weight_map,
        weight_path,
        device,
    )
