# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
import tqdm
from safetensors import safe_open
from torchtitan.components.checkpoint import ModelWrapper


def load_incremental_hf(path) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def aggregate_moe_experts(
    state_dict: Dict[str, torch.Tensor],
    num_experts: int,
    leftover_experts: List[Dict[str, torch.Tensor]],
) -> List[Tuple[bool, Dict[str, torch.Tensor]]]:
    """
    Collect MoE expert tensors from a single safetensor file, grouped by (layer, weight_type).

    HF checkpoints split experts across files. This function accumulates expert tensors
    and tracks which groups are complete (have all num_experts).

    Groups by (layer, abstract_key) where:
        - layer: first int in key (e.g. 5 from "model.layers.5.mlp.experts.0...")
        - abstract_key: key with all ints replaced by {} (for pattern matching)

    Args:
        state_dict: Tensors from a single safetensor file
        num_experts: Number of experts per layer
        leftover_experts: Incomplete groups from previous files (list of {key: tensor} dicts)

    Returns:
        List of [is_complete, {original_key: tensor}] per (layer, abstract_key) group.
        Pass incomplete groups (is_complete=False) as leftover_experts to next call.
    """
    import re

    def get_layer(k: str) -> int:
        """Extract layer number (first int in key)."""
        m = re.search(r"\d+", k)
        return int(m.group()) if m else -1

    def abstract_key(k: str) -> str:
        """Remove all ints from key."""
        return re.sub(r"\d+", "{}", k)

    # Group by (layer, abstract_key)
    by_layer_abstract = {}

    # Merge leftovers
    for d in leftover_experts:
        for key, tensor in list(d.items()):
            if "mlp.experts" in key or ".moe" in key:
                layer = get_layer(key)
                ak = abstract_key(key)
                by_layer_abstract.setdefault((layer, ak), {})[key] = d.pop(key)

    # Add current file's moe tensors
    for key, tensor in list(state_dict.items()):
        if "mlp.experts" in key or ".moe" in key:
            layer = get_layer(key)
            ak = abstract_key(key)
            by_layer_abstract.setdefault((layer, ak), {})[key] = state_dict.pop(key)

    # Return [is_complete, experts_dict] per (layer, abstract_key)
    result = []
    for (layer, ak), experts in by_layer_abstract.items():
        is_complete = len(experts) >= num_experts
        print(
            f"Layer {layer}, Abstract Key {ak}, Is Complete: {is_complete}, Number of Experts: {len(experts)}"
        )
        result.append([is_complete, experts])

    return result


@torch.inference_mode()
def convert_from_hf(input_dir, output_dir, model_name, model_flavor):
    if model_name == "flux":
        import torchtitan.experiments.flux  # noqa: F401
    # initialize model to allocate memory for state dict
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    # Do it one at a time
    model_args.n_layers = 1
    try:
        num_experts = model_args.moe_args.num_experts
    except AttributeError:
        num_experts = 1
    print(f"Number of experts: {num_experts}")

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    # get state dict in tt format with allocated memory
    state_dict = model._get_state_dict()
    leftover_experts = []
    dcp_iter = 0
    meta_data = {}
    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".safetensors"):
                files_to_process.append(os.path.join(root, file))
    print(f"Found {files_to_process} files to process")
    for file in tqdm.tqdm(files_to_process, desc="Processing files"):
        hf_state_dict = load_incremental_hf(os.path.join(root, file))
        hf_state_dict = {k: v for k, v in hf_state_dict.items() if ".inv_freq" not in k}
        experts = aggregate_moe_experts(hf_state_dict, num_experts, leftover_experts)
        leftover_experts = [expert[1] for expert in experts if not expert[0]]
        for expert in experts:
            if expert[0]:
                hf_state_dict.update(expert[1])
        # convert state dict format back hf->tt and save
        state_dict = sd_adapter.from_hf(hf_state_dict)
        keys_in_file = list(state_dict.keys())
        meta_data[dcp_iter] = keys_in_file
        dcp.save(
            state_dict,
            checkpoint_id=os.path.join(output_dir, str(dcp_iter)),
        )
        dcp_iter += 1
    with open(os.path.join(output_dir, "meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with HF checkpoint"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    args = parser.parse_args()

    convert_from_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
    )
