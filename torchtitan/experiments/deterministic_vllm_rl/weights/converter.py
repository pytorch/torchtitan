# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal weight converter between vLLM and TorchTitan formats for Qwen3-1.7B.

This script provides bidirectional weight conversion:
- vllm_to_torchtitan: Load weights from vLLM format and convert to TorchTitan format
- torchtitan_to_vllm: Load weights from TorchTitan format and convert to vLLM format
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# Weight name mapping from HuggingFace/vLLM to TorchTitan
VLLM_TO_TITAN_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    # Attention weights
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
    # MLP weights
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    # Layer norms
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    # Final norm and output
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def vllm_to_torchtitan(
    vllm_path_or_state: str | dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Load weights from vLLM format (HuggingFace) and convert to TorchTitan format.

    Args:
        vllm_path_or_state: Either a path to vLLM model directory (contains .safetensors or .bin files)
                            OR a vLLM state dict

    Returns:
        Dictionary with TorchTitan-formatted state dict
    """
    # Check if input is a state dict or a path
    if isinstance(vllm_path_or_state, dict):
        vllm_state = vllm_path_or_state
        print(f"Using provided vLLM state dict with {len(vllm_state)} weights")
    else:
        vllm_path = Path(vllm_path_or_state)

        # Load weights from vLLM format (try safetensors first, then .bin)
        vllm_state = {}
        safetensor_files = sorted(vllm_path.glob("*.safetensors"))

        if safetensor_files:
            print(f"Loading {len(safetensor_files)} safetensors files...")
            for st_file in safetensor_files:
                if "index" not in st_file.name:  # Skip index files
                    vllm_state.update(load_file(str(st_file)))
        else:
            # Fallback to .bin files
            bin_files = sorted(vllm_path.glob("*.bin"))
            print(f"Loading {len(bin_files)} .bin files...")
            for bin_file in bin_files:
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                vllm_state.update(state)

        print(f"Loaded {len(vllm_state)} weights from vLLM format")

    # Convert to TorchTitan format
    titan_state = {}

    for vllm_key, tensor in vllm_state.items():
        # Skip rotary embedding frequencies (not needed in TorchTitan)
        if "rotary_emb.inv_freq" in vllm_key:
            continue

        # Check if it's a layer-specific weight
        if "layers." in vllm_key:
            # Extract layer number
            parts = vllm_key.split(".")
            layer_idx = parts[2]

            # Create abstract key with placeholder
            abstract_vllm_key = vllm_key.replace(f".{layer_idx}.", ".{}.")

            # Look up in mapping
            if abstract_vllm_key in VLLM_TO_TITAN_MAP:
                abstract_titan_key = VLLM_TO_TITAN_MAP[abstract_vllm_key]
                titan_key = abstract_titan_key.format(layer_idx)
                titan_state[titan_key] = tensor
            else:
                print(f"Warning: No mapping found for {vllm_key}")
        else:
            # Non-layer weight
            if vllm_key in VLLM_TO_TITAN_MAP:
                titan_key = VLLM_TO_TITAN_MAP[vllm_key]
                titan_state[titan_key] = tensor
            else:
                print(f"Warning: No mapping found for {vllm_key}")

    print(f"Converted to {len(titan_state)} TorchTitan weights")
    return titan_state


def torchtitan_to_vllm(titan_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert weights from TorchTitan format to vLLM format (HuggingFace).

    Args:
        titan_state: TorchTitan state dict (can be in vLLM-compat format with gate_up_proj)

    Returns:
        Dictionary with vLLM/HuggingFace-formatted state dict
    """
    # Create reverse mapping
    titan_to_vllm_map = {v: k for k, v in VLLM_TO_TITAN_MAP.items()}

    vllm_state = {}

    for titan_key, tensor in titan_state.items():
        # Handle merged gate_up_proj (vLLM-compat format) -> split into gate_proj + up_proj
        if ".feed_forward.gate_up_proj.weight" in titan_key:
            # Split into gate_proj (first half) and up_proj (second half)
            hidden_dim = tensor.shape[0] // 2
            # CLONE to avoid aliasing - these are views into the original tensor
            gate_weight = tensor[:hidden_dim].clone()
            up_weight = tensor[hidden_dim:].clone()

            # Extract layer number
            parts = titan_key.split(".")
            layer_idx = parts[1]

            # Create vLLM keys
            gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"

            vllm_state[gate_key] = gate_weight
            vllm_state[up_key] = up_weight
            continue

        # Handle down_proj (vLLM-compat format)
        if ".feed_forward.down_proj.weight" in titan_key:
            parts = titan_key.split(".")
            layer_idx = parts[1]
            vllm_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            # CLONE to avoid aliasing
            vllm_state[vllm_key] = tensor.clone()
            continue

        # Check if it's a layer-specific weight
        if "layers." in titan_key:
            # Extract layer number
            parts = titan_key.split(".")
            layer_idx = parts[1]

            # Create abstract key with placeholder
            abstract_titan_key = titan_key.replace(f".{layer_idx}.", ".{}.")

            # Look up in reverse mapping
            if abstract_titan_key in titan_to_vllm_map:
                abstract_vllm_key = titan_to_vllm_map[abstract_titan_key]
                vllm_key = abstract_vllm_key.format(layer_idx)
                # CLONE to avoid aliasing
                vllm_state[vllm_key] = tensor.clone()
            else:
                print(f"Warning: No mapping found for {titan_key}")
        else:
            # Non-layer weight
            if titan_key in titan_to_vllm_map:
                vllm_key = titan_to_vllm_map[titan_key]
                # CLONE to avoid aliasing
                vllm_state[vllm_key] = tensor.clone()
            else:
                print(f"Warning: No mapping found for {titan_key}")

    print(f"Converted to {len(vllm_state)} vLLM weights")
    return vllm_state


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  Convert vLLM to TorchTitan:")
        print(
            "    python weight_converter.py vllm_to_titan <vllm_model_path> <output_path>"
        )
        print("  Convert TorchTitan to vLLM:")
        print(
            "    python weight_converter.py titan_to_vllm <titan_checkpoint_path> <output_path>"
        )
        sys.exit(1)

    mode = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if mode == "vllm_to_titan":
        # Convert vLLM to TorchTitan
        titan_state = vllm_to_torchtitan(input_path)

        # Save as safetensors
        print(f"Saving to {output_path}...")
        save_file(titan_state, output_path)
        print("Done!")

    elif mode == "titan_to_vllm":
        # Load TorchTitan checkpoint
        print(f"Loading TorchTitan checkpoint from {input_path}...")
        titan_state = load_file(input_path)

        # Convert to vLLM
        vllm_state = torchtitan_to_vllm(titan_state)

        # Save as safetensors
        print(f"Saving to {output_path}...")
        save_file(vllm_state, output_path)
        print("Done!")

    else:
        print(f"Unknown mode: {mode}")
        print("Use 'vllm_to_titan' or 'titan_to_vllm'")
        sys.exit(1)
