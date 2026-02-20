# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight conversion utilities for Qwen3VLLMCompatModel.

Converts between:
- TorchTitan format (separate w1, w2, w3 in FFN)
- vLLM compat format (merged gate_up_proj = [w1; w3])
"""

import torch


def torchtitan_to_vllm_compat(
    torchtitan_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Convert TorchTitan Qwen3 state dict to vLLM-compatible format.

    Main changes:
    - Merge feed_forward.w1 and feed_forward.w3 into feed_forward.gate_up_proj
    - Rename feed_forward.w2 to feed_forward.down_proj
    """
    vllm_compat_state_dict = {}

    for key, tensor in torchtitan_state_dict.items():
        # Handle FFN weight merging
        if ".feed_forward.w1.weight" in key:
            # Get the corresponding w3 weight
            w3_key = key.replace(".feed_forward.w1.weight", ".feed_forward.w3.weight")
            w1_weight = tensor
            w3_weight = torchtitan_state_dict[w3_key]

            # Merge: gate_up_proj = [w1; w3] (concatenate along output dim)
            # torch.cat creates a new tensor, so no need to clone
            gate_up_weight = torch.cat([w1_weight, w3_weight], dim=0)

            # Save with new key
            new_key = key.replace(
                ".feed_forward.w1.weight", ".feed_forward.gate_up_proj.weight"
            )
            vllm_compat_state_dict[new_key] = gate_up_weight

        elif ".feed_forward.w3.weight" in key:
            # Skip w3, already merged with w1
            continue

        elif ".feed_forward.w2.weight" in key:
            # Rename w2 to down_proj
            new_key = key.replace(
                ".feed_forward.w2.weight", ".feed_forward.down_proj.weight"
            )
            # CLONE to avoid aliasing
            vllm_compat_state_dict[new_key] = tensor.clone()

        else:
            # Keep all other weights as-is
            # CLONE to avoid aliasing
            vllm_compat_state_dict[key] = tensor.clone()

    return vllm_compat_state_dict


def vllm_compat_to_torchtitan(
    vllm_compat_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Convert vLLM-compatible state dict back to TorchTitan format.

    Main changes:
    - Split feed_forward.gate_up_proj into separate w1 and w3
    - Rename feed_forward.down_proj to w2
    """
    torchtitan_state_dict = {}

    for key, tensor in vllm_compat_state_dict.items():
        # Handle FFN weight splitting
        if ".feed_forward.gate_up_proj.weight" in key:
            # Split into w1 (first half) and w3 (second half)
            hidden_dim = tensor.shape[0] // 2
            # CLONE to avoid aliasing - these are views into the original tensor
            w1_weight = tensor[:hidden_dim].clone()
            w3_weight = tensor[hidden_dim:].clone()

            # Save with new keys
            w1_key = key.replace(
                ".feed_forward.gate_up_proj.weight", ".feed_forward.w1.weight"
            )
            w3_key = key.replace(
                ".feed_forward.gate_up_proj.weight", ".feed_forward.w3.weight"
            )
            torchtitan_state_dict[w1_key] = w1_weight
            torchtitan_state_dict[w3_key] = w3_weight

        elif ".feed_forward.down_proj.weight" in key:
            # Rename down_proj to w2
            new_key = key.replace(
                ".feed_forward.down_proj.weight", ".feed_forward.w2.weight"
            )
            # CLONE to avoid aliasing
            torchtitan_state_dict[new_key] = tensor.clone()

        else:
            # Keep all other weights as-is
            # CLONE to avoid aliasing
            torchtitan_state_dict[key] = tensor.clone()

    return torchtitan_state_dict


if __name__ == "__main__":
    # Test conversion
    from safetensors.torch import load_file, save_file

    print("Loading TorchTitan checkpoint...")
    checkpoint_path = "./converted/qwen3_torchtitan.safetensors"
    titan_state = load_file(checkpoint_path)

    print(f"\nOriginal TorchTitan state dict has {len(titan_state)} keys")
    print("Sample keys:")
    for i, key in enumerate(list(titan_state.keys())[:10]):
        print(f"  {key}")

    print("\nConverting to vLLM-compat format...")
    vllm_compat_state = torchtitan_to_vllm_compat(titan_state)

    print(f"\nvLLM-compat state dict has {len(vllm_compat_state)} keys")
    print("Sample keys:")
    for i, key in enumerate(list(vllm_compat_state.keys())[:10]):
        print(f"  {key}")

    # Save converted checkpoint
    output_path = "./converted/qwen3_vllm_compat.safetensors"
    print(f"\nSaving to {output_path}...")
    save_file(vllm_compat_state, output_path)
    print("Done!")
