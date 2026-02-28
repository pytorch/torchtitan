#!/usr/bin/env python3
"""
Convert a TorchTitan distributed checkpoint to HuggingFace safetensors format.

Usage:
    python scripts/convert_checkpoint_to_hf.py \
        --checkpoint ./outputs/persona_local_2gpu_8192_cpu_offload/checkpoint_persona_8192_cpu_offload/step-65 \
        --output ./outputs/gpt-oss-20b-persona \
        --hf-base /mnt/models/gpt-oss-20b
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

# Add torchtitan to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchtitan.models.gpt_oss.model.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.models.gpt_oss.model.args import GptOssModelArgs


# Custom mapping for unquantized export (without _blocks suffix)
UNQUANTIZED_HF_MAP = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    # Attention module
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wq.bias": "model.layers.{}.self_attn.q_proj.bias",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wk.bias": "model.layers.{}.self_attn.k_proj.bias",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wv.bias": "model.layers.{}.self_attn.v_proj.bias",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.wo.bias": "model.layers.{}.self_attn.o_proj.bias",
    "layers.{}.attention.sinks": "model.layers.{}.self_attn.sinks",
    # Transformer layer norms
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    # MoE - UNQUANTIZED format (no _blocks suffix)
    "layers.{}.moe.experts.mlp1_weight": "model.layers.{}.mlp.experts.gate_up_proj",
    "layers.{}.moe.experts.mlp1_bias": "model.layers.{}.mlp.experts.gate_up_proj_bias",
    "layers.{}.moe.experts.mlp2_weight": "model.layers.{}.mlp.experts.down_proj",
    "layers.{}.moe.experts.mlp2_bias": "model.layers.{}.mlp.experts.down_proj_bias",
    "layers.{}.moe.router.gate.weight": "model.layers.{}.mlp.router.weight",
    "layers.{}.moe.router.gate.bias": "model.layers.{}.mlp.router.bias",
    # Output
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# QAT-wrapped model mapping (extra .experts nesting from QAT wrapper)
QAT_HF_MAP = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    # Attention module (same as unquantized - attention is not QAT wrapped)
    "layers.{}.attention.wq.weight": "model.layers.{}.self_attn.q_proj.weight",
    "layers.{}.attention.wq.bias": "model.layers.{}.self_attn.q_proj.bias",
    "layers.{}.attention.wk.weight": "model.layers.{}.self_attn.k_proj.weight",
    "layers.{}.attention.wk.bias": "model.layers.{}.self_attn.k_proj.bias",
    "layers.{}.attention.wv.weight": "model.layers.{}.self_attn.v_proj.weight",
    "layers.{}.attention.wv.bias": "model.layers.{}.self_attn.v_proj.bias",
    "layers.{}.attention.wo.weight": "model.layers.{}.self_attn.o_proj.weight",
    "layers.{}.attention.wo.bias": "model.layers.{}.self_attn.o_proj.bias",
    "layers.{}.attention.sinks": "model.layers.{}.self_attn.sinks",
    # Transformer layer norms
    "layers.{}.attention_norm.weight": "model.layers.{}.input_layernorm.weight",
    "layers.{}.ffn_norm.weight": "model.layers.{}.post_attention_layernorm.weight",
    # MoE - QAT wrapped format (extra .experts from QAT GroupedMLP wrapper)
    "layers.{}.moe.experts.experts.mlp1_weight": "model.layers.{}.mlp.experts.gate_up_proj",
    "layers.{}.moe.experts.experts.mlp1_bias": "model.layers.{}.mlp.experts.gate_up_proj_bias",
    "layers.{}.moe.experts.experts.mlp2_weight": "model.layers.{}.mlp.experts.down_proj",
    "layers.{}.moe.experts.experts.mlp2_bias": "model.layers.{}.mlp.experts.down_proj_bias",
    "layers.{}.moe.router.gate.weight": "model.layers.{}.mlp.router.weight",
    "layers.{}.moe.router.gate.bias": "model.layers.{}.mlp.router.bias",
    # Output
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def convert_to_hf_unquantized(state_dict: dict, is_qat: bool = False) -> dict:
    """Convert TorchTitan state dict to HF format for unquantized models.

    Args:
        state_dict: The model state dict
        is_qat: If True, use QAT mapping (extra .experts nesting)
    """
    import re
    hf_state_dict = {}

    # Select the appropriate mapping
    hf_map = QAT_HF_MAP if is_qat else UNQUANTIZED_HF_MAP

    # Keys that need transposition for OSS package compatibility
    # TorchTitan stores: (experts, out_dim, in_dim) for y = x @ W^T pattern
    # HuggingFace transformers/vLLM expect: (experts, in_dim, out_dim)
    # NOTE: This differs from MXFP4 quantized format which uses (experts, out_dim, in_dim)
    if is_qat:
        transpose_keys = {
            "layers.{}.moe.experts.experts.mlp1_weight",  # gate_up_proj (QAT)
            "layers.{}.moe.experts.experts.mlp2_weight",  # down_proj (QAT)
        }
    else:
        transpose_keys = {
            "layers.{}.moe.experts.mlp1_weight",  # gate_up_proj
            "layers.{}.moe.experts.mlp2_weight",  # down_proj
        }

    for key, value in state_dict.items():
        if "layers" in key:
            # Extract layer number
            match = re.search(r"layers\.(\d+)\.", key)
            if not match:
                continue
            layer_num = match.group(1)
            # Create abstract key with {} placeholder
            abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", key)

            if abstract_key not in hf_map:
                print(f"  Warning: No mapping for {key}")
                continue

            hf_key = hf_map[abstract_key].format(layer_num)

            # Transpose expert weights for OSS package compatibility
            if abstract_key in transpose_keys and value.dim() == 3:
                # TorchTitan: (experts, out_dim, in_dim) -> HF: (experts, in_dim, out_dim)
                value = value.transpose(1, 2).contiguous()
                print(f"  Transposed {key}: {value.shape}")

            hf_state_dict[hf_key] = value
        else:
            if key not in hf_map:
                print(f"  Warning: No mapping for {key}")
                continue
            hf_key = hf_map[key]
            hf_state_dict[hf_key] = value

    return hf_state_dict


def detect_qat_model(state_dict: dict) -> bool:
    """Detect if the state dict is from a QAT-wrapped model."""
    # QAT models have the extra .experts nesting
    for key in state_dict.keys():
        if ".moe.experts.experts." in key:
            return True
    return False


def load_distributed_checkpoint(checkpoint_path: str) -> dict:
    """Load a distributed checkpoint into a single state dict."""
    print(f"Loading distributed checkpoint from {checkpoint_path}...")

    from torch.distributed.checkpoint.filesystem import FileSystemReader

    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    print(f"Checkpoint contains {len(metadata.state_dict_metadata)} tensors")

    # Get model keys (exclude optimizer, train_state, lr_scheduler, dataloader)
    all_keys = list(metadata.state_dict_metadata.keys())
    model_keys = [k for k in all_keys
                  if not k.startswith('optimizer.')
                  and not k.startswith('train_state')
                  and not k.startswith('lr_scheduler')
                  and not k.startswith('dataloader')]

    print(f"Found {len(model_keys)} model keys")

    # Create a state dict with empty tensors to load into
    state_dict = {}
    for key in model_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        # Get dtype from tensor properties
        dtype = torch.bfloat16  # Default
        if hasattr(tensor_meta, 'properties') and tensor_meta.properties:
            dtype = tensor_meta.properties.dtype
        # Create empty tensor
        state_dict[key] = torch.empty(tensor_meta.size, dtype=dtype)

    # Load using dcp.load
    print("Loading tensors...")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    return state_dict


def convert_to_hf_safetensors(
    checkpoint_path: str,
    output_path: str,
    hf_base_path: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Convert checkpoint to HuggingFace format."""

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model args from the HF base
    config_path = os.path.join(hf_base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Load checkpoint
    state_dict = load_distributed_checkpoint(checkpoint_path)
    print(f"Loaded {len(state_dict)} tensors from checkpoint")

    # Detect if this is a QAT-wrapped model
    is_qat = detect_qat_model(state_dict)
    if is_qat:
        print("Detected QAT-wrapped model (extra .experts nesting)")

    # Convert to HF format (unquantized)
    print("Converting to HuggingFace format (unquantized)...")
    hf_state_dict = convert_to_hf_unquantized(state_dict, is_qat=is_qat)
    print(f"Converted to {len(hf_state_dict)} HF tensors")

    # Preserve original dtypes from checkpoint (don't force uniform dtype)
    # This maintains the exact precision used during training
    print(f"Preserving checkpoint dtypes (not converting to uniform {dtype})")
    dtype_dist = {}
    for key, tensor in hf_state_dict.items():
        dtype_str = str(tensor.dtype)
        dtype_dist[dtype_str] = dtype_dist.get(dtype_str, 0) + 1
    print("  Dtype distribution:")
    for dt, count in sorted(dtype_dist.items(), key=lambda x: -x[1]):
        print(f"    {dt}: {count} tensors")

    # Split into multiple files if needed (similar to original)
    # For ~20B model, we'll use 3 shards like the original
    total_size = sum(t.numel() * t.element_size() for t in hf_state_dict.values())
    print(f"Total model size: {total_size / 1e9:.2f} GB")

    # Save as safetensors (single file for simplicity, or split)
    if total_size > 5e9:  # Split if > 5GB
        # Split into shards
        shards = []
        current_shard = {}
        current_size = 0
        shard_size = total_size // 3 + 1

        for key, tensor in sorted(hf_state_dict.items()):
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            current_shard[key] = tensor
            current_size += tensor_size

        if current_shard:
            shards.append(current_shard)

        # Save each shard
        weight_map = {}
        for i, shard in enumerate(shards):
            shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            shard_path = output_dir / shard_name
            print(f"Saving {shard_path} ({len(shard)} tensors)...")
            save_file(shard, str(shard_path))
            for key in shard:
                weight_map[key] = shard_name

        # Save index
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Saved index to {index_path}")
    else:
        # Single file
        safetensors_path = output_dir / "model.safetensors"
        print(f"Saving {safetensors_path}...")
        save_file(hf_state_dict, str(safetensors_path))

    # Copy config and tokenizer files from base
    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]

    for filename in files_to_copy:
        src = os.path.join(hf_base_path, filename)
        if os.path.exists(src):
            dst = output_dir / filename
            shutil.copy2(src, dst)
            print(f"Copied {filename}")

    # Remove quantization_config from config.json since we're unquantized
    config_path = output_dir / "config.json"
    with open(config_path, "r") as f:
        config_data = json.load(f)
    if "quantization_config" in config_data:
        del config_data["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print("Removed quantization_config from config.json")

    print(f"\nConversion complete! Model saved to {output_dir}")
    print(f"You can load it with: AutoModelForCausalLM.from_pretrained('{output_dir}')")


def main():
    parser = argparse.ArgumentParser(description="Convert TorchTitan checkpoint to HF format")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to TorchTitan checkpoint directory (e.g., .../step-65)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for HF model"
    )
    parser.add_argument(
        "--hf-base",
        type=str,
        default="/mnt/models/gpt-oss-20b",
        help="Path to base HF model (for config/tokenizer)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Output dtype"
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    convert_to_hf_safetensors(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        hf_base_path=args.hf_base,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
