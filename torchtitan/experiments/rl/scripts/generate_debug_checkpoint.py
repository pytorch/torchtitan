# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate a fake HF-format checkpoint for torchtitan debug models that
have no real HuggingFace counterpart (e.g. debugmodel_moe).

This creates the minimal set of files needed by the RL loop:
  - config.json (for vLLM model config parsing)
  - tokenizer.json + tokenizer_config.json (copied from test assets)
  - model.safetensors (random weights in HF format)

Usage:
    python -m torchtitan.experiments.rl.scripts.generate_debug_checkpoint \
        --model_name debugmodel_moe \
        --output_dir torchtitan/experiments/rl/example_checkpoint/debugmodel_moe
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from torchtitan.models.qwen3 import model_registry


# Map from torchtitan model config to HF config.json fields.
# Only includes fields that vLLM actually reads.
def _build_hf_config(model_name: str, model_spec) -> dict:
    """Build a minimal HF config.json from the torchtitan model spec."""
    model_config = model_spec.model
    config = model_config.build_config if hasattr(model_config, "build_config") else None

    # Extract dims from the model config directly
    mc = model_spec.model
    layer0 = mc.layers[0]
    attn = layer0.attention

    hf_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": mc.dim,
        "num_hidden_layers": len(mc.layers),
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads or attn.n_heads,
        "head_dim": attn.head_dim or mc.dim // attn.n_heads,
        "vocab_size": mc.vocab_size,
        "rms_norm_eps": 1e-6,
        "rope_theta": mc.rope.theta,
        "max_position_embeddings": mc.rope.max_seq_len,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
    }

    # Add MoE fields if present
    if layer0.moe is not None:
        hf_config["architectures"] = ["Qwen3MoeForCausalLM"]
        hf_config["model_type"] = "qwen3_moe"
        hf_config["num_experts"] = layer0.moe.num_experts
        hf_config["num_experts_per_tok"] = layer0.moe.router.top_k
        hf_config["intermediate_size"] = layer0.moe.experts.hidden_dim

    # Dense MLP
    if layer0.feed_forward is not None:
        ff = layer0.feed_forward
        hf_config["intermediate_size"] = ff.w1.out_features

    return hf_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fake HF checkpoint for a torchtitan debug model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name from qwen3 model registry (e.g. debugmodel, debugmodel_moe)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the generated checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tests/assets/tokenizer",
        help="Path to tokenizer files to copy",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build model and get random weights
    print(f"Building model '{args.model_name}'...")
    spec = model_registry(args.model_name)
    model = spec.model.build()
    model.init_weights(buffer_device=None)

    # Convert to HF format via state_dict_adapter
    print("Converting state dict to HF format...")
    sd_adapter = spec.state_dict_adapter(spec.model, None)
    hf_sd = sd_adapter.to_hf(model.state_dict())

    # Save weights
    weights_path = os.path.join(args.output_dir, "model.safetensors")
    save_file({k: v for k, v in hf_sd.items()}, weights_path)
    print(f"Saved weights to {weights_path}")

    # Save config.json
    config_path = os.path.join(args.output_dir, "config.json")
    hf_config = _build_hf_config(args.model_name, spec)
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Copy tokenizer files
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(args.tokenizer_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            print(f"Copied {fname}")
        else:
            print(f"Warning: {src} not found, skipping")

    print(f"\nCheckpoint ready at: {args.output_dir}")
    print(f"Use with: --hf_assets_path {args.output_dir}")


if __name__ == "__main__":
    main()
