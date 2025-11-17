import argparse
import glob
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP

from huggingface_hub import snapshot_download
from safetensors import safe_open
from torchtitan.tools.logging import init_logger, logger
from tqdm import trange

from transformers import AutoConfig, AutoTokenizer  # noqa F401


@torch.inference_mode()
def convert_deepseekv3_weights(deepseek_model, output_dir):
    # Download the model files locally
    if os.path.exists(deepseek_model):
        local_path = deepseek_model
    else:
        local_path = snapshot_download(
            repo_id=deepseek_model,
            allow_patterns=["*.safetensors", "config.json", "tokenizer*", "*.py"],
        )
    tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    dim = config.hidden_size

    logger.info(
        f"Loading original DeepSeek V3 weights from {deepseek_model} using safetensors"
    )

    # Find all safetensors files
    safetensors_files = glob.glob(f"{local_path}/*.safetensors")
    if not safetensors_files:
        raise ValueError(
            "No safetensors files found in the downloaded model directory. Ensure the model uses safetensors format."
        )

    # Load state dict directly from safetensors files
    hf_state_dict = {}
    for file in safetensors_files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                logger.info(f"Read {key}, shape={tensor.shape}, dtype{tensor.dtype}")
                hf_state_dict[key] = tensor

    logger.info("Converting...")
    state_dict = {}
    for layer in trange(n_layers):
        state_dict[f"layers.{layer}.attention_norm.weight"] = hf_state_dict[
            f"model.layers.{layer}.input_layernorm.weight"
        ]
        state_dict[f"layers.{layer}.ffn_norm.weight"] = hf_state_dict[
            f"model.layers.{layer}.post_attention_layernorm.weight"
        ]

        # Attention weights
        if config.q_lora_rank is None:
            state_dict[f"layers.{layer}.attention.wq.weight"] = hf_state_dict[
                f"model.layers.{layer}.self_attn.q_proj.weight"
            ]
        else:
            state_dict[f"layers.{layer}.attention.wq_a.weight"] = hf_state_dict[
                f"model.layers.{layer}.self_attn.q_a_proj.weight"
            ]
            state_dict[f"layers.{layer}.attention.q_norm.weight"] = hf_state_dict[
                f"model.layers.{layer}.self_attn.q_a_layernorm.weight"
            ]
            state_dict[f"layers.{layer}.attention.wq_b.weight"] = hf_state_dict[
                f"model.layers.{layer}.self_attn.q_b_proj.weight"
            ]

        state_dict[f"layers.{layer}.attention.wkv_a.weight"] = hf_state_dict[
            f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
        ]
        state_dict[f"layers.{layer}.attention.kv_norm.weight"] = hf_state_dict[
            f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"
        ]
        state_dict[f"layers.{layer}.attention.wkv_b.weight"] = hf_state_dict[
            f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        ]
        state_dict[f"layers.{layer}.attention.wo.weight"] = hf_state_dict[
            f"model.layers.{layer}.self_attn.o_proj.weight"
        ]

        # MLP or MoE
        if layer < config.first_k_dense_replace:
            # Regular MLP (FeedForward)
            state_dict[f"layers.{layer}.feed_forward.w1.weight"] = hf_state_dict[
                f"model.layers.{layer}.mlp.gate_proj.weight"
            ]
            state_dict[f"layers.{layer}.feed_forward.w3.weight"] = hf_state_dict[
                f"model.layers.{layer}.mlp.up_proj.weight"
            ]
            state_dict[f"layers.{layer}.feed_forward.w2.weight"] = hf_state_dict[
                f"model.layers.{layer}.mlp.down_proj.weight"
            ]
        else:
            # MoE
            num_experts = config.n_routed_experts
            hidden_dim = config.moe_intermediate_size

            # Router
            state_dict[f"layers.{layer}.moe.router.gate.weight"] = hf_state_dict[
                f"model.layers.{layer}.mlp.gate.weight"
            ]

            # Experts
            w1_list = []
            w2_list = []
            w3_list = []
            for i in range(num_experts):
                w1_list.append(
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.experts.{i}.gate_proj.weight"
                    ]
                )
                w3_list.append(
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.experts.{i}.up_proj.weight"
                    ]
                )
                w2_list.append(
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.experts.{i}.down_proj.weight"
                    ]
                )

            state_dict[f"layers.{layer}.moe.experts.w1"] = torch.stack(w1_list)
            state_dict[f"layers.{layer}.moe.experts.w3"] = torch.stack(w3_list)
            state_dict[f"layers.{layer}.moe.experts.w2"] = torch.stack(w2_list)

            bias_key = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
            if bias_key in hf_state_dict:
                state_dict[f"layers.{layer}.moe.expert_bias"] = hf_state_dict[
                    bias_key
                ]
                # Ephemeral for training, but torchtitan expects it
                state_dict[f"layers.{layer}.moe.tokens_per_expert"] = torch.zeros(
                    num_experts, dtype=torch.float32
                )

            # Shared expert (if exists)
            if config.n_shared_experts is not None:
                state_dict[f"layers.{layer}.moe.shared_experts.w1.weight"] = (
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
                    ]
                )
                state_dict[f"layers.{layer}.moe.shared_experts.w3.weight"] = (
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.shared_experts.up_proj.weight"
                    ]
                )
                state_dict[f"layers.{layer}.moe.shared_experts.w2.weight"] = (
                    hf_state_dict[
                        f"model.layers.{layer}.mlp.shared_experts.down_proj.weight"
                    ]
                )

    state_dict["norm.weight"] = hf_state_dict["model.norm.weight"]
    state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"]
    state_dict["output.weight"] = hf_state_dict["lm_head.weight"]

    logger.info(f"Writing to DCP at '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir, thread_count=1)
    DCP.save(state_dict, storage_writer=storage_writer, no_dist=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek V3 weights to DCP format."
    )
    parser.add_argument(
        "deepseek_model", type=str, help="HF Model in DeepSeek V3 format"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    args = parser.parse_args()

    convert_deepseekv3_weights(args.deepseek_model, args.output_dir)