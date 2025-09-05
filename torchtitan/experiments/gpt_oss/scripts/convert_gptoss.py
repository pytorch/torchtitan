"""
Convert checkpoints between TorchTitan and HuggingFace.

# Convert HF to TorchTitan DCP
uv run torchtitan/experiments/gpt_oss/scripts/convert_gptoss.py hf-to-dcp --input-path  openai/gpt-oss-20b --output-path gptoss_dcp/

# Convert TorchTitan DCP to HF
uv run torchtitan/experiments/gpt_oss/scripts/convert_gptoss.py dcp-to-hf --input-path gptoss_dcp/ --output-path gptoss_hf/
"""

import re
import tempfile
from pathlib import Path
from typing import Union, Tuple, Optional

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torchtitan.datasets.transformation import get_tokenizer_with_chat_template
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from torchtitan.models.llama3.model import precompute_freqs_cis
from tqdm import tqdm
from tyro.extras import SubcommandApp

from torchtitan.tools.logging import init_logger, logger

app = SubcommandApp()



def validate_hf_keys(hf_state_dict, model_config, model_name):
    """Validate that all expected weight keys exist in the HF state dict."""
    missing_keys = []
    n_layers = model_config.num_hidden_layers

    # Check basic weights
    required_keys = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight"
    ]

    for key in required_keys:
        if key not in hf_state_dict:
            missing_keys.append(key)

    # Check layer weights
    for layer_idx in range(n_layers):
        layer_prefix = f'model.layers.{layer_idx}'

        # Check attention weights
        attention_keys = [
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
            f"{layer_prefix}.self_attn.o_proj.weight",
            f"{layer_prefix}.self_attn.q_proj.bias",
            f"{layer_prefix}.self_attn.k_proj.bias",
            f"{layer_prefix}.self_attn.v_proj.bias",
            f"{layer_prefix}.self_attn.o_proj.bias",
            f"{layer_prefix}.input_layernorm.weight",
            f"{layer_prefix}.post_attention_layernorm.weight",
        ]

        for key in attention_keys:
            if key not in hf_state_dict:
                missing_keys.append(key)

        # Check MoE weights
        mlp_keys = [
            f"{layer_prefix}.mlp.router.weight",
            f"{layer_prefix}.mlp.router.bias",
            f"{layer_prefix}.mlp.experts.gate_up_proj",
            f"{layer_prefix}.mlp.experts.gate_up_proj_bias",
            f"{layer_prefix}.mlp.experts.down_proj",
            f"{layer_prefix}.mlp.experts.down_proj_bias",
        ]

        for key in mlp_keys:
            if key not in hf_state_dict:
                missing_keys.append(key)

    if missing_keys:
        logger.error(f"Missing {len(missing_keys)} expected weight keys in HF model:")
        for key in missing_keys[:10]:  # Show first 10
            logger.error(f"  - {key}")
        if len(missing_keys) > 10:
            logger.error(f"  ... and {len(missing_keys) - 10} more")

        # Try to diagnose the issue
        logger.info("Available keys in HF model:")
        available_keys = list(hf_state_dict.keys())
        for key in available_keys[:20]:  # Show first 20
            logger.info(f"  - {key}")
        if len(available_keys) > 20:
            logger.info(f"  ... and {len(available_keys) - 20} more")

        raise ValueError(f"HF model '{model_name}' is missing expected weight keys. "
                        f"This suggests the model architecture doesn't match expectations.")

    logger.info(f"✓ Weight key validation passed - found all expected keys")


def map_hf_to_torchtitan(hf_state_dict, model_config, max_seq_len=131072, rope_theta=500000.0, model_name="meta-llama/Llama-3.1-8B"):
    """Map HuggingFace state dict to TorchTitan format.

    Note: TorchTitan and HuggingFace use different RoPE implementations:
    - TorchTitan: Adjacent element pairing with complex arithmetic
    - HuggingFace: First/second half pairing with cos/sin arithmetic

    This difference is architectural, not a bug. Converted models will have
    slightly different positional encoding but typically minimal impact on performance.
    """

    # Validate that all expected keys exist
    validate_hf_keys(hf_state_dict, model_config, model_name)

    n_layers = model_config.num_hidden_layers
    n_heads = model_config.num_attention_heads
    dim = model_config.hidden_size
    dims_per_head = dim // n_heads

    # Fix: Corrected model family detection logic
    if "llama" in model_name.lower():
        model_family = "llama3"
    elif "qwen" in model_name.lower():
        model_family = "qwen3"
        max_seq_len = model_config.max_position_embeddings
        rope_theta = model_config.rope_theta
    elif "gpt-oss" in model_name.lower():
        model_family = "gptoss"
        max_seq_len = model_config.max_position_embeddings
        rope_theta = model_config.rope_theta
    else:
        raise ValueError(f"Unsupported HuggingFace model for conversion: {model_name}")

    # Determine n_kv_heads for GQA models
    n_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    print(f"Model info: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, model_family={model_family}, max_seq_len={max_seq_len}, rope_theta={rope_theta}")
    torchtitan_state_dict = {}

    # Convert embeddings and output
    torchtitan_state_dict["tok_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"].clone()
    torchtitan_state_dict["output.weight"] = hf_state_dict["lm_head.weight"].clone()
    torchtitan_state_dict["norm.weight"] = hf_state_dict["model.norm.weight"].clone()

    def permute(w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Convert layers
    for layer_idx in tqdm(range(n_layers), desc="Converting layers"):
        hf_layer_prefix = f'model.layers.{layer_idx}'
        layer_prefix = f'layers.{layer_idx}'

        wq = hf_state_dict[f'{hf_layer_prefix}.self_attn.q_proj.weight']
        torchtitan_state_dict[f'{layer_prefix}.attention.wq.weight'] = wq.clone()
        wq_bias = hf_state_dict[f'{hf_layer_prefix}.self_attn.q_proj.bias']
        torchtitan_state_dict[f'{layer_prefix}.attention.wq.bias'] = wq_bias.clone()

        wk = hf_state_dict[f'{hf_layer_prefix}.self_attn.k_proj.weight']
        torchtitan_state_dict[f'{layer_prefix}.attention.wk.weight'] = wk.clone()
        wk_bias = hf_state_dict[f'{hf_layer_prefix}.self_attn.k_proj.bias']
        torchtitan_state_dict[f'{layer_prefix}.attention.wk.bias'] = wk_bias.clone()

        wv = hf_state_dict[f'{hf_layer_prefix}.self_attn.v_proj.weight']
        torchtitan_state_dict[f'{layer_prefix}.attention.wv.weight'] = wv.clone()
        wv_bias = hf_state_dict[f'{hf_layer_prefix}.self_attn.v_proj.bias']
        torchtitan_state_dict[f'{layer_prefix}.attention.wv.bias'] = wv_bias.clone()

        wo = hf_state_dict[f'{hf_layer_prefix}.self_attn.o_proj.weight']
        torchtitan_state_dict[f'{layer_prefix}.attention.wo.weight'] = wo.clone()
        wo_bias = hf_state_dict[f'{hf_layer_prefix}.self_attn.o_proj.bias']
        torchtitan_state_dict[f'{layer_prefix}.attention.wo.bias'] = wo_bias.clone()

        sinks = hf_state_dict[f'{hf_layer_prefix}.self_attn.sinks']
        torchtitan_state_dict[f'{layer_prefix}.attention.sinks'] = sinks.clone()

        # MoE weights
        mlp1 = hf_state_dict[f'{hf_layer_prefix}.mlp.experts.gate_up_proj']
        torchtitan_state_dict[f'{layer_prefix}.moe.experts.mlp1_weight'] = mlp1.clone()

        mlp1_bias = hf_state_dict[f'{hf_layer_prefix}.mlp.experts.gate_up_proj_bias']
        torchtitan_state_dict[f'{layer_prefix}.moe.experts.mlp1_bias'] = mlp1_bias.clone()

        mlp2 = hf_state_dict[f'{hf_layer_prefix}.mlp.experts.down_proj']
        torchtitan_state_dict[f'{layer_prefix}.moe.experts.mlp2_weight'] = mlp2.clone()

        mlp2_bias = hf_state_dict[f'{hf_layer_prefix}.mlp.experts.down_proj_bias']
        torchtitan_state_dict[f'{layer_prefix}.moe.experts.mlp2_bias'] = mlp2_bias.clone()

        # router
        gate = hf_state_dict[f'{hf_layer_prefix}.mlp.router.weight']
        torchtitan_state_dict[f'{layer_prefix}.moe.router.gate.weight'] = gate.clone()
        router_bias = hf_state_dict[f'{hf_layer_prefix}.mlp.router.bias']
        torchtitan_state_dict[f'{layer_prefix}.moe.router.gate.bias'] = router_bias.clone()

        # # @vwxyzjn: This is technically not needed, but we added here because we haven't figured out
        # # how to tell torchtitan to ignore this parameter.
        # tokens_per_expert = torch.zeros_like(expert_bias)
        # torchtitan_state_dict[f'{layer_prefix}.moe.tokens_per_expert'] = tokens_per_expert.clone()

        # Layer norms
        attention_norm = hf_state_dict[f'{hf_layer_prefix}.input_layernorm.weight']
        torchtitan_state_dict[f'{layer_prefix}.attention_norm.weight'] = attention_norm.clone()
        ffn_norm = hf_state_dict[f'{hf_layer_prefix}.post_attention_layernorm.weight']
        torchtitan_state_dict[f'{layer_prefix}.ffn_norm.weight'] = ffn_norm.clone()

    # Precompute RoPE frequencies
    # NOTE: we no longer precompute RoPE frequencies in TorchTitan
    # this `model_config` is HF but needs to be TT (to include e.g. beta_fast)
    # torchtitan_state_dict["freqs_cis"] = precompute_freqs_cis(model_config)

    print(f"Converted {len(torchtitan_state_dict)} parameters from HuggingFace to TorchTitan format")
    return torchtitan_state_dict


def map_torchtitan_to_hf_per_param(name: str, weight: torch.Tensor, model_family: str = "llama3") -> Tuple[Optional[str], Optional[torch.Tensor]]:
    """Map a single TorchTitan parameter to HuggingFace format.

    Args:
        name: Parameter name in TorchTitan format
        weight: Parameter tensor
        model_family: Model family ("llama3", "qwen3", or "gptoss")

    Returns:
        Tuple of (hf_name, hf_weight) or (None, None) if parameter should be skipped
    """
    # Skip freqs_cis as it's computed dynamically in HF
    if name == "freqs_cis":
        return None, None

    assert model_family in ("llama3", "qwen3", "gptoss"), f"Unsupported model family: {model_family}"

    # HuggingFace permutation function (exact copy from their conversion script)
    def permute(w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Handle embeddings and output weights
    if name == "tok_embeddings.weight":
        return "model.embed_tokens.weight", weight.clone()
    elif name == "output.weight":
        return "lm_head.weight", weight.clone()
    elif name == "norm.weight":
        return "model.norm.weight", weight.clone()

    # Handle layer-specific parameters
    layer_match = re.match(r"layers\.(\d+)\.", name)
    if not layer_match:
        return None, None

    layer_idx = layer_match.group(1)
    layer_suffix = name[len(f"layers.{layer_idx}."):]
    hf_layer_prefix = f"model.layers.{layer_idx}"

    if model_family == "gptoss":
        mapping = {
            "attention.wq.weight": "self_attn.q_proj.weight",
            "attention.wq.bias": "self_attn.q_proj.bias",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wk.bias": "self_attn.k_proj.bias",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wv.bias": "self_attn.v_proj.bias",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.wo.bias": "self_attn.o_proj.bias",
            "attention.sinks": "self_attn.sinks",
            "moe.experts.mlp1_weight": "mlp.experts.gate_up_proj",
            "moe.experts.mlp1_bias": "mlp.experts.gate_up_proj_bias",
            "moe.experts.mlp2_weight": "mlp.experts.down_proj",
            "moe.experts.mlp2_bias": "mlp.experts.down_proj_bias",
            "moe.router.gate.weight": "mlp.router.weight",
            "moe.router.gate.bias": "mlp.router.bias",
            "moe.expert_bias": "mlp.router.bias", # NOTE: this gets added into router bias
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        hf_suffix = mapping.get(layer_suffix)
        if hf_suffix:
            return f"{hf_layer_prefix}.{hf_suffix}", weight.clone()
        return None, None

    # Handle attention weights
    if layer_suffix == "attention.wq.weight":
        if model_family == "llama3":
            # For query weights, assume standard head_dim=128
            dim = weight.shape[1]
            head_dim = 128
            n_heads = dim // head_dim
            transformed_weight = permute(weight, n_heads)
        else:
            transformed_weight = weight
        return f"{hf_layer_prefix}.self_attn.q_proj.weight", transformed_weight.clone()

    elif layer_suffix == "attention.wk.weight":
        if model_family == "llama3":
            # For key weights, infer n_kv_heads from weight shape
            dim = weight.shape[1]
            head_dim = 128
            n_kv_heads = weight.shape[0] // head_dim
            key_value_dim = n_kv_heads * head_dim
            transformed_weight = permute(weight, n_kv_heads, key_value_dim, dim)
        else:
            transformed_weight = weight
        return f"{hf_layer_prefix}.self_attn.k_proj.weight", transformed_weight.clone()

    elif layer_suffix == "attention.wv.weight":
        return f"{hf_layer_prefix}.self_attn.v_proj.weight", weight.clone()

    elif layer_suffix == "attention.wo.weight":
        return f"{hf_layer_prefix}.self_attn.o_proj.weight", weight.clone()

    # Handle qwen3-specific attention norms
    elif layer_suffix == "attention.q_norm.weight" and model_family == "qwen3":
        return f"{hf_layer_prefix}.self_attn.q_norm.weight", weight.clone()

    elif layer_suffix == "attention.k_norm.weight" and model_family == "qwen3":
        return f"{hf_layer_prefix}.self_attn.k_norm.weight", weight.clone()

    # Handle MLP weights
    elif layer_suffix == "feed_forward.w1.weight":
        return f"{hf_layer_prefix}.mlp.gate_proj.weight", weight.clone()

    elif layer_suffix == "feed_forward.w2.weight":
        return f"{hf_layer_prefix}.mlp.down_proj.weight", weight.clone()

    elif layer_suffix == "feed_forward.w3.weight":
        return f"{hf_layer_prefix}.mlp.up_proj.weight", weight.clone()

    # Handle layer norms
    elif layer_suffix == "attention_norm.weight":
        return f"{hf_layer_prefix}.input_layernorm.weight", weight.clone()

    elif layer_suffix == "ffn_norm.weight":
        return f"{hf_layer_prefix}.post_attention_layernorm.weight", weight.clone()

    # If no mapping found, return None
    return None, None


def map_torchtitan_to_hf(torchtitan_state_dict, max_seq_len=131072, rope_theta=500000.0):
    """Map TorchTitan state dict to HuggingFace format."""
    if any(k.endswith('.attention.q_norm.weight') for k in torchtitan_state_dict):
        model_family = 'qwen3'
    elif any(k.endswith('.attention.wq.bias') for k in torchtitan_state_dict):
        model_family = 'gptoss'
    else:
        model_family = 'llama3'

    layer_keys = [k for k in torchtitan_state_dict.keys() if k.startswith("layers.")]
    assert len(layer_keys) > 0, "No layers found in state dict"
    n_layers = max([int(k.split(".")[1]) for k in layer_keys]) + 1
    hf_state_dict = {}

    # Get model info from sample weight
    sample_wq_key = next(k for k in torchtitan_state_dict.keys() if k.endswith('.attention.wq.weight'))
    wq_weight = torchtitan_state_dict[sample_wq_key]
    dim = wq_weight.shape[1]  # input dimension

    # Check if we have a key weight to determine n_kv_heads
    sample_wk_key = next(k for k in torchtitan_state_dict.keys() if k.endswith('.attention.wk.weight'))
    wk_weight = torchtitan_state_dict[sample_wk_key]

    # Standard Llama head dim is 128 for the 3B, 8B, 70B and 405B models
    # NOTE: The only exception is the 1B model: https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json#L9
    # But let's ignore that for now
    head_dim = 128
    n_heads = dim // head_dim

    # For GQA models, n_kv_heads might be different
    n_kv_heads = wk_weight.shape[0] // head_dim

    print(f"Model info: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, model_family={model_family}")

    # HuggingFace permutation function (exact copy from their conversion script)
    def permute(w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Convert embeddings and output (no permutation needed)
    if 'tok_embeddings.weight' in torchtitan_state_dict:
        hf_state_dict['model.embed_tokens.weight'] = torchtitan_state_dict['tok_embeddings.weight'].clone()
    if 'output.weight' in torchtitan_state_dict:
        hf_state_dict['lm_head.weight'] = torchtitan_state_dict['output.weight'].clone()
    if 'norm.weight' in torchtitan_state_dict:
        hf_state_dict['model.norm.weight'] = torchtitan_state_dict['norm.weight'].clone()

    # Convert layers
    for layer_idx in tqdm(range(n_layers), desc="Converting layers"):
        layer_prefix = f'layers.{layer_idx}'
        hf_layer_prefix = f'model.layers.{layer_idx}'

        if model_family == 'gptoss':
            # Attention projections and biases
            mappings = {
                f'{layer_prefix}.attention.wq.weight': f'{hf_layer_prefix}.self_attn.q_proj.weight',
                f'{layer_prefix}.attention.wq.bias': f'{hf_layer_prefix}.self_attn.q_proj.bias',
                f'{layer_prefix}.attention.wk.weight': f'{hf_layer_prefix}.self_attn.k_proj.weight',
                f'{layer_prefix}.attention.wk.bias': f'{hf_layer_prefix}.self_attn.k_proj.bias',
                f'{layer_prefix}.attention.wv.weight': f'{hf_layer_prefix}.self_attn.v_proj.weight',
                f'{layer_prefix}.attention.wv.bias': f'{hf_layer_prefix}.self_attn.v_proj.bias',
                f'{layer_prefix}.attention.wo.weight': f'{hf_layer_prefix}.self_attn.o_proj.weight',
                f'{layer_prefix}.attention.wo.bias': f'{hf_layer_prefix}.self_attn.o_proj.bias',
                f'{layer_prefix}.attention.sinks': f'{hf_layer_prefix}.self_attn.sinks',
                f'{layer_prefix}.moe.experts.mlp1_weight': f'{hf_layer_prefix}.mlp.experts.gate_up_proj',
                f'{layer_prefix}.moe.experts.mlp1_bias': f'{hf_layer_prefix}.mlp.experts.gate_up_proj_bias',
                f'{layer_prefix}.moe.experts.mlp2_weight': f'{hf_layer_prefix}.mlp.experts.down_proj',
                f'{layer_prefix}.moe.experts.mlp2_bias': f'{hf_layer_prefix}.mlp.experts.down_proj_bias',
                f'{layer_prefix}.moe.router.gate.weight': f'{hf_layer_prefix}.mlp.router.weight',
                f'{layer_prefix}.attention_norm.weight': f'{hf_layer_prefix}.input_layernorm.weight',
                f'{layer_prefix}.ffn_norm.weight': f'{hf_layer_prefix}.post_attention_layernorm.weight',
            }
            for tt_key, hf_key in mappings.items():
                if tt_key in torchtitan_state_dict:
                    hf_state_dict[hf_key] = torchtitan_state_dict[tt_key].clone()
            # Combine router gate bias with expert bias (if present)
            router_bias_key = f'{layer_prefix}.moe.router.gate.bias'
            expert_bias_key = f'{layer_prefix}.moe.expert_bias'
            if (
                router_bias_key in torchtitan_state_dict
                or expert_bias_key in torchtitan_state_dict
            ):
                if router_bias_key in torchtitan_state_dict:
                    bias = torchtitan_state_dict[router_bias_key].clone()
                else:
                    bias = torch.zeros_like(torchtitan_state_dict[expert_bias_key])
                if expert_bias_key in torchtitan_state_dict:
                    bias = bias + torchtitan_state_dict[expert_bias_key]
                hf_state_dict[f'{hf_layer_prefix}.mlp.router.bias'] = bias
            continue

        # Attention weights with proper permutation
        if f'{layer_prefix}.attention.wq.weight' in torchtitan_state_dict:
            wq = torchtitan_state_dict[f'{layer_prefix}.attention.wq.weight']
            if model_family == "llama3":
                wq = permute(wq, n_heads)
            hf_state_dict[f'{hf_layer_prefix}.self_attn.q_proj.weight'] = wq.clone()

        if f'{layer_prefix}.attention.wk.weight' in torchtitan_state_dict:
            wk = torchtitan_state_dict[f'{layer_prefix}.attention.wk.weight']
            key_value_dim = n_kv_heads * head_dim
            if model_family == "llama3":
                wk = permute(wk, n_kv_heads, key_value_dim, dim)
            hf_state_dict[f'{hf_layer_prefix}.self_attn.k_proj.weight'] = wk.clone()

        if f'{layer_prefix}.attention.wv.weight' in torchtitan_state_dict:
            # Value weights don't get permuted
            hf_state_dict[f'{hf_layer_prefix}.self_attn.v_proj.weight'] = torchtitan_state_dict[f'{layer_prefix}.attention.wv.weight'].clone()

        if model_family == "qwen3":
            if f'{layer_prefix}.attention.q_norm.weight' in torchtitan_state_dict:
                hf_state_dict[f'{hf_layer_prefix}.self_attn.q_norm.weight'] = torchtitan_state_dict[f'{layer_prefix}.attention.q_norm.weight'].clone()
            if f'{layer_prefix}.attention.k_norm.weight' in torchtitan_state_dict:
                hf_state_dict[f'{hf_layer_prefix}.self_attn.k_norm.weight'] = torchtitan_state_dict[f'{layer_prefix}.attention.k_norm.weight'].clone()

        if f'{layer_prefix}.attention.wo.weight' in torchtitan_state_dict:
            # Output projection doesn't get permuted
            hf_state_dict[f'{hf_layer_prefix}.self_attn.o_proj.weight'] = torchtitan_state_dict[f'{layer_prefix}.attention.wo.weight'].clone()

        # MLP weights (no permutation)
        mlp_mappings = {
            f'{layer_prefix}.feed_forward.w1.weight': f'{hf_layer_prefix}.mlp.gate_proj.weight',
            f'{layer_prefix}.feed_forward.w2.weight': f'{hf_layer_prefix}.mlp.down_proj.weight',
            f'{layer_prefix}.feed_forward.w3.weight': f'{hf_layer_prefix}.mlp.up_proj.weight',
        }

        for tt_key, hf_key in mlp_mappings.items():
            if tt_key in torchtitan_state_dict:
                hf_state_dict[hf_key] = torchtitan_state_dict[tt_key].clone()

        # Layer norms (no permutation)
        norm_mappings = {
            f'{layer_prefix}.attention_norm.weight': f'{hf_layer_prefix}.input_layernorm.weight',
            f'{layer_prefix}.ffn_norm.weight': f'{hf_layer_prefix}.post_attention_layernorm.weight',
        }

        for tt_key, hf_key in norm_mappings.items():
            if tt_key in torchtitan_state_dict:
                hf_state_dict[hf_key] = torchtitan_state_dict[tt_key].clone()

    print(f"Converted {len(hf_state_dict)} parameters from TorchTitan to HuggingFace format")
    return hf_state_dict


def map_torchtitan_to_hf2(torchtitan_state_dict, max_seq_len=131072, rope_theta=500000.0, validate_against_original=True):
    """Map TorchTitan state dict to HuggingFace format using per-parameter function."""

    # Auto-detect model family
    if any(k.endswith('.attention.q_norm.weight') for k in torchtitan_state_dict):
        model_family = "qwen3"
    elif any(k.endswith('.attention.wq.bias') for k in torchtitan_state_dict):
        model_family = "gptoss"
    else:
        model_family = "llama3"

    logger.info(f"Converting using per-parameter function with model_family={model_family}")

    hf_state_dict = {}
    skipped_params = []

    # Convert each parameter individually
    for name, weight in tqdm(torchtitan_state_dict.items(), desc="Converting parameters"):
        hf_name, hf_weight = map_torchtitan_to_hf_per_param(name, weight, model_family)
        if hf_name is not None:
            if hf_name in hf_state_dict:
                hf_state_dict[hf_name] = hf_state_dict[hf_name] + hf_weight # NOTE: adds expert_bias into router bias
            else:
                hf_state_dict[hf_name] = hf_weight
        else:
            skipped_params.append(name)

    logger.info(f"Converted {len(hf_state_dict)} parameters, skipped {len(skipped_params)} parameters")
    if skipped_params:
        logger.info(f"Skipped parameters: {skipped_params}")

    # Validation against original function
    if validate_against_original:
        logger.info("Validating against original conversion function...")

        # Get original result
        original_hf_state_dict = map_torchtitan_to_hf(torchtitan_state_dict, max_seq_len, rope_theta)

        # Compare keys
        new_keys = set(hf_state_dict.keys())
        original_keys = set(original_hf_state_dict.keys())

        if new_keys != original_keys:
            missing_in_new = original_keys - new_keys
            extra_in_new = new_keys - original_keys
            logger.error(f"Key mismatch! Missing in new: {missing_in_new}, Extra in new: {extra_in_new}")
            raise ValueError("Key sets don't match between implementations")

        # Compare tensor values
        mismatched_tensors = []
        for key in original_keys:
            if not torch.allclose(hf_state_dict[key], original_hf_state_dict[key], rtol=1e-5, atol=1e-8):
                mismatched_tensors.append(key)

        if mismatched_tensors:
            logger.error(f"Tensor value mismatches in: {mismatched_tensors}")
            # Show details for first mismatch
            key = mismatched_tensors[0]
            logger.error(f"First mismatch in {key}:")
            logger.error(f"  Max abs diff: {torch.max(torch.abs(hf_state_dict[key] - original_hf_state_dict[key]))}")
            logger.error(f"  Original shape: {original_hf_state_dict[key].shape}")
            logger.error(f"  New shape: {hf_state_dict[key].shape}")
            raise ValueError("Tensor values don't match between implementations")

        logger.info("✓ Validation passed! New implementation matches original.")

    return hf_state_dict


@app.command(name="hf_to_dcp")
@torch.inference_mode()
def convert_hf_to_dcp(input_path: str, output_path: Path, max_seq_len: int = 131072, rope_theta: float = 500000.0, dtype: str = "auto"):
    """Convert HuggingFace model to TorchTitan DCP format.

    Args:
        input_path: HuggingFace model name or path
        output_path: Output DCP checkpoint path
        max_seq_len: Max sequence length for RoPE
        rope_theta: RoPE theta parameter
        dtype: Data type to use ("auto" to preserve original, or specific dtype like "float32")
    """
    logger.info(f"Loading model from {input_path}")

    # Load model with original dtype if "auto", otherwise use specified dtype
    hf_model = AutoModelForCausalLM.from_pretrained(input_path, torch_dtype=torch.bfloat16)

    hf_state_dict = hf_model.state_dict()
    logger.info(f"Loaded model with dtype: {next(iter(hf_state_dict.values())).dtype}")

    logger.info("Converting weights to TorchTitan format")
    torchtitan_state_dict = map_hf_to_torchtitan(hf_state_dict, hf_model.config, max_seq_len, rope_theta, input_path)

    logger.info(f"Writing to DCP at '{output_path}'")
    output_path.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_path, thread_count=8)
    DCP.save({"model": torchtitan_state_dict}, storage_writer=storage_writer)
    logger.info("Conversion complete!")


@app.command(name="dcp_to_hf")
@torch.inference_mode()
def convert_dcp_to_hf(input_path: str, output_path: Path, max_seq_len: int = 131072, rope_theta: float = 500000.0, default_model: str = "meta-llama/Meta-Llama-3.1-8B", validate_against_original: bool = False):
    """Convert TorchTitan DCP format to HuggingFace model.

    Args:
        input_path: Input DCP checkpoint path
        output_path: Output HuggingFace model path
        max_seq_len: Max sequence length for RoPE
        rope_theta: RoPE theta parameter
        default_model: Default HuggingFace model for config
    """

    if str(input_path).startswith("s3://"):
        import s3_utils
        local_path = s3_utils.sync_to_nvme(str(input_path))
        input_path = Path(local_path)

    logger.info(f"Loading DCP checkpoint from {input_path}")

    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    # Load DCP input_path
    state_dict = {}
    _load_state_dict(
        state_dict,
        storage_reader=DCP.filesystem.FileSystemReader(input_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    torchtitan_state_dict = state_dict["model"]
    logger.info("Converting weights to HuggingFace format")
    hf_state_dict = map_torchtitan_to_hf2(torchtitan_state_dict, max_seq_len, rope_theta, validate_against_original=validate_against_original)

    if '/' not in default_model:
        if 'qwen' in default_model.lower():
            default_model = f'Qwen/{default_model}'
        elif 'llama' in default_model.lower():
            default_model = f'meta-llama/{default_model}'
        else:
            raise ValueError(f"Unsupported model: {default_model}")

    # Create HuggingFace config
    hf_config = AutoConfig.from_pretrained(default_model)

    # Create and load model
    logger.info("Creating HuggingFace model")
    # tokenizer = AutoTokenizer.from_pretrained(default_model)
    tokenizer = get_tokenizer_with_chat_template(default_model, "tulu", override=True)
    hf_model = AutoModelForCausalLM.from_pretrained(default_model, device_map="auto") # NOTE: need device_map="auto" to avoid CPU OOM

    # load state dict
    logger.info("Loading state dict")
    hf_model.load_state_dict(hf_state_dict, strict=True)

    # Save model
    logger.info(f"Saving model to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Conversion complete!")


if __name__ == "__main__":
    init_logger()
    app.cli()
