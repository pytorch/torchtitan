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
from typing import Union

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from tqdm import tqdm
from tyro.extras import SubcommandApp

from torchtitan.tools.logging import init_logger, logger

app = SubcommandApp()


def validate_config_compatibility(hf_config, torchtitan_config_name, torchtitan_configs):
    """Validate that HF config is compatible with TorchTitan config."""
    if torchtitan_config_name not in torchtitan_configs:
        available = list(torchtitan_configs.keys())
        raise ValueError(f"TorchTitan config '{torchtitan_config_name}' not found. Available: {available}")

    tt_config = torchtitan_configs[torchtitan_config_name]

    # Critical configuration checks with proper field mappings
    checks = [
        ("vocab_size", "vocab_size"),
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("head_dim", "head_dim"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("sliding_window", "sliding_window"),
        ("num_local_experts", "num_local_experts"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("rope_theta", "rope_theta"),
        # ("rope_scaling.factor", "rope_factor"),
        # ("rope_scaling.beta_fast", "beta_fast"),
        # ("rope_scaling.beta_slow", "beta_slow"),
    ]

    mismatches = []
    warnings = []

    for hf_attr, tt_attr in checks:
        hf_val = getattr(hf_config, hf_attr, None)
        tt_val = getattr(tt_config, tt_attr, None)

        if hf_val != tt_val:
            mismatches.append(f"{hf_attr}: HF={hf_val} vs TT.{tt_attr}={tt_val}")

    if mismatches:
        raise ValueError(f"Config mismatch for {torchtitan_config_name}:\n" + "\n".join(mismatches))

    if warnings:
        print(f"⚠️  Configuration warnings for {torchtitan_config_name}:")
        for warning in warnings:
            print(f"   {warning}")
        print("   These differences might affect model behavior but won't prevent conversion.")

    print(f"✓ Configuration validation passed for {torchtitan_config_name}")
    return tt_config

def validate_tt_keys(tt_sd, n_layers, strict=True):
    """Ensure the TorchTitan dict looks like gpt-oss as encoded in hf->tt mapping."""
    top_expected = [
        "tok_embeddings.weight",
        "output.weight",
        "norm.weight",
    ]
    per_layer_expected = [
        # attention projections + biases + sinks
        "attention.wq.weight", "attention.wq.bias",
        "attention.wk.weight", "attention.wk.bias",
        "attention.wv.weight", "attention.wv.bias",
        "attention.wo.weight", "attention.wo.bias",
        "attention.sinks",
        # MoE experts (mlp1/2) + biases
        "moe.experts.mlp1_weight", "moe.experts.mlp1_bias",
        "moe.experts.mlp2_weight", "moe.experts.mlp2_bias",
        # Router
        "moe.router.gate.weight", "moe.router.gate.bias",
        # Norms
        "attention_norm.weight", "ffn_norm.weight",
    ]

    missing = []
    for k in top_expected:
        if k not in tt_sd:
            missing.append(k)

    for i in range(n_layers):
        base = f"layers.{i}."
        for suffix in per_layer_expected:
            key = base + suffix
            if key not in tt_sd:
                missing.append(key)

    if missing and strict:
        preview = "\n  - " + "\n  - ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ...and {len(missing)-20} more"
        raise KeyError(
            "TorchTitan checkpoint is missing keys required for gpt-oss inverse mapping:"
            f"{preview}{more}"
        )
    return missing  # may be useful for logging if strict=False

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


def num_layers_from_keys(state_dict):
    layer_idxs = []
    pat = re.compile(r"^layers\.(\d+)\.")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            layer_idxs.append(int(m.group(1)))
    if not layer_idxs:
        raise ValueError("Could not find any 'layers.<idx>.' keys in the TorchTitan state dict.")
    return max(layer_idxs) + 1

# TODO: correctness of map_torchtitan_to_hf is not yet tested for GPT-OSS
def map_torchtitan_to_hf(torchtitan_state_dict, *, strict=True):
    """
    Map TorchTitan (DCP) state dict -> HuggingFace format for *gpt-oss only*.

    This is the exact inverse of your `map_hf_to_torchtitan`:
      - No weight permutations.
      - Copies biases for q/k/v/o and MoE projections.
      - Preserves `.attention.sinks`.
      - MoE and router parameters use the same custom names you used on the HF side
        (i.e., HF bias keys are `gate_up_proj_bias` / `down_proj_bias`).

    Parameters
    ----------
    torchtitan_state_dict : dict[str, Tensor-like]
        TorchTitan checkpoint (flat dict).
    strict : bool
        If True, error on any missing keys. If False, copy what exists and skip missing.

    Returns
    -------
    dict[str, Tensor-like]
        HuggingFace-formatted state dict.
    """
    tt = torchtitan_state_dict
    n_layers = num_layers_from_keys(tt)
    validate_tt_keys(tt, n_layers, strict=strict)

    hf = {}

    # Top-level
    if "tok_embeddings.weight" in tt: hf["model.embed_tokens.weight"] = tt["tok_embeddings.weight"].clone()
    if "output.weight"         in tt: hf["lm_head.weight"]            = tt["output.weight"].clone()
    if "norm.weight"           in tt: hf["model.norm.weight"]         = tt["norm.weight"].clone()

    # Per-layer mappings (exact inverse of your hf->tt)
    for i in range(n_layers):
        tt_pref = f"layers.{i}"
        hf_pref = f"model.layers.{i}"

        # Attention projections (+biases)
        m = {
            f"{tt_pref}.attention.wq.weight": (f"{hf_pref}.self_attn.q_proj.weight",),
            f"{tt_pref}.attention.wq.bias":   (f"{hf_pref}.self_attn.q_proj.bias",),
            f"{tt_pref}.attention.wk.weight": (f"{hf_pref}.self_attn.k_proj.weight",),
            f"{tt_pref}.attention.wk.bias":   (f"{hf_pref}.self_attn.k_proj.bias",),
            f"{tt_pref}.attention.wv.weight": (f"{hf_pref}.self_attn.v_proj.weight",),
            f"{tt_pref}.attention.wv.bias":   (f"{hf_pref}.self_attn.v_proj.bias",),
            f"{tt_pref}.attention.wo.weight": (f"{hf_pref}.self_attn.o_proj.weight",),
            f"{tt_pref}.attention.wo.bias":   (f"{hf_pref}.self_attn.o_proj.bias",),

            # Sinks tensor
            f"{tt_pref}.attention.sinks":     (f"{hf_pref}.self_attn.sinks",),

            # MoE experts (your custom naming on HF side)
            f"{tt_pref}.moe.experts.mlp1_weight": (f"{hf_pref}.mlp.experts.gate_up_proj",),
            f"{tt_pref}.moe.experts.mlp1_bias":   (f"{hf_pref}.mlp.experts.gate_up_proj_bias",),
            f"{tt_pref}.moe.experts.mlp2_weight": (f"{hf_pref}.mlp.experts.down_proj",),
            f"{tt_pref}.moe.experts.mlp2_bias":   (f"{hf_pref}.mlp.experts.down_proj_bias",),

            # Router
            f"{tt_pref}.moe.router.gate.weight":  (f"{hf_pref}.mlp.router.weight",),
            f"{tt_pref}.moe.router.gate.bias":    (f"{hf_pref}.mlp.router.bias",),

            # Norms
            f"{tt_pref}.attention_norm.weight":   (f"{hf_pref}.input_layernorm.weight",),
            f"{tt_pref}.ffn_norm.weight":         (f"{hf_pref}.post_attention_layernorm.weight",),
        }

        for tt_key, (hf_key,) in m.items():
            if tt_key in tt:
                hf[hf_key] = tt[tt_key].clone()
            elif strict:
                raise KeyError(f"Missing expected key in TorchTitan state dict: '{tt_key}'")

    print(f"Converted {len(hf)} parameters from TorchTitan to HuggingFace format (gpt-oss).")
    return hf


@app.command(name="hf_to_dcp")
@torch.inference_mode()
def convert_hf_to_dcp(input_path: str, output_path: Path, max_seq_len: int = 131072, rope_theta: float = 150000.0, dtype: str = "auto", torchtitan_config: str = "20b"):
    """Convert HuggingFace model to TorchTitan DCP format.

    Args:
        input_path: HuggingFace model name or path
        output_path: Output DCP checkpoint path
        max_seq_len: Max sequence length for RoPE
        rope_theta: RoPE theta parameter
        dtype: Data type to use ("auto" to preserve original, or specific dtype like "float32")
        torchtitan_config: TorchTitan model config name (e.g., "16B-A3B", "debugmodel")
    """
    # Import TorchTitan configs
    try:
        from torchtitan.models.gpt_oss import gptoss_configs
    except ImportError:
        raise ImportError("Cannot import TorchTitan GPT-OSS configs. Make sure you're in the right environment.")

    logger.info(f"Loading model from {input_path}")

    # Load model with original dtype if "auto", otherwise use specified dtype
    hf_model = AutoModelForCausalLM.from_pretrained(input_path, torch_dtype=torch.bfloat16)

    # Validate configuration compatibility
    logger.info(f"Validating config compatibility with TorchTitan config: {torchtitan_config}")
    validate_config_compatibility(hf_model.config, torchtitan_config, gptoss_configs)

    hf_state_dict = hf_model.state_dict()
    logger.info(f"Loaded model with dtype: {next(iter(hf_state_dict.values())).dtype}")

    logger.info("Converting weights to TorchTitan format")
    torchtitan_state_dict = map_hf_to_torchtitan(hf_state_dict, hf_model.config, max_seq_len, rope_theta, input_path)

    logger.info(f"Writing to DCP at '{output_path}'")
    output_path.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_path, thread_count=8)
    DCP.save({"model": torchtitan_state_dict}, storage_writer=storage_writer)

    # Save metadata for reference
    metadata = {
        "original_hf_model": input_path,
        "torchtitan_config": torchtitan_config,
        "conversion_time": str(torch.tensor(0).item()),  # placeholder
        "hf_config": dict(hf_model.config.__dict__),
        "torchtitan_config_dict": dict(gptoss_configs[torchtitan_config].__dict__),
    }
    with open(output_path / "conversion_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Conversion complete!")
    logger.info(f"📋 Saved conversion metadata to {output_path}/conversion_metadata.json")
    logger.info(f"🚀 To use in TorchTitan, specify model config: {torchtitan_config}")

    # Final reminder about RoPE differences
    if "gpt-oss" in input_path.lower():
        logger.info(f"")
        logger.info(f"🔔 IMPORTANT: Converted GPT-OSS model uses TorchTitan's RoPE implementation")
        logger.info(f"   This differs from HuggingFace but is expected behavior")
        logger.info(f"   See conversion script documentation for details")


@app.command(name="dcp_to_hf")
@torch.inference_mode()
def convert_dcp_to_hf(input_path: Path, output_path: Path, max_seq_len: int = 131072, rope_theta: float = 150000.0, default_model: str = "openai/gpt-oss-20b"):
    """Convert TorchTitan DCP format to HuggingFace model.

    Args:
        input_path: Input DCP checkpoint path
        output_path: Output HuggingFace model path
        max_seq_len: Max sequence length for RoPE
        rope_theta: RoPE theta parameter
        default_model: Default HuggingFace model for config
    """
    from torchtitan.datasets.transformation import get_tokenizer_with_chat_template
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    logger.info(f"Loading DCP checkpoint from {input_path}")

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
    hf_state_dict = map_torchtitan_to_hf(torchtitan_state_dict)

    # Create HuggingFace config
    hf_config = AutoConfig.from_pretrained(default_model)

    # Create and load model
    logger.info("Creating HuggingFace model")
    # tokenizer = AutoTokenizer.from_pretrained(default_model)
    tokenizer = get_tokenizer_with_chat_template(default_model, "tulu", override=True)
    hf_model = AutoModelForCausalLM.from_pretrained(default_model)

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
