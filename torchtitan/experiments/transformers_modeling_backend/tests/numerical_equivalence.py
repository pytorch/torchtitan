# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical equivalence test: HF native MoE vs titan-replaced MoE.

For each supported HF MoE model, this script:
1. Creates a tiny HF model with random weights
2. Runs forward on the HF-native MoE block
3. Builds the equivalent native titan MoE, transfers weights from HF
4. Runs forward on the titan MoE block with the same input
5. Compares outputs via KL divergence, cosine similarity, and max abs diff

Usage:
    python -m torchtitan.experiments.transformers_modeling_backend.tests.numerical_equivalence
"""

import argparse

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "qwen3_moe": dict(
        model_type="qwen3_moe",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        decoder_sparse_step=1,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "mixtral": dict(
        model_type="mixtral",
        hidden_size=64,
        intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "deepseek_v3": dict(
        model_type="deepseek_v3",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=64,
        first_k_dense_replace=0,
        n_routed_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "olmoe": dict(
        model_type="olmoe",
        hidden_size=64,
        intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "deepseek_v2": dict(
        model_type="deepseek_v2",
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=64,
        first_k_dense_replace=0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=None,
        topk_group=None,
        n_shared_experts=1,
        mla_type="deepseek_v2",
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        attn_implementation="sdpa",
        use_cache=False,
    ),
}

# Models that require AutoConfig.from_pretrained (remote config).
# Keys are overrides applied after loading the pretrained config.
_PRETRAINED_MODEL_CONFIGS = {
    "glm4_moe": dict(
        hf_model_id="zai-org/GLM-4.7",
        overrides=dict(
            num_hidden_layers=4,
            use_cache=False,
        ),
    ),
    "glm_moe_dsa": dict(
        hf_model_id="zai-org/GLM-5",
        overrides=dict(
            num_hidden_layers=4,
            use_cache=False,
        ),
    ),
}


def _create_hf_model(model_type: str):
    """Create a tiny HF MoE model for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if model_type in _MODEL_CONFIGS:
        kwargs = _MODEL_CONFIGS[model_type]
        config = AutoConfig.for_model(**kwargs)
    elif model_type in _PRETRAINED_MODEL_CONFIGS:
        entry = _PRETRAINED_MODEL_CONFIGS[model_type]
        config = AutoConfig.from_pretrained(
            entry["hf_model_id"], trust_remote_code=True
        )
        for k, v in entry["overrides"].items():
            setattr(config, k, v)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model, config


# ---------------------------------------------------------------------------
# Weight transfer: HF MoE → titan MoE
# ---------------------------------------------------------------------------


def _transfer_weights_via_adapter(hf_moe_block, titan_moe):
    """Transfer weights from HF MoE block to titan MoE via state dict adapter.

    Uses ``hf_to_titan_moe_state_dict`` to convert the HF state dict keys
    and values, then loads into the titan module.
    """
    from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
        hf_to_titan_moe_state_dict,
    )

    hf_sd = hf_moe_block.state_dict()

    # Prefix all keys with "mlp." to match the layer-level namespace
    hf_sd_prefixed = {f"mlp.{k}": v for k, v in hf_sd.items()}
    titan_sd = hf_to_titan_moe_state_dict(hf_sd_prefixed)

    # Strip the "mlp." prefix to match titan_moe's own state dict
    titan_sd_stripped = {k.removeprefix("mlp."): v for k, v in titan_sd.items()}

    titan_moe.load_state_dict(titan_sd_stripped, strict=False)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def _compare_outputs(
    hf_out: torch.Tensor, tt_out: torch.Tensor, model_name: str
) -> dict:
    """Compare two output tensors and return metrics."""
    hf_flat = hf_out.flatten().float()
    tt_flat = tt_out.flatten().float()

    # KL divergence
    kl_div = F.kl_div(
        F.log_softmax(tt_flat.unsqueeze(0), dim=-1),
        F.softmax(hf_flat.unsqueeze(0), dim=-1),
        reduction="batchmean",
    ).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(hf_flat.unsqueeze(0), tt_flat.unsqueeze(0)).item()

    # Max absolute difference
    max_abs_diff = (hf_out.float() - tt_out.float()).abs().max().item()

    # Mean absolute difference
    mean_abs_diff = (hf_out.float() - tt_out.float()).abs().mean().item()

    return {
        "model": model_name,
        "kl_div": kl_div,
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


# ---------------------------------------------------------------------------
# Per-model test
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_model(model_type: str, device: torch.device, seed: int = 42) -> dict:
    """Test numerical equivalence for a single model type."""
    from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
        _build_moe_config,
        _probe_hf_moe_block,
    )

    torch.manual_seed(seed)

    # 1. Create HF model
    model, config = _create_hf_model(model_type)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # Find first MoE layer
    hf_moe_block = None
    for layer in model.model.layers:
        has_gate = hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "router")
        if has_gate and hasattr(layer.mlp, "experts"):
            hf_moe_block = layer.mlp
            break

    if hf_moe_block is None:
        return {"model": model_type, "error": "No MoE layer found"}

    # 2. Forward through HF MoE block
    torch.manual_seed(seed)
    x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.bfloat16)
    hf_out = hf_moe_block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # 3. Build titan MoE with same config
    config.load_balance_coeff = None  # disable load balancing for comparison
    config.comm_backend = "standard"
    params = _probe_hf_moe_block(hf_moe_block, config)
    params["load_balance_coeff"] = None
    moe_config = _build_moe_config(params, config)

    with torch.device("meta"):
        titan_moe = moe_config.build()
    titan_moe.to_empty(device=device)
    titan_moe.init_states(buffer_device=device)

    # 4. Transfer weights from HF to titan via state dict adapter
    _transfer_weights_via_adapter(hf_moe_block, titan_moe)
    titan_moe = titan_moe.to(dtype=torch.bfloat16).eval()

    # 5. Forward through titan MoE
    tt_out = titan_moe(x)

    # 6. Compare
    return _compare_outputs(hf_out, tt_out, model_type)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run numerical equivalence and round-trip adapter tests."""
    parser = argparse.ArgumentParser(
        description="Numerical equivalence: HF MoE vs titan MoE"
    )
    all_models = list(_MODEL_CONFIGS.keys()) + list(_PRETRAINED_MODEL_CONFIGS.keys())
    parser.add_argument(
        "--models",
        nargs="*",
        default=all_models,
        choices=all_models,
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []
    all_passed = True

    print(f"\nNumerical equivalence test (device={device}, seed={args.seed})")
    print("=" * 80)

    for model_type in args.models:
        try:
            result = test_model(model_type, device, args.seed)
            results.append(result)

            if "error" in result:
                status = "SKIP"
                detail = result["error"]
            elif result["kl_div"] < 1e-6:
                status = "PASS"
                detail = f"KL={result['kl_div']:.2e}  cos={result['cos_sim']:.6f}  max_diff={result['max_abs_diff']:.2e}"
            elif result["kl_div"] < 1e-3:
                status = "WARN"
                detail = f"KL={result['kl_div']:.2e}  cos={result['cos_sim']:.6f}  max_diff={result['max_abs_diff']:.2e}"
            else:
                status = "FAIL"
                detail = f"KL={result['kl_div']:.2e}  cos={result['cos_sim']:.6f}  max_diff={result['max_abs_diff']:.2e}"
                all_passed = False

            print(f"  {status:5s}  {model_type:20s}  {detail}")

        except Exception as e:
            print(f"  ERROR  {model_type:20s}  {e}")
            all_passed = False

    # Round-trip adapter test: HF → titan → HF
    print("")
    print("State dict adapter round-trip test")
    print("-" * 80)

    from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
        hf_to_titan_moe_state_dict,
        titan_to_hf_moe_state_dict,
    )

    for model_type in args.models:
        try:
            model, config = _create_hf_model(model_type)
            hf_sd = model.state_dict()
            titan_sd = hf_to_titan_moe_state_dict(hf_sd)
            roundtrip_sd = titan_to_hf_moe_state_dict(titan_sd)

            # Check all original keys are present
            missing = set(hf_sd.keys()) - set(roundtrip_sd.keys())
            extra = set(roundtrip_sd.keys()) - set(hf_sd.keys())

            # Check values match
            max_diff = 0.0
            for k in hf_sd:
                if k in roundtrip_sd:
                    diff = (
                        (hf_sd[k].float() - roundtrip_sd[k].float()).abs().max().item()
                    )
                    max_diff = max(max_diff, diff)

            if missing or extra:
                status = "FAIL"
                detail = f"missing={len(missing)} extra={len(extra)}"
                if missing:
                    detail += f" missing_keys={list(missing)[:3]}"
                all_passed = False
            elif max_diff > 1e-6:
                status = "FAIL"
                detail = f"round-trip max_diff={max_diff:.2e}"
                all_passed = False
            else:
                status = "PASS"
                detail = f"round-trip max_diff={max_diff:.2e}"

            print(f"  {status:5s}  {model_type:20s}  {detail}")

        except Exception as e:
            print(f"  ERROR  {model_type:20s}  {e}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("All models passed numerical equivalence.")
    else:
        print("Some models FAILED — see above.")
        exit(1)


if __name__ == "__main__":
    main()
