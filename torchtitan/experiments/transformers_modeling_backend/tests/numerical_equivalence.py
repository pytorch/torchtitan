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


def _create_hf_model(model_type: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = _MODEL_CONFIGS[model_type]
    config = AutoConfig.for_model(**kwargs)
    model = AutoModelForCausalLM.from_config(config)
    return model, config


# ---------------------------------------------------------------------------
# Weight transfer: HF MoE → titan MoE
# ---------------------------------------------------------------------------


def _transfer_weights(hf_moe_block, titan_moe, config):
    """Transfer weights from HF MoE block to native titan MoE.

    Handles the fused gate_up_proj → separate w1/w3 split and
    router weight transfer.
    """
    experts = hf_moe_block.experts

    # Expert weights: split fused gate_up_proj into w1 (gate) and w3 (up)
    gate_up = experts.gate_up_proj.data  # (E, 2*I, H)
    intermediate_size = gate_up.shape[1] // 2
    titan_moe.experts.w1.data.copy_(gate_up[:, :intermediate_size, :])
    titan_moe.experts.w3.data.copy_(gate_up[:, intermediate_size:, :])
    titan_moe.experts.w2.data.copy_(experts.down_proj.data)

    # Router gate weight
    gate = getattr(hf_moe_block, "gate", None) or getattr(hf_moe_block, "router", None)
    titan_moe.router.gate.weight.data.copy_(gate.weight.data)

    # Shared experts (if present)
    if titan_moe.shared_experts is not None:
        shared = None
        for name in ("shared_expert", "shared_experts", "shared_mlp"):
            shared = getattr(hf_moe_block, name, None)
            if shared is not None:
                break

        if shared is not None:
            # Determine the titan shared expert module
            titan_shared = titan_moe.shared_experts
            # Handle GatedSharedExpert wrapper
            if hasattr(titan_shared, "ffn"):
                titan_ffn = titan_shared.ffn
                # Transfer gate
                shared_gate = getattr(hf_moe_block, "shared_expert_gate", None)
                if shared_gate is not None:
                    titan_shared.gate.weight.data.copy_(shared_gate.weight.data)
            else:
                titan_ffn = titan_shared

            titan_ffn.w1.weight.data.copy_(shared.gate_proj.weight.data)
            titan_ffn.w3.weight.data.copy_(shared.up_proj.weight.data)
            titan_ffn.w2.weight.data.copy_(shared.down_proj.weight.data)

    # Expert bias (DeepSeek V3 e_score_correction_bias)
    if (
        gate is not None
        and "e_score_correction_bias" in getattr(gate, "_buffers", {})
        and titan_moe.expert_bias is not None
    ):
        titan_moe.expert_bias.data.copy_(gate.e_score_correction_bias.data)


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
    model = model.to(device).eval()

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
    x = torch.randn(2, 16, config.hidden_size, device=device)
    hf_out = hf_moe_block(x.clone())
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

    # 4. Transfer weights from HF to titan
    _transfer_weights(hf_moe_block, titan_moe, config)
    titan_moe.eval()

    # 5. Forward through titan MoE
    tt_out = titan_moe(x.clone())

    # 6. Compare
    return _compare_outputs(hf_out, tt_out, model_type)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Numerical equivalence: HF MoE vs titan MoE"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(_MODEL_CONFIGS.keys()),
        choices=list(_MODEL_CONFIGS.keys()),
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

    print("=" * 80)
    if all_passed:
        print("All models passed numerical equivalence.")
    else:
        print("Some models FAILED — see above.")
        exit(1)


if __name__ == "__main__":
    main()
