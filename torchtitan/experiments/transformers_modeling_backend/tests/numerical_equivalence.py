# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical equivalence test: HF's own MoE vs Titan-replaced MoE.

For each supported HF MoE model, this script:
1. Creates a single-layer HF model with production-scale dimensions and random weights
2. Runs forward on the HF model's own MoE block (using grouped_mm experts)
3. Builds the equivalent Titan MoE, transfers weights from HF
4. Runs forward on the Titan MoE block with the same input
5. Compares outputs via KL divergence, cosine similarity, and max abs diff

Note on numerical precision:
    Results are NOT bit-exact due to three differences in core torchtitan.
    With all three fixed, all models produce max_diff=0.00 (verified on
    full pretrained Mixtral-8x7B, Qwen3-30B-A3B, DeepSeek-V2-Lite,
    OLMoE-1B-7B).

    1. ``LocalTokenDispatcher.combine()`` casts score-weighted output to
       bf16 before ``scatter_add``, so accumulation happens in bf16. Fix:
       keep in f32 through accumulation, cast only at the end.
    2. ``scatter_add`` accumulates in expert-sorted order; HF uses
       ``reshape(N, K, D).sum(dim=1)`` in token order. Fix: unsort back
       to token order and use reshape+sum instead of scatter_add.
    3. ``TokenChoiceTopKRouter`` uses ``topk(sorted=False)``; HF defaults
       to ``sorted=True``. Different order causes different f32 sum in
       route_norm. Fix: use ``sorted=True``.

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
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        decoder_sparse_step=1,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    # Qwen2-MoE: sigmoid-gated shared expert (out = sigmoid(gate(x)) * ffn(x)).
    # Exercises the SigmoidGatedFeedForward path; deepseek_v2/v3 cover additive shared
    # experts, so this is the only config that hits the gated variant.
    "qwen2_moe": dict(
        model_type="qwen2_moe",
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        shared_expert_intermediate_size=2048,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=False,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "mixtral": dict(
        model_type="mixtral",
        hidden_size=4096,
        intermediate_size=14336,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "deepseek_v3": dict(
        model_type="deepseek_v3",
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=256,
        max_position_embeddings=64,
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_local_experts=8,
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
        hidden_size=2048,
        intermediate_size=1024,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        vocab_size=256,
        max_position_embeddings=64,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "deepseek_v2": dict(
        model_type="deepseek_v2",
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=256,
        max_position_embeddings=64,
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=None,
        topk_group=None,
        n_shared_experts=2,
        mla_type="deepseek_v2",
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "glm4_moe": dict(
        model_type="glm4_moe",
        hidden_size=5120,
        intermediate_size=12288,
        moe_intermediate_size=1536,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=64,
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "glm_moe_dsa": dict(
        model_type="glm_moe_dsa",
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=256,
        max_position_embeddings=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
        mlp_layer_types=["sparse"],
        attn_implementation="sdpa",
        use_cache=False,
    ),
    "gemma4_text": dict(
        model_type="gemma4_text",
        hidden_size=2816,
        intermediate_size=2112,
        moe_intermediate_size=704,
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=256,
        max_position_embeddings=64,
        enable_moe_block=True,
        attn_implementation="sdpa",
        use_cache=False,
    ),
}

# Models that require AutoConfig.from_pretrained (remote config).
# Keys are overrides applied after loading the pretrained config.
_PRETRAINED_MODEL_CONFIGS = {}


def _create_hf_model(model_type: str):
    """Create a tiny HF MoE model for testing.

    Uses ``grouped_mm`` expert implementation so HF and titan both use
    ``torch._grouped_mm``, isolating weight-transfer and routing logic
    from kernel-level numerical differences.
    """
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

    config._experts_implementation = "grouped_mm"
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
        _TITAN_TO_ORIGINAL_HF_KEY,
        hf_to_titan_moe_state_dict,
    )

    # Clear global adapter state from any previous model's conversion
    _TITAN_TO_ORIGINAL_HF_KEY.clear()

    hf_sd = hf_moe_block.state_dict()

    # Prefix all keys with "mlp." to match the layer-level namespace
    hf_sd_prefixed = {f"mlp.{k}": v for k, v in hf_sd.items()}
    titan_sd = hf_to_titan_moe_state_dict(hf_sd_prefixed)

    # Strip the "mlp." prefix to match titan_moe's own state dict
    titan_sd_stripped = {k.removeprefix("mlp."): v for k, v in titan_sd.items()}

    titan_moe.load_state_dict(titan_sd_stripped, strict=False)


def _transfer_weights_layer_level(hf_layer, titan_moe):
    """Transfer weights for layer-level MoE (Gemma4).

    Router and experts are at the layer level in HF.  The Titan MoE
    has ``router.gate``, ``experts.{w1,w2,w3}``.  We collect the relevant
    HF state dict entries, apply the adapter, and load into titan.
    """
    from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
        _TITAN_TO_ORIGINAL_HF_KEY,
        hf_to_titan_moe_state_dict,
    )

    # Clear global adapter state from any previous model's conversion to
    # prevent cross-model key mappings from leaking between models.
    _TITAN_TO_ORIGINAL_HF_KEY.clear()

    # Collect router + experts state dicts
    hf_sd = {}
    router = getattr(hf_layer, "router", None) or getattr(hf_layer, "gate", None)
    if router is not None:
        for k, v in router.state_dict().items():
            hf_sd[f"router.{k}"] = v
    for k, v in hf_layer.experts.state_dict().items():
        hf_sd[f"experts.{k}"] = v

    # Prefix with "mlp." to match adapter namespace
    hf_sd_prefixed = {f"mlp.{k}": v for k, v in hf_sd.items()}
    titan_sd = hf_to_titan_moe_state_dict(hf_sd_prefixed)

    # Strip the "mlp." prefix
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
        _probe_layer_level_moe,
    )

    torch.manual_seed(seed)

    # 1. Create HF model
    model, config = _create_hf_model(model_type)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    # Find first MoE layer.
    # Standard: gate/router + experts inside ``mlp`` or ``feed_forward``.
    # Layer-level (Gemma4): router + experts are siblings of the dense MLP
    # at the decoder layer level.
    hf_moe_block = None
    hf_moe_layer = None
    is_layer_level = False
    for layer in model.model.layers:
        for attr_name in ("mlp", "feed_forward"):
            moe_module = getattr(layer, attr_name, None)
            if moe_module is None:
                continue
            has_gate = hasattr(moe_module, "gate") or hasattr(moe_module, "router")
            if has_gate and hasattr(moe_module, "experts"):
                hf_moe_block = moe_module
                break
        if hf_moe_block is not None:
            break
        # Check for layer-level MoE
        has_layer_router = hasattr(layer, "router") or hasattr(layer, "gate")
        has_layer_experts = hasattr(layer, "experts")
        if has_layer_router and has_layer_experts:
            hf_moe_layer = layer
            is_layer_level = True
            break

    if hf_moe_block is None and hf_moe_layer is None:
        return {"model": model_type, "error": "No MoE layer found"}

    # 2. Forward through HF MoE block
    torch.manual_seed(seed)
    x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.bfloat16)

    if is_layer_level:
        # Layer-level MoE (Gemma4): run just the routed experts path.
        # The test compares the routed-expert-only forward (no shared expert
        # / dense MLP) to isolate the MoE weight transfer and routing.
        hf_router = getattr(hf_moe_layer, "router", None) or getattr(
            hf_moe_layer, "gate", None
        )
        hf_experts = hf_moe_layer.experts

        # Cast router to float32 to match titan's autocast
        hf_router.float()
        hf_router.register_forward_pre_hook(
            lambda mod, args: tuple(
                a.float()
                if isinstance(a, torch.Tensor) and a.is_floating_point()
                else a
                for a in args
            )
        )

        x_flat = x.reshape(-1, x.shape[-1])
        _, top_k_weights, top_k_index = hf_router(x_flat)
        hf_out = hf_experts(x_flat, top_k_index, top_k_weights)
        hf_out = hf_out.reshape(x.shape)

        # 3. Build titan MoE (experts only, no shared expert for this test)
        config.load_balance_coeff = None
        config.comm_backend = "standard"
        params = _probe_layer_level_moe(hf_moe_layer, config)
        params["load_balance_coeff"] = None
        # Remove shared expert for this comparison — we only test routed experts
        params["shared_expert_info"] = None
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            titan_moe = moe_config.build()
        titan_moe.to_empty(device=device)
        buffer_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        titan_moe.init_states(buffer_device=buffer_device)

        # 4. Transfer weights — layer-level MoE keys need manual prefix
        _transfer_weights_layer_level(hf_moe_layer, titan_moe)
        titan_moe = titan_moe.to(dtype=torch.bfloat16).eval()

    else:
        # Standard MoE block
        # Cast the HF router gate module to float32, matching titan's
        # torch.autocast(dtype=float32) on the gate linear.
        gate = getattr(hf_moe_block, "gate", None) or getattr(
            hf_moe_block, "router", None
        )
        if gate is not None:
            gate.float()
            gate.register_forward_pre_hook(
                lambda mod, args: tuple(
                    a.float()
                    if isinstance(a, torch.Tensor) and a.is_floating_point()
                    else a
                    for a in args
                )
            )

        hf_out = hf_moe_block(x)
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

        # 3. Build titan MoE with same config
        config.load_balance_coeff = None
        config.comm_backend = "standard"
        params = _probe_hf_moe_block(hf_moe_block, config)
        params["load_balance_coeff"] = None
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            titan_moe = moe_config.build()
        titan_moe.to_empty(device=device)
        buffer_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        titan_moe.init_states(buffer_device=buffer_device)

        # 4. Transfer weights from HF to titan via state dict adapter
        _transfer_weights_via_adapter(hf_moe_block, titan_moe)
        titan_moe = titan_moe.to(dtype=torch.bfloat16).eval()

    # 5. Forward through titan MoE
    tt_out = titan_moe(x)

    # 6. Compare — reshape to common shape if needed (HF MoE may flatten
    # batch and sequence dims while titan preserves them)
    if hf_out.shape != tt_out.shape:
        hf_out = hf_out.reshape(tt_out.shape)

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
