# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical equivalence test with full pretrained weights.

Loads real HF checkpoints, extracts the first MoE block, transfers weights
to a Titan MoE, and compares forward outputs. This validates that
the weight transfer and routing logic are correct with production weights,
not just random initialization.

Usage:
    python -m torchtitan.experiments.transformers_modeling_backend.tests.numerical_equivalence_pretrained \
        --model_dir /tmp/mreso/models/OLMoE-1B-7B-0924
"""

import argparse
import os

import torch
import torch.nn.functional as F


_COMPARE_METRICS = ("kl_div", "cos_sim", "max_abs_diff", "mean_abs_diff")


@torch.no_grad()
def test_pretrained(model_dir: str, device: torch.device, seed: int = 42) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
        _build_moe_config,
        _probe_hf_moe_block,
    )
    from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
        hf_to_titan_moe_state_dict,
    )

    model_name = os.path.basename(model_dir)

    # Try loading without trust_remote_code first; fall back if needed
    try:
        config = AutoConfig.from_pretrained(model_dir)
        trust_remote = False
    except Exception:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        trust_remote = True
    config._experts_implementation = "grouped_mm"

    print(f"  Loading {model_name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.bfloat16,
        trust_remote_code=trust_remote,
    )
    model = model.to(device=device).eval()

    # Find first MoE layer
    hf_moe_block = None
    layer_idx = -1
    for i, layer in enumerate(model.model.layers):
        has_gate = hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "router")
        if has_gate and hasattr(layer.mlp, "experts"):
            hf_moe_block = layer.mlp
            layer_idx = i
            break

    if hf_moe_block is None:
        return {"model": model_name, "error": "No MoE layer found"}

    print(f"  Using MoE layer {layer_idx}", flush=True)

    # Cast gate to float32 to match titan's autocast
    gate = getattr(hf_moe_block, "gate", None) or getattr(hf_moe_block, "router", None)
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

    # Forward through HF MoE block
    torch.manual_seed(seed)
    x = torch.randn(2, 16, config.hidden_size, device=device, dtype=torch.bfloat16)
    hf_out = hf_moe_block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Build titan MoE
    config.load_balance_coeff = None
    config.comm_backend = "standard"
    params = _probe_hf_moe_block(hf_moe_block, config)
    params["load_balance_coeff"] = None
    moe_config = _build_moe_config(params, config)

    with torch.device("meta"):
        titan_moe = moe_config.build()
    titan_moe.to_empty(device=device)
    titan_moe.init_states(
        buffer_device=torch.device(device) if isinstance(device, str) else device
    )

    # Transfer weights
    hf_sd = hf_moe_block.state_dict()
    hf_sd_prefixed = {f"mlp.{k}": v for k, v in hf_sd.items()}
    titan_sd = hf_to_titan_moe_state_dict(hf_sd_prefixed)
    titan_sd_stripped = {k.removeprefix("mlp."): v for k, v in titan_sd.items()}
    titan_moe.load_state_dict(titan_sd_stripped, strict=False)
    titan_moe = titan_moe.to(dtype=torch.bfloat16).eval()

    # Forward through titan MoE
    tt_out = titan_moe(x)

    # Compare
    hf_flat = hf_out.flatten().float()
    tt_flat = tt_out.flatten().float()

    kl_div = F.kl_div(
        F.log_softmax(tt_flat.unsqueeze(0), dim=-1),
        F.softmax(hf_flat.unsqueeze(0), dim=-1),
        reduction="batchmean",
    ).item()
    cos_sim = F.cosine_similarity(hf_flat.unsqueeze(0), tt_flat.unsqueeze(0)).item()
    max_abs_diff = (hf_out.float() - tt_out.float()).abs().max().item()
    mean_abs_diff = (hf_out.float() - tt_out.float()).abs().mean().item()

    # Free GPU memory before next model
    del model, hf_moe_block, titan_moe
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "kl_div": kl_div,
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Numerical equivalence with pretrained weights"
    )
    parser.add_argument(
        "--model_dir",
        nargs="+",
        required=True,
        help="Path(s) to HF model directories",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    all_passed = True

    print(f"\nPretrained MoE numerical equivalence (device={device}, seed={args.seed})")
    print("=" * 80)

    for model_dir in args.model_dir:
        try:
            result = test_pretrained(model_dir, device, args.seed)
            if "error" in result:
                status = "SKIP"
                detail = result["error"]
            elif result["kl_div"] < 1e-3:
                status = "PASS"
                detail = (
                    f"KL={result['kl_div']:.2e}  cos={result['cos_sim']:.6f}  "
                    f"max_diff={result['max_abs_diff']:.2e}"
                )
            else:
                status = "FAIL"
                detail = (
                    f"KL={result['kl_div']:.2e}  cos={result['cos_sim']:.6f}  "
                    f"max_diff={result['max_abs_diff']:.2e}"
                )
                all_passed = False

            print(f"  {status:5s}  {result['model']:30s}  {detail}")

        except Exception as e:
            print(f"  ERROR  {os.path.basename(model_dir):30s}  {e}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("All models passed.")
    else:
        print("Some models FAILED.")
        exit(1)


if __name__ == "__main__":
    main()
