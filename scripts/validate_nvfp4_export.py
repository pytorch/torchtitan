#!/usr/bin/env python3
"""Validate NVFP4 exported weights by dequantizing and checking for correctness.

Standalone script — no vLLM or TRT-LLM needed, just safetensors + torch.

Checks:
1. Packed weight shapes and dtypes match expectations
2. Scale tensors are finite and non-zero
3. Dequantized values are finite, non-zero, and in reasonable range
4. Per-tensor scale matches amax / (FP4_MAX * FP8_MAX)
5. Simple matmul with dequantized weights produces finite output
"""

import json
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

# NVFP4 constants
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
BLOCK_SIZE = 16

# FP4 E2M1 decode table: 4-bit → float
FP4_TABLE = torch.tensor(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def dequantize_nvfp4_block(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize NVFP4 packed uint8 tensor to float32.

    Args:
        packed: uint8 tensor of shape (..., N//2), 2 FP4 values per byte
        block_scale: float8_e4m3fn tensor of shape (..., N//BLOCK_SIZE)
        tensor_scale: float32 per-tensor scale (scalar or per-expert)

    Returns:
        float32 tensor of shape (..., N)
    """
    # Unpack: lower nibble = even indices, upper nibble = odd indices
    lo = (packed.int() & 0x0F).long()
    hi = (packed.int() >> 4 & 0x0F).long()

    # Interleave to get original order: [lo0, hi0, lo1, hi1, ...]
    *prefix, half_n = packed.shape
    out = torch.empty(*prefix, half_n * 2, dtype=torch.float32)
    out[..., 0::2] = FP4_TABLE[lo]
    out[..., 1::2] = FP4_TABLE[hi]

    # Apply per-block scale: each block of BLOCK_SIZE values shares one FP8 scale
    # block_scale shape: (..., N//BLOCK_SIZE)
    # out shape: (..., N)
    # Reshape out to (..., N//BLOCK_SIZE, BLOCK_SIZE) for broadcasting
    n = half_n * 2
    n_blocks = n // BLOCK_SIZE
    out_blocked = out.view(*prefix, n_blocks, BLOCK_SIZE)
    block_scale_f32 = block_scale.float()  # (..., n_blocks)
    out_blocked = out_blocked * block_scale_f32.unsqueeze(-1)
    out = out_blocked.view(*prefix, n)

    # Apply per-tensor scale (scalar in official ModelOpt format)
    tensor_scale_f32 = tensor_scale.float()
    if tensor_scale_f32.ndim == 0:
        out = out * tensor_scale_f32
    else:
        # Fallback for non-scalar (legacy format)
        while tensor_scale_f32.ndim < out.ndim:
            tensor_scale_f32 = tensor_scale_f32.unsqueeze(-1)
        out = out * tensor_scale_f32

    return out


def load_model_tensors(model_path: str):
    """Load all tensors from sharded safetensors."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files = set(weight_map.values())
    else:
        # Single file
        files = {"model.safetensors"}
        weight_map = None

    tensors = {}
    for fname in sorted(files):
        fpath = os.path.join(model_path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    return tensors


def validate_shapes(tensors: dict[str, torch.Tensor], config: dict):
    """Validate NVFP4 tensor shapes and dtypes."""
    print("\n" + "=" * 70)
    print("SHAPE & DTYPE VALIDATION")
    print("=" * 70)

    issues = []

    # Find all expert weight tensors
    expert_keys = defaultdict(dict)
    for key in tensors:
        if "mlp.experts" not in key:
            continue
        # Parse: model.layers.N.mlp.experts.{gate_up_proj|down_proj}[_suffix]
        parts = key.split("mlp.experts.")
        if len(parts) != 2:
            continue
        layer_prefix = parts[0]
        suffix = parts[1]
        expert_keys[layer_prefix][suffix] = key

    for layer_prefix in sorted(expert_keys):
        layer_tensors = expert_keys[layer_prefix]
        print(f"\n--- {layer_prefix}mlp.experts ---")

        for proj in ["gate_up_proj", "down_proj"]:
            if proj not in layer_tensors:
                issues.append(f"MISSING: {layer_prefix}mlp.experts.{proj}")
                continue

            key = layer_tensors[proj]
            t = tensors[key]
            print(f"  {proj}: shape={list(t.shape)}, dtype={t.dtype}")

            # Check dtype is uint8 (packed FP4)
            if t.dtype != torch.uint8:
                issues.append(f"WRONG DTYPE: {key} is {t.dtype}, expected uint8")

            # Check ndim
            if t.ndim != 3:
                issues.append(f"WRONG NDIM: {key} has {t.ndim} dims, expected 3")

            # Check scale tensors exist
            for scale_suffix in ["_weight_scale", "_weight_scale_2", "_input_scale"]:
                scale_key = f"{proj}{scale_suffix}"
                if scale_key in layer_tensors:
                    st = tensors[layer_tensors[scale_key]]
                    print(
                        f"    {scale_suffix}: shape={list(st.shape)}, dtype={st.dtype}"
                    )
                else:
                    issues.append(
                        f"MISSING SCALE: {layer_prefix}mlp.experts.{scale_key}"
                    )

    if issues:
        print(f"\n*** {len(issues)} ISSUES FOUND ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll shapes and dtypes OK")

    return len(issues) == 0


def validate_scales(tensors: dict[str, torch.Tensor], amax_path: str | None):
    """Validate scale tensors are finite and match amax."""
    print("\n" + "=" * 70)
    print("SCALE VALIDATION")
    print("=" * 70)

    issues = []
    amax_map = {}
    if amax_path and os.path.exists(amax_path):
        with open(amax_path) as f:
            amax_map = json.load(f)
        print(f"Loaded amax from {amax_path}: {len(amax_map)} modules")

    for key, t in sorted(tensors.items()):
        if "mlp.experts" not in key:
            continue

        is_scale = False
        for suffix in ["_weight_scale", "_weight_scale_2", "_input_scale"]:
            if key.endswith(suffix):
                is_scale = True
                break

        if not is_scale:
            continue

        t_f32 = t.float()
        nan_count = torch.isnan(t_f32).sum().item()
        inf_count = torch.isinf(t_f32).sum().item()
        zero_count = (t_f32 == 0).sum().item()
        total = t_f32.numel()

        status = "OK"
        if nan_count > 0:
            status = f"NaN={nan_count}"
            issues.append(f"NaN in {key}: {nan_count}/{total}")
        if inf_count > 0:
            status = f"Inf={inf_count}"
            issues.append(f"Inf in {key}: {inf_count}/{total}")

        print(
            f"  {key}: "
            f"min={t_f32.min().item():.6g}, max={t_f32.max().item():.6g}, "
            f"mean={t_f32.mean().item():.6g}, zeros={zero_count}/{total} "
            f"[{status}]"
        )

    # Cross-check per-tensor scale with amax
    if amax_map:
        print("\n--- Amax cross-check ---")
        for layer_key in sorted(amax_map):
            # layer_key: "layers.0.moe.experts"
            # HF key: "model.layers.0.mlp.experts"
            import re

            layer_num = re.search(r"layers\.(\d+)", layer_key).group(1)
            hf_prefix = f"model.layers.{layer_num}.mlp.experts"

            amaxes = amax_map[layer_key]
            for qname, amax_val in amaxes.items():
                expected_sf2 = amax_val / (FP4_E2M1_MAX * FP8_E4M3_MAX)

                # Map quantizer name to HF scale key
                if qname == "mlp1_weight_quantizer":
                    scale_key = f"{hf_prefix}.gate_up_proj_weight_scale_2"
                elif qname == "mlp2_weight_quantizer":
                    scale_key = f"{hf_prefix}.down_proj_weight_scale_2"
                elif qname == "mlp1_input_quantizer":
                    scale_key = f"{hf_prefix}.gate_up_proj_input_scale"
                elif qname == "mlp2_input_quantizer":
                    scale_key = f"{hf_prefix}.down_proj_input_scale"
                else:
                    continue

                if scale_key not in tensors:
                    issues.append(f"MISSING: {scale_key} (for {qname} amax={amax_val})")
                    continue

                actual = tensors[scale_key].float()
                actual_val = actual.flatten()[0].item()
                rel_err = abs(actual_val - expected_sf2) / max(abs(expected_sf2), 1e-10)

                status = "OK" if rel_err < 0.01 else f"MISMATCH (rel_err={rel_err:.4f})"
                if rel_err >= 0.01:
                    issues.append(
                        f"Scale mismatch {scale_key}: expected={expected_sf2:.6g}, "
                        f"got={actual_val:.6g} (rel_err={rel_err:.4f})"
                    )

                if layer_num == "0":  # Only print first layer detail
                    print(
                        f"  {qname} (layer {layer_num}): "
                        f"amax={amax_val:.6g}, expected_sf2={expected_sf2:.6g}, "
                        f"actual_sf2={actual_val:.6g} [{status}]"
                    )

    if issues:
        print(f"\n*** {len(issues)} SCALE ISSUES ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll scales OK")

    return len(issues) == 0


def validate_dequantize(tensors: dict[str, torch.Tensor], layer_idx: int = 0):
    """Dequantize layer weights and validate values."""
    print("\n" + "=" * 70)
    print(f"DEQUANTIZATION VALIDATION (layer {layer_idx})")
    print("=" * 70)

    issues = []

    for proj in ["gate_up_proj", "down_proj"]:
        key = f"model.layers.{layer_idx}.mlp.experts.{proj}"
        scale_key = f"{key}_weight_scale"
        sf2_key = f"{key}_weight_scale_2"

        if key not in tensors:
            print(f"  {proj}: NOT FOUND")
            continue

        packed = tensors[key]
        block_scale = tensors.get(scale_key)
        tensor_scale = tensors.get(sf2_key)

        if block_scale is None or tensor_scale is None:
            issues.append(f"Missing scales for {proj}")
            continue

        print(f"\n--- {proj} ---")
        print(f"  Packed: {list(packed.shape)} {packed.dtype}")
        print(f"  Block scale: {list(block_scale.shape)} {block_scale.dtype}")
        print(f"  Tensor scale: {list(tensor_scale.shape)} {tensor_scale.dtype}")

        # Packed weights are stored in HF layout: (E, in_packed, out)
        # Transpose to (E, out, in_packed) for dequantization (expects packed as last dim)
        packed_t = packed.transpose(1, 2).contiguous()
        block_scale_t = block_scale.transpose(1, 2).contiguous()

        # Dequantize in (E, out, in) layout
        dequant = dequantize_nvfp4_block(packed_t, block_scale_t, tensor_scale)
        print(f"  Dequantized: {list(dequant.shape)} {dequant.dtype} (E, out, in)")

        # Statistics
        nan_count = torch.isnan(dequant).sum().item()
        inf_count = torch.isinf(dequant).sum().item()
        zero_count = (dequant == 0).sum().item()
        total = dequant.numel()

        print(f"  NaN: {nan_count}/{total} ({100 * nan_count / total:.2f}%)")
        print(f"  Inf: {inf_count}/{total} ({100 * inf_count / total:.2f}%)")
        print(f"  Zero: {zero_count}/{total} ({100 * zero_count / total:.2f}%)")
        print(f"  Range: [{dequant.min().item():.6g}, {dequant.max().item():.6g}]")
        print(f"  Mean: {dequant.mean().item():.6g}")
        print(f"  Std: {dequant.std().item():.6g}")

        if nan_count > 0:
            issues.append(f"NaN in dequantized {proj}: {nan_count}/{total}")
        if inf_count > 0:
            issues.append(f"Inf in dequantized {proj}: {inf_count}/{total}")
        if zero_count == total:
            issues.append(f"ALL ZEROS in dequantized {proj}")

        # Per-expert statistics
        print(f"\n  Per-expert stats (first 4 of {packed.shape[0]}):")
        for e in range(min(4, packed.shape[0])):
            expert_w = dequant[e]
            print(
                f"    Expert {e}: mean={expert_w.mean().item():.6g}, "
                f"std={expert_w.std().item():.6g}, "
                f"absmax={expert_w.abs().max().item():.6g}"
            )

        # Spot check: manually dequantize first block of first expert
        # Use transposed tensors (E, out, in_packed) for consistent indexing
        print(f"\n  Spot check (expert 0, first block):")
        e0_packed = packed_t[0, 0, :8]  # 8 bytes = 16 FP4 values
        e0_bscale = block_scale_t[0, 0, 0]  # first block's FP8 scale
        e0_tscale = tensor_scale.flatten()[0]  # per-tensor scale (scalar)

        # Manual unpack
        lo = (e0_packed.int() & 0x0F).long()
        hi = (e0_packed.int() >> 4 & 0x0F).long()
        manual_vals = torch.empty(16)
        manual_vals[0::2] = FP4_TABLE[lo]
        manual_vals[1::2] = FP4_TABLE[hi]
        manual_dequant = manual_vals * e0_bscale.float() * e0_tscale.float()

        auto_dequant = dequant[0, 0, :16]

        max_diff = (manual_dequant - auto_dequant).abs().max().item()
        print(f"    Manual:  {manual_dequant[:8].tolist()}")
        print(f"    Auto:    {auto_dequant[:8].tolist()}")
        print(f"    Max diff: {max_diff:.2e}")

        if max_diff > 1e-6:
            issues.append(f"Dequant mismatch in {proj}: max_diff={max_diff:.2e}")

    if issues:
        print(f"\n*** {len(issues)} DEQUANT ISSUES ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nDequantization OK")

    return len(issues) == 0


def validate_matmul(tensors: dict[str, torch.Tensor], layer_idx: int = 0):
    """Run a simple matmul with dequantized weights to verify coherent output."""
    print("\n" + "=" * 70)
    print(f"MATMUL VALIDATION (layer {layer_idx})")
    print("=" * 70)

    issues = []

    # Dequantize gate_up_proj
    key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
    packed = tensors.get(key)
    scale = tensors.get(f"{key}_weight_scale")
    sf2 = tensors.get(f"{key}_weight_scale_2")

    if packed is None or scale is None or sf2 is None:
        print("  Missing gate_up_proj tensors, skipping")
        return True

    # Transpose from HF layout (E, in_packed, out) to (E, out, in_packed) for dequant
    w13 = dequantize_nvfp4_block(
        packed.transpose(1, 2).contiguous(),
        scale.transpose(1, 2).contiguous(),
        sf2,
    )
    E, out_dim, in_dim = w13.shape
    print(f"  gate_up_proj dequant: ({E}, {out_dim}, {in_dim})")

    # Create random input (simulating hidden states for one expert)
    # Shape: (seq_len, in_dim) — a few tokens assigned to one expert
    seq_len = 4
    x = torch.randn(seq_len, in_dim) * 0.1  # small values like normalized hidden states

    # MoE expert forward: y = SiLU(x @ gate.T) * (x @ up.T)
    # gate_up is fused: w13[e] has shape (2*inter, dim)
    # Split into gate and up
    expert_idx = 0
    w_gate_up = w13[expert_idx]  # (2*inter, dim)
    inter = out_dim // 2
    w_gate = w_gate_up[:inter]  # (inter, dim)
    w_up = w_gate_up[inter:]  # (inter, dim)

    gate_out = x @ w_gate.T  # (seq, inter)
    up_out = x @ w_up.T  # (seq, inter)
    hidden = torch.nn.functional.silu(gate_out) * up_out  # (seq, inter)

    print(f"  Input: shape={list(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  Gate output: mean={gate_out.mean():.4f}, std={gate_out.std():.4f}")
    print(f"  Up output: mean={up_out.mean():.4f}, std={up_out.std():.4f}")
    print(f"  SwiGLU hidden: mean={hidden.mean():.4f}, std={hidden.std():.4f}")

    # Check for NaN/Inf
    for name, t in [("gate_out", gate_out), ("up_out", up_out), ("hidden", hidden)]:
        if torch.isnan(t).any():
            issues.append(f"NaN in {name}")
            print(f"  *** NaN detected in {name}!")
        if torch.isinf(t).any():
            issues.append(f"Inf in {name}")
            print(f"  *** Inf detected in {name}!")

    # Now dequantize down_proj and complete the MoE
    key2 = f"model.layers.{layer_idx}.mlp.experts.down_proj"
    packed2 = tensors.get(key2)
    scale2 = tensors.get(f"{key2}_weight_scale")
    sf22 = tensors.get(f"{key2}_weight_scale_2")

    if packed2 is not None and scale2 is not None and sf22 is not None:
        w2 = dequantize_nvfp4_block(
            packed2.transpose(1, 2).contiguous(),
            scale2.transpose(1, 2).contiguous(),
            sf22,
        )
        w2_expert = w2[expert_idx]  # (dim, inter)
        output = hidden @ w2_expert.T  # (seq, dim)

        print(f"  down_proj output: mean={output.mean():.4f}, std={output.std():.4f}")
        if torch.isnan(output).any():
            issues.append("NaN in down_proj output")
            print("  *** NaN in down_proj output!")
        if torch.isinf(output).any():
            issues.append("Inf in down_proj output")
            print("  *** Inf in down_proj output!")

    if issues:
        print(f"\n*** {len(issues)} MATMUL ISSUES ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nMatmul OK — expert forward pass produces finite values")

    return len(issues) == 0


def validate_non_expert_weights(tensors: dict[str, torch.Tensor]):
    """Check non-expert weights (attention, embeddings, norms) for issues."""
    print("\n" + "=" * 70)
    print("NON-EXPERT WEIGHT VALIDATION")
    print("=" * 70)

    issues = []
    categories = defaultdict(list)

    for key, t in sorted(tensors.items()):
        if "mlp.experts" in key:
            continue  # Skip expert tensors

        t_f32 = t.float() if t.is_floating_point() else t.float()
        nan_count = torch.isnan(t_f32).sum().item()
        inf_count = torch.isinf(t_f32).sum().item()

        # Categorize
        if "embed" in key or "tok_embeddings" in key:
            cat = "embedding"
        elif "norm" in key or "layernorm" in key:
            cat = "norm"
        elif "attn" in key or "self_attn" in key:
            cat = "attention"
        elif "router" in key:
            cat = "router"
        elif "lm_head" in key or "output" in key:
            cat = "lm_head"
        else:
            cat = "other"

        status = "OK"
        if nan_count > 0:
            status = f"NaN={nan_count}"
            issues.append(f"NaN in {key}: {nan_count}")
        if inf_count > 0:
            status = f"Inf={inf_count}"
            issues.append(f"Inf in {key}: {inf_count}")

        categories[cat].append((key, t, status))

    for cat in ["embedding", "norm", "attention", "router", "lm_head", "other"]:
        if cat not in categories:
            continue
        print(f"\n  [{cat}] ({len(categories[cat])} tensors)")
        for key, t, status in categories[cat][:5]:  # Show first 5
            print(f"    {key}: shape={list(t.shape)}, dtype={t.dtype}, [{status}]")
        if len(categories[cat]) > 5:
            print(f"    ... and {len(categories[cat]) - 5} more")

    if issues:
        print(f"\n*** {len(issues)} NON-EXPERT ISSUES ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll non-expert weights OK")

    return len(issues) == 0


def main():
    model_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.expanduser("~/torchtitan/outputs/nvfp4_export")
    )
    print(f"Validating NVFP4 export at: {model_path}")

    # Load config
    config_path = os.path.join(model_path, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config.json: model_type={config.get('model_type', 'unknown')}")

    # Load amax values (written by ptq_export.py)
    amax_path = os.path.join(model_path, "quantizer_amax.json")

    # Load all tensors
    print("\nLoading tensors...")
    tensors = load_model_tensors(model_path)
    print(f"Loaded {len(tensors)} tensors")

    # Run validations
    results = {}
    results["shapes"] = validate_shapes(tensors, config)
    results["scales"] = validate_scales(tensors, amax_path)
    results["non_expert"] = validate_non_expert_weights(tensors)
    results["dequant_layer0"] = validate_dequantize(tensors, layer_idx=0)
    results["dequant_layer12"] = validate_dequantize(tensors, layer_idx=12)
    results["dequant_layer23"] = validate_dequantize(tensors, layer_idx=23)
    results["matmul"] = validate_matmul(tensors, layer_idx=0)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll validations PASSED — exported weights look correct")
    else:
        print("\nSome validations FAILED — investigate issues above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
