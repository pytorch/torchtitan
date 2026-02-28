#!/usr/bin/env python3
"""Find where NaN first appears in the NVFP4 model forward pass."""

import torch
import sys


def main():
    model_path = (
        sys.argv[1] if len(sys.argv) > 1 else "/home/w/torchtitan/outputs/nvfp4_export"
    )

    # Set up vLLM offline
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )

    # Get access to the model through the engine core process
    # For vLLM v1, we need to probe via generation with hooks
    # Instead, let's use a simple approach: register forward hooks on all modules
    # in the engine process

    # Actually, with vLLM v1's multiprocessing, we can't easily hook the model
    # from the main process. Let's instead check the safetensors weights
    # by simulating what vLLM does with them.

    print("\n=== Simulating NVFP4 MoE forward pass ===")

    # Load weights directly
    from safetensors import safe_open
    import json

    with open(f"{model_path}/config.json") as f:
        config = json.load(f)

    # Load layer 0 MoE weights
    f = safe_open(f"{model_path}/model-00001-of-00003.safetensors", framework="pt")

    w13 = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj").cuda()
    w13_scale = f.get_tensor(
        "model.layers.0.mlp.experts.gate_up_proj_weight_scale"
    ).cuda()
    w13_scale_2 = f.get_tensor(
        "model.layers.0.mlp.experts.gate_up_proj_weight_scale_2"
    ).cuda()
    a13_scale = f.get_tensor(
        "model.layers.0.mlp.experts.gate_up_proj_input_scale"
    ).cuda()
    w2 = f.get_tensor("model.layers.0.mlp.experts.down_proj").cuda()
    w2_scale = f.get_tensor("model.layers.0.mlp.experts.down_proj_weight_scale").cuda()
    w2_scale_2 = f.get_tensor(
        "model.layers.0.mlp.experts.down_proj_weight_scale_2"
    ).cuda()
    a2_scale = f.get_tensor("model.layers.0.mlp.experts.down_proj_input_scale").cuda()

    print(f"w13: {w13.shape} {w13.dtype}")
    print(f"w13_scale: {w13_scale.shape} {w13_scale.dtype}")
    print(f"w13_scale_2: {w13_scale_2.shape} {w13_scale_2.dtype}")
    print(f"a13_scale: {a13_scale.shape} {a13_scale.dtype}")
    print(f"w2: {w2.shape} {w2.dtype}")
    print(f"w2_scale: {w2_scale.shape} {w2_scale.dtype}")
    print(f"w2_scale_2: {w2_scale_2.shape} {w2_scale_2.dtype}")
    print(f"a2_scale: {a2_scale.shape} {a2_scale.dtype}")

    # Simulate process_weights_after_loading
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        swizzle_blockscale,
    )

    print("\n--- Swizzling scales ---")
    w13_scale_sw = swizzle_blockscale(w13_scale)
    print(
        f"w13_scale swizzled: {w13_scale_sw.shape}, nan={torch.isnan(w13_scale_sw.float()).any()}"
    )

    pad_size = w13_scale_sw.size(1) - w13.size(1)
    print(f"pad_size: {pad_size}")

    w2_scale_sw = swizzle_blockscale(w2_scale)
    print(
        f"w2_scale swizzled: {w2_scale_sw.shape}, nan={torch.isnan(w2_scale_sw.float()).any()}"
    )

    # Collapse per-tensor scales
    w13_s2 = w13_scale_2[:, 0].contiguous()
    a13_s = a13_scale.max(dim=1).values.to(torch.float32)
    print(f"\nw13_scale_2 collapsed: {w13_s2.shape} vals={w13_s2[:3].tolist()}")
    print(f"a13_scale collapsed: {a13_s.shape} vals={a13_s[:3].tolist()}")

    # Manual dequantize of one expert's one block to verify
    print("\n--- Manual dequantize check ---")
    expert_idx = 0
    # Take first block of 16 elements from w13
    packed_block = w13[expert_idx, 0, :8]  # 8 uint8 = 16 fp4 values
    block_scale = w13_scale[expert_idx, 0, 0]  # fp8 scale for this block
    tensor_scale = w13_scale_2[expert_idx, 0]  # fp32 per-tensor scale

    print(f"Packed bytes: {packed_block.tolist()}")
    print(f"Block scale (fp8): {block_scale.float().item()}")
    print(f"Tensor scale (fp32): {tensor_scale.item()}")

    # Unpack FP4: lower nibble first, upper nibble second
    unpacked = []
    for byte_val in packed_block.tolist():
        lo = byte_val & 0x0F
        hi = (byte_val >> 4) & 0x0F
        unpacked.extend([lo, hi])

    # FP4 E2M1 decode table
    fp4_table = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    fp4_vals = [fp4_table[v] for v in unpacked]
    print(f"FP4 values: {fp4_vals}")

    # Dequantize
    combined_scale = block_scale.float().item() * tensor_scale.item()
    dequant = [v * combined_scale for v in fp4_vals]
    print(f"Dequantized: {dequant}")
    print(f"Any NaN in dequant: {any(x != x for x in dequant)}")


if __name__ == "__main__":
    main()
