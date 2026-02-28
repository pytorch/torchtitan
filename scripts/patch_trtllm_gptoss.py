#!/usr/bin/env python3
"""Patch TRT-LLM's modeling_gpt_oss.py to support NVFP4 MoE weight loading.

TRT-LLM 1.1.0's load_hf_weights() only handles BF16 and MXFP4 MoE weights.
Our NVFP4 export uses different tensor names and formats:
  - gate_up_proj (uint8 packed) + gate_up_proj_weight_scale (fp8)
  - gate_up_proj_weight_scale_2 (fp32 per-expert) + gate_up_proj_input_scale (fp32)

This patch adds an NVFP4 branch that converts fused gate_up_proj tensors into
per-expert VANILLA-mode keys that the existing NVFP4CutlassFusedMoEMethod
can load correctly.

Usage:
    # Run inside the TRT-LLM Docker container:
    python3 /workspace/patch_trtllm_gptoss.py

    # Or from host:
    docker exec trtllm_debug python3 /workspace/patch_trtllm_gptoss.py
"""

import sys

FILEPATH = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_gpt_oss.py"

# The old MoE loading block in load_hf_weights() — from "if isinstance(module, MoE):"
# through the end of the try/except block, just BEFORE "module.load_weights(weights=[moe_weights])"
OLD_BLOCK = """\
            if isinstance(module, MoE):
                try:
                    # For BF16 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights['gate_up_proj']
                    gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :,
                                                                         1::2]
                    gate_up_weight = torch.cat([gate, up], dim=-1)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate, up], dim=-1)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight.to(self.model.dtype)[i, :, :]
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj'][i, :, :].to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ]
                    }
                except:
                    # For MXFP4 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights[
                        'gate_up_proj_blocks'].flatten(-2, -1)
                    gate_weight, up_weight = gate_up_weight[:, ::
                                                            2, :], gate_up_weight[:,
                                                                                  1::
                                                                                  2, :]
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=-2)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate_bias, up_bias = gate_up_bias[:, ::
                                                      2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate_bias, up_bias], dim=-1)
                    gate_up_weight_scale = module_weights['gate_up_proj_scales']
                    gate_weight_scale, up_weight_scale = gate_up_weight_scale[:, ::
                                                                              2, :], gate_up_weight_scale[:,
                                                                                                          1::
                                                                                                          2, :]
                    gate_up_weight_scale = torch.cat(
                        [gate_weight_scale, up_weight_scale], dim=-2)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj_blocks'].flatten(
                                -2, -1)[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ],
                        'gate_up_proj_weight_scale': [
                            gate_up_weight_scale[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj_weight_scale': [
                            module_weights['down_proj_scales']
                            [i, :, :].transpose(0, 1) for i in range(num_expert)
                        ]
                    }

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[
                                f"{i}.w1.weight_scale_inv"] = gate_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w3.weight_scale_inv"] = up_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w2.weight_scale_inv"] = module_weights[
                                    'down_proj_scales'][i, :, :]

                module.load_weights(weights=[moe_weights])"""

NEW_BLOCK = """\
            if isinstance(module, MoE):
                # Detect NVFP4 format: uint8 packed weights + fp8 block scales
                _is_nvfp4 = ('gate_up_proj' in module_weights
                             and module_weights['gate_up_proj'].dtype == torch.uint8
                             and 'gate_up_proj_weight_scale' in module_weights)

                if _is_nvfp4:
                    # NVFP4 W4A4 path: split fused gate_up into per-expert
                    # VANILLA-mode keys for NVFP4CutlassFusedMoEMethod.
                    #
                    # HF format stores packed weights as (E, in_packed, out)
                    # and scale as (E, in_packed//8, out). Transpose to
                    # (E, out, in_packed) since TRT-LLM expects (out, in_packed).
                    _gu = module_weights['gate_up_proj'].transpose(1, 2).contiguous()
                    _gu_ws = module_weights['gate_up_proj_weight_scale'].transpose(1, 2).contiguous()
                    _gu_ws2 = module_weights['gate_up_proj_weight_scale_2']  # scalar
                    _gu_is = module_weights['gate_up_proj_input_scale']  # scalar
                    _gu_b = module_weights['gate_up_proj_bias']
                    _dp = module_weights['down_proj'].transpose(1, 2).contiguous()
                    _dp_ws = module_weights['down_proj_weight_scale'].transpose(1, 2).contiguous()
                    _dp_ws2 = module_weights['down_proj_weight_scale_2']  # scalar
                    _dp_is = module_weights['down_proj_input_scale']  # scalar
                    _dp_b = module_weights['down_proj_bias']
                    # After transpose: _gu is (E, 2*inter, in_packed)
                    _half = _gu.shape[1] // 2

                    moe_weights = {}
                    for i in range(num_expert):
                        # Packed uint8 weights — gate=[:_half], up=[_half:]
                        moe_weights[f"{i}.w1.weight"] = _gu[i, :_half, :]
                        moe_weights[f"{i}.w3.weight"] = _gu[i, _half:, :]
                        moe_weights[f"{i}.w2.weight"] = _dp[i]
                        # FP8 per-block scales
                        moe_weights[f"{i}.w1.weight_scale"] = _gu_ws[i, :_half, :]
                        moe_weights[f"{i}.w3.weight_scale"] = _gu_ws[i, _half:, :]
                        moe_weights[f"{i}.w2.weight_scale"] = _dp_ws[i]
                        # FP32 per-tensor scales (scalar — same for gate and up)
                        moe_weights[f"{i}.w1.weight_scale_2"] = _gu_ws2
                        moe_weights[f"{i}.w3.weight_scale_2"] = _gu_ws2
                        moe_weights[f"{i}.w2.weight_scale_2"] = _dp_ws2
                        # Input scales (scalar — same for gate and up)
                        moe_weights[f"{i}.w1.input_scale"] = _gu_is
                        moe_weights[f"{i}.w3.input_scale"] = _gu_is
                        moe_weights[f"{i}.w2.input_scale"] = _dp_is
                        # Bias — these remain (E, 2*inter), split at half
                        _b_half = _gu_b.shape[1] // 2
                        moe_weights[f"{i}.w1.bias"] = _gu_b[i, :_b_half]
                        moe_weights[f"{i}.w3.bias"] = _gu_b[i, _b_half:]
                        moe_weights[f"{i}.w2.bias"] = _dp_b[i]

                    # Switch to VANILLA mode for per-expert scale loading
                    module.weight_loading_mode = MoEWeightLoadingMode.VANILLA
                else:
                    try:
                        # For BF16 ckpt.
                        # Deinterleave for gate and up.
                        gate_up_weight = module_weights['gate_up_proj']
                        gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :,
                                                                             1::2]
                        gate_up_weight = torch.cat([gate, up], dim=-1)
                        gate_up_bias = module_weights['gate_up_proj_bias']
                        gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                        gate_up_bias = torch.cat([gate, up], dim=-1)
                        moe_weights = {
                            'gate_up_proj': [
                                gate_up_weight.to(self.model.dtype)[i, :, :]
                                for i in range(num_expert)
                            ],
                            'down_proj': [
                                module_weights['down_proj'][i, :, :].to(
                                    self.model.dtype) for i in range(num_expert)
                            ],
                            'gate_up_proj.bias':
                            [gate_up_bias[i, :] for i in range(num_expert)],
                            'down_proj.bias': [
                                module_weights['down_proj_bias'][i, :]
                                for i in range(num_expert)
                            ]
                        }
                    except:
                        # For MXFP4 ckpt.
                        # Deinterleave for gate and up.
                        gate_up_weight = module_weights[
                            'gate_up_proj_blocks'].flatten(-2, -1)
                        gate_weight, up_weight = gate_up_weight[:, ::
                                                                2, :], gate_up_weight[:,
                                                                                      1::
                                                                                      2, :]
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=-2)
                        gate_up_bias = module_weights['gate_up_proj_bias']
                        gate_bias, up_bias = gate_up_bias[:, ::
                                                          2], gate_up_bias[:, 1::2]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=-1)
                        gate_up_weight_scale = module_weights['gate_up_proj_scales']
                        gate_weight_scale, up_weight_scale = gate_up_weight_scale[:, ::
                                                                                  2, :], gate_up_weight_scale[:,
                                                                                                              1::
                                                                                                              2, :]
                        gate_up_weight_scale = torch.cat(
                            [gate_weight_scale, up_weight_scale], dim=-2)
                        moe_weights = {
                            'gate_up_proj': [
                                gate_up_weight[i, :, :].transpose(0, 1)
                                for i in range(num_expert)
                            ],
                            'down_proj': [
                                module_weights['down_proj_blocks'].flatten(
                                    -2, -1)[i, :, :].transpose(0, 1)
                                for i in range(num_expert)
                            ],
                            'gate_up_proj.bias':
                            [gate_up_bias[i, :] for i in range(num_expert)],
                            'down_proj.bias': [
                                module_weights['down_proj_bias'][i, :]
                                for i in range(num_expert)
                            ],
                            'gate_up_proj_weight_scale': [
                                gate_up_weight_scale[i, :, :].transpose(0, 1)
                                for i in range(num_expert)
                            ],
                            'down_proj_weight_scale': [
                                module_weights['down_proj_scales']
                                [i, :, :].transpose(0, 1) for i in range(num_expert)
                            ]
                        }

                        if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                            for i in range(num_expert):
                                moe_weights[
                                    f"{i}.w1.weight_scale_inv"] = gate_weight_scale[
                                        i, :, :]
                                moe_weights[
                                    f"{i}.w3.weight_scale_inv"] = up_weight_scale[
                                        i, :, :]
                                moe_weights[
                                    f"{i}.w2.weight_scale_inv"] = module_weights[
                                        'down_proj_scales'][i, :, :]

                module.load_weights(weights=[moe_weights])"""


def main():
    print(f"Reading {FILEPATH}...")
    with open(FILEPATH) as f:
        content = f.read()

    if "_is_nvfp4" in content:
        print("Already patched! Skipping.")
        return 0

    if OLD_BLOCK not in content:
        print("ERROR: Could not find the expected MoE loading block.")
        print("The file may have been modified or is a different version.")
        # Try to find approximate location
        if "isinstance(module, MoE)" in content:
            print("Found 'isinstance(module, MoE)' — the block structure may differ.")
        else:
            print("Could not find MoE loading code at all!")
        return 1

    new_content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)

    if new_content == content:
        print("ERROR: Replacement had no effect.")
        return 1

    # Write backup
    backup_path = FILEPATH + ".bak"
    print(f"Writing backup to {backup_path}...")
    with open(backup_path, "w") as f:
        f.write(content)

    print(f"Writing patched file to {FILEPATH}...")
    with open(FILEPATH, "w") as f:
        f.write(new_content)

    # Verify
    with open(FILEPATH) as f:
        verify = f.read()

    if "_is_nvfp4" in verify and "NVFP4 W4A4 path" in verify:
        print("SUCCESS: NVFP4 MoE loading patch applied.")
        return 0
    else:
        print("ERROR: Verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
