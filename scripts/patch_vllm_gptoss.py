#!/usr/bin/env python3
"""Patch vLLM's gpt_oss.py _load_weights_nvfp4 for official ModelOpt format.

Three categories of fixes:

1. MoE weights/scales layout: Official ModelOpt exports use (E, in_packed, out)
   but vLLM expects (E, out, in_packed). Add .permute(0, 2, 1) after loading.

2. Scalar scales: Official ModelOpt exports weight_scale_2 and input_scale as
   scalar tensors (shape []), but vLLM allocates (E, 2) or (E,) buffers.
   Use .fill_() to broadcast.

3. Attention NVFP4: Official ModelOpt also quantizes attention layers. vLLM's
   attention TP slicing needs adjustment for packed uint8 / fp8 scale dims,
   plus scalar scale handling (weight_scale_2, input_scale, k_scale, v_scale).

Usage:
    python3 patch_vllm_gptoss.py                          # default path
    python3 patch_vllm_gptoss.py /path/to/gpt_oss.py      # custom path
"""

import shutil
import sys

DEFAULT_PATH = (
    "/home/w/vllm_venv/lib/python3.12/site-packages/"
    "vllm/model_executor/models/gpt_oss.py"
)

# Each (old, new) pair is a targeted replacement within _load_weights_nvfp4.

REPLACEMENTS = [
    # ── Fix 1a: w13_weight_scale — permute from (E, in/16, out) ──
    (
        '            if ".w13_weight_scale" in name and "_scale_2" not in name:\n'
        "                # Per-block FP8 scale: (E, inter*2, hidden//block_size)\n"
        "                if use_ep:",
        '            if ".w13_weight_scale" in name and "_scale_2" not in name:\n'
        "                # Per-block FP8 scale: ckpt (E, in/16, out) → (E, out, in/16)\n"
        "                weight = weight.permute(0, 2, 1).contiguous()\n"
        "                if use_ep:",
    ),
    # ── Fix 1b: w2_weight_scale — permute from (E, in/16, out) ──
    (
        '            elif ".w2_weight_scale" in name and "_scale_2" not in name:\n'
        "                # Per-block FP8 scale: (E, hidden, inter//block_size)\n"
        "                if use_ep:",
        '            elif ".w2_weight_scale" in name and "_scale_2" not in name:\n'
        "                # Per-block FP8 scale: ckpt (E, in/16, out) → (E, out, in/16)\n"
        "                weight = weight.permute(0, 2, 1).contiguous()\n"
        "                if use_ep:",
    ),
    # ── Fix 1c: w13_weight — permute from (E, in/2, out) ──
    (
        '            elif ".w13_weight" in name and "_scale" not in name:\n'
        "                # Packed uint8: (E, inter*2, hidden//2)\n"
        "                if use_ep:",
        '            elif ".w13_weight" in name and "_scale" not in name:\n'
        "                # Packed uint8: ckpt (E, in/2, out) → (E, out, in/2)\n"
        "                weight = weight.permute(0, 2, 1).contiguous()\n"
        "                if use_ep:",
    ),
    # ── Fix 1d: w2_weight — permute from (E, in/2, out) ──
    (
        '            elif ".w2_weight" in name and "_scale" not in name:\n'
        "                # Packed uint8: (E, hidden, inter//2)\n"
        "                if use_ep:",
        '            elif ".w2_weight" in name and "_scale" not in name:\n'
        "                # Packed uint8: ckpt (E, in/2, out) → (E, out, in/2)\n"
        "                weight = weight.permute(0, 2, 1).contiguous()\n"
        "                if use_ep:",
    ),
    # ── Fix 2: Scalar MoE scale expansion ──
    (
        "                # Per-tensor scales and router: no TP narrowing needed\n"
        "                param = params_dict.get(name)\n"
        "                if param is not None:\n"
        "                    param.data.copy_(weight)\n"
        "                    loaded_params.add(name)\n"
        "                continue",
        "                # Per-tensor scales and router: no TP narrowing needed\n"
        "                param = params_dict.get(name)\n"
        "                if param is not None:\n"
        "                    # Official ModelOpt: scalar [] scales → broadcast to (E,2)/(E,)\n"
        "                    if weight.dim() == 0 and param.data.dim() > 0:\n"
        "                        param.data.fill_(weight.item())\n"
        "                    else:\n"
        "                        param.data.copy_(weight)\n"
        "                    loaded_params.add(name)\n"
        "                continue",
    ),
    # ── Fix 3a: Attention q/k/v — handle scalar scales ──
    (
        '            if "q_proj" in name or "k_proj" in name or "v_proj" in name:\n'
        "                # QKV: shard by heads\n"
        '                if name.endswith(".bias"):',
        '            if "q_proj" in name or "k_proj" in name or "v_proj" in name:\n'
        "                # QKV: shard by heads\n"
        "                if weight.dim() == 0:\n"
        "                    # Scalar (weight_scale_2, input_scale, k_scale, v_scale)\n"
        "                    if param.dim() == 0:\n"
        "                        param.data.copy_(weight)\n"
        "                    else:\n"
        "                        param.data.fill_(weight.item())\n"
        "                    loaded_params.add(name)\n"
        '                elif name.endswith(".bias"):',
    ),
    # ── Fix 3b: Attention o_proj — handle packed weights + scalar scales ──
    (
        '            elif "o_proj" in name:\n'
        '                if name.endswith(".bias"):\n'
        "                    param.data.copy_(weight)\n"
        "                else:\n"
        "                    head_start_idx = head_start * self.config.head_dim\n"
        "                    head_end_idx = (head_start + heads_per_rank) * self.config.head_dim\n"
        "                    narrow_weight = weight[:, head_start_idx:head_end_idx]\n"
        "                    param.data.copy_(narrow_weight)\n"
        "                loaded_params.add(name)",
        '            elif "o_proj" in name:\n'
        "                if weight.dim() == 0:\n"
        "                    # Scalar (weight_scale_2, input_scale)\n"
        "                    if param.dim() == 0:\n"
        "                        param.data.copy_(weight)\n"
        "                    else:\n"
        "                        param.data.fill_(weight.item())\n"
        '                elif name.endswith(".bias"):\n'
        "                    param.data.copy_(weight)\n"
        "                else:\n"
        "                    head_start_idx = head_start * self.config.head_dim\n"
        "                    head_end_idx = (head_start + heads_per_rank) * self.config.head_dim\n"
        "                    if weight.dtype == torch.uint8:\n"
        "                        # NVFP4 packed: in_features ÷2\n"
        "                        narrow_weight = weight[:, head_start_idx // 2:head_end_idx // 2]\n"
        "                    elif weight.element_size() == 1 and weight.dim() == 2:\n"
        "                        # FP8 block scale: in_features ÷ block_size\n"
        "                        narrow_weight = weight[:, head_start_idx // nvfp4_block:head_end_idx // nvfp4_block]\n"
        "                    else:\n"
        "                        narrow_weight = weight[:, head_start_idx:head_end_idx]\n"
        "                    param.data.copy_(narrow_weight)\n"
        "                loaded_params.add(name)",
    ),
]


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    print(f"Reading {filepath}...")
    with open(filepath) as f:
        content = f.read()

    # Check if already patched
    if "ckpt (E, in/16, out)" in content:
        print("Already patched! Skipping.")
        return 0

    # Check prerequisite: _load_weights_nvfp4 must exist
    if "_load_weights_nvfp4" not in content:
        print("ERROR: _load_weights_nvfp4 method not found in file.")
        print("This patch is for vLLM versions that already have NVFP4 support.")
        return 1

    # Apply all replacements
    for i, (old, new) in enumerate(REPLACEMENTS):
        if old not in content:
            print(f"ERROR: Replacement {i} — could not find expected block.")
            # Show first line to help debug
            first_line = old.strip().splitlines()[0]
            print(f"  Looking for: {first_line!r}")
            return 1
        content = content.replace(old, new, 1)
        desc = new.strip().splitlines()[0][:72]
        print(f"  [{i}] {desc}")

    # Write backup
    backup_path = filepath + ".bak"
    print(f"\nWriting backup to {backup_path}...")
    shutil.copy2(filepath, backup_path)

    print(f"Writing patched file to {filepath}...")
    with open(filepath, "w") as f:
        f.write(content)

    # Verify
    with open(filepath) as f:
        verify = f.read()

    checks = [
        "weight.permute(0, 2, 1).contiguous()",  # MoE permute
        "param.data.fill_(weight.item())",  # scalar broadcast
        "weight.dtype == torch.uint8",  # o_proj packed
    ]
    all_ok = all(c in verify for c in checks)

    if all_ok:
        print(f"\nSUCCESS: All {len(REPLACEMENTS)} patches applied and verified.")
        return 0
    else:
        missing = [c for c in checks if c not in verify]
        print(f"\nERROR: Verification failed! Missing: {missing}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
