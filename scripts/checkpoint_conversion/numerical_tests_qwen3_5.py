#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end numerical correctness test for Qwen3.5 checkpoint conversion.

Compares HuggingFace and TorchTitan next-token logits on multimodal inputs
(random image + text prompt). Each pipeline uses its own image preprocessing
so the test validates the full path: pixels → vision encoder → decoder → logits.

Usage:
    python -m scripts.checkpoint_conversion.numerical_tests_qwen3_5 \
        --hf_model_path hf_assets/Qwen/Qwen3.5-2B \
        --tt_checkpoint_path outputs/Qwen/qwen3_5_2b_dcp \
        --model_flavor 2B
"""

import argparse
import os
import types
from typing import Any

import einops as E
import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from PIL import Image

torch._dynamo.config.disable = True

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.utils.image import (
    process_image,
    vision_to_patches,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3_5 import model_registry, QWEN3_5_SPECIAL_TOKENS
from transformers import AutoModelForImageTextToText, AutoProcessor


# ============================================================
# Metrics
# ============================================================


def kl_divergence(logits_a, logits_b):
    """KL(softmax(b) || softmax(a)) — F.kl_div(log Q, P) computes KL(P || Q)."""
    return F.kl_div(
        F.log_softmax(logits_a, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction="batchmean",
    )


def top_k_match(logits_a, logits_b, k=5):
    """Return (top1_match_rate, topk_overlap_rate)."""
    topk_a = logits_a.topk(k, dim=-1).indices
    topk_b = logits_b.topk(k, dim=-1).indices
    top1 = (topk_a[..., 0] == topk_b[..., 0]).float().mean().item()
    overlap = (
        sum(
            (topk_a[..., i : i + 1] == topk_b).any(dim=-1).float().mean().item()
            for i in range(k)
        )
        / k
    )
    return top1, overlap


# ============================================================
# Input preparation
# ============================================================


def build_inputs(hf_model_path, model_flavor, num_samples, image_size=224):
    """Build paired HF / TT inputs from random images.

    Returns:
        hf_inputs: list of dicts (processor output, ready for HF model)
        tt_inputs: list of (input_ids, pixel_values, grid_thw, mrope_positions)
        pixel_comparisons: list of per-sample pixel diff stats
        special_tokens: {"image_id", "video_id"} from the tokenizer
    """
    # Annotate as Any: AutoProcessor.from_pretrained is typed Optional, which
    # trips the .apply_chat_template call on environments without transformers stubs.
    processor: Any = AutoProcessor.from_pretrained(hf_model_path)

    model_config = model_registry(model_flavor).model
    # pyrefly: ignore [missing-attribute]
    encoder_config = model_config.vision_encoder
    patch_size = encoder_config.patch_size
    temporal_patch_size = encoder_config.temporal_patch_size
    merge_size = encoder_config.spatial_merge_size

    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        QWEN3_5_SPECIAL_TOKENS["image_token"]
    )
    video_token_id = processor.tokenizer.convert_tokens_to_ids(
        QWEN3_5_SPECIAL_TOKENS["video_token"]
    )
    special_tokens = {"image_id": image_token_id, "video_id": video_token_id}
    # Reuse the collator's MRoPE builder (only needs spatial_merge_size) so the
    # 3D position IDs match the training path exactly.
    mrope_builder = types.SimpleNamespace(spatial_merge_size=merge_size)

    hf_inputs, tt_inputs, pixel_comparisons = [], [], []

    for i in range(num_samples):
        rng = torch.Generator().manual_seed(42 + i)
        img_array = (
            torch.randint(0, 256, (image_size, image_size, 3), generator=rng)
            .numpy()
            .astype("uint8")
        )
        pil_image = Image.fromarray(img_array)

        # --- HF path ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        hf_in = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        hf_inputs.append(hf_in)

        # --- TT path ---
        img_tensor = process_image(
            pil_image, patch_size=patch_size, merge_size=merge_size
        )
        assert img_tensor is not None, f"process_image failed for sample {i}"
        patches, grid_thw = vision_to_patches(
            img_tensor,
            patch_size,
            temporal_patch_size,
            merge_size,
        )
        # 3D MRoPE positions (1, S, 3); positions=None → single document.
        mrope_positions = MultiModalCollator._build_mrope_positions(
            mrope_builder,
            hf_in["input_ids"],
            grid_thw.unsqueeze(0),
            None,
            None,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        tt_inputs.append(
            (
                hf_in["input_ids"],
                patches.unsqueeze(0),
                grid_thw.unsqueeze(0),
                mrope_positions,
            )
        )

        # --- Compare pixel values in image space ---
        # Reshape patches back to image space for pixel-level comparison.
        # Only compare frame 0 (actual image); frame 1 is temporal padding.
        hf_pv = hf_in["pixel_values"]
        t_p, h_p, w_p = grid_thw.tolist()
        pattern = "(t bh bw m n) (c pt ph pw) -> (t pt) (bh m ph) (bw n pw) c"
        kwargs = dict(
            t=t_p,
            bh=h_p // merge_size,
            bw=w_p // merge_size,
            m=merge_size,
            n=merge_size,
            c=3,
            pt=temporal_patch_size,
            ph=patch_size,
            pw=patch_size,
        )
        hf_img = E.rearrange(hf_pv, pattern, **kwargs)
        tt_img = E.rearrange(patches, pattern, **kwargs)
        pixel_comparisons.append(_compare_images(hf_img[:1], tt_img[:1], i))

    return hf_inputs, tt_inputs, pixel_comparisons, special_tokens


def _compare_images(hf_img, tt_img, sample_idx):
    diff = (hf_img.float() - tt_img.float()).abs()
    return {
        "sample": sample_idx,
        "max_diff": diff.max().item(),
        "num_differ": (diff > 1e-6).sum().item(),
        "total_pixels": diff.numel(),
    }


def print_pixel_comparisons(comparisons):
    print(f"\n{'=' * 60}")
    print("Pixel Value Comparison (reconstructed to image space)")
    print(f"{'=' * 60}")

    for c in comparisons:
        print(
            f"  Sample {c['sample'] + 1:2d}:  "
            f"differing pixels: {c['num_differ']}/{c['total_pixels']}  "
            f"max_diff={c['max_diff']:.2e}"
        )

    total_differ = sum(c["num_differ"] for c in comparisons)
    total_pixels = sum(c["total_pixels"] for c in comparisons)
    avg_max = sum(c["max_diff"] for c in comparisons) / len(comparisons)
    print(
        f"\n  Total: {total_differ}/{total_pixels} pixels differ, avg max_diff={avg_max:.2e}"
    )
    if avg_max < 1e-2:
        print("  Pixel preprocessing: MATCH")
    else:
        print("  Pixel preprocessing: DIFFER (check pipelines)")


# ============================================================
# HuggingFace inference
# ============================================================


@torch.no_grad()
def run_hf(model_path, hf_inputs, device):
    """Run HF model, return last-token logits per sample."""
    print(f"Loading HuggingFace model on {device} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    outputs = []
    for i, inp in enumerate(hf_inputs):
        inp = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inp.items()
        }
        logits = model(**inp).logits
        outputs.append(logits[:, -1:, :].cpu())
        print(f"  HF  {i + 1}/{len(hf_inputs)}")

    del model
    torch.cuda.empty_cache()
    return outputs


# ============================================================
# TorchTitan inference
# ============================================================


@torch.no_grad()
def run_tt(model_flavor, checkpoint_path, tt_inputs, special_tokens, device):
    """Run TT model, return last-token logits per sample."""
    print(f"Loading TorchTitan model on {device} ...")

    model_config = model_registry(model_flavor).model
    with torch.device("meta"):
        model = model_config.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    model.half()

    state_dict = ModelWrapper(model)._get_state_dict()
    print(f"  Loading checkpoint: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    model.to(device)

    # Replace FlexAttention with SDPA for single-process inference
    # (unfused FlexAttention without torch.compile has poor fp16 numerics).
    for layer in model.layers.values():
        if layer.full_attn:
            layer.attn.inner_attention = ScaledDotProductAttention.Config().build()

    class _BidirectionalSDPA(torch.nn.Module):
        def forward(self, q, k, v, **kwargs):
            # q/k/v: (bs, seq, heads, dim) -> transpose to (bs, heads, seq, dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            return out.transpose(1, 2)

    for blk in model.vision_encoder.layers.values():
        blk.attn.flex_attention = _BidirectionalSDPA()

    model.eval()

    outputs = []
    for i, (tokens, pixel_values, grid_thw, mrope_positions) in enumerate(tt_inputs):
        logits = model(
            tokens.to(device),
            pixel_values=pixel_values.half().to(device),
            grid_thw=grid_thw.to(device),
            mrope_positions=mrope_positions.to(device),
            special_tokens=special_tokens,
        )
        outputs.append(logits[:, -1:, :].cpu())
        print(f"  TT  {i + 1}/{len(tt_inputs)}")

    del model
    torch.cuda.empty_cache()
    return outputs


# ============================================================
# Comparison
# ============================================================


def compare(hf_outputs, tt_outputs):
    """Print per-sample and average logit comparison metrics."""
    print(f"\n{'=' * 60}")
    print("Logit Comparison")
    print(f"{'=' * 60}")

    total_kl, total_top1, total_top5 = 0.0, 0.0, 0.0
    for i, (hf, tt) in enumerate(zip(hf_outputs, tt_outputs)):
        hf, tt = hf.float(), tt.float()
        kl = kl_divergence(hf, tt).item()
        top1, top5 = top_k_match(hf, tt)
        cos = F.cosine_similarity(hf.flatten(), tt.flatten(), dim=0).item()
        max_diff = (hf - tt).abs().max().item()
        total_kl += kl
        total_top1 += top1
        total_top5 += top5
        print(
            f"  Sample {i + 1:2d}:  KL={kl:.4e}  cos={cos:.6f}  "
            f"max_diff={max_diff:.4e}  top1={top1:.0%}  top5={top5:.0%}"
        )

    n = len(hf_outputs)
    avg_kl = total_kl / n
    print(
        f"\n  Average:   KL={avg_kl:.4e}  "
        f"top1={total_top1 / n:.0%}  top5={total_top5 / n:.0%}"
    )

    if avg_kl < 1e-6:
        print("  Excellent: near-exact match")
    elif avg_kl < 1e-3:
        print("  WARN: small numerical differences")
    else:
        print("  FAIL: significant differences")

    return avg_kl


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end numerical correctness test for Qwen3.5.",
    )
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--tt_checkpoint_path", type=str, required=True)
    parser.add_argument("--model_flavor", type=str, default="4B")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    assert os.path.exists(args.hf_model_path), f"Not found: {args.hf_model_path}"
    assert os.path.exists(
        args.tt_checkpoint_path
    ), f"Not found: {args.tt_checkpoint_path}"

    device = torch.device("cuda:0")
    print(f"Using {torch.cuda.get_device_name(0)}")

    print(f"\nBuilding {args.num_samples} test samples ...")
    hf_inputs, tt_inputs, pixel_comparisons, special_tokens = build_inputs(
        args.hf_model_path,
        args.model_flavor,
        args.num_samples,
    )
    print_pixel_comparisons(pixel_comparisons)

    print("\nRunning HuggingFace inference ...")
    hf_outputs = run_hf(args.hf_model_path, hf_inputs, device)

    print("\nRunning TorchTitan inference ...")
    tt_outputs = run_tt(
        args.model_flavor,
        args.tt_checkpoint_path,
        tt_inputs,
        special_tokens,
        device,
    )

    compare(hf_outputs, tt_outputs)


if __name__ == "__main__":
    main()
