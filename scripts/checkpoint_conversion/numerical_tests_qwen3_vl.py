#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerical correctness test for Qwen3-VL checkpoint conversion.

Compares the next-token probability distributions of a HuggingFace Qwen3-VL
model with a TorchTitan Qwen3-VL model loaded from a converted DCP checkpoint,
using multimodal inputs (image + text).

Usage:
    # Basic usage (auto-splits GPUs between HF and TT):
    python scripts/checkpoint_conversion/numerical_tests_qwen3_vl.py \
        --hf_model_path /path/to/Qwen3-VL-2B-Instruct \
        --tt_checkpoint_path /path/to/qwen3_vl_2b_dcp

    # Specify model flavor and number of samples:
    python scripts/checkpoint_conversion/numerical_tests_qwen3_vl.py \
        --hf_model_path /path/to/Qwen3-VL-8B-Instruct \
        --tt_checkpoint_path /path/to/qwen3_vl_8b_dcp \
        --model_flavor 8B \
        --num_samples 10

    # Explicit GPU assignment:
    python scripts/checkpoint_conversion/numerical_tests_qwen3_vl.py \
        --hf_model_path /path/to/Qwen3-VL-2B-Instruct \
        --tt_checkpoint_path /path/to/qwen3_vl_2b_dcp \
        --hf_gpus 0 1 2 3 \
        --tt_gpus 4 5 6 7

Available model flavors: 2B, 8B, 30B-A3B, 235B-A22B
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
torchtitan_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, torchtitan_dir)

import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

# Disable torch.compile for eager-mode inference testing
torch._dynamo.config.disable = True
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.experiments.qwen3_vl import model_registry
from torchtitan.experiments.qwen3_vl.model import SpecialTokens
from transformers import AutoProcessor


def loss_fn(logits1, logits2):
    """Calculate KL divergence loss between two sets of logits."""
    probs1 = F.log_softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    return F.kl_div(probs1, probs2, reduction="mean")


def top_k_match(logits1, logits2, k=5):
    """Check if top-k token predictions match."""
    topk1 = logits1.topk(k, dim=-1).indices
    topk2 = logits2.topk(k, dim=-1).indices
    top1_match = (topk1[..., 0] == topk2[..., 0]).float().mean().item()
    topk_overlap = 0.0
    for i in range(k):
        topk_overlap += (
            (topk1[..., i : i + 1] == topk2).any(dim=-1).float().mean().item()
        )
    topk_overlap /= k
    return top1_match, topk_overlap


# ============================================================
# HuggingFace inference
# ============================================================


def _load_hf_model(model_path, gpus):
    """Load HuggingFace Qwen3-VL model across specified GPUs."""
    from transformers import AutoModelForImageTextToText

    max_memory = {gpu: "80GiB" for gpu in gpus}
    print(f"Loading HuggingFace model on GPUs: {gpus}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@torch.no_grad()
def forward_hf(model_path, hf_inputs_list, gpus):
    """Run HuggingFace multimodal inference."""
    model = _load_hf_model(model_path, gpus)
    input_device = next(model.parameters()).device

    outputs_list = []
    for i, inputs in enumerate(hf_inputs_list):
        inputs = {
            k: v.to(input_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        logits = model(**inputs).logits
        outputs_list.append(logits[:, -1:, :].cpu())
        print(f"  HF sample {i + 1}/{len(hf_inputs_list)} done")

    del model
    torch.cuda.empty_cache()
    return outputs_list


# ============================================================
# TorchTitan inference
# ============================================================


def _load_tt_model(model_flavor, checkpoint_path, gpus):
    """Load TorchTitan model and shard across specified GPUs."""
    print(f"Loading TorchTitan model on GPUs: {gpus}")

    model_spec = model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("meta"):
        model = model_config.build()

    model.to_empty(device="cpu")
    model.init_weights(buffer_device="cpu")
    model.half()

    model_wrapper = ModelWrapper(model)
    state_dict = model_wrapper._get_state_dict()
    print(f"Loading checkpoint from: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    if len(gpus) == 1:
        model = model.to(torch.device(f"cuda:{gpus[0]}"))
    else:
        first_device = torch.device(f"cuda:{gpus[0]}")
        last_device = torch.device(f"cuda:{gpus[-1]}")

        if model.tok_embeddings is not None:
            model.tok_embeddings = model.tok_embeddings.to(first_device)
        if hasattr(model, "visual") and model.visual is not None:
            model.visual = model.visual.to(first_device)

        layer_keys = list(model.layers.keys())
        n_layers = len(layer_keys)
        n_gpus = len(gpus)
        layers_per_gpu = (n_layers + n_gpus - 1) // n_gpus

        for i, key in enumerate(layer_keys):
            gpu_idx = min(i // layers_per_gpu, n_gpus - 1)
            device = torch.device(f"cuda:{gpus[gpu_idx]}")
            model.layers[key] = model.layers[key].to(device)

        if model.norm is not None:
            model.norm = model.norm.to(last_device)
        if model.output is not None:
            model.output = model.output.to(last_device)

        print(
            f"  Layers distributed: {layers_per_gpu} layers per GPU "
            f"({n_layers} total across {n_gpus} GPUs)"
        )

    model.eval()
    return model, model_config


def _tt_forward(model, tokens, gpus, **kwargs):
    """Run forward through a multi-GPU sharded TT model."""
    if len(gpus) == 1:
        device = torch.device(f"cuda:{gpus[0]}")
        tokens = tokens.to(device)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
        return model(tokens, **kwargs)

    # Get device for the visual encoder (may differ from tok_embeddings due to weight tying)
    if hasattr(model, "visual") and model.visual is not None:
        visual_device = next(model.visual.parameters()).device
    else:
        visual_device = next(model.tok_embeddings.parameters()).device
    embed_device = next(model.tok_embeddings.parameters()).device
    tokens = tokens.to(embed_device)

    special_tokens = kwargs.get("special_tokens", None)
    pixel_values = kwargs.get("pixel_values", None)
    grid_thw = kwargs.get("grid_thw", None)

    # Move vision tensors to the visual encoder's device
    if pixel_values is not None:
        pixel_values = pixel_values.to(visual_device)
    if grid_thw is not None:
        grid_thw = grid_thw.to(visual_device)

    inputs_embeds, visual_pos_masks, deepstack_visual_embeds = model._process_vision(
        tokens,
        pixel_values,
        None,
        grid_thw,
        None,
        special_tokens,
    )

    hidden_states = inputs_embeds
    for layer_idx, layer in model.layers.items():
        layer_device = next(layer.parameters()).device
        if hidden_states.device != layer_device:
            hidden_states = hidden_states.to(layer_device)

        freqs_cis = model.freqs_cis
        if freqs_cis is not None and freqs_cis.device != layer_device:
            model.freqs_cis = freqs_cis.to(layer_device)

        positions = kwargs.get("positions", None)
        if positions is not None:
            positions = positions.to(layer_device)

        hidden_states = layer(hidden_states, model.freqs_cis, None, positions)

        layer_idx_int = int(layer_idx)
        if layer_idx_int in model.deepstack_layer_indices:
            ds_idx = model.deepstack_layer_indices.index(layer_idx_int)
            if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                if ds_idx < len(deepstack_visual_embeds):
                    vis_mask = visual_pos_masks.to(layer_device)
                    vis_embed = deepstack_visual_embeds[ds_idx].to(layer_device)
                    hidden_states = model._deepstack_process(
                        hidden_states,
                        vis_mask,
                        vis_embed,
                    )

    if model.norm is not None:
        norm_device = next(model.norm.parameters()).device
        if hidden_states.device != norm_device:
            hidden_states = hidden_states.to(norm_device)
        hidden_states = model.norm(hidden_states)

    if model.output is not None:
        out_device = model.output.weight.device
        if hidden_states.device != out_device:
            hidden_states = hidden_states.to(out_device)
        hidden_states = model.output(hidden_states)

    return hidden_states


@torch.no_grad()
def forward_tt(model_flavor, checkpoint_path, tt_inputs_list, gpus):
    """Run TorchTitan multimodal inference."""
    model, model_config = _load_tt_model(model_flavor, checkpoint_path, gpus)

    special_tokens = SpecialTokens(
        img_token="<|image_pad|>",
        img_id=model_config.image_token_id,  # pyrefly: ignore[missing-attribute]
        vision_start_token="<|vision_start|>",
        vision_start_id=model_config.vision_start_token_id,  # pyrefly: ignore[missing-attribute]
        vision_end_token="<|vision_end|>",
        vision_end_id=model_config.vision_end_token_id,  # pyrefly: ignore[missing-attribute]
        pad_token="<|endoftext|>",
        pad_id=model_config.eos_id if hasattr(model_config, "eos_id") else 151643,
    )

    outputs_list = []
    for i, (tokens, pixel_values, grid_thw) in enumerate(tt_inputs_list):
        logits = _tt_forward(
            model,
            tokens,
            gpus,
            pixel_values=pixel_values.half(),
            grid_thw=grid_thw,
            special_tokens=special_tokens,
        )
        outputs_list.append(logits[:, -1:, :].cpu())
        print(f"  TT sample {i + 1}/{len(tt_inputs_list)} done")

    del model
    torch.cuda.empty_cache()
    return outputs_list


# ============================================================
# Test input preparation
# ============================================================


def print_token_structure(
    input_ids,
    processor,
    image_token_id=151655,
    vision_start_id=151652,
    vision_end_id=151653,
):
    """Debug helper: print the token structure showing image placeholder positions."""
    tokens = input_ids.squeeze().tolist()
    tokenizer = processor.tokenizer

    print(f"\n  Total tokens: {len(tokens)}")

    # Find vision token positions
    img_positions = [i for i, t in enumerate(tokens) if t == image_token_id]
    start_positions = [i for i, t in enumerate(tokens) if t == vision_start_id]
    end_positions = [i for i, t in enumerate(tokens) if t == vision_end_id]

    print(f"  <|vision_start|> positions: {start_positions}")
    print(f"  <|image_pad|> count: {len(img_positions)}")
    if img_positions:
        print(f"  <|image_pad|> range: [{img_positions[0]}, {img_positions[-1]}]")
    print(f"  <|vision_end|> positions: {end_positions}")

    # Show a snippet around the vision tokens
    if start_positions:
        start = max(0, start_positions[0] - 3)
        end = (
            min(len(tokens), end_positions[0] + 5)
            if end_positions
            else min(len(tokens), start_positions[0] + 60)
        )
        snippet_ids = tokens[start:end]

        print(f"\n  Token structure (positions {start}-{end - 1}):")
        for idx, tid in enumerate(snippet_ids):
            pos = start + idx
            if tid == image_token_id:
                if idx == 0 or snippet_ids[idx - 1] != image_token_id:
                    count = sum(1 for t in snippet_ids[idx:] if t == image_token_id)
                    print(f"    [{pos:4d}] <|image_pad|> x {count}")
            elif tid == vision_start_id:
                print(f"    [{pos:4d}] <|vision_start|>")
            elif tid == vision_end_id:
                print(f"    [{pos:4d}] <|vision_end|>")
            elif tid != image_token_id:
                decoded = tokenizer.decode([tid])
                print(f"    [{pos:4d}] {tid:6d} -> {repr(decoded)}")


def build_multimodal_inputs(hf_model_path, num_samples=3, image_size=224, verbose=True):
    """Build multimodal test inputs (image + text).

    Returns:
        hf_inputs: list of dicts with HF processor outputs
        tt_inputs: list of (tokens, pixel_values, grid_thw) for TT model
    """
    from PIL import Image
    from torchtitan.experiments.qwen3_vl.datasets.utils.image import (
        image_to_patches,
        process_image,
    )

    processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)

    hf_inputs_list = []
    tt_inputs_list = []

    for i in range(num_samples):
        rng = torch.Generator().manual_seed(42 + i)
        img_array = (
            torch.randint(0, 256, (image_size, image_size, 3), generator=rng)
            .numpy()
            .astype("uint8")
        )
        pil_image = Image.fromarray(img_array)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        hf_inputs = processor.apply_chat_template(  # pyrefly: ignore[missing-attribute]
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        hf_inputs_list.append(hf_inputs)

        # Print token structure for first sample
        if verbose and i == 0:
            print(f"\n  Sample {i} token structure:")
            print_token_structure(hf_inputs["input_ids"], processor)

        input_ids = hf_inputs["input_ids"]

        patch_size = 16
        temporal_patch_size = 2
        merge_size = 2
        img_tensor = process_image(
            pil_image,
            patch_size=patch_size,
            merge_size=merge_size,
        )
        if img_tensor is None:
            print(f"Warning: image {i} processing failed, skipping")
            continue
        patches, grid_thw = image_to_patches(
            img_tensor,
            patch_size,
            temporal_patch_size,
            merge_size,
        )
        pixel_values = patches.unsqueeze(0)
        grid_thw = grid_thw.unsqueeze(0)

        tt_inputs_list.append((input_ids, pixel_values, grid_thw))

    return hf_inputs_list, tt_inputs_list


# ============================================================
# Comparison
# ============================================================


def compare_outputs(hf_outputs, tt_outputs):
    """Compare HF and TT outputs and print metrics."""
    print(f"\n{'=' * 60}")
    print("Comparing HuggingFace vs TorchTitan outputs...")
    print(f"{'=' * 60}")

    total_loss: float = 0
    total_top1: float = 0
    total_topk: float = 0

    for i, (hf_out, tt_out) in enumerate(zip(hf_outputs, tt_outputs)):
        sample_loss = loss_fn(hf_out.float(), tt_out.float())
        top1, topk = top_k_match(hf_out.float(), tt_out.float())
        total_loss += sample_loss.item()
        total_top1 += top1
        total_topk += topk

        cos_sim = F.cosine_similarity(
            hf_out.float().flatten(), tt_out.float().flatten(), dim=0
        ).item()
        print(
            f"  Sample {i + 1}: KL={sample_loss.item():.10e}, "
            f"cos_sim={cos_sim:.10f}, top1_match={top1:.2f}, top5_overlap={topk:.2f}"
        )

    n = len(hf_outputs)
    avg_loss = total_loss / n
    avg_top1 = total_top1 / n
    avg_topk = total_topk / n

    print(f"\n  Average KL divergence: {avg_loss:.10e}")
    print(f"  Average top-1 match:   {avg_top1:.2%}")
    print(f"  Average top-5 overlap: {avg_topk:.2%}")

    if avg_loss < 0.001:
        print("  PASS: Models are numerically very similar")
    elif avg_loss < 0.1:
        print("  WARN: Models have some differences")
    else:
        print("  FAIL: Models are significantly different")

    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Numerical correctness test for Qwen3-VL checkpoint conversion. "
        "Compares HuggingFace and TorchTitan model outputs on multimodal inputs.",
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to HuggingFace Qwen3-VL model directory",
    )
    parser.add_argument(
        "--tt_checkpoint_path",
        type=str,
        required=True,
        help="Path to converted TorchTitan DCP checkpoint",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="2B",
        choices=["2B", "8B", "30B-A3B", "235B-A22B"],
        help="TorchTitan model flavor (default: 2B)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of multimodal test samples (default: 10)",
    )
    parser.add_argument(
        "--hf_gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs for HuggingFace model (default: auto first half)",
    )
    parser.add_argument(
        "--tt_gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs for TorchTitan model (default: auto second half)",
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    hf_gpus = args.hf_gpus
    tt_gpus = args.tt_gpus
    if hf_gpus is None or tt_gpus is None:
        mid = max(num_gpus // 2, 1)
        hf_gpus = hf_gpus or list(range(0, mid))
        tt_gpus = tt_gpus or list(range(mid, max(num_gpus, mid + 1)))

    print(f"HuggingFace GPUs: {hf_gpus}")
    print(f"TorchTitan GPUs:  {tt_gpus}")

    if not os.path.exists(args.hf_model_path):
        print(f"Error: HF model path not found: {args.hf_model_path}")
        return
    if not os.path.exists(args.tt_checkpoint_path):
        print(f"Error: TT checkpoint path not found: {args.tt_checkpoint_path}")
        return

    try:
        print("\n" + "=" * 60)
        print("MULTIMODAL NUMERICAL CORRECTNESS TEST")
        print("=" * 60)

        print(f"\nBuilding {args.num_samples} multimodal test samples...")
        hf_inputs, tt_inputs = build_multimodal_inputs(
            args.hf_model_path, args.num_samples
        )

        print("\nRunning HuggingFace inference...")
        hf_outputs = forward_hf(args.hf_model_path, hf_inputs, hf_gpus)

        torch.cuda.empty_cache()

        print("\nRunning TorchTitan inference...")
        tt_outputs = forward_tt(
            args.model_flavor, args.tt_checkpoint_path, tt_inputs, tt_gpus
        )

        compare_outputs(hf_outputs, tt_outputs)

    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
