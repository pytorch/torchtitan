#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full text+image logit parity: TorchTitan Kimi K3 vs HuggingFace Kimi K3.

Runs the released HuggingFace model code and TorchTitan in one process on the
same text+image prompt. Each side performs its own image preprocessing, so the
comparison covers preprocessing, vision, projector, scatter, and decoder.

The released checkpoint is MXFP4-quantized and is not loaded. Instead, the
script reduces the HuggingFace config to TorchTitan's debug model, initializes
TorchTitan, and strictly transfers its state dict to HuggingFace.

The script downloads the config, modeling, processor, and tokenizer assets from
a pinned HuggingFace revision without downloading the released weight shards.
The released code requires ``transformers==4.56.2`` and ``tiktoken``.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m \
        scripts.checkpoint_conversion.numerical_tests_kimi_k3 \
        --dtype float32
"""

import argparse
from typing import Any, cast

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image

from torchtitan.hf_datasets.multimodal.utils.image import (
    process_image,
    resize_to_patch_budget,
    vision_to_patches,
)
from torchtitan.models.kimi_k3 import model_registry
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


_HF_REPO_ID = "moonshotai/Kimi-K3"
_HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_MEDIA_TOKEN_ID = 163605
_PATCH_SIZE = 14
_MERGE_SIZE = 2
_MAX_PATCHES = 65536
_MAX_PATCHES_PER_SIDE = 512
_PROMPT = (
    "<|kimi_image_placeholder|>\n" "What is shown in this image? Describe it briefly."
)


def _reduce_hf_config(hf_config, tt_config, hf_model_path: str) -> None:
    """Reduce the released HuggingFace config to the TorchTitan debug model."""
    text_config = hf_config.text_config
    hf_full_attention_layers = [
        layer_idx + 1
        for layer_idx, layer in enumerate(tt_config.layers)
        if layer.attention is not None
    ]
    hf_kda_layers = [
        layer_idx + 1
        for layer_idx, layer in enumerate(tt_config.layers)
        if layer.delta_attention is not None
    ]
    mla = next(layer.attention for layer in tt_config.layers if layer.attention)
    kda = next(
        layer.delta_attention for layer in tt_config.layers if layer.delta_attention
    )
    dense_ffn = next(
        layer.feed_forward for layer in tt_config.layers if layer.feed_forward
    )
    moe = next(layer.moe for layer in tt_config.layers if layer.moe)

    text_overrides = {
        "vocab_size": tt_config.vocab_size,
        "hidden_size": tt_config.dim,
        "intermediate_size": dense_ffn.w1.out_features,
        "num_hidden_layers": len(tt_config.layers),
        "num_attention_heads": mla.n_heads,
        "num_key_value_heads": mla.n_heads,
        "rms_norm_eps": tt_config.norm.eps,
        "q_lora_rank": mla.wq_a.out_features,
        "kv_lora_rank": mla.kv_lora_rank,
        "qk_nope_head_dim": mla.qk_nope_head_dim,
        "qk_rope_head_dim": mla.qk_rope_head_dim,
        "v_head_dim": mla.v_head_dim,
        "activation_situ_beta": dense_ffn.beta,
        "activation_situ_linear_beta": dense_ffn.linear_beta,
        "num_experts": moe.num_experts,
        "num_experts_per_token": moe.router.top_k,
        "num_shared_experts": (
            moe.shared_experts.w1.out_features
            // moe.routed_experts.inner_experts.hidden_dim
        ),
        "moe_renormalize": moe.router.route_norm,
        "moe_intermediate_size": moe.routed_experts.inner_experts.hidden_dim,
        "routed_expert_hidden_size": moe.routed_down.out_features,
        "routed_scaling_factor": moe.router.route_scale,
        "first_k_dense_replace": next(
            layer_idx for layer_idx, layer in enumerate(tt_config.layers) if layer.moe
        ),
        "attn_res_block_size": tt_config.layers[0].attn_res_block_size,
        "linear_attn_config": {
            "full_attn_layers": hf_full_attention_layers,
            "kda_layers": hf_kda_layers,
            "head_dim": kda.head_dim,
            "num_heads": kda.num_heads,
            "short_conv_kernel_size": kda.conv_kernel_size,
            "gate_lower_bound": kda.kernel.lower_bound,
            "use_full_rank_gate": True,
        },
    }
    for name, value in text_overrides.items():
        setattr(text_config, name, value)

    vision = tt_config.vision_encoder
    assert vision is not None
    vision_overrides = {
        "patch_size": vision.patch_size,
        "init_pos_emb_height": vision.init_pos_emb_height,
        "init_pos_emb_width": vision.init_pos_emb_width,
        "init_pos_emb_time": vision.max_num_frames,
        "vt_num_attention_heads": vision.block.attn.num_heads,
        "vt_num_hidden_layers": vision.num_layers,
        "vt_hidden_size": vision.dim,
        "vt_intermediate_size": vision.block.mlp.fc1.out_features,
        "merge_kernel_size": vision.merge_kernel_size,
        "mm_hidden_size": vision.dim,
        "qkv_hidden_size": vision.block.attn.qkv_dim,
        "text_hidden_size": tt_config.dim,
        "pos_emb_interpolation_mode": vision.interpolation_mode,
    }
    for name, value in vision_overrides.items():
        setattr(hf_config.vision_config, name, value)

    for config in (hf_config, text_config):
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
    text_config._attn_implementation = "eager"
    hf_config.vision_config._attn_implementation = "eager"
    text_config._name_or_path = hf_model_path
    hf_config._name_or_path = hf_model_path


def _build_tt_model(tt_config, dtype: torch.dtype) -> KimiK3Model:
    with torch.device("meta"):
        model = tt_config.build()
    model.to_empty(device="cpu")
    model.to(dtype=dtype)
    model.init_states(buffer_device=torch.device("cpu"))
    return model.eval()


def _build_hf_model(
    hf_model_path: str,
    tt_config,
    hf_state_dict: dict[str, Any],
    dtype: torch.dtype,
):
    hf_config = AutoConfig.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    _reduce_hf_config(hf_config, tt_config, hf_model_path)
    model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    model.language_model.config._attn_implementation = "eager"
    model.to(dtype=dtype)
    model.load_state_dict(hf_state_dict, strict=True)
    return model.eval()


@torch.no_grad()
def run_hf(
    hf_model_path: str,
    tt_config,
    hf_state_dict: dict[str, Any],
    image_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """Run HuggingFace preprocessing and the reduced HuggingFace model."""
    print(f"Loading released HuggingFace Kimi K3 code on {device} ...")
    processor: Any = AutoProcessor.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = _build_hf_model(hf_model_path, tt_config, hf_state_dict, dtype).to(device)

    raw_image = (
        torch.linspace(0, 255, image_size * image_size * 3)
        .reshape(image_size, image_size, 3)
        .to(torch.uint8)
    )
    pil_image = Image.fromarray(raw_image.numpy())
    batch = processor(
        medias=[{"type": "image", "image": pil_image}],  # codespell:ignore medias
        text=_PROMPT,
        return_tensors="pt",
    )

    vision_features: dict[str, torch.Tensor] = {}

    def record_vision_features(_module, _inputs, output) -> None:
        features = output[0] if isinstance(output, (list, tuple)) else output
        vision_features["output"] = features.detach().float().cpu()

    model.mm_projector.register_forward_hook(record_vision_features)

    expert_indices: dict[int, torch.Tensor] = {}
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None:
            moe.gate.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=layer_idx: (
                    expert_indices.__setitem__(
                        layer_idx,
                        output[0].detach().cpu(),
                    )
                )
            )

    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    output = model(**inputs, use_cache=False)
    ref = {
        "input_ids": batch["input_ids"].cpu(),
        "raw_image": raw_image,
        "last_logits": output.logits[:, -1, :].float().cpu(),
        "pixel_values": batch["pixel_values"].float().cpu(),
        "grid_thws": batch["grid_thws"].cpu(),
        "vision_features": vision_features["output"],
        "expert_indices": expert_indices,
    }
    del model
    torch.cuda.empty_cache()
    return ref


def _expand_image_placeholder(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_vision_tokens: int,
) -> torch.Tensor:
    """Expand the single HF media placeholder for TorchTitan's scatter path."""
    if input_ids.shape[0] != 1:
        raise ValueError("The Kimi K3 numerical test expects a batch size of one.")
    positions = (input_ids[0] == image_token_id).nonzero().flatten()
    if positions.numel() != 1:
        raise ValueError(f"Expected one image placeholder, found {positions.numel()}.")
    position = positions.item()
    image_tokens = input_ids.new_full((1, num_vision_tokens), image_token_id)
    return torch.cat(
        (input_ids[:, :position], image_tokens, input_ids[:, position + 1 :]),
        dim=1,
    )


def _print_routing_comparison(
    hf_expert_indices: dict[int, torch.Tensor],
    tt_expert_indices: dict[int, torch.Tensor],
) -> None:
    hf_layers = set(hf_expert_indices)
    tt_layers = set(tt_expert_indices)
    if not hf_layers:
        raise ValueError("No MoE routing choices were recorded.")
    if hf_layers != tt_layers:
        raise ValueError(
            "Routing layers differ: "
            f"HF-only {sorted(hf_layers - tt_layers)}, "
            f"TorchTitan-only {sorted(tt_layers - hf_layers)}."
        )

    num_matching = 0
    num_routings = 0
    for layer_idx in sorted(hf_layers):
        top_k = hf_expert_indices[layer_idx].shape[-1]
        hf_ids = hf_expert_indices[layer_idx].reshape(-1, top_k).sort(dim=-1).values
        tt_ids = tt_expert_indices[layer_idx].reshape(-1, top_k).sort(dim=-1).values
        if hf_ids.shape != tt_ids.shape:
            raise ValueError(
                f"Layer {layer_idx} routing shapes differ: "
                f"HF {tuple(hf_ids.shape)} vs TT {tuple(tt_ids.shape)}."
            )
        num_matching += int((hf_ids == tt_ids).sum().item())
        num_routings += hf_ids.numel()
    match_rate = num_matching / num_routings if num_routings else 0.0
    print(f"router choices: {num_matching}/{num_routings} match " f"({match_rate:.1%})")


@torch.no_grad()
def run_tt(
    model: KimiK3Model,
    ref: dict[str, Any],
    vision_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Run TorchTitan preprocessing and the reduced TorchTitan model."""
    print(f"Loading TorchTitan Kimi K3 (debugmodel) on {device} ...")
    model.to(device)
    assert model.vision_encoder is not None
    model.vision_encoder.to(vision_dtype)

    expert_indices: dict[int, torch.Tensor] = {}
    for layer_idx, layer in model.layers.items():
        if layer.moe is not None:
            # pyrefly: ignore [missing-attribute]
            layer.moe.router.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=int(layer_idx): (
                    expert_indices.__setitem__(
                        layer_idx,
                        output[1].detach().cpu(),
                    )
                )
            )

    image = process_image(
        Image.fromarray(ref["raw_image"].numpy()),
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        resize_fn=resize_to_patch_budget,
        max_patches=_MAX_PATCHES,
        max_patches_per_side=_MAX_PATCHES_PER_SIDE,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    if image is None:
        raise ValueError("TorchTitan failed to process the numerical test image.")
    patches, grid = vision_to_patches(
        image,
        patch_size=_PATCH_SIZE,
        temporal_patch_size=1,
        merge_size=_MERGE_SIZE,
        patch_order="raster",
    )
    pixel_values = patches.unsqueeze(0).to(device=device, dtype=vision_dtype)
    grid_thw = grid.unsqueeze(0).to(device)
    num_vision_tokens = (grid[1] // _MERGE_SIZE) * (grid[2] // _MERGE_SIZE)
    tokens = _expand_image_placeholder(
        ref["input_ids"],
        _MEDIA_TOKEN_ID,
        int(num_vision_tokens.item()),
    ).to(device)
    positions = torch.arange(
        tokens.shape[1],
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    attention_masks = model.get_attention_masks(positions)

    print(
        f"tokens={tuple(tokens.shape)} pixel_values={tuple(pixel_values.shape)} "
        f"grid_thw={grid_thw.tolist()} vision_tokens={num_vision_tokens.item()}"
    )

    hf_pixels = ref["pixel_values"].flatten(1)
    pixel_diff = (hf_pixels - patches.float()).abs()
    print(
        f"pixel values: max_diff={pixel_diff.max().item():.3e} "
        f"num_differ={(pixel_diff > 1e-6).sum().item()}/{pixel_diff.numel()}"
    )

    tt_features = model.vision_encoder(pixel_values, grid_thw=grid_thw)
    hf_features = ref["vision_features"].reshape(-1, tt_features.shape[-1])
    tt_features = tt_features.float().cpu().reshape(-1, tt_features.shape[-1])
    vision_cos = F.cosine_similarity(
        hf_features.flatten(), tt_features.flatten(), dim=0
    ).item()
    vision_max_diff = (hf_features - tt_features).abs().max().item()
    print(
        f"vision features: shape={tuple(tt_features.shape)} "
        f"cos={vision_cos:.6f} max_diff={vision_max_diff:.3e}"
    )

    logits = model(
        tokens,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        special_tokens={"image_id": _MEDIA_TOKEN_ID},
        positions=positions,
        attention_masks=attention_masks,
    )
    _print_routing_comparison(ref["expert_indices"], expert_indices)
    return logits[:, -1, :].float().cpu().squeeze()


def compare(ref_logits: torch.Tensor, tt_logits: torch.Tensor) -> bool:
    """Print last-token metrics and return whether KL is below tolerance."""
    ref = ref_logits.squeeze()
    tt = tt_logits.squeeze()
    log_ref = F.log_softmax(ref, dim=-1)
    log_tt = F.log_softmax(tt, dim=-1)
    kl = F.kl_div(log_tt, log_ref, log_target=True, reduction="sum").item()
    cosine = F.cosine_similarity(ref, tt, dim=-1).item()
    max_diff = (ref - tt).abs().max().item()
    top1 = (ref.argmax() == tt.argmax()).item()
    top5_overlap = (
        len(set(ref.topk(5).indices.tolist()) & set(tt.topk(5).indices.tolist())) / 5
    )
    print("\nFull multimodal last-token logit parity (TorchTitan vs HuggingFace)")
    print(
        f"KL={kl:.4e} cos={cosine:.6f} max_diff={max_diff:.4e} "
        f"top1={'Y' if top1 else 'N'} top5={top5_overlap:.0%}"
    )
    passed = abs(kl) < 1e-3  # pyrefly: ignore [bad-argument-type]
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return passed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_flavor", default="debugmodel")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument(
        "--hf_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--vision_dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("Kimi K3 numerical parity requires a CUDA GPU.")

    hf_model_path = snapshot_download(
        repo_id=_HF_REPO_ID,
        revision=_HF_REVISION,
        allow_patterns=["*.json", "*.py", "tiktoken.model"],
    )
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    vision_dtype = getattr(torch, args.vision_dtype) if args.vision_dtype else dtype
    hf_dtype = getattr(torch, args.hf_dtype)
    print(
        f"hf_dtype={args.hf_dtype} titan text={args.dtype} "
        f"titan vision={args.vision_dtype or args.dtype}"
    )

    tt_config = cast(KimiK3Model.Config, model_registry(args.model_flavor).model)
    torch.manual_seed(args.seed)
    tt_model = _build_tt_model(tt_config, dtype)
    hf_state_dict = KimiK3StateDictAdapter(tt_config, hf_assets_path=None).to_hf(
        tt_model.state_dict()
    )

    ref = run_hf(
        hf_model_path,
        tt_config,
        hf_state_dict,
        args.image_size,
        hf_dtype,
        device,
    )
    del hf_state_dict
    tt_logits = run_tt(tt_model, ref, vision_dtype, device)
    if not compare(ref["last_logits"], tt_logits):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
