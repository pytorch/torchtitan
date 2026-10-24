#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pinned BF16 forward parity: TorchTitan MiniMax M3 vs HuggingFace MiniMax M3.

Runs HuggingFace's ``minimax_m3_vl`` model and TorchTitan in one process on the
same text+image prompt. Each side performs its own image preprocessing, so the
comparison covers preprocessing, vision, projector, scatter, the sparse
attention block selection, and the decoder.

The released checkpoint is not loaded. Instead, the script reduces the
HuggingFace config to TorchTitan's debug model, initializes TorchTitan, and
strictly transfers its state dict to HuggingFace. The prompt is longer than the
debug model's top-k block budget, so the sparse layers attend to a subset of
their key blocks.

The script downloads only the config, processor, and tokenizer files of a
pinned HuggingFace revision.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m \
        scripts.checkpoint_conversion.numerical_tests_minimax_m3

Add ``--force-hf-routing`` to impose HuggingFace's expert IDs and
``--force-hf-blocks`` to impose its attended key blocks, which isolates the
arithmetic from selection flips.
"""

import argparse
import os
import tempfile
from dataclasses import replace
from typing import Any, cast

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import save_file

from torchtitan.components.data.multimodal.utils.image import (
    process_image,
    vision_to_patches,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe import MoE
from torchtitan.models.minimax_m3 import (
    attention as minimax_attention,
    build_model_config,
    MINIMAX_M3_SPECIAL_TOKENS,
)
from torchtitan.models.minimax_m3.attention import (
    MiniMaxSparseAttention,
    MSAInnerAttention,
)
from torchtitan.models.minimax_m3.model import MiniMaxM3Model
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter
from transformers import AutoConfig
from transformers.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3VLProcessor,
)


_HF_REPO_ID = "MiniMaxAI/MiniMax-M3"
_HF_REVISION = "f0e1c1e04d40177e4673a22097036854f536e9c0"
_HF_ALLOW_PATTERNS = [
    "*.json",
    "chat_template.jinja",
    "merges.txt",
]
_HF_ATTN_BACKEND = "eager"
_PATCH_SIZE = 14
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2
_MIN_PIXELS = 3136
_MAX_PIXELS = 451584
_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
_PROMPT = "What is shown in this image? Describe it briefly."


def _reduce_hf_config(hf_config, tt_config: MiniMaxM3Model.Config) -> None:
    """Reduce the released HuggingFace config to the TorchTitan debug model."""
    layers = tt_config.layers
    attention = layers[0].attention
    rope = attention.rope
    assert attention.head_dim is not None and rope is not None
    sparse = next(
        layer.attention
        for layer in layers
        if isinstance(layer.attention, MiniMaxSparseAttention.Config)
    )
    inner_attention = cast(MSAInnerAttention.Config, sparse.inner_attention)
    dense_ffn = next(
        layer.feed_forward
        for layer in layers
        if not isinstance(layer.feed_forward, MoE.Config)
    )
    moe = next(
        layer.feed_forward
        for layer in layers
        if isinstance(layer.feed_forward, MoE.Config)
    )
    assert moe.shared_experts is not None

    text_overrides = {
        "vocab_size": tt_config.vocab_size,
        "hidden_size": tt_config.dim,
        "num_hidden_layers": len(layers),
        "num_attention_heads": attention.n_heads,
        "num_key_value_heads": attention.n_kv_heads,
        "head_dim": attention.head_dim,
        "rope_parameters": {
            "rope_theta": rope.theta,
            "partial_rotary_factor": rope.dim / attention.head_dim,
            "rope_type": "default",
        },
        "rotary_dim": rope.dim,
        "rms_norm_eps": tt_config.norm.eps,
        "dense_intermediate_size": dense_ffn.w13.out_features,
        "intermediate_size": moe.routed_experts.w2.in_features,
        "shared_intermediate_size": moe.shared_experts.w13.out_features,
        "num_local_experts": moe.num_experts,
        "num_experts_per_tok": moe.router.top_k,
        "routed_scaling_factor": moe.router.route_scale,
        "index_n_heads": sparse.indexer.wq.out_features // sparse.indexer.head_dim,
        "index_head_dim": sparse.indexer.head_dim,
        "index_block_size": inner_attention.index_block_size,
        "index_topk_blocks": inner_attention.topk_blocks,
        "index_local_blocks": 1,
        "layer_types": [
            (
                "minimax_m3_sparse"
                if isinstance(layer.attention, MiniMaxSparseAttention.Config)
                else "full_attention"
            )
            for layer in layers
        ],
        "mlp_layer_types": [
            "sparse" if isinstance(layer.feed_forward, MoE.Config) else "dense"
            for layer in layers
        ],
    }
    for name, value in text_overrides.items():
        setattr(hf_config.text_config, name, value)

    vision = tt_config.vision_encoder
    assert vision is not None
    vision_overrides = {
        "hidden_size": vision.dim,
        "intermediate_size": vision.block.mlp.fc1.out_features,
        "num_hidden_layers": vision.num_layers,
        "num_attention_heads": vision.num_heads,
        "patch_size": vision.patch_size,
        "temporal_patch_size": vision.temporal_patch_size,
        "spatial_merge_size": vision.spatial_merge_size,
        "layer_norm_eps": vision.pre_norm.eps,
    }
    for name, value in vision_overrides.items():
        setattr(hf_config.vision_config, name, value)
    hf_config.projector_hidden_size = tt_config.dim


def _use_released_vocab(
    tt_config: MiniMaxM3Model.Config, hf_model_path: str
) -> MiniMaxM3Model.Config:
    """Keep released tokenizer IDs valid in the reduced TorchTitan model."""
    hf_config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
    vocab_size = hf_config.text_config.vocab_size
    return replace(
        tt_config,
        vocab_size=vocab_size,
        tok_embeddings=replace(tt_config.tok_embeddings, num_embeddings=vocab_size),
        lm_head=replace(tt_config.lm_head, out_features=vocab_size),
    )


def _build_tt_model(tt_config: MiniMaxM3Model.Config, dtype: torch.dtype):
    with torch.device("meta"):
        model = tt_config.build()
    model.to_empty(device="cpu")
    model.to(dtype=dtype)
    model.init_states(buffer_device=torch.device("cpu"))
    return model.eval()


def _build_hf_model(
    hf_model_path: str,
    tt_config: MiniMaxM3Model.Config,
    hf_state_dict: dict[str, Any],
    dtype: torch.dtype,
    experts_implementation: str,
):
    hf_config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
    _reduce_hf_config(hf_config, tt_config)
    # Load through HuggingFace's checkpoint conversion, as a released checkpoint is.
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        hf_config.save_pretrained(checkpoint_dir)
        save_file(
            {key: value.contiguous() for key, value in hf_state_dict.items()},
            os.path.join(checkpoint_dir, "model.safetensors"),
        )
        # pyrefly: ignore [not-iterable]
        model, loading_info = MiniMaxM3SparseForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            dtype=dtype,
            attn_implementation=_HF_ATTN_BACKEND,
            experts_implementation=experts_implementation,
            output_loading_info=True,
        )
    problems = {name: keys for name, keys in loading_info.items() if keys}
    if problems:
        raise RuntimeError(f"HF checkpoint load is not exact: {problems}")
    return model.eval()


def _record_hf_selections(model) -> tuple[dict[int, Any], dict[int, Any]]:
    """Record each sparse layer's key blocks and each MoE layer's expert IDs."""
    blocks: dict[int, torch.Tensor] = {}
    experts: dict[int, torch.Tensor] = {}
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        indexer = getattr(layer.self_attn, "indexer", None)
        if indexer is not None:
            indexer.register_forward_hook(
                lambda _m, _i, out, layer_idx=layer_idx: blocks.__setitem__(
                    layer_idx, out[0].transpose(0, 1).detach().cpu()
                )
            )
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            gate.register_forward_hook(
                lambda _m, _i, out, layer_idx=layer_idx: experts.__setitem__(
                    layer_idx, out[2].detach().cpu()
                )
            )
    return blocks, experts


@torch.no_grad()
def run_hf(
    hf_model_path: str,
    tt_config: MiniMaxM3Model.Config,
    hf_state_dict: dict[str, Any],
    image_size: int,
    dtype: torch.dtype,
    device: torch.device,
    experts_implementation: str,
) -> dict[str, Any]:
    """Run HuggingFace preprocessing and the reduced HuggingFace model."""
    print(f"Loading HuggingFace MiniMax M3 code on {device} ...")
    processor: Any = MiniMaxM3VLProcessor.from_pretrained(
        hf_model_path, local_files_only=True
    )
    model = _build_hf_model(
        hf_model_path, tt_config, hf_state_dict, dtype, experts_implementation
    ).to(device)

    raw_image = (
        torch.linspace(0, 255, image_size * image_size * 3)
        .reshape(image_size, image_size, 3)
        .to(torch.uint8)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(raw_image.numpy())},
                {"type": "text", "text": _PROMPT},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    vision_features: dict[str, torch.Tensor] = {}
    model.model.multi_modal_projector.register_forward_hook(
        lambda _m, _i, out: vision_features.__setitem__(
            "output", out.detach().float().cpu()
        )
    )
    blocks, experts = _record_hf_selections(model)

    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    output = model(**inputs, use_cache=False)
    ref = {
        "input_ids": batch["input_ids"].cpu(),
        "raw_image": raw_image,
        "logits": output.logits[0].float().cpu(),
        "pixel_values": batch["pixel_values"].float().cpu(),
        "image_grid_thw": batch["image_grid_thw"].cpu(),
        "vision_features": vision_features["output"],
        "blocks": blocks,
        "experts": experts,
    }
    del model
    torch.cuda.empty_cache()
    return ref


def _compare_selections(
    name: str, hf: dict[int, torch.Tensor], tt: dict[int, torch.Tensor]
) -> None:
    """Print how many per-token selections agree as sets, layer by layer."""
    if sorted(hf) != sorted(tt):
        raise ValueError(
            f"{name}: layers differ, HF {sorted(hf)} vs TorchTitan {sorted(tt)}."
        )
    if not hf:
        raise ValueError(f"No {name} were recorded.")
    num_matching = num_rows = 0
    for layer_idx in sorted(hf):
        hf_ids = hf[layer_idx].flatten(0, -2).sort(dim=-1).values
        tt_ids = tt[layer_idx].flatten(0, -2).sort(dim=-1).values
        if hf_ids.shape != tt_ids.shape:
            raise ValueError(
                f"{name}: layer {layer_idx} shapes differ, "
                f"HF {tuple(hf_ids.shape)} vs TorchTitan {tuple(tt_ids.shape)}."
            )
        num_matching += int((hf_ids == tt_ids).all(dim=-1).sum().item())
        num_rows += hf_ids.shape[0]
    print(
        f"{name}: {num_matching}/{num_rows} selections match "
        f"({num_matching / num_rows:.1%})"
    )


def _pad_hf_blocks(blocks_TGK: torch.Tensor, topk_blocks: int) -> torch.Tensor:
    """Pad HF's ``min(topk, num_blocks)`` slots with unused (-1) slots."""
    missing = topk_blocks - blocks_TGK.shape[-1]
    return F.pad(blocks_TGK, (0, missing), value=-1) if missing > 0 else blocks_TGK


@torch.no_grad()
def run_tt(
    model: MiniMaxM3Model,
    ref: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
    special_tokens: dict[str, int],
    *,
    force_hf_routing: bool,
    force_hf_blocks: bool,
) -> torch.Tensor:
    """Run TorchTitan preprocessing and the reduced TorchTitan model."""
    print(f"Loading TorchTitan MiniMax M3 (debugmodel) on {device} ...")
    model.to(device)
    assert model.vision_encoder is not None

    sparse_layers = [
        int(name.split(".")[1])
        for name, module in model.layers.named_modules(prefix="layers")
        if isinstance(module, MSAInnerAttention)
    ]
    blocks: dict[int, torch.Tensor] = {}
    select_msa_blocks = minimax_attention.select_msa_blocks

    def recording_select(*args, **kwargs) -> torch.Tensor:
        layer_idx = sparse_layers[len(blocks)]
        if force_hf_blocks:
            selected = _pad_hf_blocks(
                ref["blocks"][layer_idx].to(device), kwargs["topk_blocks"]
            )
        else:
            selected = select_msa_blocks(*args, **kwargs)
        blocks[layer_idx] = selected.cpu()
        return selected

    minimax_attention.select_msa_blocks = recording_select

    experts: dict[int, torch.Tensor] = {}
    for layer_idx, layer in model.layers.items():
        moe = layer.feed_forward
        if not isinstance(moe, MoE):
            continue
        router = moe.router
        if force_hf_routing:
            ids = ref["experts"][int(layer_idx)].to(device)
            original = router.forward

            def forced_forward(
                x_TD,
                expert_bias_E=None,
                *,
                _router=router,
                _orig=original,
                _ids=ids,
                **kw,
            ):
                scores_TK, _, routing_map_TE = _orig(x_TD, expert_bias_E, **kw)
                scores_TE = _router.score_func(_router.gate(x_TD)).float()
                weights_TK = scores_TE.gather(dim=-1, index=_ids)
                weights_TK = weights_TK / weights_TK.sum(dim=-1, keepdim=True)
                routing_map_TE = torch.zeros_like(routing_map_TE).scatter_(
                    -1, _ids, True
                )
                return (
                    (weights_TK * _router.route_scale).to(scores_TK.dtype),
                    _ids,
                    routing_map_TE,
                )

            router.forward = forced_forward
        router.register_forward_hook(
            lambda _m, _i, out, layer_idx=int(layer_idx): experts.__setitem__(
                layer_idx, out[1].detach().cpu()
            )
        )

    image = process_image(
        Image.fromarray(ref["raw_image"].numpy()),
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        min_pixels=_MIN_PIXELS,
        max_pixels=_MAX_PIXELS,
        image_mean=_IMAGE_MEAN,
        image_std=_IMAGE_STD,
    )
    if image is None:
        raise ValueError("TorchTitan failed to process the numerical test image.")
    patches, grid = vision_to_patches(
        image,
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
    )
    pixel_values = patches.to(device=device, dtype=dtype)
    grid_thw = grid.unsqueeze(0).to(device)
    tokens = ref["input_ids"][0].to(device)
    positions = torch.arange(tokens.shape[0], dtype=torch.int32, device=device)
    attention_metadata = model._get_attention_metadata(positions=positions)
    print(
        f"tokens={tuple(tokens.shape)} pixel_values={tuple(pixel_values.shape)} "
        f"grid_thw={grid_thw.tolist()} HF grid_thw={ref['image_grid_thw'].tolist()}"
    )

    pixel_diff = (ref["pixel_values"] - patches.float()).abs()
    print(
        f"pixel values: max_diff={pixel_diff.max().item():.3e} "
        f"num_differ={(pixel_diff > 1e-6).sum().item()}/{pixel_diff.numel()}"
    )
    tt_features = model.vision_encoder(pixel_values, grid_thw=grid_thw).float().cpu()
    hf_features = ref["vision_features"]
    print(
        f"vision features: shape={tuple(tt_features.shape)} "
        f"cos={F.cosine_similarity(hf_features.flatten().double(), tt_features.flatten().double(), dim=0).item():.6f} "
        f"max_diff={(hf_features - tt_features).abs().max().item():.3e}"
    )

    logits = model(
        tokens,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        positions=positions,
        attention_metadata=attention_metadata,
        special_tokens=special_tokens,
    )
    minimax_attention.select_msa_blocks = select_msa_blocks
    _compare_selections("sparse attention key blocks", ref["blocks"], blocks)
    _compare_selections("router choices", ref["experts"], experts)
    return logits.float().cpu()


def compare(ref_logits: torch.Tensor, tt_logits: torch.Tensor) -> None:
    """Print last-token and all-position parity metrics."""
    for label, ref, tt in (
        ("last-token", ref_logits[-1:], tt_logits[-1:]),
        ("all-position", ref_logits, tt_logits),
    ):
        log_ref = F.log_softmax(ref, dim=-1)
        log_tt = F.log_softmax(tt, dim=-1)
        kl = F.kl_div(log_tt, log_ref, log_target=True, reduction="batchmean").item()
        cosine = F.cosine_similarity(
            ref.flatten().double(), tt.flatten().double(), dim=0
        ).item()
        max_diff = (ref - tt).abs().max().item()
        top1 = (ref.argmax(-1) == tt.argmax(-1)).float().mean().item()
        print(
            f"{label} logits: KL={kl:.4e} cos={cosine:.6f} "
            f"max_diff={max_diff:.4e} top1={top1:.1%}"
        )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_flavor", default="debugmodel")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument(
        "--experts_implementation",
        choices=["eager", "grouped_mm"],
        default="grouped_mm",
        help="HuggingFace MoE expert implementation.",
    )
    parser.add_argument(
        "--force-hf-routing",
        action="store_true",
        help="Use HF expert selections with TorchTitan router scores.",
    )
    parser.add_argument(
        "--force-hf-blocks",
        action="store_true",
        help="Use HF sparse attention key blocks in TorchTitan.",
    )
    parser.add_argument(
        "--attn_backend",
        choices=["varlen", "flex"],
        default="varlen",
        help="Backend of the full-attention layers; sparse layers always use flex.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="float32 isolates the selections from bfloat16 rounding.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    if dtype == torch.float32 and args.attn_backend == "varlen":
        parser.error("varlen attention computes in bfloat16; use --attn_backend flex.")

    if not torch.cuda.is_available():
        parser.error("MiniMax M3 numerical parity requires a CUDA GPU.")
    # HF's Conv3d patch embedding otherwise runs float32 inputs in TF32.
    torch.backends.cudnn.allow_tf32 = False

    hf_model_path = snapshot_download(
        repo_id=_HF_REPO_ID,
        revision=_HF_REVISION,
        allow_patterns=_HF_ALLOW_PATTERNS,
    )
    device = torch.device("cuda")
    print(
        f"revision={_HF_REVISION} seed={args.seed} dtype={dtype} "
        f"hf_attn={_HF_ATTN_BACKEND} tt_attn={args.attn_backend} "
        f"hf_experts={args.experts_implementation}"
    )

    tt_config = _use_released_vocab(
        build_model_config(
            args.model_flavor,
            attn_backend=args.attn_backend,
            parallelism=ParallelismConfig(),
        ),
        hf_model_path,
    )
    torch.manual_seed(args.seed)
    tt_model = _build_tt_model(tt_config, dtype)
    hf_state_dict = MiniMaxM3StateDictAdapter(tt_config, hf_assets_path=None).to_hf(
        tt_model.state_dict()
    )

    ref = run_hf(
        hf_model_path,
        tt_config,
        hf_state_dict,
        args.image_size,
        dtype,
        device,
        args.experts_implementation,
    )
    del hf_state_dict
    tokenizer = MiniMaxM3VLProcessor.from_pretrained(
        hf_model_path, local_files_only=True
    ).tokenizer
    special_tokens = {
        "image_id": tokenizer.convert_tokens_to_ids(
            MINIMAX_M3_SPECIAL_TOKENS["image_token"]
        ),
        "video_id": tokenizer.convert_tokens_to_ids(
            MINIMAX_M3_SPECIAL_TOKENS["video_token"]
        ),
    }
    print(f"special tokens: {special_tokens}")
    tt_logits = run_tt(
        tt_model,
        ref,
        dtype,
        device,
        special_tokens,
        force_hf_routing=args.force_hf_routing,
        force_hf_blocks=args.force_hf_blocks,
    )
    compare(ref["logits"], tt_logits)


if __name__ == "__main__":
    main()
