# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full text+image e2e logit parity: torchtitan Kimi-VL vs the released HF model.

Runs the released HF Kimi-VL (``trust_remote_code``) and torchtitan in ONE
process on the same text+image prompt and compares last-token logits. torchtitan
does its OWN Kimi image processing (``process_image`` + ``vision_to_patches``,
raster order), so the full pipeline is exercised (preprocessing + vision +
projector + scatter + DeepSeek-V3 text tower), not just the forward.

The released remote code targets transformers ~4.50.x and does NOT import on 5.x,
so run this in an env with ``transformers==4.50.3`` + ``tiktoken`` + ``blobfile``.

Precision: the HF reference runs at ``--hf_dtype`` (default float32, the model's
"true" output); torchtitan runs at ``--dtype`` (text) with ``--vision_dtype``
overriding only its vision encoder. ``--dtype float32`` is the correctness gate;
``--dtype bfloat16 --vision_dtype float16`` is the realistic config (~1e-2 KL).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m \\
        scripts.checkpoint_conversion.numerical_tests_kimi \\
        --hf_model_path ~/hf_assets/moonshotai/Kimi-VL-A3B-Instruct \\
        --tt_checkpoint_path outputs/kimi/kimi_vl_a3b_dcp --dtype float32

Add ``--force-hf-routing`` to make titan use HF's exact per-token expert
selections (diagnostic: removes the MoE routing-flip divergence).
"""

import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.hf_datasets.multimodal.utils.image import (
    process_image,
    resize_to_patch_budget,
    vision_to_patches,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.kimi_k2_5 import model_registry
from transformers import AutoModelForCausalLM, AutoProcessor

_MEDIA_TOKEN_ID = 163605
_PATCH_SIZE = 14
_MERGE_SIZE = 2
_PROMPT = "<|media_pad|>\nWhat is shown in this image? Describe it briefly."


class _VisionSDPA(nn.Module):
    """Bidirectional SDPA for one image (no padding -> full attention is exact)."""

    def forward(self, q, k, v, **kwargs):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return out.transpose(1, 2)


@torch.no_grad()
def run_hf(hf_model_path, image_size, dtype, device):
    """Released HF Kimi-VL (trust_remote_code) on a text+image prompt.

    Returns the raw image, tokenized input_ids, last-token logits, the projector
    output (scattered vision features), and per-MoE-layer top-k expert indices.
    """
    print(f"Loading released HF Kimi-VL on {device} ...")
    proc = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            hf_model_path, trust_remote_code=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    raw_image = (
        torch.linspace(0, 255, image_size * image_size * 3)
        .reshape(image_size, image_size, 3)
        .to(torch.uint8)
    )
    batch = proc(
        text=[_PROMPT], images=[Image.fromarray(raw_image.numpy())], return_tensors="pt"
    )

    vis = {}
    model.multi_modal_projector.register_forward_hook(
        lambda m, i, o: vis.__setitem__("f", o.detach().float().cpu())
    )
    # MoE layers have a ``gate`` (MoEGate) whose forward returns
    # (topk_idx, topk_weight, aux_loss); ``experts`` is a ModuleList (never
    # called directly). Hook the gate to record the per-token expert selection.
    expert_indices: dict[int, torch.Tensor] = {}
    for i, layer in enumerate(model.language_model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(
                lambda m, inp, out, i=i: expert_indices.__setitem__(
                    i, out[0].detach().cpu()
                )
            )

    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    out = model(**inputs)
    ref = {
        "input_ids": batch["input_ids"].cpu(),
        "raw_image": raw_image,
        "last_logits": out.logits[:, -1, :].float().cpu(),
        "vision_features": vis["f"],
        "expert_indices": expert_indices,
    }
    del model
    torch.cuda.empty_cache()
    return ref


def _force_hf_routing(model, expert_indices, device):
    """Monkeypatch each MoE router to use HF's recorded top-k experts.

    Diagnostic: titan routes every token to exactly the experts HF chose (still
    computing its own gating weights at those experts via the real scores), so the
    discrete routing-flip divergence is removed and any residual is the non-routing
    math (attention / expert FFN / fp). ``expert_indices`` maps the HF layer index
    to a ``(num_tokens, top_k)`` LongTensor.
    """
    forced = 0
    for key, layer in model.layers.items():
        if not getattr(layer, "moe_enabled", False) or int(key) not in expert_indices:
            continue
        ids = expert_indices[int(key)]
        ids = ids.view(1, ids.shape[0], ids.shape[-1]).to(device)  # (1, L, K)
        router = layer.moe.router
        orig = router.forward

        def forced_forward(x_BLD, expert_bias_E=None, _r=router, _o=orig, _ids=ids):
            # Reuse the real score computation; override only the selection.
            _, _, scores_BLE = _o(x_BLD, expert_bias_E)
            topk = scores_BLE.gather(dim=-1, index=_ids)
            if _r.route_norm:
                topk = topk / (topk.sum(dim=-1, keepdim=True) + 1e-20)
            topk = topk * _r.route_scale
            return topk, _ids, scores_BLE

        router.forward = forced_forward
        forced += 1
    return forced


@torch.no_grad()
def run_tt(model_flavor, checkpoint_path, ref, dtype, vision_dtype, force_hf_routing):
    """torchtitan Kimi-VL: its own image processing + forward on the same image."""
    device = torch.device("cuda")
    print(f"Loading torchtitan Kimi-VL ({model_flavor}) on {device} ...")
    model_config = model_registry(model_flavor).model
    with torch.device("meta"):
        model = model_config.build()
    model.to_empty(device="cpu")
    # Cast before init_states so the complex64 ComplexRoPE cache survives.
    model.to(dtype)
    model.init_states(buffer_device=torch.device("cpu"))
    state_dict = ModelWrapper(model)._get_state_dict()
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    model.to(device)

    model.vision_encoder.to(vision_dtype)  # mixed precision: ViT in vision_dtype
    for layer in model.layers.values():
        layer.attention.inner_attention = ScaledDotProductAttention.Config().build()
    for layer in model.vision_encoder.layers.values():
        layer.attn.flex_attention = _VisionSDPA()
    model.eval()

    if force_hf_routing:
        n = _force_hf_routing(model, ref["expert_indices"], device)
        print(f"forced HF routing on {n} MoE layers")

    # titan's OWN Kimi image processing on the raw image (the real data path).
    img = process_image(
        Image.fromarray(ref["raw_image"].numpy()),
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        resize_fn=resize_to_patch_budget,
        max_patches=4096,
        max_patches_per_side=512,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    # MoonViT3d consumes raster-order patches (matching the released processor).
    patches, grid = vision_to_patches(
        img,
        patch_size=_PATCH_SIZE,
        temporal_patch_size=1,
        merge_size=_MERGE_SIZE,
        patch_order="raster",
    )
    pixel_values = patches.unsqueeze(0).to(device=device, dtype=vision_dtype)
    grid_thw = grid.unsqueeze(0).to(device)
    tokens = ref["input_ids"].to(device)

    # Sanity: titan's grid must yield the same vision-token count as HF produced
    # (the placeholder run in input_ids), else the scatter compares different seqs.
    n_titan = ((grid_thw[0, 1] // _MERGE_SIZE) * (grid_thw[0, 2] // _MERGE_SIZE)).item()
    n_placeholders = (tokens == _MEDIA_TOKEN_ID).sum().item()
    print(
        f"tokens={tuple(tokens.shape)}  pixel_values={tuple(pixel_values.shape)}  "
        f"grid_thw={grid_thw.tolist()}  titan_vis_tokens={n_titan}  "
        f"hf_placeholders={n_placeholders}"
    )
    assert n_titan == n_placeholders, (
        f"titan produced {n_titan} vision tokens but HF input_ids has "
        f"{n_placeholders} placeholders -- preprocessing grids differ"
    )

    # Localize: titan's vision features (projector output) vs HF's, pre-scatter.
    tt_feats = model.vision_encoder(pixel_values, grid_thw=grid_thw)
    ref_feats = ref["vision_features"].float().reshape(-1, tt_feats.shape[-1])
    tt_feats = tt_feats.float().cpu().reshape(-1, tt_feats.shape[-1])
    vcos = F.cosine_similarity(ref_feats.flatten(), tt_feats.flatten(), dim=0).item()
    vmax = (ref_feats - tt_feats).abs().max().item()
    print(
        f"vision features (pre-scatter): shape={tuple(tt_feats.shape)}  "
        f"cos={vcos:.6f}  max_diff={vmax:.3e}"
    )

    logits = model(
        tokens,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        special_tokens={"image_id": _MEDIA_TOKEN_ID, "video_id": _MEDIA_TOKEN_ID},
    )
    return logits[:, -1, :].float().cpu().squeeze()


def compare(ref_logits, tt_logits):
    ref, tt = ref_logits.squeeze(), tt_logits.squeeze()
    pq = F.log_softmax(ref, dim=-1)
    qq = F.log_softmax(tt, dim=-1)
    kl = F.kl_div(qq, pq, log_target=True, reduction="sum").item()
    cos = F.cosine_similarity(ref, tt, dim=-1).item()
    max_diff = (ref - tt).abs().max().item()
    top1 = (ref.argmax() == tt.argmax()).item()
    ov5 = len(set(ref.topk(5).indices.tolist()) & set(tt.topk(5).indices.tolist())) / 5
    print(f"\n{'=' * 60}\nFull MM logit parity (torchtitan vs HF Kimi-VL)\n{'=' * 60}")
    print(
        f"  KL={kl:.4e}  cos={cos:.6f}  max_diff={max_diff:.4e}  "
        f"top1={'Y' if top1 else 'N'}  top5={ov5:.0%}"
    )
    print(
        "RESULT: PASS (KL < 1e-3 -- fp noise)."
        if abs(kl) < 1e-3
        else "RESULT: FAIL (KL >= 1e-3)."
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf_model_path",
        default=os.path.expanduser("~/hf_assets/moonshotai/Kimi-VL-A3B-Instruct"),
    )
    p.add_argument("--tt_checkpoint_path", default="outputs/kimi/kimi_vl_a3b_dcp")
    p.add_argument("--model_flavor", default="Kimi-VL-A3B")
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument(
        "--hf_dtype", default="float32", choices=["float32", "bfloat16", "float16"]
    )
    p.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16", "float16"]
    )
    # Mixed precision: bf16 is poor for the ViT's high-dynamic-range activations,
    # so fp16 is the usual choice. Defaults to --dtype. (titan side only.)
    p.add_argument(
        "--vision_dtype", default=None, choices=["float32", "bfloat16", "float16"]
    )
    p.add_argument(
        "--force-hf-routing",
        action="store_true",
        help="Force titan MoE routers to use HF's recorded expert selections "
        "(diagnostic: removes routing-flip divergence to isolate the rest).",
    )
    args = p.parse_args()
    device, dtype = torch.device("cuda"), getattr(torch, args.dtype)
    vision_dtype = getattr(torch, args.vision_dtype) if args.vision_dtype else dtype
    hf_dtype = getattr(torch, args.hf_dtype)
    print(
        f"hf_dtype={args.hf_dtype}  titan text={args.dtype}  "
        f"titan vision={args.vision_dtype or args.dtype}"
    )

    ref = run_hf(args.hf_model_path, args.image_size, hf_dtype, device)
    tt_logits = run_tt(
        args.model_flavor,
        args.tt_checkpoint_path,
        ref,
        dtype,
        vision_dtype,
        args.force_hf_routing,
    )
    compare(ref["last_logits"], tt_logits)


if __name__ == "__main__":
    main()
