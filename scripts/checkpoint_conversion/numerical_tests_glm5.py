#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward parity for TorchTitan GLM-5.2 and Hugging Face GLM-MoE-DSA.

The released model is too large for a single-device correctness test. This
script instead reduces the Hugging Face configuration to TorchTitan's
``glm5_2_debugmodel``, initializes TorchTitan deterministically, converts its
state dict to Hugging Face format, and compares logits for identical tokens.
The strict Hugging Face load also validates every checkpoint mapping.

Install the pinned Transformers reference before running::

    pip install "git+https://github.com/huggingface/transformers.git@\
01269bc99768ad45f8aeaeb701e1315371e25239"

Usage::

    python -m scripts.checkpoint_conversion.numerical_tests_glm5
"""

import argparse
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.moe import MoE
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.flavors import Glm5Router
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM


_DTYPE = torch.float32
_DEFAULT_FLAVOR = "glm5_2_debugmodel"


def _build_hf_config(tt_config) -> GlmMoeDsaConfig:
    """Translate the TorchTitan debug architecture into an HF config."""
    attention = tt_config.layers[0].attention
    dense_ffn = next(
        layer.feed_forward
        for layer in tt_config.layers
        if isinstance(layer.feed_forward, DenseFeedForward.Config)
    )
    moe = next(
        layer.feed_forward
        for layer in tt_config.layers
        if isinstance(layer.feed_forward, MoE.Config)
    )
    if moe.shared_experts is None:
        raise ValueError("GLM-5 numerical parity requires shared experts.")
    if not isinstance(moe.router, Glm5Router.Config):
        raise ValueError("GLM-5 numerical parity requires the GLM group router.")

    routed_hidden_dim = moe.routed_experts.w13.out_features
    shared_hidden_dim = moe.shared_experts.w13.out_features
    if shared_hidden_dim % routed_hidden_dim:
        raise ValueError(
            "shared-expert hidden size must be a multiple of routed hidden size"
        )

    mlp_layer_types = [
        "sparse" if isinstance(layer.feed_forward, MoE.Config) else "dense"
        for layer in tt_config.layers
    ]
    indexer_types = [
        "full" if source == layer_idx else "shared"
        for layer_idx, source in enumerate(tt_config.index_sources)
    ]
    config = GlmMoeDsaConfig(
        vocab_size=tt_config.vocab_size,
        hidden_size=tt_config.dim,
        intermediate_size=dense_ffn.w13.out_features,
        moe_intermediate_size=routed_hidden_dim,
        num_hidden_layers=len(tt_config.layers),
        num_attention_heads=attention.n_heads,
        num_key_value_heads=attention.n_heads,
        n_shared_experts=shared_hidden_dim // routed_hidden_dim,
        n_routed_experts=moe.num_experts,
        routed_scaling_factor=moe.router.route_scale,
        kv_lora_rank=attention.kv_lora_rank,
        q_lora_rank=attention.q_lora_rank,
        qk_rope_head_dim=attention.qk_rope_head_dim,
        v_head_dim=attention.v_head_dim,
        qk_nope_head_dim=attention.qk_nope_head_dim,
        n_group=moe.router.num_expert_groups,
        topk_group=moe.router.num_limited_groups,
        num_experts_per_tok=moe.router.top_k,
        norm_topk_prob=moe.router.route_norm,
        max_position_embeddings=tt_config.max_context_length,
        rms_norm_eps=tt_config.norm.eps,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": attention.rope.theta,
        },
        mlp_layer_types=mlp_layer_types,
        index_topk=attention.indexer.index_topk,
        index_head_dim=attention.indexer.head_dim,
        index_n_heads=attention.indexer.n_heads,
        indexer_types=indexer_types,
        first_k_dense_replace=mlp_layer_types.count("dense"),
        attention_dropout=attention.inner_attention.attention_dropout,
        use_cache=False,
        tie_word_embeddings=False,
    )
    return config


def _build_tt_model(tt_config, device: torch.device):
    with torch.device("meta"):
        model = tt_config.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    return model.to(device=device, dtype=_DTYPE).eval()


def _build_hf_model(
    tt_config,
    tt_state_dict: dict[str, Any],
    device: torch.device,
):
    hf_config = _build_hf_config(tt_config)
    model = GlmMoeDsaForCausalLM(hf_config).to(dtype=_DTYPE)
    hf_state_dict = Glm5StateDictAdapter(tt_config, hf_assets_path=None).to_hf(
        tt_state_dict
    )
    model.load_state_dict(hf_state_dict, strict=True)
    return model.to(device).eval()


def _record_routing(model, *, is_hf: bool) -> dict[int, torch.Tensor]:
    choices: dict[int, torch.Tensor] = {}
    layers = model.model.layers if is_hf else model.layers.values()
    for layer_idx, layer in enumerate(layers):
        feed_forward = layer.mlp if is_hf else layer.feed_forward
        router = getattr(feed_forward, "gate" if is_hf else "router", None)
        if router is None:
            continue
        output_index = 2 if is_hf else 1

        def record_choices(
            _module,
            _inputs,
            output,
            layer_idx=layer_idx,
            output_index=output_index,
        ) -> None:
            choices[layer_idx] = output[output_index].detach().cpu()

        router.register_forward_hook(record_choices)
    return choices


def _print_routing_comparison(
    hf_choices: dict[int, torch.Tensor],
    tt_choices: dict[int, torch.Tensor],
) -> float:
    if set(hf_choices) != set(tt_choices):
        raise ValueError(
            "routing layers differ: "
            f"HF={sorted(hf_choices)}, TorchTitan={sorted(tt_choices)}"
        )
    num_matching = 0
    num_choices = 0
    for layer_idx in sorted(hf_choices):
        hf_ids = hf_choices[layer_idx].sort(dim=-1).values
        tt_ids = tt_choices[layer_idx].sort(dim=-1).values
        if hf_ids.shape != tt_ids.shape:
            raise ValueError(
                f"layer {layer_idx} routing shapes differ: "
                f"HF={tuple(hf_ids.shape)}, TorchTitan={tuple(tt_ids.shape)}"
            )
        num_matching += int((hf_ids == tt_ids).sum().item())
        num_choices += hf_ids.numel()
    match_rate = num_matching / num_choices if num_choices else 1.0
    print(f"router choices: {num_matching}/{num_choices} match ({match_rate:.1%})")
    return match_rate


def _compare_logits(
    hf_logits: torch.Tensor, tt_logits: torch.Tensor
) -> tuple[float, float, float]:
    hf = hf_logits.float().reshape(-1, hf_logits.shape[-1])
    tt = tt_logits.float().reshape(-1, tt_logits.shape[-1])
    if hf.shape != tt.shape:
        raise ValueError(
            f"logit shapes differ: HF={tuple(hf.shape)}, TorchTitan={tuple(tt.shape)}"
        )
    log_hf = F.log_softmax(hf, dim=-1)
    log_tt = F.log_softmax(tt, dim=-1)
    kl = F.kl_div(log_tt, log_hf, log_target=True, reduction="batchmean").item()
    cosine = F.cosine_similarity(hf.flatten(), tt.flatten(), dim=0).item()
    max_diff = (hf - tt).abs().max().item()
    top1 = (hf.argmax(dim=-1) == tt.argmax(dim=-1)).float().mean().item()
    print(
        "logits: "
        f"KL={kl:.4e} cos={cosine:.8f} max_diff={max_diff:.4e} "
        f"top1={top1:.1%}"
    )
    return kl, cosine, top1


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TorchTitan and Hugging Face GLM-5.2 logits."
    )
    parser.add_argument("--model_flavor", default=_DEFAULT_FLAVOR)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--min_router_match", type=float, default=0.99)
    parser.add_argument("--max_kl", type=float, default=1e-4)
    parser.add_argument("--min_cosine", type=float, default=0.9999)
    parser.add_argument("--min_top1", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tt_config = build_model_config(
        args.model_flavor,
        parallelism=ParallelismConfig(),
        seq_len=args.seq_len,
    )
    tt_model = _build_tt_model(tt_config, device)
    hf_model = _build_hf_model(tt_config, tt_model.state_dict(), device)
    hf_choices = _record_routing(hf_model, is_hf=True)
    tt_choices = _record_routing(tt_model, is_hf=False)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    tokens = torch.randint(
        0,
        tt_config.vocab_size,
        (args.seq_len,),
        generator=generator,
        device="cpu",
    ).to(device)
    positions = torch.arange(args.seq_len, device=device)
    attention_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=device)

    hf_logits = hf_model(
        input_ids=tokens.unsqueeze(0),
        attention_mask=attention_mask,
        position_ids=positions.unsqueeze(0),
        use_cache=False,
    ).logits
    tt_logits = tt_model(tokens, positions=positions).unsqueeze(0)

    router_match = _print_routing_comparison(hf_choices, tt_choices)
    kl, cosine, top1 = _compare_logits(hf_logits, tt_logits)
    if router_match < args.min_router_match:
        raise RuntimeError(
            f"router match {router_match:.4%} is below {args.min_router_match:.4%}"
        )
    if kl > args.max_kl:
        raise RuntimeError(f"KL divergence {kl:.4e} exceeds {args.max_kl:.4e}")
    if cosine < args.min_cosine:
        raise RuntimeError(f"cosine {cosine:.8f} is below {args.min_cosine:.8f}")
    if top1 < args.min_top1:
        raise RuntimeError(f"top-1 match {top1:.4%} is below {args.min_top1:.4%}")


if __name__ == "__main__":
    main()
