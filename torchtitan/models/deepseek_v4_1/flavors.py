# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The released Flash architecture and a feature-preserving debug architecture."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import spmd_types as spmd
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    HiMidLoLinear,
    Linear,
    RMSNorm,
    RowParallelLinear,
    SqrtSoftplus,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import (
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    colwise_sharding_config,
    dense_activation_placement,
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.moe.sharding import router_gate_sharding_config
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionAttention,
)
from torchtitan.models.common.vision_encoder_sharding import (
    invariant_norm_sharding_config,
    multimodal_token_embedding_sharding_config,
    vision_attention_sharding_config,
    vision_colwise_sharding_config,
    vision_inner_attention_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rowwise_sharding_config,
)
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.deepseek_v4.mhc import HcPost
from torchtitan.models.deepseek_v4.sharding import (
    attn_sink_sharding_config,
    hc_post_sharding_config,
    wo_a_sharding_config,
)

from .attention import Compressor, CSA2InnerAttention, DeepSeekV41Attention, Indexer
from .engram import (
    compressed_token_map,
    Engram,
    EngramHash,
    FeatureParallelEmbedding,
    hash_primes,
)
from .mhc import HcCoefficients
from .model import DeepSeekV41Model, DeepSeekV41TransformerBlock
from .moe import Float32LimitedSwiGLU, ModalityMoE, ModalityRouter
from .reference_attention import ReferenceFlexInnerAttention
from .sharding import (
    compressor_sharding_config,
    csa2_inner_sharding_config,
    engram_embedding_sharding_config,
    hc_coefficients_sharding_config,
    indexer_sharding_config,
    modality_moe_sharding_config,
    modality_router_sharding_config,
    shared_attention_sharding_config,
)
from .vision import Aligner, VisionEncoder, VisionTransformerBlock

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_HC_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": nn.init.zeros_,
    "hc_scale": partial(nn.init.constant_, val=0.01),
}


def _replicated_linear(in_features, out_features):
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        inplace_wgrad_accum=False,
        param_init=_LINEAR_INIT,
        sharding_config=replicated_state_sharding_config("weight"),
    )


def _norm_config(dim, eps, *, sharding=None):
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=eps,
        param_init=_NORM_INIT,
        sharding_config=sharding or replicated_state_sharding_config("weight"),
    )


def _vision_configs(*, dim, released, attention_backend):
    vision_dim, heads, num_layers, intermediate, patch = (
        (1024, 16, 32, 2816, 14) if released else (64, 4, 2, 128, 2)
    )

    def linear(in_features, out_features, *, sharding, bias=True):
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            param_init=_LINEAR_INIT,
            sharding_config=sharding,
        )

    norm = _norm_config(vision_dim, 1e-6, sharding=invariant_norm_sharding_config())
    blocks = []
    for _ in range(num_layers):
        gate_up_sharding = vision_colwise_sharding_config()
        gate_up_sharding.state_shardings["weight"] = spmd.SpmdType(
            {"dp": spmd.R, "cp": spmd.I, "tp": spmd.S(1)}
        )
        gate_up_sharding.out_src_shardings = spmd.SpmdType(
            {"dp": spmd.V, "cp": spmd.I, "tp": spmd.S(2)}
        )
        gate_up_sharding.local_spmd = True
        blocks.append(
            VisionTransformerBlock.Config(
                norm1=replace(norm),
                norm2=replace(norm),
                attn=VisionAttention.Config(
                    dim=vision_dim,
                    num_heads=heads,
                    wq=linear(
                        vision_dim,
                        vision_dim,
                        sharding=vision_colwise_sharding_config(input_tp=spmd.R),
                    ),
                    wk=linear(
                        vision_dim,
                        vision_dim,
                        sharding=vision_colwise_sharding_config(input_tp=spmd.R),
                    ),
                    wv=linear(
                        vision_dim,
                        vision_dim,
                        sharding=vision_colwise_sharding_config(input_tp=spmd.R),
                    ),
                    proj=InvariantRowParallelLinear.Config(
                        in_features=vision_dim,
                        out_features=vision_dim,
                        bias=True,
                        param_init=_LINEAR_INIT,
                        sharding_config=vision_rowwise_sharding_config(),
                    ),
                    inner_attention=(
                        FlexInnerAttention
                        if attention_backend == "flex"
                        else ReferenceFlexInnerAttention
                    ).Config(sharding_config=vision_inner_attention_sharding_config()),
                    sharding_config=vision_attention_sharding_config(
                        rope_cache_dp=spmd.V
                    ),
                ),
                mlp=DenseFeedForward.Config(
                    w13=Linear.Config(
                        in_features=vision_dim,
                        out_features=intermediate,
                        num_linears=2,
                        bias=False,
                        param_init=_LINEAR_INIT,
                        sharding_config=gate_up_sharding,
                    ),
                    w2=InvariantRowParallelLinear.Config(
                        in_features=intermediate,
                        out_features=vision_dim,
                        bias=False,
                        param_init=_LINEAR_INIT,
                        sharding_config=vision_rowwise_sharding_config(),
                    ),
                ),
            )
        )
    vision = VisionEncoder.Config(
        dim=vision_dim,
        num_heads=heads,
        rope_theta=10000.0,
        patch_embed=linear(
            3 * patch**2,
            vision_dim,
            sharding=vision_invariant_linear_sharding_config(),
        ),
        layers=blocks,
        norm=norm,
    )
    aligner = Aligner.Config(
        downsample_ratio=3,
        w1=linear(
            9 * vision_dim, dim, sharding=vision_invariant_linear_sharding_config()
        ),
        w2=linear(dim, dim, sharding=vision_invariant_linear_sharding_config()),
    )
    image_embeddings = Embedding.Config(
        num_embeddings=3,
        embedding_dim=dim,
        param_init={"weight": partial(nn.init.normal_, std=0.02)},
        sharding_config=replicated_state_sharding_config("weight"),
    )
    return vision, aligner, image_embeddings


def _make_config(
    *,
    seq_len,
    parallelism,
    released,
    tokenizer_path,
    quantize,
    indexer_loss_coeff,
    vision_enabled,
    vision_attention_backend,
    attention_backend,
):
    if (
        parallelism.tensor_parallel_degree > 1
        or parallelism.expert_parallel_degree > 1
        or parallelism.pipeline_parallel_degree > 1
    ):
        raise ValueError("V4.1 currently supports FSDP with TP=EP=PP=1.")
    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1
    if parallelism.context_parallel_degree > 1:
        raise ValueError(
            "CSA2 shared pools currently require context_parallel_degree=1."
        )
    if parallelism.tensor_parallel_degree > parallelism.expert_parallel_degree:
        raise ValueError(
            "V4.1 requires expert_parallel_degree >= tensor_parallel_degree."
        )
    dim, vocab_size = (5120, 129280) if released else (128, 2048)
    num_layers, num_encoder_layers = (40, 20) if released else (12, 8)
    num_heads, head_dim, rope_dim = (64, 512, 64) if released else (8, 64, 16)
    q_rank, o_rank, num_groups = (1280, 1024, 8) if released else (64, 32, 2)
    index_heads, index_dim, index_topk = (32, 128, 512) if released else (4, 32, 4)
    window, num_experts, top_k, inter_dim = (
        (128, 384, 6, 2304) if released else (8, 8, 2, 128)
    )
    ratios = (
        (0, 0)
        + (2,) * (num_encoder_layers - 2)
        + (1,) * (num_layers - num_encoder_layers)
    )
    kv_sources = (2, 8, 14, 20) if released else (2, 5, 8)
    index_sources = (2, 8, 14, 20, 24, 28, 32, 36) if released else (2, 5, 8, 10)
    candidate_source = num_encoder_layers
    engram_layers = (1, 14) if released else (1, 5)
    engram_heads, engram_dim, bucket_size = (
        (8, 256, 16000000) if released else (2, 32, 128)
    )
    hc_mult, norm_eps = 4, 1e-20
    if (
        num_heads % parallelism.tensor_parallel_degree
        or num_groups % parallelism.tensor_parallel_degree
    ):
        raise ValueError(
            "Tensor parallel degree must divide attention heads and output groups."
        )
    primes = hash_primes(
        num_layers=len(engram_layers),
        max_ngram_size=4,
        num_heads=engram_heads,
        bucket_size=bucket_size,
    )
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        token_map, compressed_vocab = compressed_token_map(
            AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        )
        if len(token_map) != vocab_size or (released and compressed_vocab != 99092):
            raise ValueError(
                "Tokenizer does not match the V4.1 Flash Engram vocabulary."
            )
    else:
        token_map, compressed_vocab = (
            ((), 99092) if released else (tuple(range(vocab_size)), vocab_size)
        )
    hash_config = EngramHash.Config(
        layer_ids=engram_layers,
        primes=primes,
        max_ngram_size=4,
        compressed_vocab_size=compressed_vocab,
        pad_id=2,
        token_map=token_map,
    )
    replicated = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    layers = []
    for layer_id, ratio in enumerate(ratios):
        rope = ComplexRoPE.Config(
            dim=rope_dim,
            max_context_length=seq_len,
            theta=160000.0 if ratio else 10000.0,
            scaling="yarn" if ratio else "none",
            rope_factor=16.0,
            original_seq_len=65536,
            beta_fast=32.0,
            beta_slow=1.0,
            sharding_config=rope_cache_sharding_config(),
        )
        compressor = None
        if layer_id in kv_sources:
            compressor = Compressor.Config(
                ratio=ratio,
                wkv=_replicated_linear(dim, head_dim),
                wgate=_replicated_linear(dim, head_dim) if ratio == 2 else None,
                norm=_norm_config(head_dim, norm_eps),
                sharding_config=compressor_sharding_config(),
            )
        indexer = None
        if layer_id in index_sources:
            indexer = Indexer.Config(
                impl=attention_backend,
                num_heads=index_heads,
                head_dim=index_dim,
                rope_head_dim=rope_dim,
                ratio=ratio,
                topk=index_topk,
                wq_b=_replicated_linear(q_rank, index_heads * index_dim),
                weights_proj=_replicated_linear(dim, index_heads),
                wk=_replicated_linear(head_dim, index_dim)
                if layer_id in kv_sources
                else None,
                k_norm=_norm_config(index_dim, norm_eps)
                if layer_id in kv_sources
                else None,
                rope=replace(rope),
                candidate_source=layer_id == candidate_source,
                uses_candidates=layer_id > candidate_source,
                num_candidate_blocks=2048 if released else 4,
                candidate_block_size=8 if released else 2,
                quantize=quantize,
                sharding_config=indexer_sharding_config(),
            )
        inner = CSA2InnerAttention.Config(
            impl=attention_backend,
            window_size=window,
            compress_ratio=ratio or 1,
            softmax_scale=head_dim**-0.5,
            index_topk=index_topk,
            aux_loss=SparseIndexerLoss.Config(
                coeff=indexer_loss_coeff,
                reduce_mesh="loss",
                softmax_scale=head_dim**-0.5,
                num_heads=num_heads,
                compress_ratio=ratio,
            )
            if ratio and indexer_loss_coeff
            else None,
            sharding_config=csa2_inner_sharding_config(),
        )
        attention = DeepSeekV41Attention.Config(
            n_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=rope_dim,
            num_groups=num_groups,
            o_lora_rank=o_rank,
            ratio=ratio,
            rope=rope,
            inner_attention=inner,
            wq_a=_replicated_linear(dim, q_rank),
            q_norm=_norm_config(q_rank, norm_eps),
            wq_b=Linear.Config(
                in_features=q_rank,
                out_features=num_heads * head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=colwise_sharding_config(input_layout=replicated),
            ),
            wkv=_replicated_linear(dim, head_dim),
            kv_norm=_norm_config(head_dim, norm_eps),
            wo_a=Linear.Config(
                in_features=num_heads * head_dim // num_groups,
                out_features=num_groups * o_rank,
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=wo_a_sharding_config(),
            ),
            wo_b=RowParallelLinear.Config(
                in_features=num_groups * o_rank,
                out_features=dim,
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=rowwise_sharding_config(
                    output_layout=residual_activation_placement(enable_sp=enable_sp)
                ),
            ),
            attn_sink=Linear.Config(
                in_features=1,
                out_features=num_heads,
                bias=False,
                param_init={"weight": nn.init.zeros_},
                sharding_config=attn_sink_sharding_config(),
            ),
            compressor=compressor,
            indexer=indexer,
            quantize=quantize,
            sharding_config=shared_attention_sharding_config(enable_sp=enable_sp),
        )
        router = ModalityRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            gate=HiMidLoLinear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                backward_mode="hi_mid_lo",
                param_init=_LINEAR_INIT,
                sharding_config=router_gate_sharding_config(),
            ),
            score_func=SqrtSoftplus.Config(),
            route_norm=True,
            route_scale=1.5,
            sharding_config=modality_router_sharding_config(),
        )
        expert_init: dict[str, Callable] = {
            "w1_EFD": _LINEAR_INIT["weight"],
            "w3_EFD": _LINEAR_INIT["weight"],
            "w2_EDF": _LINEAR_INIT["weight"],
        }
        experts = make_routed_experts_config(
            dim=dim,
            hidden_dim=inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=expert_init,
            enable_ep=enable_ep,
        )
        experts.activation_fn = Float32LimitedSwiGLU.Config(limit=10.0)
        shared = make_shared_expert_ffn_config(
            dim=dim,
            hidden_dim=inter_dim,
            w13_param_init=_LINEAR_INIT,
            w2_param_init=_LINEAR_INIT,
            enable_sp=enable_sp,
        )
        shared.activation_fn = Float32LimitedSwiGLU.Config(limit=10.0)
        moe = ModalityMoE.Config(
            num_experts=num_experts,
            router=router,
            routed_experts=experts,
            shared_experts=shared,
            load_balance_coeff=1e-3,
            sharding_config=modality_moe_sharding_config(enable_sp=enable_sp),
        )
        hc = HcCoefficients.Config(
            dim=dim,
            hc_mult=hc_mult,
            sinkhorn_iters=20,
            eps=1e-6,
            norm_eps=norm_eps,
            param_init=_HC_INIT,
            sharding_config=hc_coefficients_sharding_config(enable_sp=enable_sp),
        )
        engram = None
        if layer_id in engram_layers:
            layout = primes[engram_layers.index(layer_id)]
            rows = sum(sum(order) for order in layout)
            engram = Engram.Config(
                dim=dim,
                hc_mult=hc_mult,
                norm_eps=norm_eps,
                embed=FeatureParallelEmbedding.Config(
                    num_embeddings=rows,
                    embedding_dim=engram_dim,
                    param_init={"weight": partial(nn.init.normal_, std=0.02)},
                    sharding_config=engram_embedding_sharding_config(
                        enable_sp=enable_sp
                    ),
                ),
                wkv=_replicated_linear(
                    3 * engram_heads * engram_dim, (hc_mult + 1) * dim
                ),
                quantize=quantize,
                param_init={"q_weight": nn.init.ones_, "k_weight": nn.init.ones_},
                sharding_config=replicated_state_sharding_config(
                    "q_weight", "k_weight"
                ),
            )
        layers.append(
            DeepSeekV41TransformerBlock.Config(
                attention=attention,
                attention_norm=_norm_config(
                    dim, norm_eps, sharding=norm_sharding_config(enable_sp=enable_sp)
                ),
                ffn_norm=_norm_config(
                    dim, norm_eps, sharding=norm_sharding_config(enable_sp=enable_sp)
                ),
                feed_forward=moe,
                hc_attn=hc,
                hc_ffn=replace(hc),
                hc_post=HcPost.Config(
                    sharding_config=hc_post_sharding_config(enable_sp=enable_sp)
                ),
                engram=engram,
            )
        )
    vision, aligner, image_embeddings = (
        _vision_configs(
            dim=dim, released=released, attention_backend=vision_attention_backend
        )
        if vision_enabled
        else (None, None, None)
    )
    return DeepSeekV41Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_context_length=seq_len,
        layers=layers,
        hc_mult=hc_mult,
        num_encoder_layers=num_encoder_layers,
        compress_ratios=ratios,
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        engram_layer_ids=engram_layers,
        engram_hash=hash_config,
        vision=vision,
        aligner=aligner,
        image_embeddings=image_embeddings,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init={"weight": partial(nn.init.normal_, std=1.0)},
            sharding_config=multimodal_token_embedding_sharding_config()
            if vision_enabled
            else token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=_norm_config(
            dim,
            norm_eps,
            sharding=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            bias=False,
            param_init=_LINEAR_INIT,
            sharding_config=lm_head_sharding_config(),
        ),
    )


def _debugmodel(**kwargs):
    return _make_config(released=False, **kwargs)


def _deepseek_v4_1_flash(**kwargs):
    return _make_config(released=True, **kwargs)


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "deepseek_v4_1_flash": (_deepseek_v4_1_flash, 1048576),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    tokenizer_path: str | None = None,
    quantize: bool = True,
    indexer_loss_coeff: float = 0.01,
    vision_enabled: bool | None = None,
    vision_attention_backend: str = "flex",
    attention_backend: str = "fused",
):
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown deepseek_v4_1 flavor {flavor}; available: {list(MODEL_FLAVORS)}"
        )
    factory, max_context_length = MODEL_FLAVORS[flavor]
    context_length = max_context_length if seq_len is None else seq_len
    if not 0 < context_length <= max_context_length:
        raise ValueError(f"seq_len must be in [1, {max_context_length}].")
    if indexer_loss_coeff < 0:
        raise ValueError("indexer_loss_coeff must be nonnegative.")
    if vision_attention_backend not in ("flex", "reference"):
        raise ValueError("vision_attention_backend must be flex or reference.")
    if attention_backend not in ("fused", "reference"):
        raise ValueError("attention_backend must be fused or reference.")
    config = factory(
        seq_len=context_length,
        parallelism=parallelism,
        tokenizer_path=tokenizer_path,
        quantize=quantize,
        indexer_loss_coeff=indexer_loss_coeff,
        vision_enabled=flavor == "deepseek_v4_1_flash"
        if vision_enabled is None
        else vision_enabled,
        vision_attention_backend=vision_attention_backend,
        attention_backend=attention_backend,
    )
    return config
