# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Literal

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.protocols.model_spec import ModelSpec
from .model import HFTransformerModel
from .parallelize import parallelize_hf_transformers
from .pipeline import pipeline_hf_transformers
from .state_dict_adapter import HFTransformerStateDictAdapter

__all__ = [
    "HFTransformerModel",
]


@dataclass
class TitanModelConfig:
    """Arguments for the base TorchTitan model.

    Two kinds of fields (see the groups below): those that mirror an HF config
    key default to None so AutoConfig's value is used; TorchTitan-only fields
    keep concrete defaults since they don't override anything from the HF config.
    """

    # Fields that map to an HF config key: default None so the value from
    # AutoConfig.from_pretrained is kept. A non-None default would be injected
    # over the HF config and force the wrong architecture (e.g. rope_theta).
    # Set explicitly only to intentionally override (e.g. debugmodel sizes).
    dim: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    intermediate_size: int | None = None
    norm_eps: float | None = None
    rope_theta: float | None = None

    # TorchTitan-only fields with no HF equivalent: they don't override anything
    # from the HF config, so they keep concrete defaults. (multiple_of and
    # ffn_dim_multiplier are only used when deriving FFN size from an explicitly
    # overridden dim; max_seq_len is set from training.seq_len.)
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


@dataclass
class TitanMoeModelConfig(TitanModelConfig):
    """MoE model config — extends the base config with routed-expert parameters."""

    num_experts: int = 128
    """Total number of routed experts in each MoE layer."""

    num_experts_per_tok: int = 8
    """Top-k routing: number of experts each token is dispatched to."""

    moe_intermediate_size: int = 768
    """Hidden dimension of each expert's MLP (per-expert intermediate size)."""

    decoder_sparse_step: int = 1
    """Replace the dense MLP with an MoE block every N decoder layers (1 = every layer is MoE)."""

    norm_topk_prob: bool = False
    """Normalize the top-k routing scores to sum to 1 (HF `norm_topk_prob` convention)."""

    num_nextn_predict_layers: int | None = None
    """
    DeepSeek V3-style multi-token prediction: number of MTP heads.
    None or 0 disables MTP.
    """

    experts_implementation: Literal[
        "grouped_mm", "batched_mm", "eager", "native"
    ] = "grouped_mm"
    """
    Selects the HF experts forward kernel via `PretrainedConfig._experts_implementation`.
    "grouped_mm" is the fused fast path; "eager" is HF's original for-loop
    (numerical reference for debugging). "grouped_mm"/"batched_mm"/"eager" require
    a model that supports a settable experts implementation (the
    `@use_experts_implementation` decorator) — requesting one on a model that does
    not (e.g. Llama4) raises. "native" uses the model's own built-in experts
    kernel unchanged (the only valid choice for non-settable models like Llama4).
    """

    load_balance_coeff: float | None = 1e-3
    """Step size for auxiliary-loss-free MoE load balancing. None disables it."""

    comm_backend: str = "standard"
    """Token dispatch backend for expert parallelism.
    "standard" uses PyTorch all-to-all collectives, "deepep" uses DeepEP
    kernels for H100/NVLink, "hybridep" uses HybridEP for GB200/NVLink72.
    """


flavors = {
    "debugmodel": HFTransformerModel.Config(
        model_config=TitanModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "debugmodel_flex": HFTransformerModel.Config(
        model_config=TitanModelConfig(
            use_flex_attn=True,
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "debugmodel_moe": HFTransformerModel.Config(
        model_config=TitanMoeModelConfig(
            dim=2048,
            n_layers=4,
            n_heads=16,
            n_kv_heads=8,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            num_nextn_predict_layers=0,
        ),
    ),
    "full": HFTransformerModel.Config(
        model_config=TitanModelConfig(),
    ),
    "full_moe": HFTransformerModel.Config(
        model_config=TitanMoeModelConfig(
            dim=2048,
            n_layers=48,
            n_heads=32,
            n_kv_heads=4,
            norm_eps=1e-6,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            norm_topk_prob=True,
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="transformers_modeling_backend",
        flavor=flavor,
        model=flavors[flavor],
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=HFTransformerStateDictAdapter,
    )
