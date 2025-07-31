# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.experiments.llama4.optimizer import build_llama4_optimizers
from torchtitan.models.llama3.infra.pipeline import pipeline_llama

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_deepseekv3
from .model.args import DeepSeekV3ModelArgs
from .model.model import DeepSeekV3Model

__all__ = [
    "parallelize_deepseekv3",
    "DeepseekV3ModelArgs",
    "DeepseekV3Model",
    "deepseekv3_configs",
]


deepseekv3_configs = {
    "debugmodel": DeepSeekV3ModelArgs(
        vocab_size=2000,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=3,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=3,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    ),
    "debugmodel_flex_attn": DeepSeekV3ModelArgs(
        vocab_size=2000,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=3,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=3,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "16B": DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=64,
        n_shared_experts=2,
        n_activated_experts=6,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    ),
    "236B": DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=5120,
        inter_dim=12288,
        moe_inter_dim=1536,
        n_layers=60,
        n_dense_layers=1,
        n_heads=128,
        n_routed_experts=160,
        n_shared_experts=2,
        n_activated_experts=6,
        n_expert_groups=8,
        n_limited_groups=3,
        route_scale=16.0,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    "671B": DeepSeekV3ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=8,
        n_expert_groups=8,
        n_limited_groups=4,
        route_scale=2.5,
        score_func="sigmoid",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dtype="fp8",
    ),
}


register_train_spec(
    TrainSpec(
        name="deepseek_v3",
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_llama4_optimizers,  # use optimizer hooks to update expert weights
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
