# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize_deepseek_v3 import parallelize_deepseekv3
from .infra.pipeline_parallel import pipeline_llm
from .deepseek_v3.args import DeepSeekV3ModelArgs
from .deepseek_v3.model import DeepSeekV3Model
from .deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3ModelArgs",
    "DeepSeekV3Model",
    "deepseekv3_args",
]


deepseekv3_args = {
    "debugmodel": DeepSeekV3ModelArgs(
        vocab_size=2048,
        dim=4096,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=12,
        n_dense_layers=12,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=2,
            top_k=3,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    ),
    "debugmodel_flex_attn": DeepSeekV3ModelArgs(
        vocab_size=2048,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=6,
        n_dense_layers=1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=2,
            top_k=3,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
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
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "236B": DeepSeekV3ModelArgs(
        vocab_size=102400,
        dim=5120,
        inter_dim=12288,
        moe_inter_dim=1536,
        n_layers=60,
        n_dense_layers=1,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=160,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            route_scale=16.0,
            score_before_experts=False,
        ),
        n_expert_groups=8,
        n_limited_groups=3,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "671B": DeepSeekV3ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
            score_before_experts=False,
        ),
        n_expert_groups=8,
        n_limited_groups=4,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
}


def get_deepseek_v3_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_args,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
