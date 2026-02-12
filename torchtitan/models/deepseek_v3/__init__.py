# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.common import FeedForward, RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_deepseekv3
from .model.model import DeepSeekV3Model
from .model.state_dict_adapter import DeepSeekV3StateDictAdapter

__all__ = [
    "parallelize_deepseekv3",
    "DeepSeekV3Model",
    "deepseekv3_configs",
]


deepseekv3_configs = {
    "debugmodel": DeepSeekV3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        n_heads=16,
        moe_config=MoE.Config(
            hidden_dim=256,
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
        ff_config=FeedForward.Config(hidden_dim=1024),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            format="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "debugmodel_flex_attn": DeepSeekV3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        n_dense_layers=1,
        n_heads=16,
        moe_config=MoE.Config(
            hidden_dim=256,
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
        attn_type="flex",
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=1024),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            format="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "16B": DeepSeekV3Model.Config(
        vocab_size=102400,
        dim=2048,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        moe_config=MoE.Config(
            hidden_dim=1408,
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
        attn_type="flex",
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=10944),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            format="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "236B": DeepSeekV3Model.Config(
        vocab_size=102400,
        dim=5120,
        n_layers=60,
        n_dense_layers=1,
        n_heads=128,
        moe_config=MoE.Config(
            hidden_dim=1536,
            num_experts=160,
            num_shared_experts=2,
            top_k=6,
            num_expert_groups=8,
            num_limited_groups=3,
            score_func="softmax",
            route_norm=False,
            route_scale=16.0,
            score_before_experts=False,
        ),
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attn_type="flex",
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=12288),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            format="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
    "671B": DeepSeekV3Model.Config(
        vocab_size=129280,
        dim=7168,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        moe_config=MoE.Config(
            hidden_dim=2048,
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            num_expert_groups=8,
            num_limited_groups=4,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
            score_before_experts=False,
        ),
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attn_type="flex",
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=18432),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=4096 * 4,
            theta=10000.0,
            format="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=deepseekv3_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
