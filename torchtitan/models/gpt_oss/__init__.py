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
from torchtitan.models.common import RoPE
from torchtitan.models.common.moe import MoE

from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_gptoss
from .model.model import GptOssModel
from .model.state_dict_adapter import GptOssStateDictAdapter

__all__ = [
    "parallelize_gptoss",
    "GptOssModel",
    "gptoss_configs",
]


gptoss_configs = {
    "debugmodel": GptOssModel.Config(
        dim=256,
        n_layers=4,
        moe_config=MoE.Config(
            hidden_dim=2880,
            num_experts=8,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            gate_bias=True,
            score_before_experts=False,
            top_k=4,
            use_grouped_mm=True,
            load_balance_coeff=1e-3,
        ),
        attn_mask_type="causal",
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            format="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
    "20b": GptOssModel.Config(
        n_layers=24,
        moe_config=MoE.Config(
            hidden_dim=2880,
            num_experts=32,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            gate_bias=True,
            score_before_experts=False,
            top_k=4,
            use_grouped_mm=True,
            load_balance_coeff=1e-3,
        ),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            format="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
    "120b": GptOssModel.Config(
        n_layers=36,
        moe_config=MoE.Config(
            hidden_dim=2880,
            num_experts=128,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            gate_bias=True,
            score_before_experts=False,
            top_k=4,
            use_grouped_mm=True,
            load_balance_coeff=1e-3,
        ),
        rope_config=RoPE.Config(
            dim=64,
            max_seq_len=131072,
            theta=150000.0,
            format="cos_sin",
            scaling="yarn",
            rope_factor=32,
            beta_slow=32.0,
            beta_fast=1.0,
            original_seq_len=4096,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=gptoss_configs,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=GptOssStateDictAdapter,
    )
