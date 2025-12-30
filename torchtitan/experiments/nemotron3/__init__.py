# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_nemotron3
from .model.args import Nemotron3ModelArgs
from .model.model import Nemotron3Model
from .model.state_dict_adapter import Nemotron3StateDictAdapter

__all__ = [
    "Nemotron3ModelArgs",
    "Nemotron3Model",
    "Nemotron3StateDictAdapter",
    "parallelize_nemotron3",
    "nemotron3_args",
]


# NemotronH model flavors
# Pattern key: M=Mamba2, *=Attention, E=MLP, O=MoE
nemotron3_args = {
    # Debug model for testing
    "debugmodel": Nemotron3ModelArgs(
        vocab_size=131072,
        dim=1024,
        hidden_dim=4096,
        n_layers=16,
        hybrid_override_pattern="M*M*M*M*M*M*M*M*",
        n_heads=16,
        head_dim=64,
        n_kv_heads=8,
        max_seq_len=4096,
        mamba_num_heads=16,
        mamba_head_dim=64,
    ),
    # NemotronH-nano-30B configuration
    # From https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json
    # Uses MoE (O) layers for feed-forward, giving ~30B total params with ~3B active per token
    "nano-30B": Nemotron3ModelArgs(
        vocab_size=131072,
        dim=2688,  # hidden_size
        hidden_dim=1856,  # intermediate_size
        n_layers=52,
        hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        n_heads=32,
        head_dim=128,
        n_kv_heads=2,  # num_key_value_heads
        max_seq_len=262144,  # max_position_embeddings
        mlp_hidden_act="relu2",
        attn_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02,
        norm_eps=1e-5,
        residual_in_fp32=False,
        # Mamba2 config
        use_mamba_kernels=True,
        ssm_state_size=128,
        mamba_num_heads=64,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=128,
        rescale_prenorm_residual=True,
        # MoE config
        n_routed_experts=128,
        n_shared_experts=1,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        num_experts_per_tok=6,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Nemotron3Model,
        model_args=nemotron3_args,
        parallelize_fn=parallelize_nemotron3,
        pipelining_fn=None,  # TODO: Implement pipelining if needed
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=Nemotron3StateDictAdapter,
    )
