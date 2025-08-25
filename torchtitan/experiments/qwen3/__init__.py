# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_qwen3
from .model.args import Qwen3ModelArgs
from .model.model import Qwen3Model

__all__ = [
    "parallelize_qwen3",
    "Qwen3ModelArgs",
    "Qwen3Model",
    "qwen3_configs",
]


# Adding different variants of the model

qwen3_configs = {
    "0.6B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "1.7B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "4B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=2560,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=9728,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "8B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=12288,
        rope_theta=1000000,
    ),
    "14B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=17408,
        rope_theta=1000000,
    ),
    "32B": Qwen3ModelArgs(
        vocab_size=151936,
        max_seq_len=4096,
        head_dim=128,
        dim=5120,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=25600,
        rope_theta=1000000,
    ),
}


register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=Qwen3Model,
        model_args=qwen3_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
