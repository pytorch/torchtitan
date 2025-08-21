# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_vlm
# from .infra.pipeline import pipeline_llama
from .model.args import Llama3Siglip2ModelArgs, Siglip2ModelArgs
from .model.model import Llama3Siglip2Transformer

__all__ = [
    "parallelize_vlm",
    # "pipeline_llama",
    "Llama3Siglip2ModelArgs",
    "Llama3Siglip2Transformer",
    "llama3_siglip2_configs",
]


siglip2_configs = {
    "debugmodel": Siglip2ModelArgs(
        dim=128,
        ffn_dim=256,
        n_layers=4,
        n_heads=2,
    )
}

llama3_siglip2_configs = {
    "debugmodel": Llama3Siglip2ModelArgs(
        encoder=siglip2_configs["debugmodel"],
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
    "debugmodel_flex_attn": Llama3Siglip2ModelArgs(
        encoder=siglip2_configs["debugmodel"],
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2000,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "8B": Llama3Siglip2ModelArgs(
        encoder=siglip2_configs["debugmodel"],
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": Llama3Siglip2ModelArgs(
        encoder=siglip2_configs["debugmodel"],
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": Llama3Siglip2ModelArgs(
        encoder=siglip2_configs["debugmodel"],
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3-siglip2",
        model_cls=Llama3Siglip2Transformer,
        model_args=llama3_siglip2_configs,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        # state_dict_adapter=Llama3StateDictAdapter,
    )
)
