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
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_lfm2
from .model.args import LFM2ModelArgs
from .model.model import LFM2Model

__all__ = [
    "parallelize_lfm2",
    "LFM2ModelArgs",
    "LFM2Model",
    "lfm2_args",
]


# LFM2 model configurations
# Based on the standard sizes from the lfm2 package
lfm2_args = {
    "debugmodel": LFM2ModelArgs(
        vocab_size=64400,
        hidden_size=256,  # Smaller for debugging
        intermediate_size=704,  # ~2.75x hidden_size
        num_conv_blocks=4,  # Reduced from 10
        num_attention_blocks=2,  # Reduced from 6
        num_attention_heads=8,  # Reduced from 16
        num_key_value_heads=2,  # Reduced from 4
        conv_kernel_size=3,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=500000,  # Match llama3 debugmodel
        attention_dropout=0.0,
        hidden_dropout=0.0,
    ),
    "350M": LFM2ModelArgs(
        vocab_size=64400,
        hidden_size=768,
        intermediate_size=2048,
        num_conv_blocks=10,
        num_attention_blocks=6,
        num_attention_heads=12,
        num_key_value_heads=3,
        conv_kernel_size=3,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    ),
    "700M": LFM2ModelArgs(
        vocab_size=64400,
        hidden_size=1024,
        intermediate_size=2816,
        num_conv_blocks=10,
        num_attention_blocks=6,
        num_attention_heads=16,
        num_key_value_heads=4,
        conv_kernel_size=3,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    ),
    "1.2B": LFM2ModelArgs(
        vocab_size=64400,
        hidden_size=1536,
        intermediate_size=4096,
        num_conv_blocks=10,
        num_attention_blocks=6,
        num_attention_heads=24,
        num_key_value_heads=6,
        conv_kernel_size=3,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    ),
}


def get_train_spec() -> TrainSpec:
    """Get the training specification for LFM2 model."""
    return TrainSpec(
        model_cls=LFM2Model,
        model_args=lfm2_args,
        parallelize_fn=parallelize_lfm2,
        pipelining_fn=None,  # Pipeline parallelism not supported yet
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,  # No HF adapter for now
    )
