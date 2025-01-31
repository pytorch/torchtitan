# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.model_spec import ModelSpec
from torchtitan.models.llama.model import ModelArgs, Transformer


llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


def build_model_spec() -> ModelSpec:
    # Avoid circular import
    from torchtitan.parallelisms.parallelize_llama import parallelize_llama
    from torchtitan.parallelisms.pipeline_llama import pipeline_llama

    return ModelSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        tokenizer="tiktoken",
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
    )
