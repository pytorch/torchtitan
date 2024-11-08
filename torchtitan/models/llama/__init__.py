# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8, use_scaled=False),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16, use_scaled=False),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32, use_scaled=False),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40, use_scaled=False),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40, use_scaled=False),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        use_scaled=False,
    ),
}

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
        use_scaled=False,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        use_scaled=False,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
        use_scaled=False,
    ),
}

llama3_1_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        use_scaled=True,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        use_scaled=True,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
        use_scaled=True,
    ),
}
