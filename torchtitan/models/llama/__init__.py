# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from torchtitan.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=2, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}
