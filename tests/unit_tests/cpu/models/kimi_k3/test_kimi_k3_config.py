# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from fnmatch import fnmatchcase
from typing import cast

import pytest
import torch

from torchtitan.components.optim import AdamW, DistMuon, OptimizersContainer
from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan_recipes.overrides.kimi_attention_residual import TritonAttentionResidual
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    kimi_k3_debugmodel,
)


_TRITON_ATTENTION_RESIDUAL = (
    "torchtitan_recipes.overrides.kimi_attention_residual.triton_attention_residual"
)


def test_kimi_k3_enables_attention_residual_local_compile() -> None:
    config = kimi_k3_debugmodel()
    model_config = cast(KimiK3Model.Config, config.model)

    assert model_config.local_compile.regions == [
        "loss",
        "gated_rmsnorm",
        "fused_binary_activation",
        "fp32_to_bf16_split",
        "attention_residual",
    ]


def test_triton_attention_residual_override_replaces_all_configs() -> None:
    config = kimi_k3_debugmodel()
    model_config = cast(KimiK3Model.Config, config.model)

    replacements = apply_overrides(
        OverrideConfig(imports=[_TRITON_ATTENTION_RESIDUAL]),
        config,
    )

    assert len(replacements) == len(model_config.layers) + 1
    assert isinstance(
        model_config.attention_residual,
        TritonAttentionResidual.Config,
    )
    assert all(
        isinstance(
            layer_config.attention_residual,
            TritonAttentionResidual.Config,
        )
        for layer_config in model_config.layers
    )


@pytest.mark.parametrize("flavor", ("debugmodel", "Kimi-K3"))
@pytest.mark.parametrize("context_parallel_degree", (1, 2))
def test_dist_muon_covers_vision_linear_weights(
    flavor: str, context_parallel_degree: int
) -> None:
    parallelism = ParallelismConfig(context_parallel_degree=context_parallel_degree)
    model_config = build_model_config(
        flavor, seq_len=16, parallelism=parallelism, attn_backend="varlen"
    )
    optimizer_config = _dist_muon_optimizer(
        model_config, muon_lr=1e-3, adamw_lr=1e-3, parallelism=parallelism
    )
    muon_config, adamw_config = optimizer_config.optimizers
    assert isinstance(muon_config, DistMuon.Config)
    assert isinstance(adamw_config, AdamW.Config)

    with torch.device("meta"):
        model = model_config.build()
    parameters = dict(model.named_parameters())
    vision_linear_names = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if name.startswith("vision_encoder.") and isinstance(module, Linear)
    }
    vision_names = {name for name in parameters if name.startswith("vision_encoder.")}
    claimed: set[str] = set()
    muon_names = set(
        OptimizersContainer._build_param_group(model, muon_config, claimed)[
            "param_names"
        ]
    )
    adamw_names = set(
        OptimizersContainer._build_param_group(model, adamw_config, claimed)[
            "param_names"
        ]
    )
    assert muon_names == set(muon_config.compute_sharding_by_fqn)
    assert muon_names & vision_names == vision_linear_names
    assert adamw_names & vision_names == vision_names - vision_linear_names

    vision_config = model_config.vision_encoder
    assert vision_config is not None
    assert len(vision_linear_names) == 6 * vision_config.num_layers + 3
    assert "vision_encoder.pos_embed" in adamw_names
    assert "vision_encoder.final_norm.weight" in adamw_names
    assert "vision_encoder.projector.post_norm.weight" in adamw_names
    attention_config = vision_config.block.attn
    head_dim = attention_config.dim // attention_config.num_heads
    for fqn in vision_linear_names:
        layout = muon_config.compute_sharding_by_fqn[fqn]
        assert layout.compute_dtype is torch.bfloat16
        sharding = layout.shardings_by_mesh_axis["dp_shard"]
        if fqn.endswith((".attn.wq.weight", ".attn.wk.weight", ".attn.wv.weight")):
            assert sharding == BlockShard(dim=0, block_sizes=(head_dim,))
            assert parameters[fqn].shape == (attention_config.dim, vision_config.dim)
            assert (
                sharding.num_blocks(parameters[fqn].shape[0])
                == attention_config.num_heads
            )
        else:
            assert isinstance(sharding, Owned)

    bucketed_names = []
    for bucket in muon_config.bucket_configs:
        matches = {
            fqn
            for fqn in muon_names
            if any(fnmatchcase(fqn, pattern) for pattern in bucket.patterns)
        }
        if matches & vision_names:
            assert matches <= vision_names
        bucketed_names.extend(matches)
    assert sorted(bucketed_names) == sorted(muon_names)


def test_dist_muon_without_vision_has_no_vision_layouts() -> None:
    config = kimi_k3_debugmodel(seq_len=16)
    model_config = replace(cast(KimiK3Model.Config, config.model), vision_encoder=None)
    optimizer_config = _dist_muon_optimizer(
        model_config, muon_lr=1e-3, adamw_lr=1e-3, parallelism=config.parallelism
    )
    muon_config = optimizer_config.optimizers[0]
    assert isinstance(muon_config, DistMuon.Config)
    assert not any(
        name.startswith("vision_encoder.")
        for name in muon_config.compute_sharding_by_fqn
    )
    assert not any(
        bucket.name.startswith("vision_encoder.")
        for bucket in muon_config.bucket_configs
    )
