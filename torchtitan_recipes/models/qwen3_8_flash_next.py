# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Report-based Flash-Next optimizer configuration shared by training recipes."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optim import Adam, AdamW, DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.qwen3_8_flash_next.attention import QSAAttention
from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextModel
from torchtitan.trainer import Trainer


# The report specifies eight Polar Express iterations; coefficients are from
# https://github.com/liangyuwang/FSDP-Canzona/blob/426e8c50363cc3401e64eeb80c9e6e2451ea2ed4/matrix_based_optimizer/optimizers/muon.py
_POLAR_EXPRESS_COEFFICIENTS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    if parallelism.expert_parallel_degree == 1:
        return ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
            compute_dtype=torch.bfloat16,
        )
    # EP owns an expert range first; edp_shard splits that local range, which
    # reverses the storage mesh's physical axis order.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value)
        },
        compute_dtype=torch.bfloat16,
    )


def make_dist_muon_optimizer(
    model_config: Qwen38FlashNextModel.Config,
    *,
    parallelism: ParallelismConfig,
    lr: float,
) -> OptimizersContainer.Config:
    """Build the report's decoder optimizer assignment for this parallelism."""
    owned = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
        compute_dtype=torch.bfloat16,
    )

    def blocks_of(*num_rows: int) -> ComputeLayout:
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(dim=0, block_sizes=num_rows)
            },
            compute_dtype=torch.bfloat16,
        )

    per_expert = _per_expert_compute_layout(parallelism)
    # Native [2, I, D] and [E, 2, I, D] weights already separate fc1's gate/up
    # matrices. DistMuon orthogonalizes each pair of trailing matrix dimensions.
    shared_gate_up = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        compute_dtype=torch.bfloat16,
    )
    compute_sharding_by_fqn = {}
    buckets = []
    for layer_id, layer in enumerate(model_config.layers):
        prefix = f"layers.{layer_id}"
        attention = layer.attention
        if isinstance(attention, GDN.Config):
            # Per key group: q, k, and one v matrix per value head. The z gate
            # and a/b scalars are separate AdamW parameters.
            attention_shardings = {
                "in_proj.projection": blocks_of(
                    attention.key_head_dim,
                    attention.key_head_dim,
                    *(
                        [attention.value_head_dim]
                        * (attention.n_heads // attention.n_key_heads)
                    ),
                ),
                "out_proj": owned,
            }
        else:
            assert isinstance(attention, QSAAttention.Config)
            attention_shardings = {
                "wq.projection": blocks_of(attention.head_dim),
                "wk": blocks_of(attention.head_dim),
                "wv": blocks_of(attention.head_dim),
                "wo": owned,
                "indexer.index_qk_proj": blocks_of(attention.indexer.head_dim),
            }
        dense_shardings = {
            f"{prefix}.attention.{field}.weight": layout
            for field, layout in attention_shardings.items()
        }
        dense_shardings.update(
            {
                f"{prefix}.feed_forward.shared_experts.w13.weight": shared_gate_up,
                f"{prefix}.feed_forward.shared_experts.w2.weight": owned,
            }
        )
        if layer.ple is not None:
            dense_shardings.update(
                {
                    f"{prefix}.ple.{field}.weight": owned
                    for field in ("key_proj", "value_proj")
                }
            )
        routed_shardings = {
            f"{prefix}.feed_forward.routed_experts.{field}.weight": per_expert
            for field in ("w13", "w2")
        }
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        # Dense and routed weights use different storage meshes under EP.
        buckets.extend(
            (
                BucketConfig(name=prefix, patterns=tuple(dense_shardings)),
                BucketConfig(
                    name=f"{prefix}.routed_experts", patterns=tuple(routed_shardings)
                ),
            )
        )
    return OptimizersContainer.Config(
        optimizers=[
            Adam.Config(
                pattern=r"\.ngram_embedding\.weight$",
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=0.0,
            ),
            DistMuon.Config(
                pattern=(
                    r"^layers\.\d+\.(?:attention\.(?:in_proj\.projection|out_proj|"
                    r"wq\.projection|wk|wv|wo|indexer\.index_qk_proj)|"
                    r"feed_forward\.(?:routed_experts|shared_experts)\.w(?:13|2)|"
                    r"ple\.(?:key_proj|value_proj))\.weight$"
                ),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                bucket_configs=tuple(buckets),
                lr=lr,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=_POLAR_EXPRESS_COEFFICIENTS,
                ns_steps=8,
                eps=1e-14,
                adjust_lr_fn="match_rms_adamw",
                ns_device="accelerator",
            ),
            # Embeddings, LM head, GR, router, output gates, decay/beta,
            # norms, convolutions and the vision encoder stay on AdamW.
            AdamW.Config(pattern=r".*", lr=lr, betas=(0.9, 0.95)),
        ],
    )


@dataclass(kw_only=True, slots=True)
class FlashNextTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        expected = _per_expert_compute_layout(self.parallelism)
        for optimizer in self.optim.optimizer.optimizers:
            if isinstance(optimizer, DistMuon.Config):
                for fqn, layout in optimizer.compute_sharding_by_fqn.items():
                    if ".routed_experts." in fqn and layout != expected:
                        raise ValueError(
                            f"DistMuon layout for {fqn} disagrees with expert parallelism; "
                            "build the optimizer and model with the same parallelism"
                        )
