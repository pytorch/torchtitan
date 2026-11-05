# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next training recipes and their report-based optimizer configuration."""

from dataclasses import dataclass, replace

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    Adam,
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, TokenDispatcherTransform
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAAttention, QSAIndexerLoss
from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextModel
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


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
    if parallelism.tensor_parallel_degree > 1:
        # TODO: remove once DistMuon supports TP storage layouts:
        # https://github.com/pytorch/torchtitan/pull/5678
        raise ValueError(
            "The Flash-Next DistMuon policy does not support tensor parallelism yet"
        )
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


def qwen38_flash_next_125b_hybridep() -> Trainer.Config:
    """Train the released 125B-A6B from its HF checkpoint on 16 GB300s.

    FSDP16 + EP16 places all 512 experts, 32 per rank, in one NVLink domain,
    and HybridEP dispatches over it. The PLE table is row-sharded over the
    same 16 DP ranks. Forward/backward replays as one accelerator graph, which
    needs non-blocking HybridEP; capacity factor 1.0 sizes its buffers for
    every token reaching every local expert, so no routed token is dropped.
    Full remat runs at 4K tokens per rank with the model's local_compile
    regions. The decoder loads from the multimodal checkpoint; the vision
    encoder is not built.
    """
    seq_len = 4096
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=16, expert_parallel_degree=16
    )
    # Text-only training drops the vision encoder: its dummy forward on text
    # batches copies grid sizes to the host, which graph capture rejects.
    model_config = replace(
        build_model_config(
            "125B-A6B",
            seq_len=seq_len,
            parallelism=parallelism,
            indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
        ),
        vision_encoder=None,
    )
    config = FlashNextTrainerConfig(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        ),
        hf_assets_path="./assets/hf/Qwen3.8-Flash-Next",
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(seq_len),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=make_dist_muon_optimizer(
                model_config, parallelism=parallelism, lr=1e-5
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=5,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            steps=30,
            # Enabled below, once HybridEP replaces the default dispatcher.
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        # Loads the HF safetensors from hf_assets_path.
        checkpointer=CheckpointManager.Config(
            initial_load_model_only=True,
            initial_load_in_hf=True,
        ),
        activation_remat=FullActivationRematConfig(),
    )
    # apply_transforms re-validates graphs against the HybridEP dispatcher.
    config.training.disable_accelerator_graphs = False
    return apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                num_max_tokens_per_rank=seq_len,
                kwargs={"non_blocking_capacity_factor": 1.0},
            )
        ],
    )
