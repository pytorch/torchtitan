# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next FSDP, EP and multimodal feature fixtures."""

from dataclasses import dataclass, fields, replace

import torch
from torch.distributed.tensor import Shard

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
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAAttention, QSAIndexerLoss
from torchtitan.models.qwen3_8_flash_next.model import Qwen38FlashNextModel
from torchtitan.observability.metrics import MetricsProcessor, TensorBoardLogger
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


def _dist_muon_optimizer(
    model_config: Qwen38FlashNextModel.Config,
    *,
    parallelism: ParallelismConfig,
    lr: float,
) -> OptimizersContainer.Config:
    """Apply the report's section 3.1 policy to a freshly built debug config."""
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
class _FlashNextTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        optimizers = []
        for optimizer in self.optim.optimizer.optimizers:
            if isinstance(optimizer, DistMuon.Config):
                # CLI overrides can change EP after the recipe is constructed.
                optimizer = replace(
                    optimizer,
                    compute_sharding_by_fqn={
                        fqn: _per_expert_compute_layout(self.parallelism)
                        if ".routed_experts." in fqn
                        else layout
                        for fqn, layout in optimizer.compute_sharding_by_fqn.items()
                    },
                )
                # TODO: Support TP's _StridedShard expert storage in DistMuon.
                # https://github.com/pytorch/torchtitan/issues/3353
                if self.parallelism.tensor_parallel_degree > 1:
                    raise ValueError(
                        "Flash-Next DistMuon requires tensor_parallel_degree=1"
                    )
            optimizers.append(optimizer)
        self.optim = replace(
            self.optim, optimizer=replace(self.optim.optimizer, optimizers=optimizers)
        )


def qwen38_flash_next_debugmodel(
    *,
    seq_len: int = 64,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    parallelism = parallelism or ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    model_config.local_compile = LocalCompileConfig(regions=[])
    optimizer = _dist_muon_optimizer(model_config, parallelism=parallelism, lr=8e-4)
    return _FlashNextTrainerConfig(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(8 * seq_len),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        optim=Optim.Config(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1, loggers=[TensorBoardLogger.Config()]
        ),
        parallelism=parallelism,
        activation_remat=None,
        debug=DebugConfig(seed=42, deterministic=True),
    )


def qwen38_flash_next_debugmodel_mm() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3_8 import qwen38_debugmodel

    config = qwen38_debugmodel(seq_len=1024)
    model_config = build_model_config(
        "debugmodel_mm",
        seq_len=1024,
        parallelism=config.parallelism,
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    config.model = model_config
    config.optim.optimizer = _dist_muon_optimizer(
        model_config, parallelism=config.parallelism, lr=5e-3
    )
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.model.vocab_size),
    )
    config.activation_remat = None
    config.training.disable_accelerator_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = [TensorBoardLogger.Config()]
    return _FlashNextTrainerConfig(
        **{
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.init
        }
    )
