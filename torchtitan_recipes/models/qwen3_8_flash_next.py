# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next training recipes and their report-based optimizer configuration."""

from dataclasses import dataclass

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
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAAttention, QSAIndexerLoss
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
            # and a/b scalars are separate AdamW parameters. With equal head
            # dims a uniform block size names the same matrices, and DistMuon
            # batches them into one Newton-Schulz call instead of one call per
            # pattern position (batch 1 when a rank owns one key group).
            if attention.key_head_dim != attention.value_head_dim:
                raise ValueError(
                    "The Flash-Next DistMuon policy expects equal GDN key and "
                    f"value head dims, got {attention.key_head_dim} and "
                    f"{attention.value_head_dim}"
                )
            attention_shardings = {
                "in_proj.projection": blocks_of(attention.key_head_dim),
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


def qwen38_flash_next_125b_dist_moe() -> Trainer.Config:
    """Fine-tune the released 125B-A6B decoder from its HF checkpoint on 16 GB300s.

    FSDP16 + EP16 places all 512 experts, 32 per rank, in one NVLink domain,
    and Dist-MoE dispatches over it. The PLE table is row-sharded over the
    same 16 DP ranks. Each step trains on 2M tokens: 8 packed 16K-token
    microbatches per rank, with documents split at the model's 4K context
    (the per-GPU work of a 33.5M-token step on 256 GPUs), under full remat,
    the model's local_compile regions and one accelerator graph for
    forward/backward. The decoder loads from the multimodal checkpoint; the
    vision encoder is not built. The 30 steps at a small learning rate
    measure throughput; set ``training.steps`` and the schedule for a real run.
    """
    seq_len = 4096
    num_tokens_per_microbatch = 4 * seq_len
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=16, expert_parallel_degree=16
    )
    model_config = build_model_config(
        "125B-A6B-text",
        seq_len=seq_len,
        parallelism=parallelism,
        # The reference gives no KL coefficient. The KL only updates the
        # indexer (its input is detached), whose DistMuon and AdamW updates do
        # not depend on the loss scale; the coefficient matters only through
        # global gradient-norm clipping.
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    config = FlashNextTrainerConfig(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        ),
        hf_assets_path="./assets/hf/Qwen3.8-Flash-Next",
        dataloader=GrainDataLoader.Config(
            # Graph replay needs a fixed document capacity. 64 is twice
            # c4_document_capacity's estimate for 16K tokens; a row that reaches
            # 64 documents is closed early and padded to 16K.
            max_num_documents=64,
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
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            num_tokens_per_train_step=8
            * num_tokens_per_microbatch
            * parallelism.data_parallel_shard_degree,
            steps=30,
            # Enabled below, once Dist-MoE replaces the blocking all-to-all
            # dispatcher that graph validation rejects.
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        # Loads the HF safetensors from hf_assets_path.
        checkpointer=CheckpointManager.Config(initial_load_in_hf=True),
        activation_remat=FullActivationRematConfig(),
        # Dist-MoE recomputes expert intermediates inside its own kernels; full
        # remat does not cover them. A 0.5 activation slot (~45 GiB, a fixed
        # allocation independent of routing) holds the full intermediates of
        # about the first half of the MoE layers in forward order (decided per
        # layer on device): +3% throughput at 243 GiB peak. Factor 1.0
        # (~94 GiB) reached 264 GiB and thrashed the allocator. Scratch covers
        # routing of up to 4x the balanced token count to one rank.
        dist_moe_runtime=DistMoeRuntime.Config(
            activation_slot_capacity_factor=0.5, scratch_capacity_factor=4.0
        ),
    )
    config.training.disable_accelerator_graphs = False
    # Triton hc_norm and one [down | inject] GEMM in every gated residual read;
    # the QSA indexer teacher in one pass from gather_attn's log-sum-exp.
    config.override.imports = [
        "torchtitan_recipes.overrides.flash_next_gated_residual.fused_gated_residual",
        "torchtitan_recipes.overrides.flash_next_qsa_teacher.fused_qsa_teacher",
    ]
    return apply_transforms(config, [DistMoeTransform()])
