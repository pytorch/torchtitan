# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-scale text-only Kimi K3 MXFP8 recipe and shared optimizer policy."""

import os
from dataclasses import replace
from typing import cast

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import CommConfig, derive, PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import transform_model_config_
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.moe import MoE, RoundRobinTokenChoiceTopKRouter
from torchtitan.models.kimi_k3 import (
    build_model_config,
    KIMI_K3_SPECIAL_TOKENS,
    KimiK3Model,
)
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import KinetoTraceProfiler, Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.dist_moe import require_dist_moe


def _on_dp_axes(placement) -> ComputeLayout:
    """A BF16 compute layout with ``placement`` on FSDP's storage axis.

    FSDP flattens dp_shard and cp into one storage axis under CP, so both
    names map to the same placement.
    """
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: placement,
            f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": placement,
        },
        compute_dtype=torch.bfloat16,
    )


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return _on_dp_axes(Shard(0))

    # Preserve exact EP-first DTensor ownership. If an EP-local expert count is
    # smaller than the edp_shard size, add balanced rank assignment only after
    # benchmarks show that the fixed nonempty edp_shard coordinates are a hotspot.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        # EP splits the expert dimension first, then edp_shard repartitions each
        # EP-local expert domain, which reverses the storage-mesh axis order.
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
        compute_dtype=torch.bfloat16,
    )


def dist_muon_optimizer(
    model_config: KimiK3Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
    extra_full_attention_block_sizes: dict[str, tuple[int, ...]] | None = None,
) -> OptimizersContainer.Config:
    """Build Kimi's per-projection BF16 Newton-Schulz and AdamW policy.

    Args:
        model_config: Kimi architecture, including fused projection layouts.
        muon_lr: Learning rate for logical matrix parameters.
        adamw_lr: Learning rate for remaining parameters.
        parallelism: Storage and compute meshes selected by the recipe.
        extra_full_attention_block_sizes: Additional full-attention projections
            and their Newton-Schulz block sizes; an empty tuple means the whole
            matrix.

    Returns:
        DistMuon for matrix parameters and AdamW for the rest.
    """
    attention = cast(KimiMLAAttention.Config, model_config.first_full_attention)
    delta_attention = next(
        layer.attention
        for layer in model_config.layers
        if isinstance(layer.attention, KDA.Config)
    )
    owned = _on_dp_axes(Owned())

    def blocks_of(*num_rows: int) -> ComputeLayout:
        """A repeating BlockShard pattern with one Muon matrix per entry."""
        return _on_dp_axes(BlockShard(dim=0, block_sizes=num_rows))

    # MLA fuses several projections into one parameter; Kimi runs Newton-Schulz
    # per logical projection, so each fused block is split (see #4692).
    attention_shardings = {
        "wq_a": owned,
        # per head: [q_nope_h; q_rope_h]
        "wq_b": blocks_of(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
        # one block: [kv_latent; k_rope]
        "wkv_a": blocks_of(attention.kv_lora_rank, attention.qk_rope_head_dim),
        # per head: [k_nope_h; v_h]
        "inner_attention.wkv_b": blocks_of(
            attention.qk_nope_head_dim, attention.v_head_dim
        ),
        "gate": blocks_of(attention.v_head_dim),
        "wo": owned,
    }
    attention_shardings |= {
        projection: blocks_of(*block_sizes) if block_sizes else owned
        for projection, block_sizes in (extra_full_attention_block_sizes or {}).items()
    }
    per_kda_head = blocks_of(delta_attention.head_dim)
    # forget_a is the shared low-rank down projection and output_proj mixes
    # heads back into the model dimension, so both are whole-matrix compute.
    delta_attention_shardings = {
        "q_proj": per_kda_head,
        "k_proj": per_kda_head,
        "v_proj": per_kda_head,
        "forget_a": owned,
        "forget_b": per_kda_head,
        "output_gate": per_kda_head,
        "output_proj": owned,
    }
    feed_forward_shardings = {"w13": _on_dp_axes(Shard(0)), "w2": owned}
    per_expert = _per_expert_compute_layout(parallelism)
    # Suffixes under each MoE layer's feed_forward, in DistMuon bucket order.
    moe_shardings = {
        "routed_experts.w13": per_expert,
        "routed_experts.w2": per_expert,
        "router.gate": owned,
        **{
            f"shared_experts.{name}": layout
            for name, layout in feed_forward_shardings.items()
        },
        "routed_down": owned,
        "routed_up": owned,
    }

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    routed_fqns_per_layer, non_routed_fqns_per_layer = [], []
    for layer_id, layer in enumerate(model_config.layers):
        attention_table = (
            attention_shardings
            if isinstance(layer.attention, KimiMLAAttention.Config)
            else delta_attention_shardings
        )
        feed_forward_table = (
            feed_forward_shardings
            if isinstance(layer.feed_forward, DenseFeedForward.Config)
            else moe_shardings
        )
        layer_shardings = {
            f"layers.{layer_id}.{module}.{name}.weight": layout
            for module, table in (
                ("attention", attention_table),
                ("feed_forward", feed_forward_table),
            )
            for name, layout in table.items()
        }
        compute_sharding_by_fqn |= layer_shardings
        routed_fqns_per_layer.append(
            [fqn for fqn, layout in layer_shardings.items() if layout is per_expert]
        )
        non_routed_fqns_per_layer.append(
            [fqn for fqn, layout in layer_shardings.items() if layout is not per_expert]
        )

    # Layer 0 has a much larger dense MLP, so keep it separate while amortizing
    # collective launch overhead across pairs of MoE layers.
    num_layers = len(model_config.layers)
    bucket_layer_ids = [[0]] + [
        list(range(first, min(first + 2, num_layers)))
        for first in range(1, num_layers, 2)
    ]
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        non_routed = tuple(f for i in layer_ids for f in non_routed_fqns_per_layer[i])
        routed = tuple(f for i in layer_ids for f in routed_fqns_per_layer[i])
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed))
        if routed:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed)
            )

    # Muon is designed for matrix parameters; AdamW handles non-matrix
    # parameters (RMSNorm, embeddings, LM head, biases, A_log, dt_bias, the
    # depthwise convolutions) and the degenerate matrices: beta's rows are
    # per-head scalar matrices and the residual projections are
    # single-row [1, D], so Newton-Schulz would only rescale them.
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)}|"
        rf"{'|'.join(delta_attention_shardings)})\.weight|"
        r"routed_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.w(?:13|2)\.weight|"
        r"feed_forward\.router\.gate\.weight|"
        r"feed_forward\.shared_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.routed_(?:down|up)\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=muon_pattern,
                bucket_configs=tuple(bucket_configs),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                lr=muon_lr,
                weight_decay=0.1,
                adjust_lr_fn="match_rms_adamw",
                ns_device="accelerator",
                # Start with 8 CTAs to reduce all-to-all memory without limiting
                # FSDP. At 64 ranks and ~12 MiB per peer/channel, the 32-to-8
                # channel connection-state estimate falls from ~23.6 to ~5.9 GiB
                # per rank; revisit the cap if redistribution is bandwidth-bound.
                max_nccl_ctas_for_comms=8,
            ),
            # The remaining parameters are embeddings, norms, biases, the LM
            # head, the KDA scalars/convolutions, and the vision tower.
            AdamW.Config(
                pattern=r".*",
                lr=adamw_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                foreach=True,
                fused=False,
            ),
        ],
    )


def kimi_k3_dist_moe_mxfp8_gb200_512() -> Trainer.Config:
    """Build the PP8/VPP1 4K MXFP8 throughput recipe; see torchtitan/models/kimi_k3/README.md."""
    dist_moe = require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=64,
        expert_parallel_degree=64,
        pipeline_parallel_degree=8,
        pipeline_parallel_schedule="Interleaved1F1B",
        num_pp_microbatches=64,
        pipeline_parallel_max_outstanding_sends=8,
        fsdp_reshard_after_forward="never",
    )
    model = build_model_config(
        "Kimi-K3", seq_len=4096, attn_backend="varlen", parallelism=parallelism
    )
    # Seven 12-layer chunks and one final 9-layer chunk, one per PP rank.
    # Explicit parts avoid the interleaved schedule's automatic VPP2 split on PP8.
    parts = [
        [
            f"layers.{layer_id}"
            for layer_id in range(first, min(first + 12, len(model.layers)))
        ]
        for first in range(0, len(model.layers), 12)
    ]
    parts[0].insert(0, "tok_embeddings")
    parts[-1].extend(["norm", "lm_head", "output_res_proj", "output_res_norm"])
    parallelism = replace(
        parallelism, pipeline_parallel_module_fqns_per_model_part=parts
    )
    for _, moe, _, _ in model.traverse(MoE.Config):
        moe.router = derive(moe.router, RoundRobinTokenChoiceTopKRouter.Config)
    # Trainer.Config validates graph-safe dispatch at construction time.
    # Select the backend before constructing that config, not afterwards.
    model = cast(
        KimiK3Model.Config,
        transform_model_config_(
            model,
            [
                DistMoeTransform(
                    expert_precision="mxfp8",
                    block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
                )
            ],
        ),
    )

    optimizer = dist_muon_optimizer(
        model, muon_lr=8e-4, adamw_lr=8e-4, parallelism=parallelism
    )
    rank = int(os.environ.get("RANK", "-1"))
    return Trainer.Config(
        model=model,
        hf_assets_path="./assets/hf/Kimi-K3",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
            max_num_documents=32,
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model))
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
        precision=PrecisionConfig(
            dtype="float32",
            param_dtype="bfloat16",
            reduce_dtype="float32",
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4096,
            steps=12,
            disable_accelerator_graphs=False,
        ),
        parallelism=parallelism,
        # The empty policy honors native retained boundaries.
        activation_remat=ActivationRematConfig(save_regions=[]),
        dist_moe_runtime=DistMoeRuntime.Config(),
        checkpointer=None,
        comm=CommConfig(init_timeout_seconds=1800, train_timeout_seconds=600),
        metrics=MetricsProcessor.Config(log_freq=1),
        # Trace one rank per PP stage: PP is the outermost mesh axis, so each
        # stage starts at a multiple of the dp_shard degree.
        profiler=Profiler.Config(
            trace=(
                KinetoTraceProfiler.Config(freq=6, warmup=2, active=1, repeat=1)
                if rank >= 0 and rank % parallelism.data_parallel_shard_degree == 0
                else None
            ),
        ),
    )
