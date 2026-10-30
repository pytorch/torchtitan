# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full-scale text-only Kimi K3 MXFP8 recipes and shared optimizer policy."""

import os
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import PipelineScheduleMulti
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
from torchtitan.config import CommConfig, derive, TrainingConfig
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
from torchtitan.distributed.spmd_types import spmd_sparse_mesh
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
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.dist_moe import require_dist_moe
from torchtitan_recipes.models.dist_muon import per_expert_compute_layout
from torchtitan_recipes.overrides.kimi_k3_activation_retention import retain_kda


def dist_muon_optimizer(
    model_config: KimiK3Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Build Kimi's per-projection BF16 Newton-Schulz and AdamW policy.

    Args:
        model_config: Kimi architecture, including fused projection layouts.
        muon_lr: Learning rate for logical matrix parameters.
        adamw_lr: Learning rate for remaining parameters.
        parallelism: Storage and compute meshes selected by the recipe.

    Returns:
        The shared optimizer config used by debug and full-scale recipes.
    """
    attention = cast(KimiMLAAttention.Config, model_config.first_full_attention)
    delta_attention = next(
        layer.attention
        for layer in model_config.layers
        if isinstance(layer.attention, KDA.Config)
    )
    # FSDP flattens dp_shard and cp into one storage axis under CP.
    owned = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
            f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Owned(),
            MeshAxisName.TP.value: Owned(),
        },
        compute_dtype=torch.bfloat16,
    )

    def blocks_of(*num_rows: int) -> ComputeLayout:
        """A repeating BlockShard pattern with one Muon matrix per entry."""
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(dim=0, block_sizes=num_rows),
                f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": BlockShard(
                    dim=0, block_sizes=num_rows
                ),
                MeshAxisName.TP.value: BlockShard(dim=0, block_sizes=num_rows),
            },
            compute_dtype=torch.bfloat16,
        )

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
    per_expert = per_expert_compute_layout(parallelism)
    feed_forward_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: Shard(0),
                f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Shard(0),
                MeshAxisName.TP.value: Shard(0),
            },
            compute_dtype=torch.bfloat16,
        ),
        "w2": owned,
    }
    expert_projections = ("w13.weight", "w2.weight")

    def compute_shardings_for_layer(layer_id: int) -> dict[str, ComputeLayout]:
        layer = model_config.layers[layer_id]
        prefix = f"layers.{layer_id}"
        shardings = {}
        if isinstance(layer.attention, KimiMLAAttention.Config):
            shardings.update(
                {
                    f"{prefix}.attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in attention_shardings.items()
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (
                        delta_attention_shardings.items()
                    )
                }
            )
        if isinstance(layer.feed_forward, DenseFeedForward.Config):
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.feed_forward.routed_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.feed_forward.router.gate.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.feed_forward.shared_experts.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": owned
                    for projection in ("routed_down", "routed_up")
                }
            )
        return shardings

    num_layers = len(model_config.layers)
    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    layer_fqns = []
    for layer_id in range(num_layers):
        layer_shardings = compute_shardings_for_layer(layer_id)
        compute_sharding_by_fqn.update(layer_shardings)
        layer_fqns.append(tuple(layer_shardings))
    # Layer 0 has a much larger dense MLP, so keep it separate while amortizing
    # collective launch overhead across pairs of MoE layers.
    bucket_layer_ids = [(0,)] + [
        tuple(range(first_layer_id, min(first_layer_id + 2, num_layers)))
        for first_layer_id in range(1, num_layers, 2)
    ]
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        fqns = [fqn for layer_id in layer_ids for fqn in layer_fqns[layer_id]]
        routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is per_expert
        )
        non_routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is not per_expert
        )
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed_fqns))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed_fqns)
            )
    adamw_kwargs = {
        "lr": adamw_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
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
                foreach=True,
                fused=False,
                **adamw_kwargs,
            ),
        ],
    )


class _PhasedRoundRobinRouter(RoundRobinTokenChoiceTopKRouter):
    """Balance each EP source's partial expert cycle across destinations."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoundRobinTokenChoiceTopKRouter.Config):
        pass

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        """Return forced expert IDs while leaving learned route scores intact."""
        del expert_bias_E, router_kwargs
        expert_offset = 0
        sparse_mesh = spmd_sparse_mesh()
        if sparse_mesh is not None:
            # Routing runs on the dense mesh; EP is registered on the sparse
            # mesh and must not change the router tensor's SPMD context.
            ep_group = sparse_mesh.get_group(MeshAxisName.EP.value)
            ep_size = ep_group.size()
            if self.num_experts % ep_size:
                raise ValueError("Phased routing requires experts divisible by EP")
            # Phase each source by one contiguous expert-ownership group. This
            # balances aggregate receive rows when T * K is not divisible by E.
            expert_offset = ep_group.rank() * (self.num_experts // ep_size)
        return (
            torch.arange(
                scores_TE.shape[0] * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            )
            .add_(expert_offset)
            .reshape(scores_TE.shape[0], self.top_k)
            % self.num_experts
        )


class _BackwardReshardKimiK3Model(KimiK3Model):
    """Use the native memory policy required by the PP4/VPP2 recipe.

    This changes pipeline/FSDP lifetimes, not the Kimi architecture or forward.
    The native PyTorch extensions are documented in KIMI_K3.md.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        pass

    def pipeline(self, **kwargs):
        """Configure rebuilt Kimi stages before any warmup or graph capture."""
        schedule, model_parts, first, last = super().pipeline(**kwargs)
        if not isinstance(schedule, PipelineScheduleMulti):
            raise ValueError(
                "The PP4/VPP2 memory policy requires a multi-stage schedule"
            )
        # Kimi rebuilds the stages to carry attention residuals. Configure the
        # final stages, not the temporary ones created by the ordinary split.
        for stage in schedule._stages:
            stage.set_reshard_after_backward(True)
        for part in model_parts:
            if not isinstance(part, FSDPModule):
                raise ValueError("The PP4/VPP2 memory policy requires FSDP stages")
            part.set_reduce_scatter_on_current_stream(True)
        return schedule, model_parts, first, last


def _kimi_k3_dist_moe_mxfp8(
    *, parallelism: ParallelismConfig, muon_max_ctas: int
) -> Trainer.Config:
    """Build the shared 4K FP32-state workload for the two full-scale recipes."""
    dist_moe = require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    model = build_model_config(
        "Kimi-K3", seq_len=4096, attn_backend="varlen", parallelism=parallelism
    )
    model = replace(model, vision_encoder=None)
    # Both layouts have seven 12-layer chunks and one final 9-layer chunk.
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
        moe.router = derive(moe.router, _PhasedRoundRobinRouter.Config)
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
    optimizer = replace(
        optimizer,
        optimizers=[
            replace(
                item,
                persistent_redistribution_buffers=False,
                max_nccl_ctas_for_comms=muon_max_ctas,
            )
            if isinstance(item, DistMuon.Config)
            else item
            for item in optimizer.optimizers
        ],
    )
    rank = int(os.environ.get("RANK", "-1"))
    config = Trainer.Config(
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
        training=TrainingConfig(
            max_context_length=4096,
            num_tokens_per_microbatch_per_dp_rank=4096,
            num_tokens_per_train_step=(
                4096
                * parallelism.num_pp_microbatches
                * parallelism.data_parallel_shard_degree
            ),
            steps=12,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_accelerator_graphs=False,
            empty_device_cache_after_step=False,
        ),
        parallelism=parallelism,
        # The empty policy honors native retained boundaries. The 256-GPU
        # variant additionally saves selected shared/KDA projection regions.
        activation_remat=ActivationRematConfig(save_regions=[]),
        dist_moe_runtime=DistMoeRuntime.Config(
            activation_slot_capacity_factor=None,
            scratch_capacity_factor=1.0,
            vmm_capacity_factor=None,
        ),
        checkpointer=None,
        comm=CommConfig(init_timeout_seconds=1800, train_timeout_seconds=600),
        metrics=MetricsProcessor.Config(log_freq=1),
        profiler=Profiler.Config(
            enable_profiling=(
                0 <= rank < parallelism.pipeline_parallel_degree * 64 and rank % 64 == 0
            ),
            profile_freq=6,
            profiler_repeat=1,
            profiler_warmup=2,
            profiler_active=1,
        ),
    )
    return config


def kimi_k3_dist_moe_mxfp8_gb200_512() -> Trainer.Config:
    """Build the PP8/VPP1 4K MXFP8 throughput recipe; see KIMI_K3.md."""
    return _kimi_k3_dist_moe_mxfp8(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=64,
            expert_parallel_degree=64,
            pipeline_parallel_degree=8,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=64,
            pp_max_unsharded_active_stages=1,
            pipeline_parallel_max_outstanding_sends=8,
            fsdp_reshard_after_forward="never",
            fsdp_enable_explicit_backward_prefetch=True,
        ),
        muon_max_ctas=8,
    )


def kimi_k3_dist_moe_mxfp8_gb300_256() -> Trainer.Config:
    """Build the PP4/VPP2 4K MXFP8 projection-retention recipe; see KIMI_K3.md."""
    if not hasattr(PipelineStage, "set_reshard_after_backward") or not hasattr(
        FSDPModule, "set_reduce_scatter_on_current_stream"
    ):
        raise ValueError(
            "The 256-GPU recipe requires native pipeline backward resharding "
            "and FSDP current-stream reduction; see KIMI_K3.md"
        )
    config = _kimi_k3_dist_moe_mxfp8(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=64,
            expert_parallel_degree=64,
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=32,
            pp_max_unsharded_active_stages=2,
            pipeline_parallel_max_outstanding_sends=4,
            fsdp_reshard_after_forward="always",
            fsdp_enable_explicit_backward_prefetch=True,
            pipeline_parallel_empty_cache_before_first_backward=True,
        ),
        muon_max_ctas=1,
    )
    model = cast(KimiK3Model.Config, config.model)
    # Retain one KDA core per physical rank, in its second 12-layer chunk.
    late_kda_fqns = []
    for first in range(48, len(model.layers), 12):
        last_kda = max(
            layer_id
            for layer_id in range(first, min(first + 12, len(model.layers)))
            if isinstance(model.layers[layer_id].attention, KDA.Config)
        )
        late_kda_fqns.append(f"layers.{last_kda}")
    policy = retain_kda(
        cast(ActivationRematConfig, config.activation_remat),
        block_fqns=late_kda_fqns,
    )
    projection_regions = [
        "attention.q_proj.linear",
        "attention.k_proj.linear",
        "attention.v_proj.linear",
        "attention.output_gate.linear",
    ]
    return replace(
        config,
        model=derive(model, _BackwardReshardKimiK3Model.Config),
        activation_remat=replace(
            policy,
            save_regions=[
                "feed_forward.shared_experts.w13.linear",
                "feed_forward.shared_experts.w2.linear",
            ],
            block_save_regions={
                fqn: [*regions, *projection_regions]
                for fqn, regions in policy.block_save_regions.items()
            },
        ),
    )
