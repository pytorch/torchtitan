# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, ParamGroupConfig
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.flex_shard.muon import (
    BenchInstrumentedOptimizers,
    FlexShardGatherMuonOptimizers,
    FlexShardMuonOptimizers,
    FSDP2MuonOptimizers,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_debugmodel,
)
from torchtitan.trainer import Trainer

from . import model_registry
from .parallelize import make_gather_muon_parallelize_fn, parallelize_deepseekv3_muon


# =====================================================================================
# Plain FlexShard debug configs (no Muon) -- the existing FlexShard surface. These keep
# the model registry's default parallelize_fn (``parallelize_deepseekv3``); the Muon
# configs below opt in by setting ``model_spec.parallelize_fn`` explicitly.
# =====================================================================================
def flex_shard_deepseek_v3_debugmodel() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel")
    return config


def flex_shard_deepseek_v3_debugmodel_dp8_ep4() -> Trainer.Config:
    config = flex_shard_deepseek_v3_debugmodel()
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=4,
    )
    return config


def flex_shard_deepseek_v3_debugmodel_dp8_ep4_ce_loss() -> Trainer.Config:
    """DP8/EP4 debug model with standard (non-chunked) CrossEntropyLoss."""
    config = flex_shard_deepseek_v3_debugmodel_dp8_ep4()
    config.loss = CrossEntropyLoss.Config()
    return config


def _muon_optimizer(
    *, muon_lr: float = 0.02, adamw_lr: float = 8e-4
) -> FlexShardMuonOptimizers.Config:
    """Full Muon: owned 2D dense -> Muon, 3D MoE experts -> GroupedMuon / GatherGroupedMuon,
    everything else (norms, embeddings, LM head) -> AdamW.

    Routing is by FlexShard placement (see :class:`FlexShardMuonOptimizers`), so the
    ``pattern`` fields are placeholders. On a dense model (no 3D experts) this reduces to
    dense Muon + AdamW. The ``_grouped_muon_optimizer`` below is the same recipe plus the
    optional MuonClip QK-clip (``qk_clip_tau``).
    """
    return FlexShardMuonOptimizers.Config(
        param_groups=[
            ParamGroupConfig(
                pattern="<owned 2D matrices>",
                optimizer_name="Muon",
                optimizer_kwargs={
                    "lr": muon_lr,
                    "weight_decay": 0.1,
                    "momentum": 0.95,
                    "nesterov": True,
                },
            ),
            ParamGroupConfig(
                pattern="<rest>",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": adamw_lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        implementation="foreach",
    )


def flex_shard_deepseek_v3_debugmodel_muon() -> Trainer.Config:
    """DeepSeek V3 debug model trained with FlexShard + communication-efficient Muon.

    Keeps the core debug parallelism (``dp_shard`` inferred from world size,
    ``dp_replicate=1``, ``expert_parallel_degree=1``). With ``ep=1`` FlexShard
    shards the experts itself on the dp mesh (gathered for compute -- not true EP;
    fine for the debug model). See ``..._ep2`` for real expert parallelism.

    **Run with NGPU <= 6** (the debug model has 6 layers; comm-efficient ``Owned``
    requires ``dp_shard <= num_layers`` so every dp rank owns a layer). No HSDP /
    TP / PP.
    """
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry("debugmodel")
    # Opt in to comm-efficient Muon: the registry default is plain FlexShard
    # (parallelize_deepseekv3), so set the Muon parallelizer explicitly here.
    config.model_spec.parallelize_fn = parallelize_deepseekv3_muon
    config.optimizer = _muon_optimizer()
    # Activation checkpointing is disabled: AC recompute does not yet compose with
    # FlexShard's Owned broadcast + FlexAttention's compiled HOP (the reshard-after-
    # forward recompute path errors on the compiled op). Re-enabling AC is future work.
    config.activation_checkpoint = None
    # Leave parallelism at the core debug default: dp_shard=-1 (inferred = NGPU),
    # dp_replicate=1, expert_parallel_degree=1.
    return config


def flex_shard_deepseek_v3_debugmodel_muon_ep2() -> Trainer.Config:
    """Real expert parallelism: experts permanently partitioned across ``ep=2``.

    The experts keep their EP partition (DTensor), are unwrapped to each rank's
    local EP shard, and FlexShard then FSDP-shards that within the EP group; dense
    weights run comm-efficient Muon on the dp mesh. ``ep`` must divide ``dp_shard`` and
    ``dp_shard <= num_layers (6)``, so e.g. **run with NGPU=4** (-> dp_shard=4,
    ep=2, efsdp=2).
    """
    config = flex_shard_deepseek_v3_debugmodel_muon()
    config.parallelism = ParallelismConfig(expert_parallel_degree=2)
    return config


def flex_shard_deepseek_v3_debugmodel_adamw() -> Trainer.Config:
    """AdamW baseline for comparison against the comm-efficient Muon config.

    Identical model / FlexShard sharding / AC-off / parallelism to
    ``flex_shard_deepseek_v3_debugmodel_muon`` -- only the optimizer differs: every
    parameter is optimized by AdamW (``default_adamw``) instead of routing 2D
    matrices to Muon. AdamW on ``Owned`` params is still comm-efficient (the owner holds
    the full param + averaged grad; empty shards on other ranks are no-ops), and is
    numerically standard AdamW, so it is a clean optimizer-only baseline.
    """
    config = flex_shard_deepseek_v3_debugmodel_muon()
    config.optimizer = default_adamw(lr=8e-4)
    return config


def _gather_muon_optimizer(
    *, muon_lr: float = 0.02, adamw_lr: float = 8e-4
) -> FlexShardGatherMuonOptimizers.Config:
    """Full Muon, gather dense distribution: sharded 2D dense -> GatherMuon, 3D experts ->
    GroupedMuon / GatherGroupedMuon, everything else -> AdamW."""
    return FlexShardGatherMuonOptimizers.Config(
        param_groups=[
            ParamGroupConfig(
                pattern="<sharded 2D matrices>",
                optimizer_name="Muon",
                optimizer_kwargs={
                    "lr": muon_lr,
                    "weight_decay": 0.1,
                    "momentum": 0.95,
                    "nesterov": True,
                },
            ),
            ParamGroupConfig(
                pattern="<rest>",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": adamw_lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        implementation="foreach",
    )


def flex_shard_deepseek_v3_debugmodel_gather_muon_shard() -> Trainer.Config:
    """Gather-for-NS Muon baseline: dense 2D matrices use plain FSDP ``Shard(0)``.

    Same model / AC-off as the comm-efficient Muon config, but dense matrices are
    row-sharded and the optimizer all-gathers each matrix before Newton-Schulz (1
    all-gather/bucket + redundant NS). Unlike comm-efficient ``Owned`` there is no
    ``num_layers >= dp_shard`` limit (sharding is within-tensor). Compare against
    ``flex_shard_deepseek_v3_debugmodel_muon`` to measure comm-efficient vs gather.
    """
    config = flex_shard_deepseek_v3_debugmodel_muon()
    config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("shard")
    config.optimizer = _gather_muon_optimizer()
    return config


def flex_shard_deepseek_v3_debugmodel_gather_muon_grouped() -> Trainer.Config:
    """Gather-for-NS Muon baseline: dense 2D matrices use byte-perfect ``GroupedRaggedShard``.

    Like ``..._gather_muon_shard`` but each layer's 2D matrices are packed into one
    byte-balanced ``GroupedRaggedShard`` bucket (cut crosses matrix boundaries), so
    every rank holds exactly ``1/world_size`` of the layer's matrices.
    """
    config = flex_shard_deepseek_v3_debugmodel_gather_muon_shard()
    config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("grouped")
    return config


# =====================================================================================
# DeepSeek V3 16B: owned + comm-efficient Muon, whole-layer placement, single-node
# (8x95GB). dsv3-16B is 27 layers and runs on <= 27 ranks single-node (world_size <=
# num_layers), so each rank owns whole layers (one Owned bucket per layer). Full Muon recipe:
# dense 2D -> Muon, experts -> GroupedMuon, embeddings/lm_head/norms -> AdamW. Two EP regimes:
# ep=1 ("flex_shard 1d") and ep=8 (+EP). (Finer per-matrix / two-level placements only differ
# from whole-layer when world_size > num_layers, which this single-node regime is not.)
# =====================================================================================
def _grouped_muon_optimizer(
    *, muon_lr: float = 0.005, adamw_lr: float = 2e-4, qk_clip_tau: float | None = None
) -> FlexShardMuonOptimizers.Config:
    """Full Muon recipe: owned 2D dense -> Muon, 3D experts -> GroupedMuon, rest -> AdamW.

    ``qk_clip_tau`` (optional) enables MuonClip's QK-clip (Kimi K2) after each step.
    """
    base = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    return FlexShardMuonOptimizers.Config(
        param_groups=base.param_groups,
        implementation=base.implementation,
        qk_clip_tau=qk_clip_tau,
    )


def _flex_shard_dsv3_16b(
    *,
    expert_parallel_degree: int = 8,
    muon_lr: float = 0.005,
    adamw_lr: float = 2e-4,
    qk_clip_tau: float | None = None,
) -> Trainer.Config:
    """Build a DeepSeek V3 16B FlexShard + comm-efficient Muon config (whole-layer Owned).

    Full Muon recipe: dense 2D -> Muon, 3D experts -> GroupedMuon, rest -> AdamW, with
    optional ``qk_clip_tau`` for MuonClip. Whole-layer Owned placement: one Owned bucket per
    layer (27 layers >= the single-node world_size, so each rank owns whole layers).
    Test-friendly: bs=1 / seq=2048, c4_test + test tokenizer, AC off, eager (no compile), no PP.
    """
    config = deepseek_v3_16b()
    config.model_spec = model_registry("16B")
    config.model_spec.parallelize_fn = parallelize_deepseekv3_muon
    # eager + AC off (FlexShard Owned is eager-only; AC/compile do not compose yet).
    config.activation_checkpoint = None
    config.compile = CompileConfig(enable=False)
    # 1D FSDP (+ optional EP); strip the 16B default's PP schedule.
    config.parallelism = ParallelismConfig(
        expert_parallel_degree=expert_parallel_degree
    )
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.training = TrainingConfig(local_batch_size=1, seq_len=2048, steps=10)

    config.optimizer = _grouped_muon_optimizer(
        muon_lr=muon_lr, adamw_lr=adamw_lr, qk_clip_tau=qk_clip_tau
    )
    return config


# Full Muon (experts -> GroupedMuon), whole-layer Owned. ep=8 (+EP) and ep=1 (flex_shard 1d).
def flex_shard_deepseek_v3_16b_muon_grouped() -> Trainer.Config:
    return _flex_shard_dsv3_16b(expert_parallel_degree=8)


# Full MuonClip recipe: full Muon (grouped experts) + QK-clip (Kimi K2, tau=100).
def flex_shard_deepseek_v3_16b_muonclip() -> Trainer.Config:
    return _flex_shard_dsv3_16b(expert_parallel_degree=8, qk_clip_tau=100.0)


def flex_shard_deepseek_v3_16b_muon_grouped_1d() -> Trainer.Config:
    return _flex_shard_dsv3_16b(expert_parallel_degree=1)


# =====================================================================================
# DeepSeek V3 16B: MuonClip across the 4 distribution strategies. Full Muon + QK-clip in 3
# dense distributions -- Owned (comm-efficient), gather_shard (Shard(0)), gather_grouped
# (GroupedRaggedShard) -- plus fsdp2 + AdamW as the production reference. All ep=8, AC off,
# eager, c4_test; sweep batch via --training.local_batch_size.
# =====================================================================================
def _dsv3_16b_test_base(*, expert_parallel_degree: int = 8) -> Trainer.Config:
    """core deepseek_v3_16b + shared test settings (AC off, eager, EP, c4_test).

    Keeps the CORE ``model_spec`` (``fully_shard`` parallelize); the FlexShard configs
    (Owned / gather) override ``model_spec`` with the flex_shard wrapper themselves.
    """
    config = deepseek_v3_16b()
    config.activation_checkpoint = None
    config.compile = CompileConfig(enable=False)
    config.parallelism = ParallelismConfig(
        expert_parallel_degree=expert_parallel_degree
    )
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.training = TrainingConfig(local_batch_size=1, seq_len=2048, steps=10)
    return config


def _gather_grouped_muon_optimizer(
    *, muon_lr: float = 0.005, adamw_lr: float = 2e-4, qk_clip_tau: float | None = None
) -> FlexShardGatherMuonOptimizers.Config:
    """Gather full Muon: GatherMuon (sharded 2D dense) + GroupedMuon (experts) + AdamW; +QK-clip.

    Same container as the plain gather optimizer (gather is now always full Muon -- experts
    -> GroupedMuon / GatherGroupedMuon); this builder just adds the MuonClip ``qk_clip_tau``.
    """
    base = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    return FlexShardGatherMuonOptimizers.Config(
        param_groups=base.param_groups,
        implementation=base.implementation,
        qk_clip_tau=qk_clip_tau,
    )


def flex_shard_deepseek_v3_16b_gather_shard_muonclip() -> Trainer.Config:
    """gather_shard MuonClip: dense 2D -> Shard(0) GatherMuon, experts -> GroupedMuon, +QK-clip."""
    config = _dsv3_16b_test_base()
    config.model_spec = model_registry("16B")
    config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("shard")
    config.optimizer = _gather_grouped_muon_optimizer(qk_clip_tau=100.0)
    return config


def flex_shard_deepseek_v3_16b_gather_grouped_muonclip() -> Trainer.Config:
    """gather_grouped MuonClip: dense 2D -> GroupedRaggedShard GatherMuon, experts -> GroupedMuon, +QK-clip."""
    config = _dsv3_16b_test_base()
    config.model_spec = model_registry("16B")
    config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("grouped")
    config.optimizer = _gather_grouped_muon_optimizer(qk_clip_tau=100.0)
    return config


def flex_shard_deepseek_v3_16b_fsdp2_adamw() -> Trainer.Config:
    """fsdp2 + AdamW production reference (core fully_shard, no Muon / QK-clip).

    Uses :class:`BenchInstrumentedOptimizers` (the stock AdamW container + the
    ``[muon-bench]`` instrumentation) so its total_iter / opt_step / comm are reported
    on the same axes as the Muon methods.
    """
    config = _dsv3_16b_test_base()
    da = default_adamw(lr=2e-4)
    config.optimizer = BenchInstrumentedOptimizers.Config(
        param_groups=da.param_groups,
        implementation=da.implementation,
    )
    return config


# =====================================================================================
# DeepSeek V3 16B: fsdp2 baseline -- core fully_shard (DTensor Shard(0)) + Muon
# that lets DTensor do the all-gather in opt.step (vs comm-efficient Owned). Same full-Muon
# recipe as Owned/gather: dense 2D + 3D experts -> DTensorMuon (single / batched NS),
# everything else -> AdamW.
# =====================================================================================
def _fsdp2_muon_optimizer(
    *, muon_lr: float = 0.005, adamw_lr: float = 2e-4
) -> FSDP2MuonOptimizers.Config:
    """DTensorMuon on dense 2D body matrices + 3D experts, AdamW on the rest (core fully_shard)."""
    base = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    return FSDP2MuonOptimizers.Config(
        param_groups=base.param_groups,
        implementation=base.implementation,
    )


def flex_shard_deepseek_v3_16b_fsdp2_muon() -> Trainer.Config:
    """fsdp2 Shard(0) + DTensor-all-gather Muon baseline for dsv3-16B (vs comm-efficient Owned).

    Keeps the core ``deepseek_v3_16b`` model + ``fully_shard`` parallelize (DTensor params);
    only the optimizer differs. Test-friendly: bs=1 / seq=2048, c4_test + test tokenizer, AC
    off, eager (no compile), EP=8.
    """
    config = deepseek_v3_16b()
    config.optimizer = _fsdp2_muon_optimizer()
    config.activation_checkpoint = None
    config.compile = CompileConfig(enable=False)
    config.parallelism = ParallelismConfig(expert_parallel_degree=8)
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.training = TrainingConfig(local_batch_size=1, seq_len=2048, steps=10)
    return config
