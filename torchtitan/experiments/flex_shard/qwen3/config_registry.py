# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard + Muon benchmark configs for the dense Qwen3 ladder.

Built for the distributed-Muon **systems** benchmark (see
``docs/muon.md``): the comm-efficient ``Owned`` Muon path across model sizes
(``..._muon`` for every flavor -> model-size / GPU-count scaling), plus the
gather-for-NS baselines and an AdamW reference at a couple of sizes (the fixed-config
optimizer-step breakdown).

Notes:
- **Weight tying is disabled** (``enable_weight_tying=False``). FlexShard assigns each
  parameter FQN to exactly one bucket, so a tied ``lm_head.weight`` /
  ``tok_embeddings.weight`` (one shared storage under two FQNs) would be sharded twice;
  untying makes them separate sharded params. Qwen3's tied flavors (0.6B/1.7B/4B)
  *skip-init* the embedding (it normally aliases the initialized LM head), so once
  untied the embedding must be initialized itself -- we set it to ``normal_(std=1.0)``,
  matching the natively-untied flavors (8B/14B/32B). This slightly inflates the small
  flavors' param count (a separate LM head); the systems metrics (``opt_step`` scales
  with NS work, ``step_comm`` with sharding) are unaffected by tying.
- **Activation checkpointing is disabled** (AC recompute does not yet compose with
  FlexShard's ``Owned`` broadcast + FlexAttention's compiled HOP; re-enabling is Stage 2).
- The test tokenizer + ``c4_test`` are used so configs run without HF asset downloads;
  token content does not affect the step-time measurements. Override batch/seq/steps on
  the CLI per size as needed.
"""

from functools import partial

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, ParamGroupConfig
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.experiments.flex_shard.muon import (
    BenchInstrumentedOptimizers,
    FlexShardGatherMuonOptimizers,
    FlexShardMuonOptimizers,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3 import model_registry as core_qwen3_model_registry
from torchtitan.trainer import Trainer

from . import model_registry
from .parallelize import (
    make_gather_muon_parallelize_fn,
    parallelize_qwen3_moe_muon,
    parallelize_qwen3_moe_muon_auto,
    parallelize_qwen3_moe_muon_twolevel,
    parallelize_qwen3_muon_auto,
    parallelize_qwen3_muon_idle,
    parallelize_qwen3_muon_permatrix,
    parallelize_qwen3_muon_twolevel,
)


def _muon_optimizer(
    *, muon_lr: float = 0.02, adamw_lr: float = 8e-4
) -> FlexShardMuonOptimizers.Config:
    """Full Muon (placement-routed): owned 2D dense -> Muon, 3D MoE experts ->
    GroupedMuon / GatherGroupedMuon, everything else -> AdamW. On a dense model (no 3D
    experts) this reduces to dense Muon + AdamW.
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


def _gather_muon_optimizer(
    *, muon_lr: float = 0.02, adamw_lr: float = 8e-4
) -> FlexShardGatherMuonOptimizers.Config:
    """Full Muon, gather dense distribution: sharded 2D dense -> GatherMuon (all-gather +
    NS), 3D experts -> GroupedMuon / GatherGroupedMuon, everything else -> AdamW. On a
    dense model (no experts) this is just gather-Muon + AdamW."""
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


def _bench_adamw_optimizer(lr: float = 8e-4) -> BenchInstrumentedOptimizers.Config:
    """AdamW with benchmark instrumentation (for the vanilla FSDP2 + AdamW reference)."""
    return BenchInstrumentedOptimizers.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        implementation="foreach",
    )


def _flex_shard_qwen3(
    flavor: str, *, variant: str, truncate_layers: int | None = None
) -> Trainer.Config:
    """Build a Qwen3 benchmark config for ``flavor`` and optimizer ``variant``.

    ``variant`` is one of ``"muon"`` (FlexShard comm-efficient ``Owned``, whole-layer
    allocation -- case 1), ``"muon_permatrix"`` (per-2D-tensor allocation -- case 2,
    finer LPT balance), ``"adamw"`` (FlexShard ``Owned`` +
    AdamW; isolates the optimizer at fixed sharding), ``"gather_shard"`` /
    ``"gather_grouped"`` (FlexShard gather-for-NS Muon baselines), or ``"fsdp2_adamw"``
    (**vanilla FSDP2 + AdamW** -- the production reference; measures the cost of *adopting*
    Owned/Muon, engine + optimizer). All share the same model/batch/seq/AC-off so only the
    sharding+optimizer differ.
    """
    # The fsdp2_adamw reference uses the core FSDP2 (fully_shard) parallelizer; every
    # other variant uses the FlexShard parallelizer (Owned by default).
    if variant == "fsdp2_adamw":
        spec = core_qwen3_model_registry(flavor)
    else:
        spec = model_registry(flavor)
    # Untie lm_head from tok_embeddings (FlexShard shards each FQN once) and re-init the
    # now-separate embedding; applied to every variant so they compare the same model.
    spec.model.enable_weight_tying = False
    spec.model.tok_embeddings.param_init = {"weight": partial(nn.init.normal_, std=1.0)}
    # Optional depth truncation -- to study the num_layers < world_size regime (where
    # two-level / per-tensor allocation differs from whole-layer) on a fixed GPU count.
    if truncate_layers is not None:
        spec.model.layers = spec.model.layers[:truncate_layers]

    config = Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_muon_optimizer(),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(local_batch_size=4, seq_len=2048, steps=10),
        checkpoint=CheckpointManager.Config(interval=1000),
        # AC off for all variants: AC does not yet compose with Owned broadcast +
        # FlexAttention's HOP, and keeping it off everywhere makes the comparison fair.
        activation_checkpoint=None,
    )

    if variant == "muon":
        config.optimizer = _muon_optimizer()
    elif variant == "muon_permatrix":
        # Per-2D-tensor Owned allocation (case 2): same Muon optimizer, finer LPT.
        config.model_spec.parallelize_fn = parallelize_qwen3_muon_permatrix
        config.optimizer = _muon_optimizer()
    elif variant == "muon_twolevel":
        # Two-level allocation: per-layer rank groups + within-layer LPT.
        config.model_spec.parallelize_fn = parallelize_qwen3_muon_twolevel
        config.optimizer = _muon_optimizer()
    elif variant == "muon_idle":
        # Whole-layer (case 1) allowing idle ranks -- reference at num_layers<world_size.
        config.model_spec.parallelize_fn = parallelize_qwen3_muon_idle
        config.optimizer = _muon_optimizer()
    elif variant == "muon_auto":
        # Unified system: auto-select the allocation level for this regime.
        config.model_spec.parallelize_fn = parallelize_qwen3_muon_auto
        config.optimizer = _muon_optimizer()
    elif variant == "adamw":
        config.optimizer = default_adamw(lr=8e-4)
    elif variant == "fsdp2_adamw":
        config.optimizer = _bench_adamw_optimizer()
    elif variant == "gather_shard":
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("shard")
        config.optimizer = _gather_muon_optimizer()
    elif variant == "gather_grouped":
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("grouped")
        config.optimizer = _gather_muon_optimizer()
    elif variant == "gather_shard_raf":
        # gather-for-NS Muon with reshard-after-forward ON (Shard(0) supports RAF; the
        # sharded matrices reshard after forward instead of staying resident).
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn(
            "shard", reshard_after_forward=True
        )
        config.optimizer = _gather_muon_optimizer()
    elif variant == "gather_grouped_raf":
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn(
            "grouped", reshard_after_forward=True
        )
        config.optimizer = _gather_muon_optimizer()
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return config


# =====================================================================================
# Representative example configs (one per capability). The full set used for Experiments
# A/B (every model size x method x granularity/batched variant) lives in the local-only
# ``config_registry_local.py``; this submitted file keeps just a few examples that show the
# pattern -- add more by following the same ``_flex_shard_qwen3`` / ``_flex_shard_qwen3_moe``
# factory calls.
# =====================================================================================
# --- Dense: comm-efficient Owned Muon (debug + 4B); add other sizes via the same factory ---
def flex_shard_qwen3_debugmodel_muon() -> Trainer.Config:
    return _flex_shard_qwen3("debugmodel", variant="muon")


def flex_shard_qwen3_4b_muon() -> Trainer.Config:
    return _flex_shard_qwen3("4B", variant="muon")


def flex_shard_qwen3_4b_muon_permatrix() -> Trainer.Config:
    return _flex_shard_qwen3("4B", variant="muon_permatrix")


def flex_shard_qwen3_4b_muon_auto() -> Trainer.Config:
    # 36 layers, 8 GPUs -> auto resolves to "layer" (W <= num_layers).
    return _flex_shard_qwen3("4B", variant="muon_auto")


# --- Vanilla FSDP2 + AdamW reference (production baseline: cost of adopting Owned/Muon) ---
def flex_shard_qwen3_4b_fsdp2_adamw() -> Trainer.Config:
    return _flex_shard_qwen3("4B", variant="fsdp2_adamw")


def flex_shard_qwen3_4b_gather_muon_shard() -> Trainer.Config:
    return _flex_shard_qwen3("4B", variant="gather_shard")


def flex_shard_qwen3_4b_gather_muon_grouped() -> Trainer.Config:
    return _flex_shard_qwen3("4B", variant="gather_grouped")


def _flex_shard_qwen3_moe(
    flavor: str,
    *,
    variant: str,
    expert_parallel_degree: int = 8,
    truncate_layers: int | None = None,
    muon_lr: float = 0.005,
    adamw_lr: float = 2e-4,
) -> Trainer.Config:
    """Build a Qwen3-MoE benchmark config for ``flavor`` and optimizer ``variant``.

    ``variant`` is one of ``"muon"`` (comm-efficient Owned whole-layer + EP),
    ``"muon_auto"`` (Owned with the auto allocation level -- per-tensor / two-level when
    ``num_layers < world_size`` -- + EP), ``"muon_twolevel"`` (Owned two-level + EP),
    ``"adamw"`` (Owned + AdamW at the same EP sharding -- optimizer-only baseline),
    ``"gather_shard"`` / ``"gather_grouped"`` (gather-for-NS Muon baselines; EP-capable --
    experts go to the EP mesh, only the dense 2D matrices are gathered), or
    ``"fsdp2_adamw"`` (vanilla FSDP2 + AdamW + EP, the
    production reference). The comm-efficient (Owned) variants are full Muon: dense 2D
    matrices -> Owned Muon, 3D experts -> GroupedMuon / GatherGroupedMuon; norms,
    embeddings and the LM head -> AdamW.
    """
    if variant == "fsdp2_adamw":
        spec = core_qwen3_model_registry(flavor)
    else:
        spec = model_registry(flavor)
    # FlexShard assigns each FQN to one bucket; untie lm_head/embedding so they are
    # separate sharded params (both are natively initialized in the MoE flavors).
    spec.model.enable_weight_tying = False
    # Optional depth truncation -- to exercise the num_layers < world_size regime (where
    # the auto/two-level dense allocation differs from whole-layer) on a fixed GPU count.
    if truncate_layers is not None:
        spec.model.layers = spec.model.layers[:truncate_layers]

    # All methods use the same EP degree so the comparison is fixed-config (dp/ep): the
    # gather baselines now route experts to the EP mesh (gathering only the dense 2D
    # matrices), so only the dense-matrix distribution differs across methods.
    ep = expert_parallel_degree

    config = Trainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=_muon_optimizer(),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        # MoE models are large: default to bs=1 / seq=2048 so the targets fit; override
        # on the CLI. opt_step / step_comm are batch-independent anyway.
        training=TrainingConfig(local_batch_size=1, seq_len=2048, steps=10),
        checkpoint=CheckpointManager.Config(interval=1000),
        activation_checkpoint=None,
        parallelism=ParallelismConfig(expert_parallel_degree=ep),
    )

    if variant == "muon":
        config.model_spec.parallelize_fn = parallelize_qwen3_moe_muon
        config.optimizer = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "muon_auto":
        config.model_spec.parallelize_fn = parallelize_qwen3_moe_muon_auto
        config.optimizer = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "muon_twolevel":
        config.model_spec.parallelize_fn = parallelize_qwen3_moe_muon_twolevel
        config.optimizer = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "muon_grouped":
        # Full Muon: dense 2D -> Owned Muon, experts -> GroupedMuon (per-expert NS).
        # FlexShardMuonOptimizers always routes experts to GroupedMuon, so the optimizer
        # is just the Owned full-Muon one (``_muon_optimizer``).
        config.model_spec.parallelize_fn = parallelize_qwen3_moe_muon_auto
        config.optimizer = _muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "adamw":
        config.model_spec.parallelize_fn = parallelize_qwen3_moe_muon
        config.optimizer = default_adamw(lr=adamw_lr)
    elif variant == "gather_shard":
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("shard")
        config.optimizer = _gather_muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "gather_grouped":
        config.model_spec.parallelize_fn = make_gather_muon_parallelize_fn("grouped")
        config.optimizer = _gather_muon_optimizer(muon_lr=muon_lr, adamw_lr=adamw_lr)
    elif variant == "fsdp2_adamw":
        config.optimizer = _bench_adamw_optimizer(adamw_lr)
    else:
        raise ValueError(f"unknown MoE variant {variant!r}")
    return config


# --- debugmodel_moe (8 layers, 64 experts): local EP + finer-allocation validation ---
def flex_shard_qwen3_debugmodel_moe_muon() -> Trainer.Config:
    return _flex_shard_qwen3_moe(
        "debugmodel_moe", variant="muon", expert_parallel_degree=2
    )


def flex_shard_qwen3_30b_a3b_muon_grouped() -> Trainer.Config:
    # Full Muon: dense 2D -> Owned Muon, experts -> GroupedMuon (per-expert batched NS).
    return _flex_shard_qwen3_moe("30B-A3B", variant="muon_grouped")


# Full Muon with the gather dense distribution (GroupedRaggedShard) -- the gather
# counterpart to muon_grouped for the dense-distribution comparison (Owned vs gather).
def flex_shard_qwen3_30b_a3b_gather_grouped() -> Trainer.Config:
    return _flex_shard_qwen3_moe("30B-A3B", variant="gather_grouped")


def flex_shard_qwen3_30b_a3b_fsdp2_adamw() -> Trainer.Config:
    return _flex_shard_qwen3_moe("30B-A3B", variant="fsdp2_adamw")
