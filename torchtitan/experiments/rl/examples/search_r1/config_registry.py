# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the full Search-R1 recipe entirely from the example's config — the core
defaults are unchanged, so every other config keeps vanilla GRPO. ``ConfigManager``
discovers these directly from the example module::

    --module search_r1 \\
        --config rl_grpo_qwen3_1_7b_search_r1
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_1_7b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Runs on 8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1), with a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.
    """
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=500,
        num_groups_per_rollout_batch=32,
        group_size=8,
        num_validation_samples=500,
        validation_freq=5,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=48, seq_len=4096),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10000,  # only the initial HF load; no mid-run checkpoints
                last_save_model_only=True,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            # cudagraph on: decode-only graphs (FULL_DECODE_ONLY) are safe at this
            # config's large batch; plain full graphs corrupted here before. See #3668.
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-8B — same recipe as the 1.7B config.

    Only the model and GPU split differ. 8 GPUs: 2 generator (TP=2) + 4 trainer
    (TP=4) + retriever on the spare GPUs. The fp32 trainer needs TP=4 to avoid OOM.
    """
    # TODO: use mixed precision (fp32 master + bf16 compute) via FSDP + activation
    # checkpointing, which is more memory-efficient and could keep the split generator-heavy.
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    # 0.6 (vs the 0.9 default) reserves room for the weight-sync memory spike, which
    # OOMs the 8B generator otherwise.
    # TODO(@meetv18): the spike is likely GPU-Direct weight transfer being on by default;
    # make the transfer device configurable (CPU default) so this cap can be raised.
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config


def rl_grpo_qwen3_32b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-32B (dense) -- single-host-per-role, multi-generator.

    Same recipe as the 1.7B/8B configs; only the model and GPU split differ.
    Trainer: single-host FSDP across 8 GPUs (``data_parallel_shard_degree=8``,
    ``tensor_parallel_degree=1``) in pure bf16 (``dtype=bfloat16``, no fp32 master;
    fp32 gradient reduce). bf16 roughly halves per-rank memory vs an fp32 master, so
    32B fits one 80GB host, and the shard group is intra-host (NVLink) so no NCCL
    group spans hosts. Full activation checkpointing frees the forward activations
    so compute_logprobs' full-vocab fp32 transient fits. Weight sync uses direct
    GPU-to-GPU RDMA (see the trainer note below for the measured per-step decompose).

    Each generator uses TP=8 = 8 GPUs (1 host) with decode-only CUDA graphs
    (FULL_DECODE_ONLY, #3668-safe) + aot_eager torch.compile (the top-level
    ``compile``) to speed up the generation-bound rollout. With
    ``--num_generators 3`` the MAST job is 1 controller + 1 trainer + 3 generator
    = 5 hosts.
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("32B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-32B"
    # Leave the trainer's per-layer torch.compile off. Note this is the *trainer*
    # compile only; the rollout's vLLM compile is driven separately by the
    # generator cudagraph below and stays on. (The earlier step-1 NCCL timeout was
    # not a compile desync -- see the FullAC note below for the real cause.)
    config.compile = dataclasses.replace(config.compile, enable=False)
    config.trainer = dataclasses.replace(
        config.trainer,
        # Pure bf16 training (no fp32 master): dtype=bfloat16 keeps the master
        # weights + AdamW states in bf16, roughly halving per-rank memory vs an
        # fp32 master -- that headroom is what lets FSDP=8 fit (the fp32-master
        # variant OOM'd in step-2 backward). mixed_precision_reduce=fp32 still
        # reduces gradients in fp32 for stability.
        training=dataclasses.replace(
            config.trainer.training,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        # Full activation checkpointing: SelectiveAC OOMs at 32B/FSDP=8,
        # FullAC fits.
        ac_config=FullAC.Config(),
        # FSDP across all 8 GPUs on one host (dp_shard=8, TP=1) shards the master
        # + AdamW 8-way so 32B fits one 80GB host; intra-host shard group -> the
        # FSDP collectives never span hosts.
        #
        # FSDP=8 + direct_rdma=True + bf16 IS runnable: validated end-to-end with 3
        # generators (single-host trainer, so no 2-host monarch_rdma fanout). Direct
        # GPU-to-GPU RDMA cuts the trainer->generator weight sync from ~133s
        # (CPU-staged) to ~4s. Measured per-step decompose (FSDP=8, bf16, RDMA, 3
        # generators; ~38s/step recent vs ~179s CPU-staged):
        #   generator_pull (weight sync)   ~4s    (CPU-staged was ~133s)
        #   trainer_push   (weight sync)   ~0s
        #   rollout (generation+retrieval) ~20s   <- the new bottleneck
        #   trainer fwd/bwd (6 micro-bw)   ~15s
        # Needs Monarch RDMA available (conda rdma-core matching Monarch's build,
        # v60 -> rdmav59 provider; see mast_rl/build_conda.sh). direct_rdma=None
        # auto-falls-back to CPU-staged where RDMA is unavailable.
        weight_sync_direct_rdma=True,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        # Direct GPU-to-GPU RDMA weight sync (must match the trainer); ~4s pull vs
        # ~133s CPU-staged. See the trainer note for the per-step decompose.
        weight_sync_direct_rdma=True,
        # Decode-only CUDA graphs (FULL_DECODE_ONLY, #3668-safe): capture
        # pure-decode batches (the bulk of the generation-bound rollout), run
        # prefill eagerly. Works now that VLLMAttentionWrapper's q/k/v args are
        # renamed to q_BLNH/k_BLNH/v_BLNH so the spmd-attention local_map
        # resolves in_dst_shardings (see attention.py).
        cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=8
        ),
    )
    # 3 generator replicas (each its own MAST role / proc mesh): the multinode
    # topology this config targets. Overridable with --num_generators.
    config.num_generators = 3
    return config


def rl_grpo_qwen3_32b_search_r1_fsdp16() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-32B (dense) -- 2-host trainer (FSDP=16) + 3 generators.

    Builds on ``rl_grpo_qwen3_32b_search_r1`` but shards the trainer across two
    hosts: ``data_parallel_shard_degree=16``, ``tensor_parallel_degree=1`` = 16 GPUs
    = 2 hosts. The launcher infers this (``infer_nodes(16, 8) = 2``), so the MAST job
    is 1 controller + 2 trainer + 3 generator = 6 hosts. This variant reverts the
    base's bf16 + direct-RDMA back to an fp32 master + CPU-staged weight sync (see
    the trainer override below for why).

    Why fp32 master here (vs the base's bf16): sharding the fp32 master + AdamW
    states 16-way roughly halves per-rank optimizer memory (~36 GB/rank), giving the
    accuracy of an fp32 master while still fitting -- and freeing room to drop to the
    faster SelectiveAC default instead of FullAC's full-forward recompute.

    Caveat: the FSDP all-gather/reduce-scatter group now spans 2 hosts. run.sh
    defaults NCCL_NET=Socket (the rlmast conda env lacks the mlx5 RDMA driver), so
    cross-host collectives go over TCP -- functional but slower than RDMA. The
    generation-bound rollout still dominates wall-clock; set NCCL_NET=IB once
    libmlx5 is available for full cross-host speed.
    """
    config = rl_grpo_qwen3_32b_search_r1()
    config.trainer = dataclasses.replace(
        config.trainer,
        # 2 hosts x 8 GPUs = 16-way FSDP shard (spans the host boundary).
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
        ),
        # FSDP=16 frees ~half the per-rank memory vs FSDP=8, so the SelectiveAC
        # default fits -- no need for FullAC's recompute overhead.
        ac_config=SelectiveAC.Config(),
        # Revert the FSDP=8 base back to this variant's validated 2-host settings:
        # fp32 master (16-way sharding gives the memory room) + CPU-staged weight
        # sync. The base's bf16 + direct_rdma=True is the single-host FSDP=8 path;
        # direct RDMA across a 2-host trainer hits the monarch_rdma fanout
        # (untested), so fsdp16 stays CPU-staged (validated: 500 steps, val ~0.48).
        training=dataclasses.replace(config.trainer.training, dtype="float32"),
        weight_sync_direct_rdma=False,
    )
    config.generator = dataclasses.replace(
        config.generator, weight_sync_direct_rdma=False
    )
    return config


def rl_grpo_qwen3_32b_search_r1_fsdp16_rdma() -> RLTrainer.Config:
    """EXPERIMENTAL: FSDP=16 (2-host trainer) + direct-RDMA weight sync.

    Identical to the working single-host ``rl_grpo_qwen3_32b_search_r1`` (pure bf16 +
    ``direct_rdma=True`` + 3 generators) EXCEPT the trainer shards 16-way across two
    hosts (``data_parallel_shard_degree=16``). The 2-host trainer is the only
    difference from the validated FSDP=8 RDMA recipe, so this isolates whether direct
    RDMA weight sync survives a multi-host trainer mesh -- i.e. whether the
    documented monarch_rdma 2-host fanout (3 generators reading the trainer's RDMA
    buffers across two hosts) actually fails, vs the single-host FSDP=8 path that
    works. Unlike ``rl_grpo_qwen3_32b_search_r1_fsdp16`` (fp32 master + CPU-staged),
    this keeps the base's bf16 + direct_rdma=True. Needs Monarch RDMA available
    (conda rdma-core matching Monarch's build; see mast_rl/build_conda.sh).
    1 controller + 2 trainer + 3 generator = 6 hosts.

    RESULT (2026-06-21): does NOT work. The initial RDMA weight sync completes
    (trainer push ~105s registration + generator pull ~7.8s), but right after
    training starts the trainer SIGSEGVs inside Monarch's RDMA layer
    (``monarch/rdmaxcel-sys/src/rdmaxcel.cpp:556`` ->
    ``IbvMemoryRegion::drop``) -- a Monarch multi-host direct-RDMA bug, not a config
    or memory issue (it is NOT the rdma-core version: v60 got the single-host FSDP=8
    RDMA path to step 71+, and bf16/16-way has ample memory). For a multi-host
    trainer, use the CPU-staged ``rl_grpo_qwen3_32b_search_r1_fsdp16`` instead;
    direct RDMA only works with the single-host FSDP=8 trainer for now.
    """
    config = rl_grpo_qwen3_32b_search_r1()
    config.trainer = dataclasses.replace(
        config.trainer,
        # 2 hosts x 8 GPUs = 16-way FSDP shard. Keeps the base's bf16 +
        # direct_rdma=True; SelectiveAC fits at 16-way (vs FullAC at FSDP=8).
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
        ),
        ac_config=SelectiveAC.Config(),
    )
    return config
