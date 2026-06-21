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
    ``tensor_parallel_degree=1``) with mixed precision (bf16 compute, fp32 reduce
    -- the default) and fp32 master weights. FSDP shards the fp32 master + AdamW
    states 8-way so 32B fits one 80GB host (~60-70GB/rank), and the shard group
    is intra-host (NVLink) so no NCCL group spans hosts -> no RoCE needed. Full
    activation checkpointing frees the forward activations so compute_logprobs'
    full-vocab fp32 transient fits. Per tianyu-l: trainer uses FSDP + mixed
    precision + fp32 master.

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
        # Per tianyu-l: FSDP + mixed precision + fp32 master weights. Set
        # explicitly (these match the defaults) so the recipe is self-documenting:
        #   dtype="float32"             -> fp32 master weights
        #   mixed_precision_param=bf16  -> bf16 compute (FSDP casts for fwd/bwd)
        #   mixed_precision_reduce=fp32 -> fp32 gradient reduce
        training=dataclasses.replace(
            config.trainer.training,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        # Full activation checkpointing: SelectiveAC OOMs at 32B/FSDP=8,
        # FullAC fits.
        ac_config=FullAC.Config(),
        # FSDP across all 8 GPUs on one host (dp_shard=8, TP=1) shards the fp32
        # master + AdamW 8-way so 32B fits one 80GB host; intra-host shard group
        # -> no cross-host NCCL.
        #
        # CPU-staged weight sync (direct_rdma=False), NOT GPU-Direct RDMA. With
        # num_generators>1 the trainer's direct-RDMA push does not fan out safely:
        # the post-step-1 trainer->generator sync hangs (NCCL watchdog abort) and
        # the 2-host trainer crashes in monarch_rdma IbvMemoryRegion. CPU-staging
        # through TorchStore is fanout-safe across all generators. See
        # generator.py TODO ">1 generator -> trainer should use direct_rdma=False".
        weight_sync_direct_rdma=False,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        # CPU-staged weight sync (fanout-safe for >1 generator); see trainer note.
        weight_sync_direct_rdma=False,
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

    Same recipe as ``rl_grpo_qwen3_32b_search_r1`` but the trainer shards across two
    hosts: ``data_parallel_shard_degree=16``, ``tensor_parallel_degree=1`` = 16 GPUs
    = 2 hosts. The launcher infers this (``infer_nodes(16, 8) = 2``), so the MAST job
    is 1 controller + 2 trainer + 3 generator = 6 hosts.

    Why 16-way: sharding the fp32 master + AdamW states 16-way (vs 8-way) roughly
    halves the trainer's per-rank optimizer memory (~36 GB/rank vs ~72 GB at FSDP=8,
    which is why FSDP=8 sat on the OOM edge). That frees enough room to drop back to
    the faster SelectiveAC default -- FSDP=16 is the memory lever instead of FullAC's
    full-forward recompute -- while still fitting compute_logprobs' full-vocab fp32
    transient.

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
        # FSDP=16 frees ~half the per-rank master+optimizer memory vs FSDP=8, so the
        # SelectiveAC default fits -- no need for FullAC's recompute overhead.
        ac_config=SelectiveAC.Config(),
    )
    return config
