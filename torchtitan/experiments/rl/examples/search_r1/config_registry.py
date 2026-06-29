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
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_1_7b_search_r1() -> Controller.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Runs on 8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1), with a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.
    """
    return Controller.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_groups_per_train_step=8,
            group_size=8,
            validation=ValidationConfig(num_samples=500),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=4096),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                # Mid-run checkpoints so a preempted run resumes; full last save
                # (not model-only) keeps it resumable; keep_latest_k caps disk.
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                ),
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


def rl_grpo_qwen3_8b_search_r1() -> Controller.Config:
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


def rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf() -> Controller.Config:
    """GRPO Search-R1 for Qwen3-30B-A3B MoE with a DeepEP v2 cudagraph generator.

    DeepEP v2 runs multi-node on H100 (NVLink intra-node + IB/RoCE inter-node), so unlike
    a HybridEP generator (whose all-to-all is intra-node only) this generator may span
    nodes. Qwen3-30B-A3B has 4 KV heads, so the generator TP must be <=4. The trainer
    keeps the compact (host-synced, backward-able) DeepEP path; the generator applies the
    ``deepep_inference`` override to switch its dispatchers to the cudagraph-able EXPAND
    layout. Applies the same ``fused_swiglu`` + ``helion_rope`` perf overrides (CUDA-only)
    as ``rl_grpo_qwen3_30b_a3b_varlen_perf``. The generator graphs pure-decode batches
    (FULL_DECODE_ONLY) and runs prefill / mixed batches eager (see the ``cudagraph`` knob).
    """
    model_spec = model_registry(
        "30B-A3B", attn_backend="varlen", moe_comm_backend="deepep"
    )

    # Same opt-in throughput overrides as rl_grpo_qwen3_30b_a3b_varlen_perf, applied
    # independently to the trainer and generator actors.
    perf_imports = [
        "torchtitan.overrides.fused_swiglu",
        "torchtitan.overrides.helion_rope",
    ]

    config = Controller.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-30B-A3B",
        # Generation is the async-loop bottleneck; run several generators (the
        # GeneratorRouter load-balances rollouts). Each is one host (TP=4/EP=4); the
        # single controller retrieval server caps how far this scales.
        num_generators=4,
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            # Cold-start GRPO grouping: group_size=32 gives more diverse siblings per
            # prompt -> fewer zero-std groups (256 rollouts/step). Keep zero-std groups
            # (drop_zero_std=False) so a step still lands at cold start, where most groups
            # are all-wrong under pure exact-match; the rare nonzero group drives gradient.
            num_groups_per_train_step=8,
            group_size=32,
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            ),
            validation=ValidationConfig(num_samples=500),
            batcher=Batcher.Config(
                # TODO: TBD local_batch_size, seq_len
                batch=BatchConfig(local_batch_size=1, seq_len=4096),
            ),
        ),
        compile=CompileConfig(enable=False),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),  # TODO: TBD
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            # FSDP=16/EP=16 (two H100 hosts): FSDP=8/EP=8 sits at the 80GB edge and a
            # step-2 forward/backward tripped a CUDA "unspecified launch failure" there;
            # sharding across 16 ranks (128 experts -> 8 per rank; DeepEP all-to-all goes
            # inter-node over IB/RoCE) roughly halves per-GPU memory.
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=16,
                tensor_parallel_degree=1,
                expert_parallel_degree=16,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # Chunked loss: a bare DAPOLoss materializes the full-vocab (151936) x seq_len
            # logits/logprobs in one shot -> forward_backward OOMs on 80GB. Slicing into
            # num_chunks bounds that transient (same DAPO clip params).
            loss=ChunkedLossWrapper.Config(
                num_chunks=16,
                loss_fn=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
            ),
            override=OverrideConfig(imports=list(perf_imports)),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(  # single node generator
                data_parallel_degree=1,
                tensor_parallel_degree=4,
                expert_parallel_degree=4,
            ),
            # FULL_DECODE_ONLY: graph pure-decode batches, run prefill / mixed eager.
            # FULL_AND_PIECEWISE (graphs prefill / mixed too) corrupted the DeepEP EXPAND
            # MoE forward -> garbage output, no search, 0 reward. Decode-only graphing
            # avoids it (#3668, as the 1.7B/8B/dense configs) and is far faster than eager.
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=512),
            # Generator override: helion_rope + the DeepEP EXPAND inference override, but
            # NOT fused_swiglu -- its fused_grouped_experts claims `moe.experts` while
            # deepep_inference claims the descendant `moe.experts.token_dispatcher`, a
            # nested-override conflict (#3778). The trainer keeps fused_swiglu.
            override=OverrideConfig(
                imports=[
                    "torchtitan.overrides.helion_rope",
                    "torchtitan.distributed.deepep.inference_override",
                ]
            ),
        ),
    )
    # Two inference knobs to set per workload (no golden default; here EP=4):
    #  * max_num_batched_tokens: vLLM's per-step token budget (default None -> vLLM's own
    #    default of 2048). Decide it from your input/rollout sequence length.
    #  * num_max_tokens_per_rank: per-rank EXPAND-dispatch capacity, REQUIRED by the
    #    deepep_inference override. For a dropless model (highest memory) set it to
    #    max_num_batched_tokens // ep; lower it gradually to save memory (trading off
    #    dropped tokens).
    config.generator.max_num_batched_tokens = 2048  # TODO: TBD
    num_max_tokens_per_rank = (
        config.generator.max_num_batched_tokens
        // config.generator.parallelism.expert_parallel_degree
    )
    for block in config.model_spec.model.layers:
        moe = getattr(block, "moe", None)
        if moe is None:
            continue
        moe.experts.token_dispatcher.num_max_tokens_per_rank = num_max_tokens_per_rank
    return config


def rl_grpo_qwen3_30b_a3b_dense_search_r1_perf() -> Controller.Config:
    """GRPO Search-R1 for Qwen3-30B-A3B MoE without expert parallelism (dense FSDP).

    Same model and Search-R1 recipe as ``rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf``,
    but the MoE experts are sharded by FSDP only -- no expert parallelism and the
    "standard" (non-DeepEP) token dispatch on both the trainer and the vLLM generator.
    This trades replicated expert compute for sidestepping the DeepEP build/runtime setup.
    The trainer runs FSDP=16 (two H100 hosts); with no EP every expert shards across all
    16 ranks, so this is more memory-comfortable than the FSDP=8 + EP=8 DeepEP config.
    The generator keeps DP=1/TP=4 (Qwen3-30B-A3B has 4 KV heads, so generator TP <=4).
    Mixed precision (fp32 master + bf16 compute) comes from the TrainingConfig defaults,
    matching the DeepEP config. Applies the same ``fused_swiglu`` + ``helion_rope`` perf
    overrides (CUDA-only).
    """
    # Default "standard" MoE comm backend (no DeepEP); experts shard under FSDP.
    model_spec = model_registry("30B-A3B", attn_backend="varlen")

    perf_imports = [
        "torchtitan.overrides.fused_swiglu",
        "torchtitan.overrides.helion_rope",
    ]

    return Controller.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-30B-A3B",
        num_generators=2,  # TODO: TBD -- number of generator proc meshes to spawn
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_groups_per_train_step=32,  # TODO: TBD
            group_size=8,  # TODO: TBD
            validation=ValidationConfig(num_samples=500),
            batcher=Batcher.Config(
                # TODO: TBD local_batch_size, seq_len
                batch=BatchConfig(local_batch_size=1, seq_len=4096),
            ),
        ),
        compile=CompileConfig(enable=False),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),  # TODO: TBD
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            parallelism=ParallelismConfig(
                # FSDP=16, no EP -> dense and expert params both shard under FSDP.
                data_parallel_shard_degree=16,  # TODO: TBD
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
            override=OverrideConfig(imports=list(perf_imports)),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(  # single node generator, no EP
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            # FULL_DECODE_ONLY (default): graph only pure-decode batches; run prefill /
            # mixed batches eager. The known-good 1.7B/8B search_r1 configs use this;
            # FULL_AND_PIECEWISE (also graphs prefill/mixed) corrupted generation here
            # (gibberish output) -- see #3668. Match the small configs to derisk.
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=512),
            override=OverrideConfig(imports=list(perf_imports)),
        ),
    )
