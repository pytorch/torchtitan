# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the **slime-aligned recipe** (the stabilizers that the reproduction needs)
entirely from the example's config — the core defaults are unchanged, so every other
config keeps vanilla GRPO. Imported into ``config_registry`` so ``ConfigManager``
discovers them via ``--module rl --config rl_grpo_qwen3_*_search_r1``.
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import GRPOLoss, RLTrainer
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_1_7b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Reproduces slime's ``examples/search-r1`` nq_test EM curve (~0.13 -> ~0.30):
    **pure-EM 0/1 reward** (the rollouter's default rubric — no format/retrieval
    weighting, matching slime), **standard GRPO with group-std-normalized advantages**,
    **256 samples/step** (32 prompts x group 8, no dynamic sampling), clip-higher
    (0.2/0.28), KL-to-reference (low_var_kl, 0.001), AdamW lr 1e-6 constant,
    temperature 1.0 / top_p 1.0, max 4 search turns, 500 steps.

    8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1) + (off-config) a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the Search-R1
    NQ/HotpotQA parquet data; see ``README.md``.
    """
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=500,
        # 32 prompts x group_size 8 = 256 rollouts/step (= slime's rollout_batch_size
        # 32, n_samples_per_prompt 8, global_batch_size 256). No dynamic sampling.
        num_groups_per_rollout_batch=32,
        group_size=group_size,
        # slime uses standard GRPO: normalize the advantage by the group reward std.
        advantage_std_normalization=True,
        # Eval on NQ test (slime-style): every 5 steps, first 500 prompts (file order).
        num_validation_samples=500,
        validation_freq=5,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        # seq_len 4096 fits the multi-turn rollouts; global_batch_size 48 packed rows
        # so the ~256 rollouts/step train in one optimizer step.
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=48, seq_len=4096),
        ),
        trainer=PolicyTrainer.Config(
            # lr 1e-6 constant (min_lr_factor=1.0 -> flat after a 2-step warmup).
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10000,  # only the initial HF load; no mid-run checkpoints
                last_save_model_only=True,
            ),
            # slime stabilizers: clip-higher (0.2/0.28) + KL-to-reference (low_var_kl,
            # 0.001). These are off by default in core; this config opts in.
            loss=GRPOLoss.Config(
                clip_eps=0.2,
                clip_eps_high=0.28,
                kl_coef=0.001,
                kl_loss_type="low_var_kl",
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            # Run vLLM EAGER (no CUDA-graph capture). Verified by two controlled runs
            # (only this flag changed): cudagraph="full" corrupts GENERATION itself —
            # the engine emits degenerate "locklock" tokens at step-0 validation, before
            # any training/loss runs. So the GRPOLoss NaN-drop (which only fixes the
            # trainer loss) cannot rescue it; eager is required and matches the reference
            # 1.7B run. The NaN-drop loss is kept for robustness. TODO: root-cause the
            # cudagraph generation corruption (likely captured-graph vs weight-sync
            # staleness in the vLLM generator) so cudagraph can be re-enabled for speed.
            cudagraph=VLLMCudagraphConfig(enable=False),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                # slime: temperature 1.0 + top_p 1.0; stop each turn at its action tag.
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
                stop=["</search>", "</answer>"],
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-8B — identical slime recipe to the 1.7B config.

    Only the model and GPU split differ (everything affecting the gradient is the
    same, so the 8B curve is directly comparable). 8 GPUs: 2 generator (TP=2) +
    4 trainer (TP=4) + retriever on the spare GPUs. The trainer runs fp32, so 8B
    needs TP=4 to avoid OOM; the bf16 vLLM generator fits on TP=2.
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config
