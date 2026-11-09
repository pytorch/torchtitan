# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA GRPO for Muse Spark Open 816B-A42B on SLAKE (medical image + question -> short answer).

Data: the English SLAKE release (Hugging Face ``BoKelvin/SLAKE``), unzipped so that
``{slake_data_dir}/train.json``, ``test.json`` and ``{slake_data_dir}/imgs/`` exist. Answers
are graded by exact match after normalization (`RewardSlakeAnswer`).

Run with ``--module torchtitan_recipes.rl.slake --config rl_muse_spark_816b_a42b_slake_lora_gb300``.
"""

from __future__ import annotations

from typing import cast

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.muse_spark import (
    build_model_config as build_muse_spark_model_config,
    MUSE_SPARK_SPECIAL_TOKENS,
    MUSE_SPARK_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.slake import RewardSlakeAnswer, SlakeDataset, SlakeEnv
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

# LoRA on the language model's attention projections and output gate. The routed experts
# (the generator's Dist-MoE experts take no LoRA), the router and the vision tower stay
# frozen.
MUSE_SPARK_LORA_TARGETS = (
    "attention.qkv_linear.wqkv",
    "attention.wo",
    "attention.output_gate",
)

# English SLAKE test questions without knowledge-graph ones (what SlakeDataset keeps).
_NUM_SLAKE_TEST_QUESTIONS = 913


def rl_muse_spark_816b_a42b_slake_lora_gb300(
    *,
    hf_assets_path: str = "./assets/hf/Muse-Spark-1.2-816B-A42B-open",
    slake_data_dir: str = "./assets/datasets/slake",
) -> Controller.Config:
    """LoRA GRPO for Muse Spark Open 816B-A42B on SLAKE, on 32 GB300 GPUs with Dist-MoE.

    The measured configuration: an 8-GPU trainer and three 8-GPU generators, 32 prompts x
    16 samples, 8K tokens per microbatch, low reasoning effort, at most 4K response tokens.
    The trainer is the bound side (~21 s per step, ~25 samples/s); two generators keep up
    when every group trains, the third covers the groups that drop_zero_std_reward_groups
    discards (most closed questions are answered right by every sample of a group).

    LoRA (rank 8, alpha 16) on the attention projections and output gate; the adapters keep
    fp32 master weights and AdamW states and compute in bf16. The frozen base is stored in
    ``param_dtype`` (bf16): in fp32 the 816B base does not fit the 8-GPU trainer. The
    generators load the base from the HF checkpoint, so every weight sync carries only the
    adapters (~51 MB).

    Trainer: FSDP(shard)=8, EP=8 (the bf16 weights take ~204 GB per GPU). Generators: DP=2,
    TP=4, EP=8 per replica, bf16 weights, FULL CUDA graphs; each replica's EP group must
    share one NVLink domain.

    Data: SLAKE's English closed and open questions, with the Muse Spark SLAKE SFT recipe's
    short-answer instruction. Validation: greedy answers on the English test questions
    before the first and after the last training step.
    """
    max_total_tokens = 8192
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        expert_parallel_degree=8,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=2,
        tensor_parallel_degree=4,
        expert_parallel_degree=8,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_muse_spark_model_config,
        "816B-A42B_mm",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=max_total_tokens,
        attn_backend="varlen",
    )
    hf_checkpoint = cast(
        CheckpointManager.Config,
        TorchCheckpointingManager.Config(initial_load_in_hf=True, load_only=True),
    )
    config = Controller.Config(
        hf_assets_path=hf_assets_path,
        dump_folder="outputs/rl/muse_spark_816b_a42b_slake_lora_gb300",
        tokenizer=MuseSparkTokenizer.Config(**MUSE_SPARK_SPECIAL_TOKENS),
        num_generators=3,
        async_loop=AsyncLoopConfig(
            num_training_steps=200,
            num_prompts_per_train_step=32,
            num_samples_per_prompt=16,
            validation=ValidationConfig(steps=_NUM_SLAKE_TEST_QUESTIONS),
            batcher=Batcher.Config(
                training_sample_builder=TrainingSampleBuilder.Config(
                    drop_zero_std_reward_groups=True
                ),
            ),
        ),
        rollouter=Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=SlakeDataset.Config(data_dir=slake_data_dir, seed=42)
            ),
            validation_dataset=SlakeDataset.Config(
                data_dir=slake_data_dir, split="test", seed=99
            ),
            worker=RolloutWorker.Config(
                rubric=Rubric.Config(reward_fns=[RewardSlakeAnswer.Config(weight=1.0)]),
                message_env=SlakeEnv.Config(),
                token_env=TokenEnv.Config(
                    max_rollout_tokens=max_total_tokens, max_num_turns=1
                ),
            ),
        ),
        vision_processor=VisionProcessor.Config(**MUSE_SPARK_VISION_PROCESSOR_KWARGS),
        renderer=MuseSparkRendererConfig(reasoning_effort="low"),
        precision=PrecisionConfig(dtype="float32", param_dtype="bfloat16"),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*", lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0, min_lr_factor=1.0
                ),
            ),
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
            ),
            parallelism=trainer_parallelism,
            dist_moe_runtime=DistMoeRuntime.Config(scratch_capacity_factor=16.0),
            checkpointer=hf_checkpoint,
            # Dist-MoE's eager expert-output callback rejects TorchDispatchMode-based
            # checkpointing; torch_remat recomputes whole blocks with saved-tensor hooks.
            activation_remat=ActivationRematConfig(save_regions=[]),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            parallelism=generator_parallelism,
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="FULL")),
            # vLLM's profiling dummy batch can route every token to one expert.
            dist_moe_runtime=DistMoeRuntime.Config(
                inference=True, scratch_capacity_factor=16.0
            ),
            # The generators load the frozen base themselves (same HF checkpoint as the
            # trainer); weight syncs then carry only the adapters.
            checkpointer=hf_checkpoint,
            gpu_memory_limit=0.9,
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=4096),
        ),
    )
    lora = LoRATransform(
        rank=8, alpha=16.0, target_modules=list(MUSE_SPARK_LORA_TARGETS)
    )
    # The routed experts are frozen: Dist-MoE accumulates weight gradients in place only
    # into parameters that need them.
    config.trainer = apply_transforms(
        config.trainer, [DistMoeTransform(inplace_wgrad_accum=False), lora]
    )
    config.generator = apply_transforms(
        config.generator, [DistMoeTransform(inplace_wgrad_accum=False), lora]
    )
    return config
