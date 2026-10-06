# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the alphabet-sort example on Intel XPU.

Kept out of ``alphabet_sort`` because these configs carry XPU-specific
settings that would be wrong defaults on CUDA. Select them with the fully
qualified module path::

    --module torchtitan_recipes.rl.alphabet_sort_xpu \
        --config rl_grpo_full_qwen3_0_6b_flex

The two configs here are controlled twins: identical global batch, local batch,
lr, sampling, parallelism, and generator settings. The only difference is
whether a ``LoRATransform`` is applied, so a full-parameter run and a LoRA run
are directly comparable.

Four values vary from job to job and are read from the environment (see
``_env_*`` below). A config recipe is nullary and the CLI sets only
``--output-dir``, ``--comm-backend``, and ``--resume-step`` -- ``--override``
swaps ``Configurable`` implementations rather than setting fields -- so the
environment is the only channel a PBS script has for them. ``qsub -v NAME`` and
``mpiexec --envall`` both forward it.
"""

import os

import torch
from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    LinearLoRAHandler,
    LoRATransform,
    ModelConfigTransformContext,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.trainer import Trainer

# vLLM is not installed in the lint environment.
# pyrefly: ignore[missing-import]
from vllm.config import AttentionConfig

from torchtitan_recipes.rl.alphabet_sort import (
    _alphabet_sort_rollouter_config,
    _build_qwen3_rl_model_config,
)

_SEQ_LEN = 2048

# Per-job knobs. The defaults reproduce a self-contained 200-step run against the
# in-repo example checkpoint; the Aurora launchers in scripts/xpu/ set all four.
_NUM_TRAINING_STEPS = int(os.environ.get("TITAN_XPU_NUM_STEPS", "200"))
_VALIDATION_NUM_SAMPLES = int(os.environ.get("TITAN_XPU_VAL_SAMPLES", "20"))
_HF_ASSETS_PATH = os.environ.get(
    "TITAN_XPU_HF_ASSETS_PATH", "torchtitan/rl/example_checkpoint/Qwen3-0.6B"
)
# 0.85 rather than vLLM's 0.9 default: vLLM sizes the KV cache from free memory
# reported at init, and on XPU that reading lags the trainer's allocations on the
# same card. A single-node LoRA run has headroom to raise it.
_GPU_MEMORY_LIMIT = float(os.environ.get("TITAN_XPU_GPU_MEMORY_LIMIT", "0.85"))


def _disable_flex_autotune() -> None:
    """Turn off inductor autotuning for the flex-attention kernel.

    The XPU flex backward asks for kernel configs that exceed the device's
    register budget and fails with OUT_OF_RESOURCES, so autotuning has to be off
    before the first compile. ``inductor_configs`` is a class variable baked into
    ``_compiled_flex_attn`` at import time, so the compiled callable has to be
    rebuilt rather than just the dict updated.

    This is process-global: it changes flex attention for every model built
    afterwards in this process. That is intended -- an XPU process cannot run the
    autotuned kernel at all.
    """
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.models.common.attention import FlexInnerAttention

    FlexInnerAttention.inductor_configs = {
        **FlexInnerAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    # pyrefly: ignore[no-matching-overload]
    FlexInnerAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexInnerAttention.inductor_configs
    )


def _xpu_grpo_qwen3_0_6b_flex(model_config: Decoder.Config) -> Controller.Config:
    """The shared body of both XPU configs (4 tiles: 2 trainer + 2 generator).

    FSDP rather than TP on the trainer side, because TP on XPU pays a
    per-microbatch allreduce over the same links the weight sync uses.
    """
    return Controller.Config(
        model=model_config,
        hf_assets_path=_HF_ASSETS_PATH,
        async_loop=AsyncLoopConfig(
            num_training_steps=_NUM_TRAINING_STEPS,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=_VALIDATION_NUM_SAMPLES),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=True)),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        # Two independent single-tile generator replicas. generator.parallelism's
        # data_parallel_degree is reserved for feeding expert parallelism.
        num_generators=2,
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=5,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * _SEQ_LEN,
                max_context_length=_SEQ_LEN,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=1,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                # Load the pretrained weights, write nothing back. Saving is off
                # because the DCP save ran out of device memory inside oneCCL on
                # multi-node XPU; interval applies again once that is fixed. Do
                # NOT express this as "no checkpointer": load() would go too,
                # and the run would silently train from random init.
                load_only=True,
                interval=50,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=_GPU_MEMORY_LIMIT,
            # No CuMem pool on XPU. It exists to keep RDMA registrations valid
            # under expandable segments, which the XPU path disables anyway
            # (see torchtitan/rl/_runtime.py).
            enable_cumem_allocator=False,
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            # Bare AttentionConfig: let vLLM pick its own XPU backend. The
            # derived default would ask for FLEX_ATTENTION, which vLLM does not
            # have there.
            attention_config=AttentionConfig(),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_full_qwen3_0_6b_flex() -> Controller.Config:
    """Full-parameter GRPO for Qwen3-0.6B with flex attention, on XPU.

    Every parameter is trained, so the optimizer state and the per-step
    trainer -> generator weight payload are the whole ~0.6B model rather than
    just the adapters.
    """
    _disable_flex_autotune()
    model_config = _build_qwen3_rl_model_config(
        "0.6B", seq_len=_SEQ_LEN, attn_backend="flex"
    )
    return _xpu_grpo_qwen3_0_6b_flex(model_config)


def rl_grpo_lora_qwen3_0_6b() -> Controller.Config:
    """LoRA GRPO for Qwen3-0.6B with flex attention, on XPU.

    rank=32 adapters on the attention projections; every other parameter is
    frozen. lr stays at the full-parameter 2e-6 so the two arms differ in one
    variable only (a standalone LoRA run would want something like 1e-4).
    """
    _disable_flex_autotune()
    model_config = _build_qwen3_rl_model_config(
        "0.6B", seq_len=_SEQ_LEN, attn_backend="flex"
    )
    config = _xpu_grpo_qwen3_0_6b_flex(model_config)
    # apply_transforms() reads training/parallelism off the config it is given.
    # This is a Controller.Config (an RL tree), where those live one level down
    # under `trainer`, so the context has to be passed explicitly.
    return apply_transforms(
        config,
        [
            LoRATransform(
                handlers=(LinearLoRAHandler(),),
                rank=32,
                alpha=64.0,
                target_modules=["wqkv", "wo"],
            )
        ],
        context=ModelConfigTransformContext(
            training=config.trainer.training,
            parallelism=config.trainer.parallelism,
        ),
    )
