# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""features integration-test suite: run configurations and their test definitions.

Each configuration function is selected by module and name in the launched
training process, so it must stay importable from this module.
"""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import Any, cast

import torch
import torch.distributed as dist
from renderers import Message, Qwen3RendererConfig
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base import BaseModel
from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
    UlyssesFlexInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.observability.sdc_replayer import SDCReplayer, SDCReplayMismatch
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel,
    llama3_debugmodel_ce_loss,
    llama3_debugmodel_lora,
    llama3_debugmodel_varlen_attn,
    sft_debugmodel,
)
from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel

from tests.integration_tests import IntegrationTestDefinition


logger = logging.getLogger(__name__)


class SDCReplayMismatchTrainingEngine(TrainingEngine):
    """Inject a gradient mismatch into the first SDC replay execution."""

    def __init__(
        self,
        config: TrainingEngine.Config,
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        output_dir: str,
    ) -> None:
        super().__init__(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )
        self._num_forward_backward_calls = 0

    def _non_pp_forward_backward_microbatch(
        self,
        *,
        inputs: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        model_kwargs: dict[str, Any],
        loss_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        loss = super()._non_pp_forward_backward_microbatch(
            inputs=inputs,
            labels=labels,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
        )
        self._num_forward_backward_calls += 1
        if self._num_forward_backward_calls != 2 or dist.get_rank() != 0:
            return loss

        with torch.no_grad():
            for model_part in self.model_parts:
                for parameter in model_part.parameters():
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad
                    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
                    if local_grad.numel() > 0:
                        # Corrupt the first replay before SDC captures its signature.
                        local_grad[(0,) * local_grad.ndim].add_(1)
                        return loss
        raise AssertionError("Could not find a local gradient to corrupt.")


class SDCReplayMismatchTrainer(Trainer):
    """Inject a replay-only gradient mismatch and verify it is fatal."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    engine_cls = SDCReplayMismatchTrainingEngine
    engine: SDCReplayMismatchTrainingEngine

    def train_step(
        self,
        data_iterator: Iterator[TrainingMicrobatch],
    ) -> None:
        try:
            super().train_step(data_iterator)
        except SDCReplayMismatch as error:
            assert error.step == 1
            assert error.local_step == 1
            assert error.replay == 1
            assert error.rank == 0
            assert error.signature_mismatch is not None
            assert error.signature_mismatch.startswith("gradient:0:")
            assert self.engine.sdc_replayer is not None
            assert self.engine.sdc_replayer.steps_since_reset == 0
            logger.info("Detected expected %s", error)
            self.engine.num_completed_steps = error.step
            return
        raise AssertionError("Expected SDC replay to detect the injected mismatch.")


def deepseek_v3_debugmodel_sdc_replay_mismatch() -> Trainer.Config:
    base_config = deepseek_v3_debugmodel(
        seq_len=2048,
        attn_backend="flex",
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2, expert_parallel_degree=2
        ),
    )
    config = SDCReplayMismatchTrainer.Config(
        **{
            config_field.name: getattr(base_config, config_field.name)
            for config_field in fields(base_config)
            if config_field.init
        }
    )
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.disable_cuda_graphs = True
    config.training.steps = 1
    config.sdc_replayer = SDCReplayer.Config()
    return config


def llama3_debugmodel_sdc_replay_cuda_graph() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.steps = 3
    config.sdc_replayer = SDCReplayer.Config()
    return config


def llama3_debugmodel_default() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.profiler.enable_profiling = True
    config.metrics.enable_tensorboard = True
    return config


def llama3_debugmodel_tp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_ce_loss_tp2() -> Trainer.Config:
    config = llama3_debugmodel_ce_loss(seq_len=2048)
    # Non-chunked CE loss does not pass SPMD type checking yet.
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_tp2_no_sp() -> Trainer.Config:
    config = llama3_debugmodel(
        seq_len=2048,
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            tensor_parallel_degree=2,
            enable_sequence_parallel=False,
        ),
    )
    _set_spmd_typechecking(config, typechecking=True)
    return config


def llama3_debugmodel_full_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        last_save_model_only=False,
    )
    return config


def llama3_debugmodel_full_checkpoint_load() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.training.steps = 20
    return config


def llama3_debugmodel_hf_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.checkpointer.folder = "hf_checkpoint"
    config.checkpointer.last_save_model_only = True
    config.checkpointer.last_save_in_hf = True
    return config


def llama3_debugmodel_hf_checkpoint_load() -> Trainer.Config:
    """Loads what ``llama3_debugmodel_hf_checkpoint_save`` wrote.

    The integration runner supplies the per-test output directory to each run.
    """
    config = llama3_debugmodel_full_checkpoint_save()
    test_output_dir = os.getenv(
        "TORCHTITAN_TEST_OUTPUT_DIR",
        os.path.join(
            os.getenv("RUNNER_TEMP", ""),
            "artifacts-to-be-uploaded/model_only_hf_checkpoint",
        ),
    )
    config.checkpointer.initial_load_path = os.path.join(
        test_output_dir,
        "hf_checkpoint/step-10/",
    )
    config.checkpointer.initial_load_model_only = True
    config.checkpointer.initial_load_in_hf = True
    return config


def llama3_debugmodel_last_save_model_only_bf16() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.checkpointer.last_save_model_only = True
    config.checkpointer.export_dtype = "bfloat16"
    return config


def llama3_debugmodel_pp2_1f1b() -> Trainer.Config:
    """PP-only, so it leaves SPMD type checking off.

    Type checking needs at least one SPMD axis greater than 1; collapsing
    every dense SPMD axis to size 1 trips DTensor's rejection of a Shard
    placement on a degenerate axis.
    """
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def muse_glimmer_debugmodel_fsdp2_pp2_deferred_gradient_reduction() -> Trainer.Config:
    config = muse_glimmer_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 65536
    return config


def _muse_glimmer_debugmodel_pp2_max_outstanding_sends(
    schedule: str,
) -> Trainer.Config:
    config = muse_glimmer_debugmodel(seq_len=2048)
    config.comm.backend = "real_pp_fake_spmd"
    config.debug.deterministic = True
    config.debug.seed = 42
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = schedule
    config.parallelism.pipeline_parallel_max_outstanding_sends = 2
    config.activation_remat = None
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def muse_glimmer_debugmodel_pp2_looped_bfs_send_budget() -> Trainer.Config:
    return _muse_glimmer_debugmodel_pp2_max_outstanding_sends("LoopedBFS")


def muse_glimmer_debugmodel_pp2_interleaved_1f1b_send_budget() -> Trainer.Config:
    return _muse_glimmer_debugmodel_pp2_max_outstanding_sends("Interleaved1F1B")


def muse_glimmer_debugmodel_fsdp2_pp2_optimizer_cuda_graph() -> Trainer.Config:
    config = muse_glimmer_debugmodel_fsdp2_pp2_deferred_gradient_reduction()
    config.optim.enable_cuda_graph = True
    return config


def muse_glimmer_debugmodel_fsdp2_optimizer_cuda_graph() -> Trainer.Config:
    config = muse_glimmer_debugmodel_fsdp2_deferred_gradient_reduction()
    config.optim.enable_cuda_graph = True
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_pp2_1f1b()
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    return config


def llama3_debugmodel_tp2_pp2_gpipe() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "GPipe"
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_save() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        last_save_model_only=False,
    )
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_load() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_tp2_pp2_save()
    config.training.steps = 20
    return config


def llama3_debugmodel_fsdp2_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def llama3_debugmodel_pp4_interleaved_1f1b() -> Trainer.Config:
    """PP-only; see ``llama3_debugmodel_pp2_1f1b`` for why type checking is off."""
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.debug.deterministic = True
    config.debug.seed = 42
    return config


def llama3_debugmodel_pp4_interleaved_1f1b_layers_per_stage() -> Trainer.Config:
    config = llama3_debugmodel_pp4_interleaved_1f1b()
    config.parallelism.pipeline_parallel_layers_per_stage = 1
    return config


def llama3_debugmodel_pp4_zero_bubble() -> Trainer.Config:
    config = llama3_debugmodel_varlen_attn(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "InterleavedZeroBubble"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.activation_remat = FullActivationRematConfig()
    config.debug.deterministic = True
    config.debug.seed = 42
    return config


def llama3_debugmodel_pp2_zbv() -> Trainer.Config:
    config = llama3_debugmodel_varlen_attn(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "ZBVZeroBubble"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.activation_remat = FullActivationRematConfig()
    config.debug.deterministic = True
    config.debug.seed = 42
    return config


def llama3_debugmodel_pp2_custom_csv() -> Trainer.Config:
    config = llama3_debugmodel_varlen_attn(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "PipelineScheduleMulti"
    config.parallelism.pipeline_parallel_schedule_csv = (
        "./tests/assets/custom_schedule.csv"
    )
    config.activation_remat = FullActivationRematConfig()
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.debug.deterministic = True
    config.debug.seed = 42
    return config


def muse_glimmer_debugmodel_optimizer_bf16_states() -> Trainer.Config:
    config = muse_glimmer_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.training.mixed_precision_reduce = "float32"
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", moment_dtype="bfloat16")]
    )
    return config


def llama3_debugmodel_ddp4() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 4
    return config


def llama3_debugmodel_hsdp2x2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    return config


def llama3_debugmodel_cp4() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048, attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.context_parallel_degree = 4
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_ulysses_cp2() -> Trainer.Config:
    """Attention reshards the CP axis onto the head dimension."""
    config = llama3_debugmodel(attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.context_parallel_degree = 2
    # Head-sharded attention has no per-rank sequence imbalance to balance.
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: UlyssesFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_ulysses_cp2_varlen() -> Trainer.Config:
    """Llama 3 with varlen Ulysses CP."""
    config = llama3_debugmodel()
    # Packed varlen metadata lacks SPMD annotations.
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.context_parallel_degree = 2
    # Ulysses does not support token reordering.
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={VarlenInnerAttention: UlyssesVarlenInnerAttention}
            )
        ],
    )


def llama3_debugmodel_fsdp2_ulysses_cp2_varlen() -> Trainer.Config:
    config = llama3_debugmodel_ulysses_cp2_varlen()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def llama3_debugmodel_fsdp2_tp2_ulysses_cp2_varlen() -> Trainer.Config:
    config = llama3_debugmodel_ulysses_cp2_varlen()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_hsdp2x2_tp2() -> Trainer.Config:
    config = llama3_debugmodel_hsdp2x2()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048, attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_ddp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048, attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_hsdp2x2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048, attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_cp2()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_reshard_always() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.fsdp_reshard_after_forward = "always"
    return config


def llama3_debugmodel_optional_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        last_save_model_only=False,
    )
    return config


def llama3_debugmodel_optional_checkpoint_load_tp2() -> Trainer.Config:
    """Loads a ``[dp:4]`` checkpoint at ``[dp:2, tp:2]``.

    The dataloader is excluded from loading to avoid errors caused by the
    mismatched dp degree.
    """
    config = llama3_debugmodel_optional_checkpoint_save()
    config.checkpointer.exclude_from_loading = [
        "lr_scheduler",
        "dataloader",
        "optimizer",
    ]
    config.parallelism.tensor_parallel_degree = 2
    config.training.steps = 20
    return config


def llama3_debugmodel_gradient_accumulation() -> Trainer.Config:
    """Two gradient accumulation steps on 2 GPUs."""
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.training.num_tokens_per_train_step = 65536
    return config


def muse_glimmer_debugmodel_fsdp2_per_group_cuda_graph() -> Trainer.Config:
    """Three fixed accumulation groups with FSDP degree 2.

    GPU unit tests cover changing group counts.
    """
    config = muse_glimmer_debugmodel(seq_len=2048)
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.cuda_graph_per_microbatch_group = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.training.num_tokens_per_train_step = 98304
    config.parallelism.data_parallel_shard_degree = 2
    return config


def muse_glimmer_debugmodel_fsdp2_deferred_gradient_reduction() -> Trainer.Config:
    config = muse_glimmer_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.training.num_tokens_per_train_step = 65536
    return config


def muse_glimmer_debugmodel_shared_eval_cuda_graph() -> Trainer.Config:
    """Train and validate with shared graph workspace on two FSDP ranks."""
    config = muse_glimmer_debugmodel(seq_len=128)
    config.debug.deterministic = True
    config.debug.seed = 42
    config.parallelism.data_parallel_shard_degree = 2
    config.optim.enable_cuda_graph = True
    config.validator = Validator.Config(
        freq=2,
        steps=1,
        dataloader=replace(config.dataloader),
    )
    return config


def llama3_debugmodel_validation_tp2_cp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048, attn_backend="flex")
    _set_spmd_typechecking(config, typechecking=False)
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.validator = Validator.Config(
        freq=5,
        steps=10,
        dataloader=replace(config.dataloader),
    )
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={FlexInnerAttention: AllGatherKVFlexInnerAttention}
            )
        ],
    )


def llama3_debugmodel_fused_swiglu_tp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.override.imports = ["torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"]
    config.parallelism.tensor_parallel_degree = 2
    return config


def deepseek_v3_debugmodel_fused_swiglu_tp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel(
        seq_len=2048,
        attn_backend="flex",
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2, expert_parallel_degree=4
        ),
    )
    _set_spmd_typechecking(config, typechecking=True)
    config.override.imports = ["torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"]
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_varlen_attn_fsdp4_sac() -> Trainer.Config:
    config = llama3_debugmodel_varlen_attn(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.activation_remat = DefaultActivationRematConfig()
    return config


def llama3_debugmodel_lora_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel_lora(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    return config


def llama3_debugmodel_sft() -> Trainer.Config:
    config = sft_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    return config


def llama3_debugmodel_sft_multiturn() -> Trainer.Config:
    config = llama3_debugmodel_sft()

    def messages(sample) -> list[Message]:
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
            {"role": "user", "content": "Repeat your answer."},
            {"role": "assistant", "content": sample["answer"]},
        ]

    dataloader = cast(GrainDataLoader.Config, config.dataloader)
    packing = cast(FirstFitPackingConfig, dataloader.dataset)
    dataset = cast(SingleDatasetConfig, packing.dataset)
    processor = cast(ChatProcessor.Config, dataset.processor)
    processor.messages_fn = messages
    processor.renderer = from_renderers(Qwen3RendererConfig())
    return config


def llama3_debugmodel_seed_checkpoint() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    config.training.disable_cuda_graphs = True
    return config


def build_features_test_list() -> list[IntegrationTestDefinition]:
    """
    Build the list of integration tests covering the core features of torchtitan.

    Each entry names one configuration per run.
    """
    return [
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_sdc_replay_cuda_graph],
            test_descr="SDC replay with CUDA graphs",
            test_name="sdc_replay_cuda_graph",
            ngpu=1,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_sdc_replay_mismatch],
            test_descr="SDC replay detects a distributed gradient mismatch",
            test_name="sdc_replay_mismatch",
            ngpu=2,
            # Cross-rank mismatch propagation (all_reduce + all_gather_object)
            # requires real collectives.
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_default],
            test_descr="default",
            test_name="default",
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_tp2,
                llama3_debugmodel_ce_loss_tp2,
            ],
            test_descr=(
                "2D eager (ChunkedLossWrapper + standard CE loss with TP+loss_parallel)"
            ),
            test_name="2d_eager",
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_tp2_no_sp],
            test_descr="2D eager (SP disabled)",
            test_name="2d_eager_no_sp",
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_full_checkpoint_save,
                llama3_debugmodel_full_checkpoint_load,
            ],
            test_descr="Checkpoint Integration Test - Save Load Full Checkpoint",
            test_name="full_checkpoint",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_hf_checkpoint_save,
                llama3_debugmodel_hf_checkpoint_load,
            ],
            test_descr=(
                "Checkpoint Integration Test - save load model only checkpoint in "
                "HF definition and format"
            ),
            test_name="model_only_hf_checkpoint",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_last_save_model_only_bf16],
            test_descr="Checkpoint Integration Test - Save Model Only bf16",
            test_name="last_save_model_only_bf16",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_pp2_1f1b],
            test_descr="PP 1D test 1F1B",
            test_name="pp_1f1b",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_fsdp2_pp2_1f1b,
                llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage,
            ],
            test_descr="PP+DP 1F1B 2D test",
            test_name="pp_dp_1f1b",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp2_pp2_deferred_gradient_reduction],
            test_descr="FSDP+PP deferred gradient reduction with accumulation",
            test_name="fsdp_pp_deferred_gradient_reduction",
            ngpu=4,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[
                muse_glimmer_debugmodel_pp2_looped_bfs_send_budget,
                muse_glimmer_debugmodel_pp2_interleaved_1f1b_send_budget,
            ],
            test_descr="PP bounded outstanding sends",
            test_name="pp_max_outstanding_sends",
            ngpu=2,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp2_pp2_optimizer_cuda_graph],
            test_descr="FSDP+PP forward-backward and optimizer CUDA graphs",
            test_name="fsdp_pp_optimizer_cuda_graph",
            ngpu=4,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp2_optimizer_cuda_graph],
            test_descr="FSDP forward-backward and optimizer CUDA graphs",
            test_name="fsdp_optimizer_cuda_graph",
            ngpu=2,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_tp2_pp2_gpipe],
            test_descr="PP+TP GPipe 2D test",
            test_name="pp_tp_gpipe",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_fsdp2_tp2_pp2_save,
                llama3_debugmodel_fsdp2_tp2_pp2_load,
            ],
            test_descr="PP+DP+TP 3D test with save/load resume ckpt",
            test_name="pp_dp_tp",
            ngpu=8,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_tp2_pp2],
            test_descr="PP+DP+TP 3D test",
            test_name="3d",
            ngpu=8,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_pp4_interleaved_1f1b,
                llama3_debugmodel_pp4_interleaved_1f1b_layers_per_stage,
            ],
            test_descr="PP looped 1F1B test",
            test_name="pp_looped_1f1b",
            ngpu=4,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_pp4_zero_bubble],
            test_descr="PP looped zero bubble test",
            test_name="pp_looped_zero_bubble",
            ngpu=4,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_pp2_zbv],
            test_descr="PP zero bubble test (v shaped)",
            test_name="pp_zbv",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_pp2_custom_csv],
            test_descr="PP with custom pipeline schedule loaded from CSV file",
            test_name="pp_custom_csv",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_optimizer_bf16_states],
            test_descr="BF16 Optimizer States Test",
            test_name="optimizer_bf16_states",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_ddp4],
            test_descr="DDP",
            test_name="ddp",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_hsdp2x2],
            test_descr="HSDP",
            test_name="hsdp",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_cp4],
            test_descr="KV-all-gather CP with Flex attention",
            test_name="cp",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_hsdp2x2_tp2],
            test_descr="HSDP+TP",
            test_name="hsdp+tp",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_cp2],
            test_descr="FSDP+CP",
            test_name="fsdp+cp",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_ulysses_cp2],
            test_descr="Ulysses CP with Flex attention",
            test_name="cp_ulysses",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_ulysses_cp2_varlen],
            test_descr="Ulysses CP with varlen attention",
            test_name="cp_ulysses_varlen",
            ngpu=2,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_ulysses_cp2_varlen],
            test_descr="FSDP + Ulysses CP with varlen attention",
            test_name="fsdp+cp_ulysses_varlen",
            ngpu=4,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_tp2_ulysses_cp2_varlen],
            test_descr="FSDP + TP + Ulysses CP with varlen attention",
            test_name="fsdp+tp+cp_ulysses_varlen",
            ngpu=8,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_ddp2_cp2],
            test_descr="HSDP+CP (without dp_shard)",
            test_name="hsdp+cp_without_dp_shard",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_hsdp2x2_cp2],
            test_descr="HSDP+CP (with dp_shard)",
            test_name="hsdp+cp_with_dp_shard",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp2_tp2_cp2],
            test_descr="FSDP+TP+CP",
            test_name="fsdp+tp+cp",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fsdp_reshard_always],
            test_descr="Test always resharding after forward pass",
            test_name="fsdp_reshard_always",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[
                llama3_debugmodel_optional_checkpoint_save,
                llama3_debugmodel_optional_checkpoint_load_tp2,
            ],
            test_descr="Optional checkpoint",
            test_name="optional_checkpoint",
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_gradient_accumulation],
            test_descr="Gradient accumulation",
            test_name="gradient_accumulation",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp2_per_group_cuda_graph],
            test_descr="Per-group CUDA graph replay with FSDP accumulation",
            test_name="fsdp_per_group_cuda_graph",
            ngpu=2,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_shared_eval_cuda_graph],
            test_descr="Shared train and eval CUDA graphs with FSDP",
            test_name="shared_eval_cuda_graph",
            ngpu=2,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_debugmodel_fsdp2_deferred_gradient_reduction],
            test_descr="FSDP deferred gradient reduction with accumulation",
            test_name="fsdp_deferred_gradient_reduction",
            ngpu=2,
            use_real_pg=True,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_validation_tp2_cp2_pp2],
            test_descr="Validation test with tp, cp, pp",
            test_name="validation_tp_cp_pp",
            ngpu=8,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_fused_swiglu_tp2],
            test_descr="Override: use Triton SwiGLU activation (FSDP2 + TP2)",
            test_name="override_fused_swiglu",
            ngpu=4,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_debugmodel_fused_swiglu_tp2_ep4],
            test_descr=(
                "Override: use Triton SwiGLU activation on deepseek_v3 "
                "(FSDP2 + TP2 dense, EP4 sparse)"
            ),
            test_name="override_fused_swiglu_moe",
            ngpu=4,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_varlen_attn_fsdp4_sac],
            test_descr="FSDP+VARLEN_ATTN + selective AC",
            test_name="fsdp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_lora_tp2_pp2],
            test_descr="LoRA training test",
            test_name="lora",
            ngpu=8,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_sft],
            test_descr="SFT ChatDataset integration and numerics test",
            test_name="sft",
            ngpu=2,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/sft.txt"
            ),
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_sft_multiturn],
            test_descr="Multi-turn SFT with renderer-provided loss masks",
            test_name="sft_multiturn",
            ngpu=2,
        ),
        IntegrationTestDefinition(
            configs=[llama3_debugmodel_seed_checkpoint],
            test_descr="Seed checkpoint creation",
            test_name="seed_checkpoint",
            ngpu=1,
            timeout=30,
            use_real_pg=True,
        ),
    ]
