# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``features`` integration test suite."""

import os
from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.sdc_replay import SDCReplayMismatch
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_ce_loss,
    llama3_debugmodel_float8_emulate_lora,
    llama3_debugmodel_varlen_attn,
    sft_debugmodel,
)
from torchtitan.tools.logging import logger
from torchtitan.trainer import (
    ForwardBackwardStepContext,
    ForwardBackwardStepFn,
    ForwardBackwardStepWrapper,
    Trainer,
)

from . import _use_spmd_types


class SDCReplayMismatchTrainer(Trainer):
    """Inject a replay-only gradient mismatch and verify it is fatal."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._num_forward_backward_calls = 0

    def make_forward_backward_step(
        self,
        *,
        step_wrappers: tuple[ForwardBackwardStepWrapper, ...] = (),
    ) -> ForwardBackwardStepFn:
        return super().make_forward_backward_step(
            step_wrappers=(self._inject_sdc_mismatch, *step_wrappers)
        )

    def _inject_sdc_mismatch(
        self,
        next_step_fn: ForwardBackwardStepFn,
        context: ForwardBackwardStepContext,
    ) -> torch.Tensor:
        loss = next_step_fn(context)
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
                        local_grad[(0,) * local_grad.ndim].add_(1)
                        return loss
        raise AssertionError("Could not find a local gradient to corrupt.")

    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> None:
        try:
            super().train_step(data_iterator)
        except SDCReplayMismatch as error:
            assert error.step == 1
            assert error.attempt == 1
            assert error.replay == 1
            assert error.rank == 0
            assert error.signature_mismatch is not None
            assert error.signature_mismatch.startswith("gradient:0:")
            assert self.sdc_attempt_step == 0
            logger.info("Detected expected %s", error)
            return
        raise AssertionError("Expected SDC replay to detect the injected mismatch.")


def deepseek_v3_debugmodel_sdc_replay_mismatch() -> Trainer.Config:
    base_config = deepseek_v3_debugmodel()
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
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.sdc_replay.enabled = True
    return config


def llama3_debugmodel_sdc_replay_cudagraph() -> Trainer.Config:
    config = llama3_debugmodel()
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.steps = 3
    config.sdc_replay.enabled = True
    return config


def llama3_debugmodel_default() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.profiler.enable_profiling = True
    config.metrics.enable_tensorboard = True
    return config


def llama3_debugmodel_compile() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.compile.enable = True
    return config


def llama3_debugmodel_compile_sac_op() -> Trainer.Config:
    config = llama3_debugmodel_compile()
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def llama3_debugmodel_tp2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_ce_loss_tp2() -> Trainer.Config:
    config = llama3_debugmodel_ce_loss()
    # Non-chunked CE loss does not pass SPMD type checking yet.
    _use_spmd_types(config, typechecking=False)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_tp2_no_sp() -> Trainer.Config:
    config = llama3_debugmodel_tp2()
    config.parallelism.enable_sequence_parallel = False
    return config


def llama3_debugmodel_tp2_compile() -> Trainer.Config:
    config = llama3_debugmodel_compile()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_tp2_asynctp_compile_spmd_types() -> Trainer.Config:
    config = llama3_debugmodel_tp2_compile()
    config.compile.enable_async_tensor_parallel = True
    return config


def llama3_debugmodel_full_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.checkpoint.enable = True
    return config


def llama3_debugmodel_full_checkpoint_load() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.training.steps = 20
    return config


def llama3_debugmodel_hf_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.checkpoint.folder = "hf_checkpoint"
    config.checkpoint.last_save_model_only = True
    config.checkpoint.last_save_in_hf = True
    return config


def llama3_debugmodel_hf_checkpoint_load() -> Trainer.Config:
    """Loads what ``llama3_debugmodel_hf_checkpoint_save`` wrote.

    The load path points into the integration runner's output directory, which
    is ``RUNNER_TEMP`` on GitHub Actions and a relative path elsewhere.
    """
    config = llama3_debugmodel_full_checkpoint_save()
    config.checkpoint.initial_load_path = os.path.join(
        os.getenv("RUNNER_TEMP", ""),
        "artifacts-to-be-uploaded/model_only_hf_checkpoint/hf_checkpoint/step-10/",
    )
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True
    return config


def llama3_debugmodel_last_save_model_only_bf16() -> Trainer.Config:
    config = llama3_debugmodel_full_checkpoint_save()
    config.checkpoint.last_save_model_only = True
    config.checkpoint.export_dtype = "bfloat16"
    return config


def llama3_debugmodel_pp2_1f1b() -> Trainer.Config:
    """PP-only, so it leaves SPMD type checking off.

    Type checking needs at least one SPMD axis greater than 1; collapsing
    every dense SPMD axis to size 1 trips DTensor's rejection of a Shard
    placement on a degenerate axis.
    """
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 1
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_pp2_1f1b()
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    return config


def llama3_debugmodel_tp2_pp2_gpipe() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "GPipe"
    config.parallelism.tensor_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_save() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.checkpoint.enable = True
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_load() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_tp2_pp2_save()
    config.training.steps = 20
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_compile() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_pp4_interleaved_1f1b() -> Trainer.Config:
    """PP-only; see ``llama3_debugmodel_pp2_1f1b`` for why type checking is off."""
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_pp4_interleaved_1f1b_layers_per_stage() -> Trainer.Config:
    config = llama3_debugmodel_pp4_interleaved_1f1b()
    config.parallelism.pipeline_parallel_layers_per_stage = 1
    return config


def llama3_debugmodel_pp4_zero_bubble() -> Trainer.Config:
    config = llama3_debugmodel_pp4_interleaved_1f1b()
    config.parallelism.pipeline_parallel_schedule = "InterleavedZeroBubble"
    config.activation_checkpoint = FullAC.Config()
    return config


def llama3_debugmodel_pp2_zbv() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "ZBVZeroBubble"
    config.activation_checkpoint = FullAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_pp2_custom_csv() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "PipelineScheduleMulti"
    config.parallelism.pipeline_parallel_schedule_csv = (
        "./tests/assets/custom_schedule.csv"
    )
    config.activation_checkpoint = FullAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_optimizer_bf16_states() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.optimizer.implementation = "fused_opt_states_bf16"
    return config


def llama3_debugmodel_ddp4() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 4
    return config


def llama3_debugmodel_hsdp2x2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    return config


def llama3_debugmodel_cp4() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.context_parallel_degree = 4
    return config


def llama3_debugmodel_hsdp2x2_tp2() -> Trainer.Config:
    config = llama3_debugmodel_hsdp2x2()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def llama3_debugmodel_ddp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def llama3_debugmodel_hsdp2x2_cp2() -> Trainer.Config:
    config = llama3_debugmodel_hsdp2x2()
    config.parallelism.context_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_cp2()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp_reshard_always() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.parallelism.fsdp_reshard_after_forward = "always"
    return config


def llama3_debugmodel_optional_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.checkpoint.enable = True
    return config


def llama3_debugmodel_optional_checkpoint_load_tp2() -> Trainer.Config:
    """Loads a ``[dp:4]`` checkpoint at ``[dp:2, tp:2]``.

    The dataloader is excluded from loading to avoid errors caused by the
    mismatched dp degree.
    """
    config = llama3_debugmodel_optional_checkpoint_save()
    config.checkpoint.exclude_from_loading = ["lr_scheduler", "dataloader", "optimizer"]
    config.parallelism.tensor_parallel_degree = 2
    config.training.steps = 20
    return config


def llama3_debugmodel_gradient_accumulation() -> Trainer.Config:
    """Two gradient accumulation steps on 2 GPUs.

    Local batch size 8 over 2 ranks gives a default global batch size of 16;
    doubling it to 32 asks for two accumulation steps.
    """
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.training.local_batch_size = 8
    config.training.global_batch_size = 32
    return config


def llama3_debugmodel_validation_tp2_cp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=False)
    config.validator.enable = True
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fused_swiglu_tp2() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.override.imports = ["torchtitan.overrides.fused_swiglu.fused_swiglu"]
    config.parallelism.tensor_parallel_degree = 2
    return config


def deepseek_v3_debugmodel_fused_grouped_experts_tp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.override.imports = [
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ]
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_varlen_attn_fsdp4_sac() -> Trainer.Config:
    config = llama3_debugmodel_varlen_attn()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_float8_emulate_lora_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel_float8_emulate_lora()
    _use_spmd_types(config, typechecking=False)
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_torchcomms_cp2_pp2_compile() -> Trainer.Config:
    """Keeps the default SPMD backend: CP with compile hits an upstream symint
    limitation under ``spmd_types``."""
    config = llama3_debugmodel()
    config.comm.mode = "torchcomms"
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_torchcomms_tp2_pp2_compile() -> Trainer.Config:
    config = llama3_debugmodel_ce_loss()
    _use_spmd_types(config, typechecking=False)
    config.comm.mode = "torchcomms"
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_sft() -> Trainer.Config:
    config = sft_debugmodel()
    _use_spmd_types(config, typechecking=True)
    return config


def llama3_debugmodel_seed_checkpoint() -> Trainer.Config:
    config = llama3_debugmodel()
    _use_spmd_types(config, typechecking=True)
    config.checkpoint.enable = True
    config.checkpoint.create_seed_checkpoint = True
    config.training.disable_cuda_graphs = True
    return config
