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
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_ce_loss,
    llama3_debugmodel_float8_emulate_lora,
    llama3_debugmodel_varlen_attn,
    sft_debugmodel,
)
from torchtitan.observability.sdc_replayer import SDCReplayer, SDCReplayMismatch
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from . import _use_spmd_types


class SDCReplayMismatchTrainer(Trainer):
    """Inject a replay-only gradient mismatch and verify it is fatal."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._num_forward_backward_calls = 0

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        loss = super().forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
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
                        # Intentionally corrupt one local gradient element to verify
                        # that SDC replay reports the injected mismatch.
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
            assert error.local_step == 1
            assert error.replay == 1
            assert error.rank == 0
            assert error.signature_mismatch is not None
            assert error.signature_mismatch.startswith("gradient:0:")
            assert self.sdc_replayer is not None
            assert self.sdc_replayer.steps_since_reset == 0
            logger.info("Detected expected %s", error)
            return
        raise AssertionError("Expected SDC replay to detect the injected mismatch.")


def deepseek_v3_debugmodel_sdc_replay_mismatch() -> Trainer.Config:
    base_config = deepseek_v3_debugmodel(seq_len=2048)
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
    config.sdc_replayer = SDCReplayer.Config()
    return config


def llama3_debugmodel_sdc_replay_cudagraph() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.debug.deterministic = True
    config.debug.seed = 42
    config.training.steps = 3
    config.sdc_replayer = SDCReplayer.Config()
    return config


def llama3_debugmodel_default() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.profiler.enable_profiling = True
    config.metrics.enable_tensorboard = True
    return config


def llama3_debugmodel_compile() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.compile.enable = True
    return config


def llama3_debugmodel_compile_sac_op() -> Trainer.Config:
    config = llama3_debugmodel_compile()
    config.activation_checkpoint = SelectiveAC.Config()
    return config


def llama3_debugmodel_tp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_ce_loss_tp2() -> Trainer.Config:
    config = llama3_debugmodel_ce_loss(seq_len=2048)
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
    config = llama3_debugmodel(seq_len=2048)
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
    config.checkpoint.initial_load_path = os.path.join(
        test_output_dir,
        "hf_checkpoint/step-10/",
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
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.parallelism.data_parallel_shard_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_pp2_1f1b()
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    return config


def llama3_debugmodel_tp2_pp2_gpipe() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "GPipe"
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_save() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.checkpoint.enable = True
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_load() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_tp2_pp2_save()
    config.training.steps = 20
    return config


def llama3_debugmodel_fsdp2_tp2_pp2_compile() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_pp4_interleaved_1f1b() -> Trainer.Config:
    """PP-only; see ``llama3_debugmodel_pp2_1f1b`` for why type checking is off."""
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
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
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "ZBVZeroBubble"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.activation_checkpoint = FullAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_pp2_custom_csv() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "PipelineScheduleMulti"
    config.parallelism.pipeline_parallel_schedule_csv = (
        "./tests/assets/custom_schedule.csv"
    )
    config.activation_checkpoint = FullAC.Config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_optimizer_bf16_states() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.optimizer.implementation = "fused_opt_states_bf16"
    return config


def llama3_debugmodel_ddp4() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 4
    return config


def llama3_debugmodel_hsdp2x2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    return config


def llama3_debugmodel_cp4() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.context_parallel_degree = 4
    return config


def llama3_debugmodel_hsdp2x2_tp2() -> Trainer.Config:
    config = llama3_debugmodel_hsdp2x2()
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_debugmodel_fsdp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def llama3_debugmodel_ddp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
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
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.parallelism.fsdp_reshard_after_forward = "always"
    return config


def llama3_debugmodel_optional_checkpoint_save() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
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
    """Two gradient accumulation steps on 2 GPUs."""
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.training.num_tokens_per_microbatch_per_dp_rank = 16384
    config.training.num_tokens_per_train_step = 65536
    return config


def llama3_debugmodel_validation_tp2_cp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.validator.enable = True
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fused_swiglu_tp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.override.imports = ["torchtitan.overrides.fused_swiglu.fused_swiglu"]
    config.parallelism.tensor_parallel_degree = 2
    return config


def deepseek_v3_debugmodel_fused_grouped_experts_tp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
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
    config = llama3_debugmodel_varlen_attn(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_float8_emulate_lora_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel_float8_emulate_lora(seq_len=2048)
    _use_spmd_types(config, typechecking=False)
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_sft() -> Trainer.Config:
    config = sft_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    return config


def llama3_debugmodel_seed_checkpoint() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    _use_spmd_types(config, typechecking=True)
    config.checkpoint.enable = True
    config.checkpoint.create_seed_checkpoint = True
    config.training.disable_cuda_graphs = True
    return config


def kimi_k3_debugmodel_pp2() -> Trainer.Config:
    """Kimi K3 text decoder split over two pipeline stages.

    The block attention residual is defined over the whole stack, so it travels
    between stages as a second stage payload alongside the hidden states.
    """
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    return config


def kimi_k3_debugmodel_pp8_vp4() -> Trainer.Config:
    """Eight ranks, four virtual stages each, on the interleaved schedule.

    One layer per stage over 32 stages: the 30 layers plus the embedding and
    the head, each counted as a layer, so the first and last stages hold no
    transformer layer and the block residual crosses every boundary the
    schedule has.
    """
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel

    config = kimi_k3_debugmodel()
    config.parallelism.pipeline_parallel_degree = 8
    config.parallelism.pipeline_parallel_layers_per_stage = 1
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 8
    return config
