# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ensure no more flags are added to CLI"""

import dataclasses
import unittest

import tyro
from torchtitan.models.llama3.config_registry import llama3_debugmodel

_FROZEN_CLI_OPTIONS = frozenset(
    {
        "activation_checkpoint.debug",
        "activation_checkpoint.determinism_check",
        "activation_checkpoint.force_recompute_mm_shapes_by_fqns",
        "activation_checkpoint.preserve_rng_state",
        "checkpoint.async_mode",
        "checkpoint.create_seed_checkpoint",
        "checkpoint.enable",
        "checkpoint.enable_first_step_checkpoint",
        "checkpoint.exclude_from_loading",
        "checkpoint.export_dtype",
        "checkpoint.folder",
        "checkpoint.initial_load_in_hf",
        "checkpoint.initial_load_in_hf_quantized",
        "checkpoint.initial_load_model_only",
        "checkpoint.initial_load_path",
        "checkpoint.interval",
        "checkpoint.keep_latest_k",
        "checkpoint.last_save_in_hf",
        "checkpoint.last_save_model_only",
        "checkpoint.load_only",
        "checkpoint.load_step",
        "comm.init_timeout_seconds",
        "comm.mode",
        "comm.save_traces_file_prefix",
        "comm.save_traces_folder",
        "comm.trace_buf_size",
        "comm.train_timeout_seconds",
        "compile.backend",
        "compile.components",
        "compile.enable",
        "dataloader.dataset",
        "dataloader.dataset_path",
        "dataloader.infinite",
        "dataloader.num_workers",
        "dataloader.persistent_workers",
        "dataloader.pin_memory",
        "dataloader.prefetch_factor",
        "debug.batch_invariant",
        "debug.detect_anomaly",
        "debug.deterministic",
        "debug.deterministic_warn_only",
        "debug.enable_structured_logging",
        "debug.moe_force_load_balance",
        "debug.print_config",
        "debug.save_config_file",
        "debug.seed",
        "debug.spmd_typechecking",
        "dump_folder",
        "hf_assets_path",
        "loss.loss_fn.global_vocab_size",
        "loss.num_chunks",
        "lr_scheduler.decay_ratio",
        "lr_scheduler.decay_type",
        "lr_scheduler.min_lr_factor",
        "lr_scheduler.total_steps",
        "lr_scheduler.warmup_steps",
        "metrics.disable_color_printing",
        "metrics.enable_tensorboard",
        "metrics.enable_wandb",
        "metrics.log_freq",
        "metrics.save_for_all_ranks",
        "metrics.save_tb_folder",
        "optimizer.implementation",
        "optimizer.param_groups",
        "override.imports",
        "parallelism.context_parallel_degree",
        "parallelism.context_parallel_load_balancer",
        "parallelism.context_parallel_ptrr_mask_key",
        "parallelism.data_parallel_replicate_degree",
        "parallelism.data_parallel_shard_degree",
        "parallelism.enable_async_tensor_parallel",
        "parallelism.enable_fsdp_symm_mem",
        "parallelism.enable_sequence_parallel",
        "parallelism.expert_parallel_degree",
        "parallelism.fsdp_reshard_after_forward",
        "parallelism.module_fqns_per_model_part",
        "parallelism.pipeline_parallel_degree",
        "parallelism.pipeline_parallel_first_stage_less_layers",
        "parallelism.pipeline_parallel_last_stage_less_layers",
        "parallelism.pipeline_parallel_layers_per_stage",
        "parallelism.pipeline_parallel_microbatch_size",
        "parallelism.pipeline_parallel_schedule",
        "parallelism.pipeline_parallel_schedule_csv",
        "parallelism.spmd_backend",
        "parallelism.tensor_parallel_degree",
        "profiler.enable_memory_snapshot",
        "profiler.enable_profiling",
        "profiler.memory_snapshot_freq",
        "profiler.memory_snapshot_max_entries",
        "profiler.profile_freq",
        "profiler.profiler_active",
        "profiler.profiler_repeat",
        "profiler.profiler_skip_first",
        "profiler.profiler_skip_first_wait",
        "profiler.profiler_warmup",
        "profiler.save_memory_snapshot_folder",
        "profiler.save_traces_folder",
        "training.dtype",
        "training.enable_cpu_offload",
        "training.gc_debug",
        "training.gc_freq",
        "training.global_batch_size",
        "training.local_batch_size",
        "training.max_norm",
        "training.mixed_precision_param",
        "training.mixed_precision_reduce",
        "training.seq_len",
        "training.steps",
        "validator.dataloader.dataset",
        "validator.dataloader.dataset_path",
        "validator.dataloader.infinite",
        "validator.dataloader.num_workers",
        "validator.dataloader.persistent_workers",
        "validator.dataloader.pin_memory",
        "validator.dataloader.prefetch_factor",
        "validator.enable",
        "validator.freq",
        "validator.steps",
    }
)


def _is_suppressed(field_type) -> bool:
    """True when tyro.conf.Suppress hides the field from the command line."""
    return any(m is tyro.conf.Suppress for m in getattr(field_type, "__metadata__", ()))


def _cli_options(config, prefix: str = "") -> set[str]:
    """Collect the ``section.option`` names tyro exposes for a config."""
    options = set()
    for f in dataclasses.fields(config):
        if _is_suppressed(f.type):
            continue
        value = getattr(config, f.name, None)
        name = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            options |= _cli_options(value, f"{name}.")
        else:
            options.add(name)
    return options


class TestCliOptionsFrozen(unittest.TestCase):
    def test_no_new_options(self):
        current = _cli_options(llama3_debugmodel())

        added = sorted(current - _FROZEN_CLI_OPTIONS)
        self.assertFalse(
            added, f"The command-line options are frozen, but this adds {added}."
        )

    def test_model_config_tree_is_off_the_cli(self):
        """The escape hatch the freeze depends on."""
        model_spec_field = next(
            f for f in dataclasses.fields(llama3_debugmodel()) if f.name == "model_spec"
        )
        self.assertTrue(
            _is_suppressed(model_spec_field.type),
            "Trainer.Config.model_spec must stay tyro.conf.Suppress: it is "
            "what keeps the model config tree off the command line, and "
            "therefore what makes the frozen CLI workable.",
        )


if __name__ == "__main__":
    unittest.main()
