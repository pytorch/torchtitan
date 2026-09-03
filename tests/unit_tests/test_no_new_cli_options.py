# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ensure no more flags are added to CLI"""

import dataclasses
import importlib
import typing
import unittest
import warnings

import tyro
from torchtitan.trainer import Trainer

_FROZEN_CLI_OPTIONS = frozenset(
    {
        "activation_checkpoint.debug",
        "activation_checkpoint.determinism_check",
        "activation_checkpoint.force_recompute_mm_shapes_by_fqns",
        "activation_checkpoint.memory_budget",
        "activation_checkpoint.preserve_rng_state",
        "activation_checkpoint.visualize_memory_budget_pareto",
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
        "compile.enable_async_tensor_parallel",
        "dataloader.build_mrope_positions",
        "dataloader.dataset",
        "dataloader.dataset_path",
        "dataloader.dataset_subset",
        "dataloader.generate_timesteps",
        "dataloader.image_mean",
        "dataloader.image_std",
        "dataloader.img_size",
        "dataloader.infinite",
        "dataloader.load_dataset_kwargs",
        "dataloader.max_images_per_batch",
        "dataloader.max_patches",
        "dataloader.max_patches_per_side",
        "dataloader.max_pixels",
        "dataloader.min_pixels",
        "dataloader.num_workers",
        "dataloader.packing_buffer_size",
        "dataloader.patch_order",
        "dataloader.patch_size",
        "dataloader.persistent_workers",
        "dataloader.pin_memory",
        "dataloader.prefetch_factor",
        "dataloader.prompt_dropout_prob",
        "dataloader.seed",
        "dataloader.sources.dataset",
        "dataloader.sources.dataset_path",
        "dataloader.sources.infinite",
        "dataloader.sources.load_dataset_kwargs",
        "dataloader.sources.num_workers",
        "dataloader.sources.persistent_workers",
        "dataloader.sources.pin_memory",
        "dataloader.sources.prefetch_factor",
        "dataloader.sources.weight",
        "dataloader.spatial_merge_size",
        "dataloader.stopping_strategy",
        "dataloader.temporal_patch_size",
        "dataloader.video_dir",
        "dataloader.video_fps",
        "dataloader.video_max_frames",
        "dataloader.video_min_frames",
        "dataloader.weight",
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
        "encoder.autoencoder_path",
        "encoder.clip_encoder",
        "encoder.random_init",
        "encoder.t5_encoder",
        "hf_assets_path",
        "inference.img_size",
        "inference.local_batch_size",
        "inference.prompts_path",
        "inference.sampling.classifier_free_guidance_scale",
        "inference.sampling.denoising_steps",
        "inference.sampling.enable_classifier_free_guidance",
        "inference.save_img_folder",
        "loss.global_vocab_size",
        "loss.loss_fn.global_vocab_size",
        "loss.loss_fn.mtp_scale",
        "loss.mtp_scale",
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
        "optimizer.optimizer_factory_kwargs_by_name",
        "optimizer.param_groups",
        "optimizer.param_groups.optimizer_kwargs",
        "optimizer.param_groups.optimizer_name",
        "optimizer.param_groups.pattern",
        "override.imports",
        "parallelism.context_parallel_degree",
        "parallelism.context_parallel_load_balancer",
        "parallelism.context_parallel_ptrr_mask_key",
        "parallelism.data_parallel_replicate_degree",
        "parallelism.data_parallel_shard_degree",
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
        "tokenizer.clip_tokenizer_path",
        "tokenizer.image_token",
        "tokenizer.max_t5_encoding_len",
        "tokenizer.pad_token",
        "tokenizer.t5_tokenizer_path",
        "tokenizer.test_mode",
        "tokenizer.video_token",
        "tokenizer.vision_end_token",
        "tokenizer.vision_start_token",
        "training.disable_cuda_graphs",
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
        "validator.all_timesteps",
        "validator.dataloader.build_mrope_positions",
        "validator.dataloader.dataset",
        "validator.dataloader.dataset_path",
        "validator.dataloader.dataset_subset",
        "validator.dataloader.generate_timesteps",
        "validator.dataloader.image_mean",
        "validator.dataloader.image_std",
        "validator.dataloader.img_size",
        "validator.dataloader.infinite",
        "validator.dataloader.load_dataset_kwargs",
        "validator.dataloader.max_images_per_batch",
        "validator.dataloader.max_patches",
        "validator.dataloader.max_patches_per_side",
        "validator.dataloader.max_pixels",
        "validator.dataloader.min_pixels",
        "validator.dataloader.num_workers",
        "validator.dataloader.packing_buffer_size",
        "validator.dataloader.patch_order",
        "validator.dataloader.patch_size",
        "validator.dataloader.persistent_workers",
        "validator.dataloader.pin_memory",
        "validator.dataloader.prefetch_factor",
        "validator.dataloader.prompt_dropout_prob",
        "validator.dataloader.seed",
        "validator.dataloader.sources.dataset",
        "validator.dataloader.sources.dataset_path",
        "validator.dataloader.sources.infinite",
        "validator.dataloader.sources.load_dataset_kwargs",
        "validator.dataloader.sources.num_workers",
        "validator.dataloader.sources.persistent_workers",
        "validator.dataloader.sources.pin_memory",
        "validator.dataloader.sources.prefetch_factor",
        "validator.dataloader.sources.weight",
        "validator.dataloader.spatial_merge_size",
        "validator.dataloader.stopping_strategy",
        "validator.dataloader.temporal_patch_size",
        "validator.dataloader.video_dir",
        "validator.dataloader.video_fps",
        "validator.dataloader.video_max_frames",
        "validator.dataloader.video_min_frames",
        "validator.dataloader.weight",
        "validator.enable",
        "validator.freq",
        "validator.sampling.classifier_free_guidance_scale",
        "validator.sampling.denoising_steps",
        "validator.sampling.enable_classifier_free_guidance",
        "validator.save_img_count",
        "validator.save_img_folder",
        "validator.steps",
    }
)


def _strip_annotated(field_type):
    """The underlying type of an ``Annotated[...]``, or the type itself."""
    return typing.get_args(field_type)[0] if _is_annotated(field_type) else field_type


def _is_annotated(field_type) -> bool:
    return hasattr(field_type, "__metadata__")


def _is_suppressed(field_type) -> bool:
    """True when tyro.conf.Suppress hides the field from the command line."""
    return any(m is tyro.conf.Suppress for m in getattr(field_type, "__metadata__", ()))


def _cli_options(config, prefix: str = "") -> set[str]:
    """Collect the ``section.option`` names tyro exposes for a config.

    Walks the instance. :func:`_declared_cli_options` starts from
    ``Trainer.Config`` and never expands subclasses of that root, so a model
    that returns its own subclass -- ``FluxTrainer.Config``, with its
    ``encoder`` and ``inference`` sections -- is only reachable this way.
    """
    options = set()
    # Resolved rather than raw: a module using ``from __future__ import
    # annotations`` stores its field types as strings, which would hide the
    # Suppress annotation. checkpoint.py is one such module.
    hints = typing.get_type_hints(type(config), include_extras=True)
    for f in dataclasses.fields(config):
        field_type = hints.get(f.name, f.type)
        if _is_suppressed(field_type):
            continue
        value = getattr(config, f.name)
        name = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(value):
            options |= _cli_options(value, f"{name}.")
        elif value is None and _config_types(field_type):
            # A section switched off, such as activation_checkpoint=None. It
            # is a subcommand rather than an option, and _declared_cli_options
            # already covers what its members expose.
            continue
        else:
            options.add(name)
    return options


def _subclasses(config_cls: type) -> set[type]:
    """``config_cls`` and every imported subclass of it defined in core.

    ``__subclasses__`` sees whatever the process has imported, so an
    experiment's config subclass would otherwise appear in the snapshot for
    any test run that happened to import it first. The freeze covers core, and
    ``torchtitan/experiments`` sets its own rules.
    """
    found = {config_cls}
    for sub in config_cls.__subclasses__():
        if sub.__module__.startswith("torchtitan.experiments."):
            continue
        found |= _subclasses(sub)
    return found


def _config_types(field_type) -> set[type]:
    """The config classes a field may hold, unwrapping Annotated and generics.

    Both unions and containers expand. tyro indexes a ``list[ParamGroupConfig]``
    per element, so ``--optimizer.param-groups.0.optimizer-kwargs.lr`` is a real
    option; the index is dropped here and the element's fields are recorded once.

    Subclasses expand too, because a field declared as a component base holds
    whichever implementation the configuration picked -- ``loss.mtp_scale``
    exists only when the loss is deepseek_v3's.
    """
    field_type = _strip_annotated(field_type)
    if dataclasses.is_dataclass(field_type):
        return _subclasses(field_type)
    found: set[type] = set()
    for arg in typing.get_args(field_type):
        arg = _strip_annotated(arg)
        if dataclasses.is_dataclass(arg):
            found |= _subclasses(arg)
    return found


def _declared_cli_options(
    config_cls, prefix: str = "", seen: frozenset[type] = frozenset()
) -> set[str]:
    """Collect the option names reachable through a config class's annotations.

    Complements :func:`_cli_options`: an instance only shows the components and
    union members it happens to hold, so everything else would go unguarded.

    ``seen`` breaks the cycles that subclass expansion creates:
    ``ChunkedLossWrapper.Config.loss_fn`` is a ``BaseLoss.Config``, which
    expands back to the wrapper.
    """
    if config_cls in seen:
        return set()
    seen = seen | {config_cls}
    options = set()
    hints = typing.get_type_hints(config_cls, include_extras=True)
    for f in dataclasses.fields(config_cls):
        field_type = hints.get(f.name, f.type)
        if _is_suppressed(field_type):
            continue
        name = f"{prefix}{f.name}"
        nested = _config_types(field_type)
        if nested:
            for member in nested:
                options |= _declared_cli_options(member, f"{name}.", seen)
        else:
            options.add(name)
    return options


_GUARDED_CONFIGS = (
    ("llama3", "llama3_debugmodel"),
    ("llama3", "sft_debugmodel"),
    ("deepseek_v3", "deepseek_v3_debugmodel"),
    ("qwen3", "qwen3_debugmodel"),
    ("qwen3_5", "qwen35_debugmodel_moe"),
    ("gpt_oss", "gpt_oss_debugmodel"),
    ("flux", "flux_debugmodel"),
    ("kimi_k2_7", "kimi_k2_5_debugmodel"),
    ("muse_glimmer", "muse_glimmer_debugmodel_mm"),
)


def _guarded_configs():
    """One entry point per model, since each exposes a different surface.

    tyro derives the command line from the selected configuration, so a
    component only reachable through one model is only guarded if that model
    is walked. A model whose registry needs a dependency this environment
    lacks is skipped: the assertion is one-directional, so seeing fewer
    options can never fail the test, only cover less.
    """
    for model, config_name in _GUARDED_CONFIGS:
        try:
            registry = importlib.import_module(
                f"torchtitan.models.{model}.config_registry"
            )
        except ImportError as e:
            warnings.warn(f"freeze snapshot skips {model}: {e}", stacklevel=2)
            continue
        yield getattr(registry, config_name)()


class TestCliOptionsFrozen(unittest.TestCase):
    def test_every_model_is_guarded(self):
        """A new model must be added to _GUARDED_CONFIGS, not silently skipped."""
        from torchtitan.models import _supported_models

        self.assertEqual(
            _supported_models - {model for model, _ in _GUARDED_CONFIGS},
            set(),
            "These models expose a command-line surface that the freeze does "
            "not cover. Add one configuration each to _GUARDED_CONFIGS.",
        )

    def test_no_new_options(self):
        # Neither walk alone is the whole surface: an instance shows the
        # components it picked, the annotations show everything a declared
        # type can hold. Build the configurations first -- the declared walk
        # expands subclasses, and a subclass is only visible once the module
        # defining it has been imported.
        configs = list(_guarded_configs())
        current = _declared_cli_options(Trainer.Config)
        for config in configs:
            current |= _cli_options(config)

        added = sorted(current - _FROZEN_CLI_OPTIONS)
        self.assertFalse(
            added, f"The command-line options are frozen, but this adds {added}."
        )

    def test_model_config_tree_is_off_the_cli(self):
        """The escape hatch the freeze depends on."""
        hints = typing.get_type_hints(Trainer.Config, include_extras=True)
        self.assertTrue(
            _is_suppressed(hints["model_spec"]),
            "Trainer.Config.model_spec must stay tyro.conf.Suppress: it is "
            "what keeps the model config tree off the command line, and "
            "therefore what makes the frozen CLI workable.",
        )


if __name__ == "__main__":
    unittest.main()
