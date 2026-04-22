# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single-process precompile entry point for graph_trainer.

Uses compile-on-one-rank (CooR) to generate a rank-agnostic compiled
artifact from a single process, which can then be loaded by all ranks
during torchrun training. This avoids the need to run torchrun with N
GPUs just for precompilation.

Supports two compile modes:
- aot: AOT joint graph export + Inductor compilation
- aot_fx_trace: make_fx tracing of fwd+loss+bwd + Inductor compilation

Usage (aot mode):
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel \
        --compile.passes full_inductor_compilation \
        --compile.joint_passes inductor_decomposition \
        --compile.precompile_artifact_dir /tmp/precompile_artifacts

Usage (aot_fx_trace mode):
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel \
        --compile.mode aot_fx_trace \
        --compile.precompile_artifact_dir /tmp/fx_trace_artifacts
"""

import contextlib
import dataclasses
import functools
from typing import Any, cast

import torch
import torch.distributed as dist

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_graph_ac,
    parallelize_inputs,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.compile import (
    _make_precompile_callback,
    _SERIALIZABLE_PASSES,
)
from torchtitan.experiments.graph_trainer.graph_utils import (
    CompiledModule,
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    joint_graph_builder,
    make_compiler_with_passes,
)
from torchtitan.experiments.graph_trainer.precompile import (
    _ARTIFACT_KEY,
    _FX_TRACE_ARTIFACT_KEY,
)
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class PrecompileFakeTrainer:
    """Lightweight trainer for single-process precompilation.

    Mirrors the model build/parallelize/init path of GraphTrainer but
    uses a fake distributed backend with compile-on-one-rank (CooR)
    instead of real multi-GPU communication.  No optimizer, dataloader,
    checkpointer, or training loop are created.
    """

    def __init__(self, config):
        self.config = config
        compile_config = config.compile

        if not compile_config.precompile_artifact_dir:
            raise ValueError(
                "PrecompileFakeTrainer requires "
                "--compile.precompile_artifact_dir to be set."
            )

        mode = compile_config.mode
        if mode not in ("aot", "aot_fx_trace"):
            raise ValueError(
                f"PrecompileFakeTrainer only supports --compile.mode aot or "
                f"aot_fx_trace, got '{mode}'."
            )

        self._init_fake_distributed(config)
        self._build_model(config, compile_config)

        self.loss_fn = self.model_spec.build_loss_fn(
            compile_config, parallel_dims=self.parallel_dims
        )
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    def _init_fake_distributed(self, config):
        """Initialize fake process group and CooR for single-process compile."""
        parallelism = config.parallelism
        dp_replicate = parallelism.data_parallel_replicate_degree
        dp_shard = parallelism.data_parallel_shard_degree
        cp = parallelism.context_parallel_degree
        tp = parallelism.tensor_parallel_degree
        pp = parallelism.pipeline_parallel_degree

        # dp_shard=-1 means "use remaining ranks" which can't be inferred
        # in single-process mode.  The compiled graph bakes in tensor shapes
        # that depend on dp_shard, so the exact value must match training.
        if dp_shard < 0:
            raise ValueError(
                "PrecompileFakeTrainer requires an explicit "
                "--parallelism.data_parallel_shard_degree (not -1). "
                "Set it to the value you will use during torchrun training."
            )
        world_size = dp_replicate * dp_shard * cp * tp * pp

        logger.info(
            f"Initializing single-process precompile with world_size={world_size}"
        )

        # rank must be 0 because --virtual-local-rank maps every torchrun rank
        # to local rank 0, so the precompiled artifact needs to match that setup.
        # Fake backend produces correct collective output shapes without real
        # communication, letting us trace distributed ops on a single process.
        dist.init_process_group("fake", rank=0, world_size=world_size)

        # CooR must be enabled globally (not just during tracing) so that the
        # parallelization phase (TP, FSDP mesh setup) also uses symbolic
        # coordinates rather than hardcoding rank-specific values.
        import torch.distributed.config as dist_config

        dist_config.compile_on_one_rank = True

        # Match the deterministic mode that the training loop will use.
        # The backward graph captures use_deterministic_algorithms() at
        # compile time and asserts it matches at runtime.
        if config.debug.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.parallel_dims = ParallelDims(
            dp_shard=dp_shard,
            dp_replicate=dp_replicate,
            cp=cp,
            tp=tp,
            pp=pp,
            ep=parallelism.expert_parallel_degree,
            etp=parallelism.expert_tensor_parallel_degree,
            world_size=world_size,
        )
        self.parallel_dims.build_mesh()

    def _build_model(self, config, compile_config):
        """Build, parallelize, and initialize the model on a single device."""
        model_spec = config.model_spec
        if model_spec is None:
            raise ValueError(
                "model_spec must be set. Pass --module to specify the model "
                "(e.g. --module graph_trainer.llama3)."
            )
        self.model_spec = model_spec

        model_config = model_spec.model
        model_config.update_from_config(trainer_config=config)
        self.model_config = model_config

        logger.info(f"Building {model_spec.name} {model_spec.flavor} on meta device")
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        model.verify_module_protocol()

        # For aot_fx_trace, apply_compile inside parallelize_fn is a no-op
        # (returns model unchanged), so we pass the real compile_config.
        # This lets side effects from parallelize_fn (e.g. apply_graph_ac
        # adding "apply_sac" to joint_passes) be visible to
        # compute_config_fingerprint later without needing a manual hack.
        # For aot, apply_compile would try to load a non-existent artifact,
        # so we suppress it with a copy that has enable=False.  This
        # complexity goes away once we complete the migration to
        # aot_fx_trace and remove the aot code path.
        if compile_config.mode == "aot":
            parallelize_compile_config = dataclasses.replace(
                compile_config, enable=False
            )
        else:
            parallelize_compile_config = compile_config

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            model_converters=config.model_converters,
            parallelism=config.parallelism,
            compile_config=parallelize_compile_config,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )

        # CooR must be disabled during init_weights because DTensor RNG ops
        # (weight initialization seeding) raise NotImplementedError under
        # compile_on_one_rank=True.  Re-enable for the tracing phase after.
        import torch.distributed.config as dist_config

        device_type = utils.device_type
        model.to_empty(device=device_type)
        dist_config.compile_on_one_rank = False
        try:
            with torch.no_grad():
                model.init_weights(buffer_device=None)
        finally:
            dist_config.compile_on_one_rank = True
        model.train()

        self.model = model
        self.compile_config = compile_config

        logger.info("Model parallelized and materialized")

    def _make_dummy_inputs(self):
        """Create dummy input and label tensors matching training shapes."""
        seq_len = self.config.training.seq_len
        local_batch_size = self.config.training.local_batch_size
        vocab_size = self.model_config.vocab_size

        dummy_inputs = torch.randint(
            0, vocab_size, (local_batch_size, seq_len), device=self.device
        )
        dummy_labels = torch.randint(
            0, vocab_size, (local_batch_size, seq_len), device=self.device
        )
        return dummy_inputs, dummy_labels

    def precompile(self):
        """Run precompilation and save the artifact to disk."""
        if self.compile_config.mode == "aot":
            self._precompile_aot()
        elif self.compile_config.mode == "aot_fx_trace":
            self._precompile_aot_fx_trace()

    def _precompile_aot(self):
        """AOT mode precompilation: joint graph export + Inductor."""
        config = self.config
        compile_config = self.compile_config

        if not (_SERIALIZABLE_PASSES & set(compile_config.passes)):
            raise ValueError(
                "PrecompileFakeTrainer requires at least one pass that produces "
                "serializable output "
                f"({', '.join(sorted(_SERIALIZABLE_PASSES))}) in --compile.passes."
            )

        # Augment compile_config with AC joint passes to match the training
        # path, which calls apply_graph_ac during parallelization.  Without
        # this the SAC pass won't run and the config fingerprint will differ.
        if config.activation_checkpoint.mode != "none":
            apply_graph_ac(compile_config, config.activation_checkpoint)

        register_blockmask_pytree_node()

        from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy

        fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            config.parallelism.fsdp_reshard_after_forward,
            self.parallel_dims.pp_enabled,
        )

        from .precompile import compute_config_fingerprint

        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
        config_fingerprint = compute_config_fingerprint(
            self.model, compile_config, self.parallel_dims
        )

        joint_custom_passes = get_joint_custom_passes_from_config(
            self.parallel_dims, compile_config, fsdp_reshard_after_forward
        )
        compiler_passes = get_compiler_passes_from_config(
            self.model, compile_config, self.parallel_dims
        )
        fw_compiler, bw_compiler = make_compiler_with_passes(
            compiler_passes, dump_folder=config.dump_folder
        )

        on_compile = _make_precompile_callback(
            self.model,
            compile_config,
            self.parallel_dims,
            storage=storage,
            config_fingerprint=config_fingerprint,
        )

        model_joint_graph_builder = functools.partial(
            joint_graph_builder,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            joint_custom_passes=joint_custom_passes,
            dump_folder=config.dump_folder,
            compile_config=compile_config,
            serializable=True,
            on_compile=on_compile,
        )

        compiled_model = CompiledModule(
            self.model,
            self.parallel_dims,
            model_joint_graph_builder,
            parallelize_inputs,
        )

        # Forward pass triggers AOT compilation; the backward graph is compiled
        # eagerly (not lazily) because serializable=True sets
        # force_non_lazy_backward_lowering=True in aot_compile_joint.
        dummy_inputs, _ = self._make_dummy_inputs()

        logger.info("Running forward pass to trigger AOT compilation...")
        compiled_model(dummy_inputs)

        logger.info(
            f"Precompile complete. Artifact saved to "
            f"{compile_config.precompile_artifact_dir}/{_ARTIFACT_KEY}.bin"
        )

    def _precompile_aot_fx_trace(self):
        """aot_fx_trace mode precompilation: make_fx tracing + Inductor."""
        from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
            precompile_fx_trace_save,
        )
        from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step

        fwd_bwd_fn = make_fwd_bwd_step(self.loss_fn)

        dummy_inputs, dummy_labels = self._make_dummy_inputs()

        # The trainer computes global_valid_tokens via dist_sum (an
        # all-reduce + .item()), which returns a Python float.  Use the
        # same type here so make_fx bakes it as a graph constant — not a
        # graph input — identical to the non-precompile runtime trace.
        global_batch_size = (
            self.config.training.local_batch_size
            * self.parallel_dims.dp_shard
            * self.parallel_dims.dp_replicate
            * self.parallel_dims.cp
        )
        dummy_global_valid_tokens = float(
            global_batch_size * self.config.training.seq_len
        )
        extra_inputs: dict[str, torch.Tensor] = {}
        extra_kwargs: dict[str, Any] = {}

        if isinstance(self.model_config, Decoder.Config) and self.model_config.layers:
            attn_config = self.model_config.layers[0].attention
            mask_type = getattr(attn_config, "mask_type", "causal")

            if mask_type == "block_causal" or self.parallel_dims.cp_enabled:
                extra_kwargs["positions"] = torch.arange(
                    0,
                    dummy_inputs.shape[1],
                    dtype=torch.int32,
                    device=dummy_inputs.device,
                ).expand(dummy_inputs.shape)

            inner_attention = getattr(attn_config, "inner_attention", None)
            if inner_attention is not None:
                from torchtitan.models.common.attention import (
                    FlexAttention,
                    VarlenAttention,
                )

                if isinstance(
                    inner_attention,
                    (FlexAttention.Config, VarlenAttention.Config),
                ):
                    assert (
                        self.tokenizer is not None
                    ), "tokenizer is required for flex/varlen attention"
                    extra_kwargs["attention_masks"] = cast(
                        Decoder, self.model
                    ).get_attention_masks(
                        input_batch=dummy_inputs,
                        tokenizer=self.tokenizer,
                        extra_inputs=extra_inputs or {},
                    )

        # TODO: Add CP support — call prepare_context_parallel_input here
        # to shard dummy_inputs/dummy_labels/extra_kwargs along the sequence
        # dimension, matching the trainer's post_dataloading_process.
        if self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "CooR precompile does not yet support context parallelism. "
                "Set --parallelism.context_parallel_degree 1."
            )

        # Enable loss_parallel when TP is active and loss_parallel is not
        # disabled.  This matches the training path which wraps tracing +
        # execution inside train_context() → loss_parallel().  Without it,
        # cross_entropy fails with "mixed torch.Tensor and DTensor" because
        # the TP-parallelized model outputs Shard'd DTensors but labels
        # remain plain tensors.
        loss_parallel_enabled = (
            self.parallel_dims.tp_enabled
            and not self.config.parallelism.disable_loss_parallel
        )
        loss_parallel_ctx = (
            torch.distributed.tensor.parallel.loss_parallel()
            if loss_parallel_enabled
            else contextlib.nullcontext()
        )

        logger.info("Tracing fwd+loss+bwd via make_fx...")
        with loss_parallel_ctx:
            traced_result = trace_train_step(fwd_bwd_fn)(
                self.model,
                dummy_inputs,
                dummy_labels,
                dummy_global_valid_tokens,
                extra_inputs,
                extra_kwargs,
            )
        logger.info(
            f"Traced graph has {len(list(traced_result.gm.graph.nodes))} nodes, "
            f"{len(traced_result.state_fqns)} state entries"
        )

        # Apply precompile-time graph passes (cleanup + regional_inductor)
        # so compiled Triton kernels are baked into the serialized artifact.
        # cudagraph is excluded — it runs at load time on each rank.
        from torchtitan.experiments.graph_trainer.passes import (
            apply_graph_passes,
            compile_time_passes,
        )

        passes = compile_time_passes(traced_result, self.config)
        traced_result.gm = apply_graph_passes(
            traced_result.gm, traced_result.example_inputs, passes
        )
        logger.info(
            f"Applied {len(passes)} precompile graph passes, "
            f"graph now has {len(list(traced_result.gm.graph.nodes))} nodes"
        )

        storage = DiskStorageAdapter(self.compile_config.precompile_artifact_dir)
        config_fingerprint = compute_config_fingerprint(
            self.model, self.compile_config, self.parallel_dims
        )

        precompile_fx_trace_save(
            traced_result,
            storage,
            config_fingerprint=config_fingerprint,
        )

        logger.info(
            f"Precompile complete. Artifact saved to "
            f"{self.compile_config.precompile_artifact_dir}/{_FX_TRACE_ARTIFACT_KEY}.bin"
        )

    def close(self):
        dist.destroy_process_group()


def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    trainer = PrecompileFakeTrainer(config)
    trainer.precompile()
    trainer.close()


if __name__ == "__main__":
    main()
