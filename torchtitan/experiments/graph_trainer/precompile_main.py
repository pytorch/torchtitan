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

from torchtitan.config import ConfigManager
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
from torchtitan.trainer import Trainer


class PrecompileFakeTrainer(Trainer):
    """Lightweight trainer for single-process precompilation.

    Subclasses Trainer to inherit the model build/parallelize/init path
    but uses a fake distributed backend with compile-on-one-rank (CooR)
    instead of real multi-GPU communication.  No optimizer, dataloader,
    checkpointer, or training loop are created.
    """

    def __init__(self, config):
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

        self.compile_config = compile_config
        self._tokenizer_lazy = None

        super().__init__(config)

    # -- Overrides for device/distributed/determinism --

    def _init_device(self) -> None:
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def init_distributed(self) -> ParallelDims:
        parallelism = self.config.parallelism
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

        parallel_dims = ParallelDims(
            dp_shard=dp_shard,
            dp_replicate=dp_replicate,
            cp=cp,
            tp=tp,
            pp=pp,
            ep=parallelism.expert_parallel_degree,
            etp=parallelism.expert_tensor_parallel_degree,
            world_size=world_size,
        )
        parallel_dims.build_mesh()
        return parallel_dims

    def _init_determinism(self) -> None:
        # Match the deterministic mode that the training loop will use.
        # The backward graph captures use_deterministic_algorithms() at
        # compile time and asserts it matches at runtime.
        # We can't call super() here because dist_utils.set_determinism
        # does DTensor RNG seeding which fails under CooR's fake PG.
        if self.config.debug.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # -- Overrides for model build hooks --

    def _build_data_pipeline(self) -> None:
        pass

    @property
    def tokenizer(self):
        if self._tokenizer_lazy is None:
            self._tokenizer_lazy = self.config.tokenizer.build(
                tokenizer_path=self.config.hf_assets_path
            )
        return self._tokenizer_lazy

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer_lazy = value

    def _get_compile_config_for_parallelize(self):
        # For aot_fx_trace, apply_compile inside parallelize_fn is a no-op
        # (returns model unchanged), so we pass the real compile_config.
        # This lets side effects from parallelize_fn (e.g. apply_graph_ac
        # adding "apply_sac" to joint_passes) be visible to
        # compute_config_fingerprint later without needing a manual hack.
        # For aot, apply_compile would try to load a non-existent artifact,
        # so we suppress it with a copy that has enable=False.  This
        # complexity goes away once we complete the migration to
        # aot_fx_trace and remove the aot code path.
        if self.compile_config.mode == "aot":
            return dataclasses.replace(self.compile_config, enable=False)
        return self.compile_config

    def _init_model_weights(
        self,
        model: torch.nn.Module,
        init_device: str,
        buffer_device: torch.device | None,
    ) -> None:
        # CooR must be disabled during init_weights because DTensor RNG ops
        # (weight initialization seeding) raise NotImplementedError under
        # compile_on_one_rank=True.  Re-enable for the tracing phase after.
        import torch.distributed.config as dist_config

        model.to_empty(device=init_device)
        dist_config.compile_on_one_rank = False
        try:
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
        finally:
            dist_config.compile_on_one_rank = True
        model.train()

    # -- Override to skip training infrastructure --

    def _build_training_infrastructure(self) -> None:
        self.model = self.model_parts[0]

    # -- Precompile-specific methods --

    def _make_dummy_inputs(self):
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
        if self.compile_config.mode == "aot":
            self._precompile_aot()
        elif self.compile_config.mode == "aot_fx_trace":
            self._precompile_aot_fx_trace()
        else:
            raise AssertionError(f"unexpected mode: {self.compile_config.mode}")

    def _precompile_aot(self):
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
        from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
            precompile_fx_trace_save,
        )
        from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step

        config = self.config
        compile_config = self.compile_config
        model = self.model
        parallel_dims = self.parallel_dims

        fwd_bwd_fn = make_fwd_bwd_step(self.loss_fn)

        dummy_inputs, dummy_labels = self._make_dummy_inputs()

        # The trainer computes global_valid_tokens via dist_sum over the
        # "batch" mesh (dp_shard * dp_replicate), using local_valid_tokens
        # counted from full sequences BEFORE CP sharding.  So CP must NOT
        # be included here — the batch mesh doesn't cover CP ranks.
        global_batch_size = (
            config.training.local_batch_size
            * parallel_dims.dp_shard
            * parallel_dims.dp_replicate
        )
        dummy_global_valid_tokens = float(global_batch_size * config.training.seq_len)
        extra_inputs: dict[str, torch.Tensor] = {}
        extra_kwargs: dict[str, Any] = {}

        if isinstance(self.model_config, Decoder.Config) and self.model_config.layers:
            attn_config = self.model_config.layers[0].attention
            mask_type = getattr(attn_config, "mask_type", "causal")

            if mask_type == "block_causal" or parallel_dims.cp_enabled:
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
                    extra_kwargs["attention_masks"] = cast(
                        Decoder, model
                    ).get_attention_masks(
                        input_batch=dummy_inputs,
                        tokenizer=self.tokenizer,
                        extra_inputs=extra_inputs,
                    )

        if parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import (
                prepare_context_parallel_input,
            )

            dummy_inputs, dummy_labels, extra_kwargs = prepare_context_parallel_input(
                dummy_inputs,
                dummy_labels,
                extra_kwargs,
                parallel_dims.get_mesh("cp"),
                self.device,
                config.parallelism.context_parallel_load_balancer,
            )

        # Enable loss_parallel when TP is active and loss_parallel is not
        # disabled.  This matches the training path which wraps tracing +
        # execution inside train_context() → loss_parallel().  Without it,
        # cross_entropy fails with "mixed torch.Tensor and DTensor" because
        # the TP-parallelized model outputs Shard'd DTensors but labels
        # remain plain tensors.
        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        loss_parallel_ctx = (
            torch.distributed.tensor.parallel.loss_parallel()
            if loss_parallel_enabled
            else contextlib.nullcontext()
        )

        logger.info("Tracing fwd+loss+bwd via make_fx...")
        with loss_parallel_ctx:
            traced_result = trace_train_step(fwd_bwd_fn)(
                model,
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

        passes = compile_time_passes(traced_result, config)
        traced_result.gm = apply_graph_passes(
            traced_result.gm, traced_result.example_inputs, passes
        )
        logger.info(
            f"Applied {len(passes)} precompile graph passes, "
            f"graph now has {len(list(traced_result.gm.graph.nodes))} nodes"
        )

        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
        config_fingerprint = compute_config_fingerprint(
            model, compile_config, parallel_dims
        )

        precompile_fx_trace_save(
            traced_result,
            storage,
            config_fingerprint=config_fingerprint,
        )

        logger.info(
            f"Precompile complete. Artifact saved to "
            f"{compile_config.precompile_artifact_dir}/{_FX_TRACE_ARTIFACT_KEY}.bin"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        super().close()
        dist.destroy_process_group()


def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    with PrecompileFakeTrainer(config) as trainer:
        trainer.precompile()


if __name__ == "__main__":
    main()
