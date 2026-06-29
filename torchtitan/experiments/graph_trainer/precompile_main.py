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

Usage (aot_fx_trace mode):
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel \
        --compile.mode aot_fx_trace \
        --compile.precompile_artifact_dir /tmp/fx_trace_artifacts
"""

import contextlib
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree

from torchtitan.components.loss import ChunkedCELoss
from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.precompile import _FX_TRACE_ARTIFACT_KEY
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def _common_setup(config, *, materialize_whole_model: bool = True):
    """Common setup for precompile: fake PG, CooR, model build.

    When ``materialize_whole_model`` is False (the GraphPP/PP save path), the
    whole-model parallelize and materialize step is skipped and the raw model
    is returned on the meta device. The PP producer then splits the model into
    stages and parallelizes/materializes each stage submodule separately.
    """
    compile_config = config.compile

    if not compile_config.precompile_artifact_dir:
        raise ValueError(
            "precompile_main requires --compile.precompile_artifact_dir to be set."
        )

    parallelism = config.parallelism
    dp_replicate = parallelism.data_parallel_replicate_degree
    dp_shard = parallelism.data_parallel_shard_degree
    cp = parallelism.context_parallel_degree
    tp = parallelism.tensor_parallel_degree
    pp = parallelism.pipeline_parallel_degree

    # dp_shard=-1 means "use remaining ranks" which can't be inferred
    # in single-process mode. The compiled graph bakes in tensor shapes
    # that depend on dp_shard, so the exact value must match training.
    if dp_shard < 0:
        raise ValueError(
            "precompile_main requires an explicit "
            "--parallelism.data_parallel_shard_degree (not -1). "
            "Set it to the value you will use during torchrun training."
        )
    world_size = dp_replicate * dp_shard * cp * tp * pp

    logger.info(f"Initializing single-process precompile with world_size={world_size}")

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

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parallel_dims = ParallelDims(
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=parallelism.expert_parallel_degree,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError(
            "model_spec must be set. Pass --module to specify the model "
            "(e.g. --module graph_trainer.llama3)."
        )

    # TODO: Factor the model setup below with the training path so precompile
    # and training share a single implementation of build/parallelize/init.
    model_config = model_spec.model
    model_config.update_from_config(config=config)

    logger.info(f"Building {model_spec.name} {model_spec.flavor} on meta device")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    model.verify_module_protocol()

    if materialize_whole_model:
        # For aot_fx_trace, apply_compile inside parallelize_fn is a no-op
        # (returns model unchanged), so we pass the real compile_config.
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )

        # CooR must be disabled during init_weights because DTensor RNG ops
        # (weight initialization seeding) raise NotImplementedError under
        # compile_on_one_rank=True. Re-enable for the tracing phase after.
        device_type = utils.device_type
        model.to_empty(device=device_type)
        dist_config.compile_on_one_rank = False
        try:
            with torch.no_grad():
                model.init_weights(buffer_device=None)
        finally:
            dist_config.compile_on_one_rank = True
        model.train()

        logger.info("Model parallelized and materialized")

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    return (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    )


def _prepare_loss_for_precompile(model, loss_fn) -> None:
    """Match Trainer's post-parallelization loss setup for precompile tracing."""
    if not isinstance(loss_fn, ChunkedCELoss):
        return

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Model must have lm_head for ChunkedCELoss precompile")

    loss_fn.set_lm_head(lm_head)
    model._skip_lm_head = True


def _precompile_aot_fx_trace(
    config,
    model,
    model_config,
    model_spec,
    compile_config,
    parallel_dims,
    device,
    tokenizer,
):
    """aot_fx_trace mode precompilation: make_fx tracing + Inductor."""
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
    from torchtitan.experiments.graph_trainer.precompile import (
        compute_config_fingerprint,
        precompile_fx_trace_save,
    )
    from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step

    loss_fn = config.loss.build(compile_config=compile_config)
    _prepare_loss_for_precompile(model, loss_fn)

    fwd_bwd_fn = make_fwd_bwd_step(model, loss_fn)

    seq_len = config.training.seq_len
    local_batch_size = config.training.local_batch_size
    vocab_size = model_config.vocab_size

    dummy_inputs = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    dummy_labels = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    # The trainer computes global_valid_tokens via dist_sum (an
    # all-reduce + .item()), which returns a Python float. Use the
    # same type here so make_fx bakes it as a graph constant -- not a
    # graph input -- identical to the non-precompile runtime trace. The
    # value is the resolved global_batch_size (which folds in gradient
    # accumulation) times seq_len, over the batch mesh (dp_replicate*dp_shard,
    # excluding cp), matching Trainer's runtime reduction.
    batch_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    global_batch_size = config.training.global_batch_size
    if global_batch_size < 0:
        global_batch_size = local_batch_size * batch_degree
    dummy_global_valid_tokens = float(global_batch_size * seq_len)
    extra_kwargs: dict[str, Any] = {}

    if isinstance(model_config, Decoder.Config) and model_config.layers:
        attn_config = model_config.layers[0].attention
        inner_attention = attn_config.inner_attention

        positions = torch.arange(
            0, dummy_inputs.shape[1], dtype=torch.int32, device=dummy_inputs.device
        ).expand(dummy_inputs.shape)

        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            extra_kwargs["attention_masks"] = cast(Decoder, model).get_attention_masks(
                positions=positions,
            )

        extra_kwargs["positions"] = positions

    # TODO: Add CP support — call prepare_context_parallel_input here
    # to shard dummy_inputs/dummy_labels/extra_kwargs along the sequence
    # dimension, matching the trainer's post_dataloading_process.
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "CooR precompile does not yet support context parallelism. "
            "Set --parallelism.context_parallel_degree 1."
        )

    loss_parallel_ctx = (
        # TODO(bobrenjc93): Migrate graph trainer to the manual loss-parallel
        # custom autograd function and remove this DTensor context manager.
        torch.distributed.tensor.parallel.loss_parallel()
        if parallel_dims.tp_enabled
        else contextlib.nullcontext()
    )

    maybe_register_blockmask_pytree_node()

    logger.info("Tracing fwd+loss+bwd via make_fx...")
    with loss_parallel_ctx:
        traced_result = minimal_fx_tracer(fwd_bwd_fn, module=model)(
            dummy_inputs,
            dummy_labels,
            dummy_global_valid_tokens,
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


def _precompile_graph_pp(
    config,
    model,
    model_config,
    model_spec,
    compile_config,
    parallel_dims,
    device,
):
    """GraphPP precompile: build every stage in one process and serialize per PP rank.

    Single-process CooR cannot reuse the live training build path, because eager
    PP metadata inference (``schedule._initialize_stages``) needs real P2P shape
    exchange between ranks. Instead we build *all* virtual stages in this one
    process, run their forwards in stage order to chain representative
    activations (so each non-first stage gets a real input without P2P), and
    trace/partition/compile each stage with ``_build_stage_graph_bundle``
    (Inductor kernels are baked in when ``--compile.enable``). The resulting
    callables are grouped by PP rank and saved; each torchrun rank loads the
    bundle for its own PP coordinate and runs it directly, with no
    recompilation.
    """
    import torch.distributed.config as dist_config
    from torch.distributed.pipelining.stage import _normalize_model_output_as_tuple

    from torchtitan.distributed.pipeline_parallel import (
        _build_get_mesh_callback,
        _generate_llm_fqn_per_model_part,
        _get_pipeline_metadata,
        _get_pp_rank_to_stage_indices_mapping,
        _split_module,
    )
    from torchtitan.distributed.utils import get_train_context
    from torchtitan.experiments.graph_trainer.graph_pp.precompile import (
        compute_graph_pp_fingerprint,
        ensure_schedule_precompilable,
        make_distributed_objects_deepcopy_safe,
        save_graph_pp_rank_bundle,
    )
    from torchtitan.experiments.graph_trainer.graph_pp.runner import (
        _build_stage_graph_bundle,
        GraphPipelineStage,
    )

    parallelism = config.parallelism
    pp_degree = parallelism.pipeline_parallel_degree
    pp_schedule = parallelism.pipeline_parallel_schedule

    ensure_schedule_precompilable(pp_schedule)
    # CooR lifts TorchBind ProcessGroup/DeviceMesh constants into the traced
    # graph; GraphPP partition deepcopies it, so make those singletons shareable.
    make_distributed_objects_deepcopy_safe()

    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "GraphPP precompile does not yet support context parallelism. "
            "Set --parallelism.context_parallel_degree 1."
        )

    loss_fn = config.loss.build(compile_config=compile_config)

    # Derive the module FQNs for every virtual stage across all PP ranks.
    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    num_stages = len(module_names_per_stage)

    pp_mesh = parallel_dims.get_mesh("pp")
    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    device_type = utils.device_type

    # Build, parallelize, and materialize every stage submodule. CooR must be
    # off during init_weights (DTensor RNG ops are unsupported under it) and on
    # everywhere else so parallelization and tracing stay rank-agnostic.
    stages: list[GraphPipelineStage] = []
    for stage_idx in range(num_stages):
        submod = _split_module(model, module_names_per_stage[stage_idx])
        submod = model_spec.parallelize_fn(
            submod,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )
        submod.to_empty(device=device_type)
        dist_config.compile_on_one_rank = False
        try:
            with torch.no_grad():
                submod.init_weights(buffer_device=None)
        finally:
            dist_config.compile_on_one_rank = True
        submod.train()
        stages.append(
            GraphPipelineStage(
                submod,
                stage_index=stage_idx,
                num_stages=num_stages,
                device=device,
                loss_fn=loss_fn,
                compile_config=compile_config,
                model_config=model_config,
                parallelism=parallelism,
                group=pp_mesh.get_group("pp"),
                get_mesh=get_mesh_cb,
            )
        )
    logger.info("GraphPP precompile built %s stages on a single process", num_stages)

    # Match Trainer's post-construction ChunkedCELoss wiring for the last stage:
    # the last stage's forward skips lm_head and the chunked loss applies it.
    # Without this the last-stage trace asserts on a missing lm_head.
    _prepare_loss_for_precompile(stages[-1].submod, loss_fn)

    # Representative microbatch matches the per-microbatch shape training traces
    # with (the schedule splits the local batch by pipeline_parallel_microbatch_size).
    seq_len = config.training.seq_len
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    vocab_size = model_config.vocab_size
    dummy_inputs = torch.randint(
        0, vocab_size, (microbatch_size, seq_len), device=device
    )
    dummy_labels = torch.randint(
        0, vocab_size, (microbatch_size, seq_len), device=device
    )

    # global_valid_tokens is the full-batch token count baked as a graph constant
    # (the last-stage loss divides by it per microbatch). It must equal the
    # trainer's value: resolved global_batch_size (which folds in gradient
    # accumulation) times seq_len, over the batch mesh (dp_replicate*dp_shard,
    # excluding cp), so it matches regardless of grad-accum steps.
    batch_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    global_batch_size = config.training.global_batch_size
    if global_batch_size < 0:
        global_batch_size = config.training.local_batch_size * batch_degree
    dummy_global_valid_tokens = float(global_batch_size * seq_len)
    loss_kwargs: dict[str, Any] = {"global_valid_tokens": dummy_global_valid_tokens}

    # positions and attention_masks are forwarded to every stage, matching
    # Trainer.post_dataloading_process (masks only for Flex/Varlen backends).
    positions = torch.arange(0, seq_len, dtype=torch.int32, device=device).expand(
        microbatch_size, seq_len
    )
    extra_kwargs: dict[str, Any] = {"positions": positions}
    if isinstance(model_config, Decoder.Config):
        inner_attention = getattr(model_config.first_attention, "inner_attention", None)
        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            extra_kwargs["attention_masks"] = cast(
                Decoder, stages[0].submod
            ).get_attention_masks(positions=positions)

    maybe_register_blockmask_pytree_node()

    train_context = get_train_context(parallel_dims=parallel_dims)

    # Chain forward in stage order so each stage traces with a real input; for
    # non-last stages synthesize output-grad tangents (ones_like the forward
    # output) instead of relying on eager PP backward metadata.
    with train_context():
        stage_args: tuple[Any, ...] = (dummy_inputs,)
        for stage_idx, stage in enumerate(stages):
            is_last = stage_idx == num_stages - 1
            stage_target = dummy_labels if is_last else None
            output_grads = None
            next_args = None
            if not is_last:
                with torch.no_grad():
                    output = stage.submod(*stage_args, **extra_kwargs)
                output_grads = pytree.tree_map_only(
                    torch.Tensor, torch.ones_like, output
                )
                next_args = _normalize_model_output_as_tuple(output)
            # Build (and, when --compile.enable, Inductor-compile) each callable
            # now so the saved bundle is in its final state and load needs no
            # recompilation. With enable off the saved graphs are uncompiled FX.
            _build_stage_graph_bundle(
                stage,
                stage_args,
                extra_kwargs,
                stage_target,
                loss_kwargs,
                compile_callables=True,
                output_grads=output_grads,
            )
            if not is_last:
                stage_args = next_args

    # Group stages by PP rank and serialize one bundle per rank. Each rank's
    # fingerprint covers only that rank's stage submodules, matching the load
    # side, which only has its own rank's parts.
    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    for pp_rank in range(pp_degree):
        stage_indices = _get_pp_rank_to_stage_indices_mapping(
            pp_rank, pp_degree, pp_schedule, num_stages
        )
        rank_stages = [stages[i] for i in stage_indices]
        config_fingerprint = compute_graph_pp_fingerprint(
            [stage.submod for stage in rank_stages],
            compile_config,
            parallel_dims,
            schedule_name=pp_schedule,
            parallelism=parallelism,
            training=config.training,
            loss_config=config.loss,
            debug_config=config.debug,
        )
        save_graph_pp_rank_bundle(
            rank_stages,
            storage,
            pp_rank=pp_rank,
            schedule_name=pp_schedule,
            config_fingerprint=config_fingerprint,
        )

    logger.info(
        "GraphPP precompile complete. Saved %s per-rank bundles to %s",
        pp_degree,
        compile_config.precompile_artifact_dir,
    )


def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    mode = config.compile.mode
    if mode != "aot_fx_trace":
        raise ValueError(
            f"precompile_main only supports --compile.mode aot_fx_trace, "
            f"got '{mode}'."
        )

    pp_enabled = config.parallelism.pipeline_parallel_degree > 1

    (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    ) = _common_setup(config, materialize_whole_model=not pp_enabled)

    if pp_enabled:
        _precompile_graph_pp(
            config,
            model,
            model_config,
            model_spec,
            compile_config,
            parallel_dims,
            device,
        )
    else:
        _precompile_aot_fx_trace(
            config,
            model,
            model_config,
            model_spec,
            compile_config,
            parallel_dims,
            device,
            tokenizer,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
