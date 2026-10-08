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

Usage:
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module torchtitan_recipes.tests.graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel
"""

import contextlib
import copy
import logging

import torch
import torch.distributed as dist

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import apply_overrides, ConfigLoader, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    validate_memory_policy_config,
)
from torchtitan.experiments.graph_trainer.precompile import (
    _FX_TRACE_ARTIFACT_KEY,
    _register_coor_ops,
    _SCHEDULED_FWD_BWD_ARTIFACT_KEY,
)
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.observability.logging import init_logger
from torchtitan.tools import utils


logger = logging.getLogger(__name__)


def _num_spmd_microbatches(config, parallelism_context: ParallelismContext) -> int:
    """Resolve the PP1 accumulation count using Trainer's token contract."""
    if parallelism_context.pp_enabled:
        return 1
    num_tokens_per_microbatch = (
        config.training.num_tokens_per_microbatch_per_dp_rank
        * parallelism_context.dp_replicate
        * parallelism_context.dp_shard
    )
    num_tokens_per_train_step = config.training.num_tokens_per_train_step
    if num_tokens_per_train_step < 0:
        return 1
    if num_tokens_per_train_step % num_tokens_per_microbatch != 0:
        raise ValueError(
            "training.num_tokens_per_train_step "
            f"({num_tokens_per_train_step}) must be divisible by the number "
            "of tokens processed globally in one PP1 microbatch "
            f"({num_tokens_per_microbatch})."
        )
    return num_tokens_per_train_step // num_tokens_per_microbatch


def _common_setup(config):
    """Common setup for precompile: fake PG, CooR, model build."""
    compile_config = config.compile

    if not compile_config.precompile_artifact_dir:
        raise ValueError(
            "precompile_main requires compile.precompile_artifact_dir in the recipe."
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
            "parallelism.data_parallel_shard_degree (not -1) in the recipe. "
            "It must match the value used during torchrun training."
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
    _register_coor_ops()

    # Match the deterministic mode that the training loop will use.
    # The backward graph captures use_deterministic_algorithms() at
    # compile time and asserts it matches at runtime.
    if config.debug.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parallelism_context = ParallelismContext(
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=parallelism.expert_parallel_degree,
        world_size=world_size,
        enable_sequence_parallel=parallelism.enable_sequence_parallel,
    )
    parallelism_context.build_mesh()

    # TODO: Factor the model setup below with the training path so precompile
    # and training share a single implementation of build/parallelize/init.
    model_config = copy.deepcopy(config.model)
    model_config.set_sharding_(config.parallelism)
    config.model = model_config
    if config.override.imports:
        apply_overrides(config.override, config)
    model_config = config.model
    logger.info(f"Building {type(model_config).__qualname__} on meta device")
    with (
        parallelism_context.activate_spmd(),
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    # For aot_fx_trace, apply_compile inside model.parallelize is a no-op
    # (returns model unchanged), so we pass the real compile_config.
    model = model.parallelize(
        parallelism_context=parallelism_context,
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
        compile_config,
        parallelism_context,
        device,
        tokenizer,
    )


def _prepare_loss_for_precompile(model, loss_fn) -> None:
    """Match Trainer's post-parallelization loss setup for precompile tracing."""
    if not isinstance(loss_fn, ChunkedLossWrapper):
        return

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Model must have lm_head for ChunkedLossWrapper precompile")

    loss_fn.set_lm_head(lm_head)
    model._skip_lm_head = True


def _build_precompile_inputs(
    config,
    model,
    parallelism_context,
    device,
    tokenizer,
):
    """Build representative graph inputs through the configured data path."""
    num_microbatches = _num_spmd_microbatches(config, parallelism_context)
    dataloader = config.dataloader.build(
        dp_world_size=(parallelism_context.dp_replicate * parallelism_context.dp_shard),
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=config.training.max_context_length,
        num_tokens_per_microbatch=(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
    )
    try:
        data_iterator = iter(dataloader)
        microbatches = [next(data_iterator) for _ in range(num_microbatches)]
    finally:
        dataloader.close()

    local_loss_token_counts = torch.zeros_like(microbatches[0].loss_token_counts)
    local_routing_token_counts = torch.zeros_like(microbatches[0].routing_token_counts)
    for microbatch in microbatches:
        local_loss_token_counts.add_(microbatch.loss_token_counts)
        local_routing_token_counts.add_(microbatch.routing_token_counts)

    # These representative rank-0 counts establish trace-time shape and dtype.
    # The actual global counts remain runtime graph inputs.
    dp_degree = parallelism_context.dp_replicate * parallelism_context.dp_shard
    global_loss_token_counts = local_loss_token_counts.to(device) * dp_degree
    input_dict = microbatches[0].to_input_dict(device, non_blocking=True)
    with parallelism_context.activate_spmd():
        inputs, labels, extra_kwargs = model.preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=config.parallelism,
            max_num_documents=config.dataloader.max_num_documents,
            max_context_length=config.training.max_context_length,
        )
    if "aux_loss_denominators" in extra_kwargs:
        extra_kwargs["aux_loss_denominators"] = (
            local_routing_token_counts.to(device) * dp_degree
        )
    return inputs, labels, global_loss_token_counts, extra_kwargs


def _precompile_aot_fx_trace(
    config,
    model,
    compile_config,
    parallelism_context,
    device,
    tokenizer,
):
    """aot_fx_trace mode precompilation: make_fx tracing + Inductor."""
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
    from torchtitan.experiments.graph_trainer.precompile import (
        compute_config_fingerprint,
        get_spmd_precompile_meshes,
        precompile_fx_trace_save,
        precompile_scheduled_fwd_bwd_save,
    )
    from torchtitan.experiments.graph_trainer.spmd_graph_builder import (
        make_fwd_bwd_step,
    )

    loss_fn = config.loss.build()
    _prepare_loss_for_precompile(model, loss_fn)

    fwd_bwd_fn = make_fwd_bwd_step(model, loss_fn)

    if parallelism_context.cp_enabled:
        raise NotImplementedError(
            "CooR precompile does not yet support context parallelism. "
            "Set parallelism.context_parallel_degree=1."
        )

    (
        dummy_inputs,
        dummy_labels,
        dummy_global_loss_token_counts,
        extra_kwargs,
    ) = _build_precompile_inputs(
        config,
        model,
        parallelism_context,
        device,
        tokenizer,
    )

    loss_parallel_ctx = (
        # TODO(bobrenjc93): Migrate graph trainer to the manual loss-parallel
        # custom autograd function and remove this DTensor context manager.
        torch.distributed.tensor.parallel.loss_parallel()
        if parallelism_context.tp_enabled
        else contextlib.nullcontext()
    )

    maybe_register_blockmask_pytree_node()

    precompile_meshes = get_spmd_precompile_meshes(parallelism_context)
    logger.info("Tracing fwd+loss+bwd via make_fx...")
    with parallelism_context.activate_spmd(), loss_parallel_ctx:
        traced_result = minimal_fx_tracer(
            fwd_bwd_fn,
            module=model,
            precompile_meshes=precompile_meshes,
        )(dummy_inputs, dummy_labels, dummy_global_loss_token_counts, extra_kwargs)
    logger.info(
        f"Traced graph has {len(list(traced_result.gm.graph.nodes))} nodes, "
        f"{len(traced_result.state_fqns)} state entries"
    )

    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    config_fingerprint = compute_config_fingerprint(
        model, compile_config, parallelism_context
    )
    num_microbatches = _num_spmd_microbatches(config, parallelism_context)
    if num_microbatches > 1:
        from torchtitan.experiments.graph_trainer.graph_pp.pipeline import (
            resolve_graph_execution_plan,
        )
        from torchtitan.experiments.graph_trainer.graph_pp.stage import (
            GraphPipelineStage,
        )
        from torchtitan.experiments.graph_trainer.spmd_gradient_accumulation_graph_builder import (
            _build_scheduled_fwd_bwd_graphs,
        )

        plan = resolve_graph_execution_plan(
            compile_config,
            num_microbatches=num_microbatches,
            parallelism=config.parallelism,
            pp_enabled=False,
            fsdp_enabled=parallelism_context.fsdp_enabled,
        )
        pp_mesh = parallelism_context.get_optional_mesh(
            "pp", include_singleton_axes=True
        )
        assert pp_mesh is not None
        stage = GraphPipelineStage(
            model,
            stage_index=0,
            num_stages=1,
            device=device,
            group=pp_mesh.get_group("pp"),
        )
        num_param_grads = sum(
            parameter.requires_grad
            for _, parameter in model.named_parameters(remove_duplicate=False)
        )
        stage_graphs = _build_scheduled_fwd_bwd_graphs(
            stage,
            traced_result,
            trainer_config=config,
            plan=plan,
            num_param_grads=num_param_grads,
        )
        precompile_scheduled_fwd_bwd_save(
            stage_graphs,
            storage,
            num_runtime_mesh_inputs=len(precompile_meshes),
            config_fingerprint=config_fingerprint,
            execution_plan=plan,
        )
        logger.info(
            "Precompile complete. Artifact saved to %s/%s.bin",
            compile_config.precompile_artifact_dir,
            _SCHEDULED_FWD_BWD_ARTIFACT_KEY,
        )
        return

    # Apply precompile-time graph passes (cleanup + regional_inductor)
    # so compiled Triton kernels are baked into the serialized artifact.
    # Opt-in passes from cuda_graph.py must be applied by the caller at runtime.
    from torchtitan.experiments.graph_trainer.passes import (
        apply_graph_passes,
        compile_time_passes,
    )

    passes = compile_time_passes(
        traced_result, config, parallelism_context=parallelism_context
    )

    traced_result.gm = apply_graph_passes(
        traced_result.gm, traced_result.example_inputs, passes
    )
    logger.info(
        f"Applied {len(passes)} precompile graph passes, "
        f"graph now has {len(list(traced_result.gm.graph.nodes))} nodes"
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


def main():
    init_logger()
    config = ConfigLoader().load()

    (
        model,
        model_config,
        compile_config,
        parallelism_context,
        device,
        tokenizer,
    ) = _common_setup(config)
    validate_memory_policy_config(compile_config)

    dist_moe_runtime = None
    if config.dist_moe is not None:
        dist_moe_runtime = config.dist_moe.build(
            model_parts=[model],
            parallelism_context=parallelism_context,
            device=device,
            num_tokens_per_microbatch_per_dp_rank=(
                config.training.num_tokens_per_microbatch_per_dp_rank
            ),
            pp_schedule=None,
            set_forward_context=None,
            functional_wgrad_dtype=TORCH_DTYPE_MAP[
                config.training.mixed_precision_param
            ],
        )
    try:
        _precompile_aot_fx_trace(
            config,
            model,
            compile_config,
            parallelism_context,
            device,
            tokenizer,
        )
    finally:
        if dist_moe_runtime is not None:
            dist_moe_runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
