# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and
# various training techniques (e.g. activation checkpointing and compile) to the
# Muse Glimmer model.

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, validate_config
from torchtitan.tools.logging import logger

from .model import MuseGlimmerModel


def parallelize_muse_glimmer(
    model: MuseGlimmerModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    if parallelism.spmd_backend != "spmd_types":
        raise NotImplementedError(
            "Muse Glimmer only supports spmd_backend='spmd_types'; "
            f"got '{parallelism.spmd_backend}'."
        )

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # When the model owns the vision stack (multimodal flavor), the encoder +
    # adapter are submodules: TP is applied by ``model.parallelize`` (driven by
    # the sharding configs set in update_from_config), and AC/compile/FSDP are
    # applied to them explicitly below, mirroring qwen3_5's parallelize_qwen3_5.
    has_vision = model.vision_encoder is not None
    if has_vision:
        assert model.vision_adapter is not None
        if parallel_dims.cp_enabled:
            raise NotImplementedError(
                "context parallel is not supported for the Muse Glimmer vision encoder."
            )
        if parallel_dims.tp_enabled:
            # pyrefly: ignore [missing-attribute]
            vision_num_heads = model.vision_encoder.num_heads
            assert vision_num_heads % parallel_dims.tp == 0, (
                f"vision num_heads ({vision_num_heads}) must be "
                f"divisible by TP degree ({parallel_dims.tp})"
            )

    validate_config(parallel_dims, model)
    model.parallelize(parallel_dims)
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=dump_folder)
        ac_policy.apply(model)
        if has_vision:
            # The vision encoder's block stack is named ``layers`` (like
            # qwen3_5), so the policy applies directly. The adapter is a 2-layer
            # MLP with no transformer-block structure, so it is left unwrapped.
            ac_policy.apply(model.vision_encoder)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )
        if has_vision:
            apply_compile(
                model.vision_encoder,  # pyrefly: ignore [bad-argument-type]
                compile_config=compile_config,
                parallel_dims=parallel_dims,
            )
            apply_compile(
                model.vision_adapter,  # pyrefly: ignore [bad-argument-type]
                compile_config=compile_config,
                parallel_dims=parallel_dims,
            )

    # Skip FSDP wrapper for inference. FSDP's forward hooks
    # are incompatible with torch.inference_mode() used by vLLM.
    # AC and compile are disabled via config (mode="none", enable=False).
    if skip_dp:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)

    # FSDP the vision encoder + adapter as single units BEFORE the decoder
    # (qwen3_5 documents this ordering): one AllGather per module is cheaper than
    # per-layer sharding for the (relatively small) vision stack.
    if has_vision:
        param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
        reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
        for module in (model.vision_encoder, model.vision_adapter):
            apply_fsdp_to_vision_encoder(
                module,  # pyrefly: ignore [bad-argument-type]
                dp_mesh,
                param_dtype,
                reduce_dtype,
                reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
                pp_enabled=parallel_dims.pp_enabled,
                dp_mesh_dims=dp_mesh_dims,
            )

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def pipeline_muse_glimmer(
    model: MuseGlimmerModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    model_config,
    **kwargs,
):
    """PP wrapper that assigns the owned vision stack to the first pipeline stage.

    Delegates to ``pipeline_llm`` after injecting the multimodal modules into the
    first stage's FQN list. The auto-generated LLM split only knows about
    ``tok_embeddings`` + decoder layers + ``norm``/``lm_head``; it does not model
    Muse Glimmer's vision modules, so without this they would be pruned to ``None`` on
    every stage.

    Muse Glimmer's owned vision stack runs inside ``MuseGlimmerModel.forward`` on the embedding
    stage (where ``tok_embeddings`` lives): the encoder + adapter encode raw
    images, and ``vision_projection`` + ``perception_emb_norm`` scatter the result
    into the token embeddings. All present vision modules must therefore live on
    stage 0. (For the standalone-encoder flavor only ``vision_projection`` +
    ``perception_emb_norm`` exist; the per-module presence check handles that.)
    """
    import dataclasses

    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
        _get_pipeline_metadata,
        pipeline_llm,
    )

    # NOTE: We cannot delegate to the generic ``pipeline_vlm`` here. That helper
    # only injects a single ``vision_encoder`` FQN into stage 0; Muse Glimmer owns
    # a multi-module vision stack (vision_encoder, vision_adapter,
    # vision_projection, perception_emb_norm) whose membership varies by flavor
    # (the standalone-encoder flavor has only vision_projection +
    # perception_emb_norm). We therefore replicate ``pipeline_vlm``'s structure but
    # inject the full, per-module presence-checked stack instead.
    if parallelism.module_fqns_per_model_part is None:
        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)
        fqn_per_part = _generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
        # The owned vision stack lives on the first stage alongside
        # tok_embeddings. This adds load to stage 0 that the auto split does not
        # model (input_weight only accounts for tok_embeddings); for a heavy
        # vision encoder, bump
        # parallelism.pipeline_parallel_first_stage_less_layers to rebalance.
        # Prepend in data-flow order so the resulting stage-0 list reads
        # encoder -> adapter -> projection -> emb_norm -> tok_embeddings.
        vision_fqns = [
            fqn
            for fqn in (
                "vision_encoder",
                "vision_adapter",
                "vision_projection",
                "perception_emb_norm",
            )
            if getattr(model, fqn, None) is not None
        ]
        fqn_per_part[0][:0] = vision_fqns
        parallelism = dataclasses.replace(
            parallelism, module_fqns_per_model_part=fqn_per_part
        )

    return pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        model_config=model_config,
        **kwargs,
    )
