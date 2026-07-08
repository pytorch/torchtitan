# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

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
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.tools.logging import logger


def _wrap_flex_kernel_cp(model: nn.Module, cp_mesh: DeviceMesh) -> None:
    """All-gather k/v across the CP axis inside each flex kernel forward.

    q/k/v reach the flex kernel seq-sharded on the CP axis (tensor dim 2 of the
    ``(b, heads, seq, dim)`` layout). Attention needs full-length k/v, so we
    all-gather them across the CP mesh with a funcol collective (autograd-aware:
    the backward reduce-scatters gradients back to the local shard). q stays
    sharded, so each rank computes attention for its seq shard against the full
    keys -- the BlockMask is Q-sharded / KV-full to match.

    This is the explicit-collective analogue of Titan's ``flex_cp_allgather``
    path: the kernel runs nested inside the attention module's local_map region
    where the CP mesh dim is no longer visible to a declarative redistribute, so
    the gather is done here on local tensors. Called before ``model.parallelize``
    so the wrap is captured inside the local_map wrapping.
    """
    import torch.distributed as dist
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        flex_cp_allgather,
    )

    pg_name = dist._get_process_group_name(cp_mesh.get_group())
    seq_dim = 2  # (b, heads, seq, dim)

    for module in model.modules():
        kernel = getattr(module, "_titan_flex_kernel", None)
        if kernel is None:
            continue

        def _make_cp_forward(orig_forward):
            def cp_forward(query, key, value, **kwargs):
                key = key.contiguous()
                value = value.contiguous()
                global_key, global_value = flex_cp_allgather(
                    key, value, seq_dim, pg_name
                )
                return orig_forward(query, global_key, global_value, **kwargs)

            return cp_forward

        kernel.forward = _make_cp_forward(kernel.forward)


# ---------------------------------------------------------------------------
# Main parallelization entry point
# ---------------------------------------------------------------------------


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply parallelism to the HF model using the titan Module protocol.

    Flow:
    1. Build and swap Titan MoE modules (sets _sharding_config on MoE tree)
    2. Convert all remaining HF nn.Modules to Module protocol via __class__ swap
    3. Set ShardingConfig on every module based on its role
    4. Single model.parallelize(parallel_dims) call — shards states, wraps forward
    5. Apply AC, compile, FSDP as usual
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # Flex attention supports FSDP, TP, and context parallelism. Under CP the
    # flex kernel's local_map redistributes k/v from seq-sharded to CP-Replicate
    # (all-gather); see _attach_flex_kernel in hf_sharding.py. The CP-sharded
    # BlockMask is built and sharded on its Q axis upstream (trainer, ptrr
    # balancer). PP is not yet wired for flex+CP.
    use_flex = getattr(model.model.config, "use_flex_attn", False)
    if use_flex and parallel_dims.cp_enabled and parallel_dims.pp_enabled:
        raise NotImplementedError(
            "use_flex_attn=True with context parallelism is not yet supported "
            "together with pipeline parallelism in the HF transformers backend."
        )

    # 0. Un-tie embedding/lm_head weights for FSDP compatibility.
    # Some models (Gemma4) share the embedding and lm_head weight
    # (tie_word_embeddings=True). FSDP2 cannot handle parameters shared
    # across FSDP groups. Un-tying here creates an independent copy for
    # the lm_head so both can be sharded separately. Since we train from
    # scratch, the un-tied weights will diverge during training (which is
    # expected — the model learns independent embedding and output layers).
    if (
        model.tok_embeddings is not None
        and model.lm_head is not None
        and any(
            p1 is p2
            for p1 in model.tok_embeddings.parameters()
            for p2 in model.lm_head.parameters()
        )
    ):
        model.lm_head.weight = nn.Parameter(
            model.lm_head.weight.clone(),
            requires_grad=model.lm_head.weight.requires_grad,
        )
        logger.info("Un-tied embedding/lm_head weights for FSDP compatibility")

    # 1. Build and swap Titan MoE (sets _sharding_config, does NOT parallelize)
    if any(getattr(b, "moe_enabled", False) for b in model.layers):
        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            build_and_swap_native_moe,
        )

        build_and_swap_native_moe(model, parallel_dims)

    # 2. Convert HF modules to Module protocol.
    # TP/EP always need it. CP-only needs it too when flex is enabled: the flex
    # kernel's local_map is what all-gathers k/v across the CP axis, and it is
    # only installed by the sharding pass below.
    needs_module_protocol = (
        parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
        or (parallel_dims.cp_enabled and use_flex)
    )
    if needs_module_protocol:
        from torchtitan.experiments.transformers_modeling_backend.hf_sharding import (
            set_hf_sharding_configs,
        )
        from torchtitan.experiments.transformers_modeling_backend.module_conversion import (
            convert_hf_to_module,
        )

        convert_hf_to_module(model)

        # 3. Set sharding configs on all non-MoE modules
        set_hf_sharding_configs(
            model,
            enable_sp=parallel_dims.tp_enabled,
            # Loss parallelism shards the lm_head output on the vocab dim (S(-1)).
            # Main removed the ParallelismConfig.disable_loss_parallel field and
            # the loss_parallel() train-context; core cross_entropy_loss now
            # detects a vocab-sharded DTensor pred and runs vocab-parallel CE
            # directly. Enable it exactly when TP is active (matches core TP
            # models, which shard the lm_head output S(-1) under TP).
            enable_loss_parallel=parallel_dims.tp_enabled,
        )

        # 3b. Under CP, wrap each flex kernel forward to all-gather k/v across
        # the CP axis (on the seq dim). Must run before model.parallelize so the
        # wrap is captured inside the local_map region and operates on the local
        # (already TP-head-sharded, CP-seq-sharded) tensors.
        if parallel_dims.cp_enabled and use_flex:
            _wrap_flex_kernel_cp(model, parallel_dims.get_mesh("cp"))

        # 4. Single parallelize call — handles TP, EP, MoE, everything
        model.parallelize(parallel_dims)

        if parallel_dims.tp_enabled:
            maybe_enable_async_tp(
                parallelism, compile_config, parallel_dims.get_mesh("tp")
            )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    # Compile after AC wrapping and before FSDP. Compile the whole transformer
    # block (including Titan MoE) via the shared core helper — the previous
    # MoE-only ``apply_compile_sparse`` workaround is obsolete now that
    # whole-block MoE compile works (pytorch/torchtitan#3409 fixed upstream).
    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh_dim_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )

    edp_mesh_names = (
        ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
    )
    edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        parallel_dims.get_mesh(dp_mesh_dim_names),
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        ep_degree=parallel_dims.ep,
        dp_mod_ep_mesh=edp_mesh,
    )

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    if parallel_dims.cp_enabled:
        model.set_cp_mesh(parallel_dims.get_mesh("cp"))
        logger.info("Applied Context Parallel to the model")

    return model


# ---------------------------------------------------------------------------
# FSDP
# ---------------------------------------------------------------------------


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_mod_ep_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
    enable_symm_mem: bool = False,
):
    """Apply data parallelism (via FSDP2) to the model.

    When EP is enabled (``ep_degree > 1``), uses flat FSDP with
    ``shard_placement_fn`` to route expert params to ``dp_mod_ep_mesh``
    and other params to ``dp_mesh`` within a single ``fully_shard`` call
    per transformer block — matching Titan's approach and avoiding
    nested FSDP hooks that cause SAC op-count mismatches during recompute.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    # When input/output embeddings are tied (e.g. Qwen3), tok_embeddings and
    # lm_head share one parameter. FSDP2 forbids a parameter being managed by
    # two FSDP groups, so they must be grouped into a single unit.
    tok_emb_weight = getattr(model.tok_embeddings, "weight", None)
    lm_head_weight = getattr(model.lm_head, "weight", None)
    # Detect tying by parameter identity (the exact thing FSDP2 checks); this
    # also skips PP nn.Identity placeholders, which have no `.weight`.
    tie_word_embeddings = (
        tok_emb_weight is not None
        and lm_head_weight is not None
        and tok_emb_weight is lm_head_weight
    )

    if tie_word_embeddings:
        fully_shard(
            [
                m
                for m in (model.tok_embeddings, model.norm, model.lm_head)
                if m is not None
            ],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    elif model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers:
        if (
            hasattr(transformer_block, "moe_enabled")
            and transformer_block.moe_enabled
            and ep_degree > 1
        ):
            from torch.distributed.fsdp._fully_shard._fsdp_common import (
                FSDPMeshInfo,
                HSDPMeshInfo,
                ShardPlacementResult,
            )

            assert dp_mod_ep_mesh is not None
            moe_module = getattr(transformer_block, "mlp", None) or getattr(
                transformer_block, "feed_forward", None
            )
            experts = moe_module.experts
            expert_params = set(experts.parameters())
            num_local_experts = experts.num_experts // ep_degree

            if dp_mod_ep_mesh.size() > num_local_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            def _get_fsdp_mesh_info(mesh: DeviceMesh) -> FSDPMeshInfo:
                if mesh.ndim == 1:
                    return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
                if mesh.ndim == 2:
                    return HSDPMeshInfo(
                        mesh=mesh, replicate_mesh_dim=0, shard_mesh_dim=1
                    )
                raise ValueError(f"Expected 1D or 2D FSDP mesh, got {mesh.ndim}D mesh.")

            edp_mesh_info = _get_fsdp_mesh_info(dp_mod_ep_mesh)
            dp_mesh_info = _get_fsdp_mesh_info(dp_mesh)

            def _shard_placement_fn(
                param: nn.Parameter,
                _expert_params: set = expert_params,
                _expert_placement: Shard = expert_shard_placement,
                _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
            ) -> ShardPlacementResult:
                if param in _expert_params:
                    return ShardPlacementResult(
                        placement=_expert_placement, mesh_info=_edp_mesh_info
                    )
                return ShardPlacementResult(placement=Shard(0), mesh_info=_dp_mesh_info)

            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_shard_placement_fn,
            )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass. When
    # weights are tied, norm/lm_head are already grouped with tok_embeddings above.
    if not tie_word_embeddings and model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP.
    if ep_degree == 1:
        return

    # forward
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and model.layers is not None:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    # backward
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if (
        model.norm is not None
        and model.lm_head is not None
        and model.layers is not None
    ):
        model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
