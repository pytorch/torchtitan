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


def _install_native_embedding(model: nn.Module) -> None:
    """Swap the HF token embedding for titan's vocab-parallel ``Embedding``.

    Under the spmd_types backend every op is type-checked, and
    HF's plain ``nn.Embedding.forward`` (``F.embedding`` on a vocab-sharded
    ``S(0)`` weight) triggers an implicit redistribution that the checker
    rejects ("No valid sharding strategy for embedding"). The native
    ``Embedding`` (models/common/embedding.py) does a masked local lookup on the
    local vocab shard (``weight.to_local()`` + index mask), producing a Partial
    output with no implicit redistribution -- the same construction native
    models use, and needed for TP correctness regardless. Only the class is
    swapped (weight/padding_idx preserved, state_dict keys unchanged); the
    embedding's ``_sharding_config`` is still set by set_hf_sharding_configs.
    Runs before convert_hf_to_module so it is skipped there (already a Module).
    """
    from torchtitan.models.common.embedding import Embedding as _NativeEmbedding

    emb = model.tok_embeddings
    if emb is None or isinstance(emb, nn.Identity):
        return
    if type(emb) is nn.Embedding:
        emb.__class__ = _NativeEmbedding
        emb.tp_group = None  # set by Embedding.parallelize; forward reads it
        # Native Embedding's vocab-parallel forward passes the global padding_idx
        # to F.embedding on the local vocab shard, which asserts it is < the local
        # num_embeddings -- so a non-None padding_idx (e.g. GLM-5) crashes under
        # TP. Drop it: the masked local lookup already zeros out-of-shard tokens,
        # and causal-LM padding is handled by attention/loss masking, so the
        # padding_idx row-zeroing is vestigial here.
        if emb.padding_idx is not None:
            emb.padding_idx = None
    else:
        logger.warning(
            f"tok_embeddings is {type(emb).__name__}, not nn.Embedding; leaving "
            "as-is (may not be typecheck-clean under spmd_types + TP)."
        )


def _wrap_rope_no_typecheck(model: nn.Module) -> None:
    """Run the HF rotary embedding's cos/sin computation under no_typecheck.

    HF computes cos/sin at runtime via a batched matmul (inv_freq @ position_ids).
    Under spmd_types typechecking with a DP-sharded batch -- e.g. EP or FSDP add a
    ``dp`` mesh axis over which the batch is Varying -- that bmm mixes ``R`` and
    ``V`` batch placements, which the checker rejects ("No valid sharding strategy
    for matmul on axis mesh_dp"). RoPE is a pure positional function with no
    shardable weights (inv_freq is a fixed buffer), so shield it exactly like
    native RoPE (models/common/rope.py wraps its helpers in @spmd.no_typecheck()).
    A no-op when typechecking is off. Called before ``model.parallelize`` so the
    wrap sits inside the Module-protocol forward.
    """
    import spmd_types as spmd

    rope = getattr(model, "rotary_emb", None)
    if rope is None or isinstance(rope, nn.Identity):
        return

    def _make_nc_forward(orig_forward):
        def nc_forward(*args, **kwargs):
            with spmd.no_typecheck():
                return orig_forward(*args, **kwargs)

        return nc_forward

    rope.forward = _make_nc_forward(rope.forward)


def _wrap_moe_experts_contiguous(model: nn.Module) -> None:
    """Make the native MoE experts tolerate a non-contiguous local input.

    ``RoutedExperts.forward`` flattens its input with ``x_BLD.view(T, D)``, which
    requires a contiguous tensor. Under sequence parallelism (enabled whenever
    TP>1), the attention output projection's rowwise reduce-scatter to the SP
    activation layout produces a *seq-major* (non-contiguous) local shard. This
    is NOT specific to this backend or a difference in our sharding: native
    ``GQAttention``'s wo output has the identical seq-major stride under the same
    ``rowwise_config(output_sp=True)`` reduce-scatter (verified). Native models
    just never route it into the experts ``.view``: native GQA models are dense
    (no MoE), and native MoE models use non-GQA attention (deepseek MLA, gpt_oss
    custom) whose experts input happens to land contiguous. This HF backend is
    the first to pair GQA-style attention (seq-major SP wo output) with the native
    MoE experts, so it is the first to feed that ``.view`` a non-contiguous shard.

    The layout is left as-is (it is correct SP behavior, matching native); we only
    make the experts tolerate it by wrapping ``RoutedExperts.forward`` (the
    routed-expert local_map region that owns the ``.view``) -- before
    ``model.parallelize`` so the wrap sits inside the local_map region and sees
    local tensors -- to ``.contiguous()`` the tensor args. This is a no-op view
    (no copy) when the input is already contiguous, so it only copies for the
    seq-major shard. The clean long-term fix is upstream (``RoutedExperts.forward``
    using ``.reshape`` instead of ``.view``); this is the experiment-side
    workaround until then. Guarded to spmd + TP+EP at the call site.
    """
    from torchtitan.models.common.moe import RoutedExperts

    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue

        def _make_contiguous_forward(orig_forward):
            def contiguous_forward(*args, **kwargs):
                args = tuple(a.contiguous() if torch.is_tensor(a) else a for a in args)
                return orig_forward(*args, **kwargs)

            return contiguous_forward

        module.forward = _make_contiguous_forward(module.forward)


def _wire_flex_kernel_tp(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Give GQA flex core kernels the TP group for the TP-aware flat q/k norm.

    OLMoE-style attention norms the full q/k projection (n_heads*head_dim) before
    the head reshape; under TP that projection is column-sharded, so
    ``HFFlexAttnCoreKernel`` must all-reduce the RMSNorm variance across the TP
    group (see ``_tp_aware_flat_rmsnorm``). Set the TP group on each GQA core
    kernel here (before ``model.parallelize``). A no-op for per-head-norm (Qwen3)
    and no-norm (Mixtral) models -- they never enter the flat-norm branch.
    """
    tp_group = tp_mesh.get_group()
    for module in model.modules():
        core = getattr(module, "_titan_flex_attn_core", None)
        if core is not None:
            core._tp_group = tp_group


def _wrap_flex_kernel_cp(model: nn.Module, cp_mesh: DeviceMesh) -> None:
    """Wire the CP k/v all-gather into each attention kernel.

    q/k/v reach attention seq-sharded on the CP axis. Attention needs full-length
    k/v, so k/v are all-gathered across the CP mesh with a funcol collective
    (autograd-aware: the backward reduce-scatters gradients back to the local
    shard). q stays sharded, so each rank computes attention for its seq shard
    against the full keys -- the BlockMask is Q-sharded / KV-full to match.

    Two kernel families need this, both handled here (called before
    ``model.parallelize`` so the effect is captured inside the local_map region,
    where the CP mesh dim is no longer visible to a declarative redistribute):

    * flex core kernels (``_titan_flex_attn_core`` / ``_titan_mla_flex_attn_core``,
      GQA/MLA under spmd_types): they reshape/RoPE then run flex inside their own
      local_map region, so the all-gather must happen inside that forward on
      local (already reshaped) tensors. We only hand them the CP process-group
      name here (``_cp_pg_name``); the kernel does the gather itself.
    * ``HFFlexKernel`` (``_titan_flex_kernel``, the DSA path and any CP-without-a-
      core-kernel case): it receives q/k/v already reshaped to
      ``(b, heads, seq, dim)``, so wrap its forward to gather k/v on seq dim 2.
    """
    import torch.distributed as dist
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        flex_cp_allgather,
    )

    pg_name = dist._get_process_group_name(cp_mesh.get_group())
    seq_dim = 2  # (b, heads, seq, dim)

    for module in model.modules():
        # Flex core kernels (GQA/MLA): gather happens inside their forward.
        for attr in ("_titan_flex_attn_core", "_titan_mla_flex_attn_core"):
            core = getattr(module, attr, None)
            if core is not None:
                core._cp_pg_name = pg_name

        # HFFlexKernel path (DSA): wrap the forward to gather already-reshaped k/v.
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

    # The HF transformers backend supports the legacy "default" (DTensor) backend
    # and "spmd_types". full_dtensor is a transitional core backend slated for
    # removal upstream, so it is intentionally not wired here -- fail loud rather
    # than silently falling through to the default path.
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError(
            "the HF transformers backend supports spmd_backend 'default' or "
            "'spmd_types'; full_dtensor is not supported."
        )

    # Flex attention supports FSDP, TP, CP, and PP (in any combination). Under CP
    # the flex kernel's local_map redistributes
    # k/v from seq-sharded to CP-Replicate (all-gather); see _attach_flex_kernel
    # in hf_sharding.py. The CP-sharded BlockMask is built and sharded on its Q
    # axis upstream (trainer, ptrr balancer). Note: the ptrr balancer requires
    # the number of Q blocks (seq_len / flex BLOCK_SIZE) to be divisible by the
    # CP degree; too-short sequences raise "num_tasks N must be divisible by
    # group_size" from the balancer -- this is a CP+ptrr constraint, independent
    # of PP.

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
    # TP/EP always need it. CP-only needs it too: the flex kernel's local_map is
    # what all-gathers k/v across the CP axis, and it is only installed by the
    # sharding pass below.
    # Under the spmd_types backend, FSDP requires every parameter to be an spmd
    # tensor on the SPMD mesh, which only the sharding pass + model.parallelize
    # produces -- so run it for any parallelism (even FSDP-only), not just
    # TP/EP/CP.
    use_spmd = parallelism.spmd_backend == "spmd_types"
    needs_module_protocol = (
        parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.cp_enabled
        or use_spmd
    )
    if needs_module_protocol:
        from torchtitan.experiments.transformers_modeling_backend.hf_sharding import (
            set_hf_sharding_configs,
        )
        from torchtitan.experiments.transformers_modeling_backend.module_conversion import (
            convert_hf_to_module,
        )

        # Under spmd_types, replace the HF embedding with the native
        # vocab-parallel Embedding before conversion (typecheck-clean lookup).
        if use_spmd:
            _install_native_embedding(model)

        convert_hf_to_module(model)

        # 3. Set sharding configs on all non-MoE modules
        set_hf_sharding_configs(
            model,
            enable_sp=parallel_dims.tp_enabled,
            spmd_backend=parallelism.spmd_backend,
            cp_enabled=parallel_dims.cp_enabled,
        )

        # 3b. Under CP, wrap each flex kernel forward to all-gather k/v across
        # the CP axis (on the seq dim). Must run before model.parallelize so the
        # wrap is captured inside the local_map region and operates on the local
        # (already TP-head-sharded, CP-seq-sharded) tensors.
        if parallel_dims.cp_enabled:
            _wrap_flex_kernel_cp(model, parallel_dims.get_mesh("cp"))

        # 3b'. Under TP, give GQA flex core kernels the TP group so an OLMoE-style
        # full-projection q/k RMSNorm all-reduces its variance across TP (the
        # projection is column-sharded). No-op for per-head / no-norm models.
        if parallel_dims.tp_enabled:
            _wire_flex_kernel_tp(model, parallel_dims.get_mesh("tp"))

        # 3c. Under spmd + combined TP+EP, the experts local_map yields a
        # non-contiguous local shard that native GroupedExperts.forward's .view
        # rejects; wrap the experts forward (before parallelize, so it is inside
        # the local region) to contiguous-ify its inputs.
        if use_spmd and parallel_dims.tp_enabled and parallel_dims.ep_enabled:
            _wrap_moe_experts_contiguous(model)

        # 3d. Under spmd, shield the HF rotary embedding's runtime cos/sin bmm
        # from typechecking (it mixes R/V batch placements when the batch is
        # DP-sharded, e.g. under EP/FSDP). See _wrap_rope_no_typecheck.
        if use_spmd:
            _wrap_rope_no_typecheck(model)

        # 4. Single parallelize call -- handles TP, EP, MoE, everything
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

    # Resolve the FSDP mesh. Under the spmd_types backend the DP mesh is a
    # submesh of the multi-axis SPMD storage mesh, selected via resolve_fsdp_mesh
    # + a DataParallelMeshDims that fully_shard flattens out; the legacy
    # "fsdp"/"efsdp" named axes only exist under the default backend. (The core
    # helpers live in distributed/full_dtensor.py but apply to spmd_types too.)
    dp_mesh_dims = None
    edp_mesh_dims = None
    if use_spmd:
        from torchtitan.distributed.full_dtensor import (
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        if parallel_dims.ep_enabled:
            edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
        else:
            edp_mesh = None
    else:
        dp_mesh_dim_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
        ep_degree=parallel_dims.ep,
        dp_mod_ep_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
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
    dp_mesh_dims=None,
    edp_mesh_dims=None,
):
    """Apply data parallelism (via FSDP2) to the model.

    When EP is enabled (``ep_degree > 1``), uses flat FSDP with
    ``shard_placement_fn`` to route expert params to ``dp_mod_ep_mesh``
    and other params to ``dp_mesh`` within a single ``fully_shard`` call
    per transformer block — matching Titan's approach and avoiding
    nested FSDP hooks that cause SAC op-count mismatches during recompute.

    Under the spmd_types backend ``dp_mesh``/``dp_mod_ep_mesh`` are multi-axis
    SPMD storage meshes and ``dp_mesh_dims``/``edp_mesh_dims``
    (``DataParallelMeshDims``) tell ``fully_shard`` which axes are the DP
    submesh to flatten. Under the default backend these are ``None`` and the
    meshes are already the plain DP mesh.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
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
            moe_module = getattr(transformer_block, "mlp", None)
            # Post-#3859 the grouped experts live under routed_experts.inner_experts
            # (with the token_dispatcher as a sibling under routed_experts).
            experts = moe_module.routed_experts.inner_experts
            expert_params = set(experts.parameters())
            num_local_experts = experts.num_experts // ep_degree

            if dp_mod_ep_mesh.size() > num_local_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            if dp_mesh_dims is not None:
                # spmd_types: meshes are multi-axis SPMD storage meshes; let
                # FSDP's mesh-info builder extract and flatten the DP submesh via
                # the DataParallelMeshDims (matches native).
                from torch.distributed.fsdp._fully_shard._fsdp_init import (
                    _get_mesh_info,
                )

                edp_mesh_info = _get_mesh_info(dp_mod_ep_mesh, edp_mesh_dims)
                dp_mesh_info = _get_mesh_info(dp_mesh, dp_mesh_dims)
                assert isinstance(edp_mesh_info, FSDPMeshInfo)
                assert isinstance(dp_mesh_info, FSDPMeshInfo)
            else:

                def _get_fsdp_mesh_info(mesh: DeviceMesh) -> FSDPMeshInfo:
                    if mesh.ndim == 1:
                        return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
                    if mesh.ndim == 2:
                        return HSDPMeshInfo(
                            mesh=mesh, replicate_mesh_dim=0, shard_mesh_dim=1
                        )
                    raise ValueError(
                        f"Expected 1D or 2D FSDP mesh, got {mesh.ndim}D mesh."
                    )

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
