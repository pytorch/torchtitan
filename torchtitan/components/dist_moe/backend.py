# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan integration for the standalone :mod:`dist_moe` package."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import (
    _analyze_pipeline_resource_liveness,
    PipelineScheduleMulti,
)
from torch.utils.hooks import RemovableHandle

from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.dist_moe import (
    _DistMoeW13ShardedTensor,
    _DistMoeW2ShardedTensor,
    _dynamic_prepared_weight,
)

from dist_moe import (
    BlockScaledFormat,
    create_context,
    dist_moe as run_dist_moe,
    DistMoeBlockScaledConfig,
    DistMoeBlockScaledKernelConfig,
    DistMoeConfig,
    DistMoeContext,
    DistMoeExecutionOptions,
    DistMoeInputScaledRMSNorm,
    DistMoePreparedWeight,
    DistMoeVmmConfig,
    DistMoeVmmPrefetch,
    plan_dist_moe_memory,
    prefetch_dist_moe_vmm,
)


logger = logging.getLogger(__name__)

# Shape suffix legend for DistMoE weights:
#   E = experts, 2 = fused gate/up projections, F = intermediate dimension,
#   D = model dimension

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims
    from torchtitan.trainer import Trainer


__all__ = [
    "DistMoeRuntime",
    "DistMoeRoutedExperts",
    "MXFP8DistMoeRoutedExperts",
    "prepare_dist_moe_runtime",
]

ActivationSlotPolicy = Literal["auto", "microbatch", "stage_microbatch"]
_DistMoeWeightOperand = torch.Tensor | DistMoePreparedWeight


@dataclass(eq=False)
class DistMoeRuntime:
    """Own the context shared by every DistMoE layer on one rank.

    The trainer owns this object because its memory plan depends on the complete
    local layer set and finalized distributed schedule. Modules only retain a
    reference used by ``forward``.

    Args:
        config: Fully resolved configuration for the standalone DistMoE runtime.
        group: Expert-parallel process group used by dispatch and combine.
        device: CUDA device on which the context will be created.
        prefetch: Optional host-backed VMM mapping prepared before model-state
            initialization.
        slots: Immutable mapping from ``(stage, microbatch)`` to activation slot
            and local layer depth. Empty when pipeline parallelism is disabled.
    """

    config: DistMoeConfig
    group: dist.ProcessGroup
    device: torch.device
    prefetch: DistMoeVmmPrefetch | None
    slots: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    context: DistMoeContext | None = None
    _selected: tuple[int, int] | None = None
    _pipeline_hooks: list[RemovableHandle] = field(default_factory=list)

    def initialize(self) -> None:
        """Create the shared context after model parameters and buffers exist.

        A prepared VMM mapping is consumed at most once. If context creation
        fails, the mapping is closed before the exception is propagated.
        Repeated successful calls are no-ops.
        """
        if self.context is None:
            prefetch = self.prefetch
            try:
                self.context = create_context(
                    group=self.group,
                    config=self.config,
                    device=self.device,
                    prefetched_vmm=prefetch,
                )
            finally:
                self.prefetch = None
                if self.context is None and prefetch is not None:
                    prefetch.close()

    def select_pipeline_slot(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Select a precomputed PP activation slot before a stage forward.

        Args:
            module: Stage-root module invoking this forward pre-hook. It is not
                otherwise inspected.
            args: Positional stage-forward arguments, returned unchanged.
            kwargs: Stage-forward keyword arguments containing the canonical
                pipeline stage and microbatch indices.

        Returns:
            Updated arguments with the pipeline-only metadata removed, or
            ``None`` when no metadata was supplied.

        Raises:
            RuntimeError: If metadata is incomplete or the context has not been
                initialized.
            ValueError: If no immutable slot was planned for the invocation.
        """
        del module
        stage_index = kwargs.get("pipeline_stage_index")
        microbatch_index = kwargs.get("pipeline_microbatch_index")
        if stage_index is None and microbatch_index is None:
            return
        if not isinstance(stage_index, int) or not isinstance(microbatch_index, int):
            raise RuntimeError(
                "Dist-MoE pipeline forwards require integer stage and microbatch IDs"
            )
        key = (stage_index, microbatch_index)
        if key != self._selected:
            try:
                slot, depth = self.slots[key]
            except KeyError as error:
                raise ValueError(
                    f"Dist-MoE has no activation slot for pipeline invocation {key}"
                ) from error
            context = self.context
            if context is None:
                raise RuntimeError("Dist-MoE context is not initialized")
            context.select_activation_slot(slot, depth)
            self._selected = key
        model_kwargs = dict(kwargs)
        del model_kwargs["pipeline_stage_index"]
        del model_kwargs["pipeline_microbatch_index"]
        return args, model_kwargs

    def close(self) -> None:
        """Idempotently remove PP hooks and release all runtime-owned storage."""
        for hook in self._pipeline_hooks:
            hook.remove()
        self._pipeline_hooks.clear()
        prefetch = self.prefetch
        self.prefetch = None
        if prefetch is not None:
            prefetch.close()
        context = self.context
        if context is not None:
            context.close()
            self.context = None
        self._selected = None


class DistMoeRoutedExperts(RoutedExperts):
    """BF16 experts whose backend owns dispatch, compute, and combine.

    Construction materializes only the W13/W2 parameters and optional
    postprocess module represented by the common routed-expert config. DistMoE
    supplies the activation and dispatcher behavior at execution time.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        """Configure BF16 DistMoE execution, memory, and kernel policy.

        Args:
            max_routing_imbalance_factor: Maximum receive-row capacity relative
                to balanced routing.
            device_memory_budget_bytes: Optional total device arena budget.
            activation_slot_policy: Pipeline activation-lifetime granularity.
                ``auto`` selects the smaller exact schedule-derived plan.
            num_activation_slots: Optional lower bound on activation slots.
            vmm_host_scratch_imbalance_factor: Total device and host-backed VMM
                scratch capacity, or ``None`` to disable VMM.
            prefetch_vmm: Prepare host-backed VMM mappings during model setup.
            num_sms: Optional number of SMs assigned to DistMoE kernels.
            wgrad_dtype: Weight-gradient output dtype.
            inplace_wgrad_accum: Accumulate WGRAD into parameter gradients.
            kernel_config: Optional named BF16 CuTe kernel schedule.
        """

        max_routing_imbalance_factor: float = 1.0
        device_memory_budget_bytes: int | None = None
        activation_slot_policy: ActivationSlotPolicy = "auto"
        num_activation_slots: int | None = None
        vmm_host_scratch_imbalance_factor: float | None = None
        prefetch_vmm: bool = False
        num_sms: int | None = None
        wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"
        inplace_wgrad_accum: bool = False
        kernel_config: str | None = None

        def __post_init__(self) -> None:
            """Validate module structure and values before construction."""
            RoutedExperts.Config.__post_init__(self)
            if (
                type(self.w13) is not GroupedLinear.Config
                or type(self.w2) is not GroupedLinear.Config
                or type(self.activation_fn) is not SwiGLU.Config
            ):
                raise TypeError(
                    "DistMoE requires the stock grouped-linear projections and "
                    "SwiGLU"
                )
            if not isinstance(self.token_dispatcher, AllToAllTokenDispatcher.Config):
                raise ValueError(
                    "DistMoE owns expert communication and requires the standard "
                    "all-to-all routed-expert config"
                )
            postprocess = self.expert_output_postprocess
            owner = None if postprocess is None else postprocess._owner
            if postprocess is not None and not callable(
                getattr(owner, "to_dist_moe_postprocess", None)
            ):
                raise TypeError(
                    f"{type(postprocess).__qualname__} cannot run inside DistMoE; "
                    "its module must define to_dist_moe_postprocess()"
                )
            if self.max_routing_imbalance_factor <= 0:
                raise ValueError("max_routing_imbalance_factor must be positive")
            if self.num_activation_slots is not None and self.num_activation_slots <= 0:
                raise ValueError("num_activation_slots must be positive")
            if (
                self.vmm_host_scratch_imbalance_factor is not None
                and self.vmm_host_scratch_imbalance_factor <= 0
            ):
                raise ValueError("vmm_host_scratch_imbalance_factor must be positive")
            if self.prefetch_vmm and self.vmm_host_scratch_imbalance_factor is None:
                raise ValueError("prefetch_vmm requires VMM to be enabled")
            if self.num_sms is not None and self.num_sms <= 0:
                raise ValueError("num_sms must be positive")
            if self.activation_slot_policy not in (
                "auto",
                "microbatch",
                "stage_microbatch",
            ):
                raise ValueError("Unsupported DistMoE activation slot policy")
            if self.wgrad_dtype not in ("bfloat16", "float32"):
                raise ValueError("Unsupported DistMoE WGRAD dtype")

    def __init__(self, config: Config):
        # DistMoE owns dispatch and SwiGLU, so only construct the parameter and
        # optional postprocess modules represented by the common config.
        Module.__init__(self)
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.expert_output_postprocess = (
            config.expert_output_postprocess.build()
            if config.expert_output_postprocess is not None
            else None
        )
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.num_experts = config.w13.group_size
        self.top_k = config.token_dispatcher.top_k
        self._dist_moe_config = config
        self._runtime: DistMoeRuntime | None = None

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Shard owned parameters without wiring the unused stock dispatcher."""
        Module.parallelize(self, parallel_dims)

    def _dist_moe_weight_operands(
        self,
    ) -> tuple[_DistMoeWeightOperand, _DistMoeWeightOperand]:
        """Return W13 and W2 operands for the standalone DistMoE call."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        return w13_EFD, w2_EDF

    def _build_dist_moe_postprocess(self) -> DistMoeInputScaledRMSNorm | None:
        """Translate the current postprocess parameters to a kernel descriptor."""
        module = self.expert_output_postprocess
        if module is None:
            return None
        factory = getattr(module, "to_dist_moe_postprocess", None)
        if not callable(factory):
            raise TypeError(
                f"{type(module).__qualname__} cannot run inside DistMoE; it must "
                "define to_dist_moe_postprocess()"
            )
        postprocess = factory()
        if not isinstance(postprocess, DistMoeInputScaledRMSNorm):
            raise TypeError(
                "to_dist_moe_postprocess() must return DistMoeInputScaledRMSNorm"
            )
        return postprocess

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused dispatch, expert compute, and combine.

        Args:
            x_TD: Local input tokens.
            topk_scores_TK: Router scores for selected experts.
            topk_expert_ids_TK: Global IDs of selected experts.
            num_local_tokens_per_expert_E: Counts maintained for load-balancing
                state. DistMoE derives its own dispatch metadata.

        Returns:
            Combined local expert output.
        """
        del num_local_tokens_per_expert_E
        runtime = self._runtime
        if runtime is None or runtime.context is None:
            raise RuntimeError("DistMoE context is not initialized")
        w13_operand, w2_operand = self._dist_moe_weight_operands()
        # FSDP replaces module-visible weights for each unshard lifetime, and
        # the postprocess descriptor captures its current parameter. Build the
        # options here so neither reference survives a reshard.
        options = DistMoeExecutionOptions(
            inplace_wgrad_accum=self._dist_moe_config.inplace_wgrad_accum,
            wgrad_parameter_owners=(self.w13.weight, self.w2.weight)
            if self._dist_moe_config.inplace_wgrad_accum
            else None,
            experts_output_postprocess=self._build_dist_moe_postprocess(),
        )
        return run_dist_moe(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_operand,
            w2_operand,
            runtime.context,
            options=options,
        )


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """MXFP8 DistMoE routed experts with FSDP-managed prepared weights.

    Construction wraps W13 and W2 in TorchTitan's shared prepared-weight
    lifecycle so each FSDP unshard produces the layouts consumed by DistMoE.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure asynchronous MXFP8 kernels and shared runtime policy."""

        pipeline: Literal["staged", "mega"] = "staged"
        fast_math: bool = False
        kernel_config: DistMoeBlockScaledKernelConfig | None = None

        def __post_init__(self) -> None:
            """Validate the selected asynchronous block-scaled pipeline."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if self.pipeline not in ("staged", "mega"):
                raise ValueError("Unsupported MXFP8 DistMoE pipeline")

    def __init__(self, config: Config):
        super().__init__(config)
        self.w13.weight = torch.nn.Parameter(
            _DistMoeW13ShardedTensor(self.w13.weight.data),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            _DistMoeW2ShardedTensor(self.w2.weight.data),
            requires_grad=self.w2.weight.requires_grad,
        )

    def _dist_moe_weight_operands(
        self,
    ) -> tuple[_DistMoeWeightOperand, _DistMoeWeightOperand]:
        """Return prepared grouped MXFP8 operands for this unshard lifetime."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_operand = (
            w13_E2FD.operands.prepared(w13_E2FD.flatten(1, 2))
            if isinstance(w13_E2FD, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(w13_E2FD, gate_up=True)
        )
        w2_operand = (
            w2_EDF.operands.prepared(w2_EDF)
            if isinstance(w2_EDF, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(w2_EDF, gate_up=False)
        )
        return w13_operand, w2_operand


def _build_dist_moe_runtime_config(
    module: DistMoeRoutedExperts,
    *,
    max_num_tokens: int,
    num_moe_layers: int,
    num_activation_slots: int,
) -> DistMoeConfig:
    """Resolve one module policy into a standalone DistMoE configuration.

    Args:
        module: Local routed-expert module supplying shapes and kernel policy.
        max_num_tokens: Maximum local input rows for one microbatch.
        num_moe_layers: Number of layers represented by each activation slot.
        num_activation_slots: Number of simultaneously live activation slots.

    Returns:
        A fully resolved configuration suitable for memory planning and context
        creation.
    """
    policy = module._dist_moe_config
    is_mxfp8 = isinstance(module, MXFP8DistMoeRoutedExperts)
    if is_mxfp8:
        assert isinstance(policy, MXFP8DistMoeRoutedExperts.Config)
        blockscaled = DistMoeBlockScaledConfig(
            format=BlockScaledFormat.MXFP8_E4M3,
            fast_math=policy.fast_math,
            pipeline=policy.pipeline,
            kernel=policy.kernel_config,
        )
        kernel_config = None
    else:
        assert isinstance(policy, DistMoeRoutedExperts.Config)
        blockscaled = None
        kernel_config = policy.kernel_config
    vmm = (
        None
        if policy.vmm_host_scratch_imbalance_factor is None
        else DistMoeVmmConfig(
            host_scratch_imbalance_factor=policy.vmm_host_scratch_imbalance_factor
        )
    )
    return DistMoeConfig(
        max_num_tokens=max_num_tokens,
        hidden_dim=module.hidden_dim,
        intermediate_dim=module.intermediate_dim,
        top_k=module.top_k,
        num_experts=module.num_experts,
        num_moe_layers=num_moe_layers,
        max_routing_imbalance_factor=policy.max_routing_imbalance_factor,
        device_memory_budget_bytes=policy.device_memory_budget_bytes,
        num_microbatch_stacks=num_activation_slots,
        vmm=vmm,
        num_sms=policy.num_sms,
        kernel_config=kernel_config,
        blockscaled=blockscaled,
        wgrad_dtype=(
            torch.bfloat16 if policy.wgrad_dtype == "bfloat16" else torch.float32
        ),
    )


@dataclass(frozen=True, slots=True)
class _PipelineActivationPlan:
    """Describe immutable PP activation ownership for the local rank."""

    granularity: Literal["microbatch", "stage_microbatch"]
    num_slots: int
    layer_depth: int
    assignments: dict[tuple[int, int], tuple[int, int]]
    stage_indices: tuple[int, ...]


def _plan_pipeline_activation_slots(
    schedule: PipelineScheduleMulti,
    *,
    pp_rank: int,
    model_parts: list[torch.nn.Module],
    policy: DistMoeRoutedExperts.Config,
) -> _PipelineActivationPlan:
    """Choose the smallest configured schedule-derived PP activation plan.

    Args:
        schedule: Final multi-stage schedule whose resource lifetimes are
            analyzed.
        pp_rank: Physical pipeline rank represented by this process.
        model_parts: Local stage modules in the same order as schedule stages.
        policy: Shared activation-slot policy for local DistMoE modules.

    Returns:
        The selected slot granularity, allocation dimensions, immutable
        stage/microbatch assignments, and participating stage indices.

    Raises:
        RuntimeError: If schedule stages and local model parts disagree or no
            local pipeline stage contains DistMoE.
    """
    stages = schedule._stages
    if len(stages) != len(model_parts):
        raise RuntimeError("pipeline schedule and model parts disagree")
    stage_modules = {
        stage.stage_index: modules
        for stage, part in zip(stages, model_parts, strict=True)
        if (
            modules := [
                module
                for module in part.modules()
                if isinstance(module, DistMoeRoutedExperts)
            ]
        )
    }
    if not stage_modules:
        raise RuntimeError("no local pipeline stage contains DistMoE")
    candidates = (
        ("microbatch", "stage_microbatch")
        if policy.activation_slot_policy == "auto"
        else (policy.activation_slot_policy,)
    )
    plans = []
    stage_indices = tuple(stage_modules)
    for granularity in candidates:
        liveness = _analyze_pipeline_resource_liveness(
            schedule,
            physical_rank=pp_rank,
            stage_indices=stage_indices,
            granularity=granularity,
        )
        depth = (
            sum(len(modules) for modules in stage_modules.values())
            if granularity == "microbatch"
            else max(len(modules) for modules in stage_modules.values())
        )
        slots = max(liveness.num_slots, policy.num_activation_slots or 0)
        plans.append((slots * depth, granularity, liveness, slots, depth))
    _, granularity, liveness, num_slots, depth = min(
        plans, key=lambda item: (item[0], item[1] != "stage_microbatch")
    )
    assignments = {
        (stage_index, microbatch_index): (
            liveness.slot_for(stage_index, microbatch_index),
            depth if granularity == "microbatch" else len(stage_modules[stage_index]),
        )
        for stage_index in stage_indices
        for microbatch_index in range(liveness.num_microbatches)
    }
    return _PipelineActivationPlan(
        granularity=granularity,
        num_slots=num_slots,
        layer_depth=depth,
        assignments=assignments,
        stage_indices=stage_indices,
    )


def prepare_dist_moe_runtime(
    *,
    config: Trainer.Config,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
    pp_schedule: object | None,
) -> DistMoeRuntime | None:
    """Prepare and attach one shared DistMoE runtime for the local rank.

    This function runs after model parallelization and schedule construction but
    before parameter materialization. It resolves the complete rank-local memory
    plan, optionally starts VMM preparation, attaches the shared runtime to every
    local DistMoE module, and installs PP slot-selection hooks. The trainer must
    call :meth:`DistMoeRuntime.initialize` after model-state initialization and
    :meth:`DistMoeRuntime.close` during teardown.

    Args:
        config: Final trainer configuration.
        model_parts: Local model or pipeline-stage modules.
        parallel_dims: Final distributed mesh dimensions.
        device: CUDA device on which DistMoE will execute.
        pp_schedule: Final pipeline schedule, or ``None`` without PP.

    Returns:
        The trainer-owned runtime, or ``None`` when the rank has no DistMoE
        modules or is creating a seed checkpoint.

    Raises:
        RuntimeError: If required EP or PP topology is unavailable.
        ValueError: If hardware, token shapes, or local module policies are
            incompatible with one shared runtime.
    """
    modules = tuple(
        dict.fromkeys(
            module
            for part in model_parts
            for module in part.modules()
            if isinstance(module, DistMoeRoutedExperts)
        )
    )
    if not modules or config.checkpoint.create_seed_checkpoint:
        return None
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("DistMoE requires an SM100-or-newer CUDA device")

    policy = modules[0]._dist_moe_config
    shared_policy = (
        policy.activation_slot_policy,
        policy.num_activation_slots,
        policy.prefetch_vmm,
    )
    if any(
        (
            module._dist_moe_config.activation_slot_policy,
            module._dist_moe_config.num_activation_slots,
            module._dist_moe_config.prefetch_vmm,
        )
        != shared_policy
        for module in modules[1:]
    ):
        raise ValueError(
            "All local DistMoE layers must share activation-slot and VMM-prefetch "
            "policy"
        )

    ep_mesh = parallel_dims.get_optional_mesh("ep", include_singleton_axes=True)
    if ep_mesh is None:
        raise RuntimeError("DistMoE requires an expert-parallel mesh")
    group = ep_mesh.get_group()
    local_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    token_shards = parallel_dims.cp * parallel_dims.tp
    if local_tokens % token_shards:
        raise ValueError("DistMoE input tokens must divide evenly across CP and SP")
    max_num_tokens = local_tokens // token_shards

    pipeline_plan: _PipelineActivationPlan | None = None
    num_slots = policy.num_activation_slots or 1
    layer_depth = len(modules)
    assignments: dict[tuple[int, int], tuple[int, int]] = {}
    if parallel_dims.pp_enabled:
        if not isinstance(pp_schedule, PipelineScheduleMulti):
            raise ValueError(
                "DistMoE pipeline activation planning requires a multi-stage schedule"
            )
        pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
        if pp_mesh is None:
            raise RuntimeError("pipeline parallelism requires a PP mesh")
        pipeline_plan = _plan_pipeline_activation_slots(
            pp_schedule,
            pp_rank=pp_mesh.get_local_rank(),
            model_parts=model_parts,
            policy=policy,
        )
        num_slots = pipeline_plan.num_slots
        layer_depth = pipeline_plan.layer_depth
        assignments = pipeline_plan.assignments

    runtime_config = _build_dist_moe_runtime_config(
        modules[0],
        max_num_tokens=max_num_tokens,
        num_moe_layers=layer_depth,
        num_activation_slots=num_slots,
    )
    if any(
        _build_dist_moe_runtime_config(
            module,
            max_num_tokens=max_num_tokens,
            num_moe_layers=layer_depth,
            num_activation_slots=num_slots,
        )
        != runtime_config
        for module in modules[1:]
    ):
        raise ValueError("All local DistMoE layers must resolve one runtime config")

    ep_size = dist.get_world_size(group)
    memory_plan = plan_dist_moe_memory(
        runtime_config,
        ep_size=ep_size,
        device=device,
    )
    prefetch = None
    runtime = None
    try:
        if memory_plan.uses_host_scratch and policy.prefetch_vmm:
            prefetch = prefetch_dist_moe_vmm(
                config=runtime_config,
                ep_size=ep_size,
                device=device,
            )
        runtime = DistMoeRuntime(
            config=runtime_config,
            group=group,
            device=device,
            prefetch=prefetch,
            slots=assignments,
        )
        for module in modules:
            module._runtime = runtime

        if pipeline_plan is not None:
            assert isinstance(pp_schedule, PipelineScheduleMulti)
            for stage, part in zip(pp_schedule._stages, model_parts, strict=True):
                if stage.stage_index not in pipeline_plan.stage_indices:
                    continue
                stage.pass_pipeline_metadata = True
                runtime._pipeline_hooks.append(
                    part.register_forward_pre_hook(
                        runtime.select_pipeline_slot,
                        with_kwargs=True,
                    )
                )
    except Exception:
        if runtime is not None:
            runtime.close()
            for module in modules:
                module._runtime = None
        elif prefetch is not None:
            prefetch.close()
        raise

    logger.info("%s", memory_plan.explain())
    if pipeline_plan is not None:
        logger.info(
            "DistMoE pipeline activation slots: policy=%s slots=%d depth=%d",
            pipeline_plan.granularity,
            pipeline_plan.num_slots,
            pipeline_plan.layer_depth,
        )
    return runtime
