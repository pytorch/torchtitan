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

from torchtitan.models.common.moe import RoutedExperts
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
    DistMoeVmmConfig,
    DistMoeVmmPrefetch,
    plan_dist_moe_memory,
    prefetch_dist_moe_vmm,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims
    from torchtitan.trainer import Trainer


__all__ = [
    "DistMoeRoutedExperts",
    "MXFP8DistMoeRoutedExperts",
    "setup_dist_moe",
]

ActivationSlotPolicy = Literal["auto", "microbatch", "stage_microbatch"]


@dataclass(eq=False)
class _DistMoeRuntime:
    """One context shared by all local Dist-MoE layers."""

    config: DistMoeConfig
    group: dist.ProcessGroup
    prefetch: DistMoeVmmPrefetch | None
    slots: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    context: DistMoeContext | None = None
    _selected: tuple[int, int] | None = None
    _pipeline_hooks: list[RemovableHandle] = field(default_factory=list)

    def initialize(self, device: torch.device) -> None:
        """Create the annex context once after module buffers are materialized."""
        if self.context is None:
            prefetch = self.prefetch
            try:
                self.context = create_context(
                    group=self.group,
                    config=self.config,
                    device=device,
                    prefetched_vmm=prefetch,
                )
            finally:
                self.prefetch = None
                if self.context is None and prefetch is not None:
                    prefetch.close()

    def select(self, stage_index: int, microbatch_index: int) -> None:
        """Select the immutable slot and stage depth for one pipeline action."""
        key = (stage_index, microbatch_index)
        if key == self._selected:
            return
        try:
            slot, depth = self.slots[key]
        except KeyError as error:
            raise ValueError(
                "Dist-MoE has no activation slot for pipeline invocation " f"{key}"
            ) from error
        context = self.context
        if context is None:
            raise RuntimeError("Dist-MoE context is not initialized")
        context.select_activation_slot(slot, depth)
        self._selected = key

    def select_from_stage_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Select one slot and consume pipeline-only forward metadata."""
        del module
        stage_index = kwargs.get("pipeline_stage_index")
        microbatch_index = kwargs.get("pipeline_microbatch_index")
        if stage_index is None and microbatch_index is None:
            return
        if not isinstance(stage_index, int) or not isinstance(microbatch_index, int):
            raise RuntimeError(
                "Dist-MoE pipeline forwards require integer stage and microbatch IDs"
            )
        self.select(stage_index, microbatch_index)
        model_kwargs = dict(kwargs)
        del model_kwargs["pipeline_stage_index"]
        del model_kwargs["pipeline_microbatch_index"]
        return args, model_kwargs

    def close(self) -> None:
        """Release context-owned symmetric and VMM storage."""
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


class _DistMoeRoutedExperts(RoutedExperts):
    """Common module contract for DistMoE compute variants."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        """Configure shared DistMoE execution and memory policy.

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

        def __post_init__(self) -> None:
            RoutedExperts.Config.__post_init__(self)
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
        self._runtime: _DistMoeRuntime | None = None

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Shard owned parameters without wiring the unused stock dispatcher."""
        Module.parallelize(self, parallel_dims)

    def _weights(self) -> tuple[Any, Any]:
        """Return the operands consumed by the configured DistMoE backend."""
        return self.w13.weight.flatten(1, 2), self.w2.weight

    def _postprocess(self) -> DistMoeInputScaledRMSNorm | None:
        """Translate the owned postprocess module to an annex descriptor."""
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

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self._runtime is None:
            if self.w13.weight.device.type == "cpu" and buffer_device is None:
                return
            raise RuntimeError("DistMoE runtime was not configured")
        self._runtime.initialize(buffer_device or self.w13.weight.device)

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
        w13, w2 = self._weights()
        options = DistMoeExecutionOptions(
            inplace_wgrad_accum=self._dist_moe_config.inplace_wgrad_accum,
            wgrad_parameter_owners=(self.w13.weight, self.w2.weight)
            if self._dist_moe_config.inplace_wgrad_accum
            else None,
            experts_output_postprocess=self._postprocess(),
        )
        return run_dist_moe(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13,
            w2,
            runtime.context,
            options=options,
        )

    def close(self) -> None:
        """Release the shared DistMoE runtime idempotently."""
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.close()


class DistMoeRoutedExperts(_DistMoeRoutedExperts):
    """BF16 routed experts whose backend owns dispatch, compute, and combine."""

    @dataclass(kw_only=True, slots=True)
    class Config(_DistMoeRoutedExperts.Config):
        """Configure BF16 DistMoE kernels in addition to shared runtime policy."""

        kernel_config: str | None = None


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """MXFP8 DistMoE routed experts with FSDP-managed prepared weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure asynchronous MXFP8 kernels and shared runtime policy."""

        pipeline: Literal["staged", "mega"] = "staged"
        fast_math: bool = False
        kernel_config: DistMoeBlockScaledKernelConfig | None = None

        def __post_init__(self) -> None:
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

    def _weights(self) -> tuple[Any, Any]:
        """Return prepared grouped MXFP8 operands for this unshard lifetime."""
        w13 = self.w13.weight
        w2 = self.w2.weight
        w13_arg = (
            w13.operands.prepared(w13.flatten(1, 2))
            if isinstance(w13, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(w13, gate_up=True)
        )
        w2_arg = (
            w2.operands.prepared(w2)
            if isinstance(w2, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(w2, gate_up=False)
        )
        return w13_arg, w2_arg


def _annex_config(
    module: _DistMoeRoutedExperts,
    *,
    max_num_tokens: int,
    num_moe_layers: int,
    num_activation_slots: int,
):
    """Translate one TorchTitan backend policy to the annex configuration."""
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


def _pipeline_stage_modules(
    schedule: PipelineScheduleMulti,
    model_parts: list[torch.nn.Module],
) -> dict[int, list[_DistMoeRoutedExperts]]:
    """Map each local logical stage to its Dist-MoE modules."""
    stages = schedule._stages
    if len(stages) != len(model_parts):
        raise RuntimeError("pipeline schedule and model parts disagree")
    result: dict[int, list[_DistMoeRoutedExperts]] = {}
    for stage, part in zip(stages, model_parts, strict=True):
        modules = [
            module
            for module in part.modules()
            if isinstance(module, _DistMoeRoutedExperts)
        ]
        if modules:
            result[stage.stage_index] = modules
    return result


def _pipeline_slots(
    schedule: PipelineScheduleMulti,
    *,
    pp_rank: int,
    stage_modules: dict[int, list[_DistMoeRoutedExperts]],
    policy: _DistMoeRoutedExperts.Config,
) -> tuple[int, int, dict[tuple[int, int], tuple[int, int]], str]:
    """Choose and materialize the configured pipeline activation-slot plan."""
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
    return num_slots, depth, assignments, granularity


@dataclass(frozen=True, slots=True)
class _DistMoeSetup:
    """Validated local context plan, ready to bind to model modules."""

    config: DistMoeConfig
    group: dist.ProcessGroup
    modules: tuple[_DistMoeRoutedExperts, ...]
    stage_modules: dict[int, list[_DistMoeRoutedExperts]]
    assignments: dict[tuple[int, int], tuple[int, int]]
    granularity: str
    num_slots: int
    layer_depth: int


def _collect_dist_moe_modules(
    model_parts: list[torch.nn.Module],
) -> tuple[_DistMoeRoutedExperts, ...]:
    """Return each DistMoE module once, preserving model traversal order."""
    modules = (
        module
        for part in model_parts
        for module in part.modules()
        if isinstance(module, _DistMoeRoutedExperts)
    )
    return tuple(dict.fromkeys(modules))


def _validate_dist_moe_runtime(
    modules: tuple[_DistMoeRoutedExperts, ...],
    *,
    device: torch.device,
) -> _DistMoeRoutedExperts.Config:
    """Validate properties that are known only after model construction."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("DistMoE requires an SM100-or-newer CUDA device")
    policy = modules[0]._dist_moe_config
    if any(module._dist_moe_config != policy for module in modules[1:]):
        raise ValueError("All local DistMoE layers must share one runtime policy")
    return policy


def _plan_dist_moe_setup(
    *,
    config: Trainer.Config,
    modules: tuple[_DistMoeRoutedExperts, ...],
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    pp_schedule: object | None,
    policy: _DistMoeRoutedExperts.Config,
) -> _DistMoeSetup:
    """Resolve topology, capacity, and schedule-derived activation ownership."""
    ep_mesh = parallel_dims.get_optional_mesh("ep", include_singleton_axes=True)
    if ep_mesh is None:
        raise RuntimeError("DistMoE requires an expert-parallel mesh")
    group = ep_mesh.get_group()

    local_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    token_shards = parallel_dims.cp * parallel_dims.tp
    if local_tokens % token_shards:
        raise ValueError("DistMoE input tokens must divide evenly across CP and SP")
    max_num_tokens = local_tokens // token_shards

    num_slots = policy.num_activation_slots or 1
    layer_depth = len(modules)
    assignments: dict[tuple[int, int], tuple[int, int]] = {}
    stage_modules: dict[int, list[_DistMoeRoutedExperts]] = {}
    granularity = "none"
    if parallel_dims.pp_enabled:
        if not isinstance(pp_schedule, PipelineScheduleMulti):
            raise ValueError(
                "DistMoE pipeline activation planning requires a multi-stage schedule"
            )
        pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
        assert pp_mesh is not None
        stage_modules = _pipeline_stage_modules(pp_schedule, model_parts)
        num_slots, layer_depth, assignments, granularity = _pipeline_slots(
            pp_schedule,
            pp_rank=pp_mesh.get_local_rank(),
            stage_modules=stage_modules,
            policy=policy,
        )

    annex_config = _annex_config(
        modules[0],
        max_num_tokens=max_num_tokens,
        num_moe_layers=layer_depth,
        num_activation_slots=num_slots,
    )
    if any(
        _annex_config(
            module,
            max_num_tokens=max_num_tokens,
            num_moe_layers=layer_depth,
            num_activation_slots=num_slots,
        )
        != annex_config
        for module in modules[1:]
    ):
        raise ValueError("All local DistMoE layers must resolve one annex config")
    return _DistMoeSetup(
        config=annex_config,
        group=group,
        modules=modules,
        stage_modules=stage_modules,
        assignments=assignments,
        granularity=granularity,
        num_slots=num_slots,
        layer_depth=layer_depth,
    )


def _bind_dist_moe_setup(
    setup: _DistMoeSetup,
    *,
    model_parts: list[torch.nn.Module],
    pp_schedule: object | None,
    device: torch.device,
) -> None:
    """Allocate one shared runtime and bind its static pipeline metadata hooks."""
    memory_plan = plan_dist_moe_memory(
        setup.config,
        ep_size=dist.get_world_size(setup.group),
        device=device,
    )
    policy = setup.modules[0]._dist_moe_config
    prefetch = (
        prefetch_dist_moe_vmm(
            config=setup.config,
            ep_size=dist.get_world_size(setup.group),
            device=device,
        )
        if memory_plan.uses_host_scratch and policy.prefetch_vmm
        else None
    )
    runtime = _DistMoeRuntime(
        setup.config,
        setup.group,
        prefetch,
        setup.assignments,
    )
    for module in setup.modules:
        module._runtime = runtime

    if setup.assignments:
        assert isinstance(pp_schedule, PipelineScheduleMulti)
        for stage, part in zip(pp_schedule._stages, model_parts, strict=True):
            if stage.stage_index not in setup.stage_modules:
                continue
            stage.pass_pipeline_metadata = True
            runtime._pipeline_hooks.append(
                part.register_forward_pre_hook(
                    runtime.select_from_stage_forward,
                    with_kwargs=True,
                )
            )

    logger.info("%s", memory_plan.explain())
    if setup.assignments:
        logger.info(
            "DistMoE pipeline activation slots: policy=%s slots=%d depth=%d",
            setup.granularity,
            setup.num_slots,
            setup.layer_depth,
        )


def setup_dist_moe(
    *,
    config: Trainer.Config,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
    pp_schedule: object | None,
) -> None:
    """Collect, validate, plan, and bind one local DistMoE context."""
    modules = _collect_dist_moe_modules(model_parts)
    if not modules or config.checkpoint.create_seed_checkpoint:
        return
    policy = _validate_dist_moe_runtime(modules, device=device)
    setup = _plan_dist_moe_setup(
        config=config,
        modules=modules,
        model_parts=model_parts,
        parallel_dims=parallel_dims,
        pp_schedule=pp_schedule,
        policy=policy,
    )
    _bind_dist_moe_setup(
        setup,
        model_parts=model_parts,
        pp_schedule=pp_schedule,
        device=device,
    )
