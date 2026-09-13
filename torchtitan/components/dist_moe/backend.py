# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan integration for the standalone :mod:`dist_moe` package."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import (
    _analyze_pipeline_resource_liveness,
    PipelineScheduleMulti,
)
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh

from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module

from dist_moe import (
    BlockScaledFormat,
    create_context,
    dist_moe as run_dist_moe,
    DistMoeBlockScaledConfig,
    DistMoeBlockScaledKernelConfig,
    DistMoeConfig,
    DistMoeContext,
    DistMoeExecutionOptions,
    DistMoeExpertPostprocess,
    DistMoeInputScaledRMSNorm,
    DistMoeVmmConfig,
    DistMoeVmmPrefetch,
    plan_dist_moe_memory,
    prefetch_dist_moe_vmm,
)

from .tensor import (
    _DistMoeW13ShardedTensor,
    _DistMoeW2ShardedTensor,
    _dynamic_prepared_weight,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims
    from torchtitan.trainer import Trainer


__all__ = [
    "cleanup_dist_moe",
    "DistMoeBackendConfig",
    "DistMoeConverter",
    "DistMoeRoutedExperts",
    "setup_dist_moe",
]

DistMoeDType = Literal["bf16", "mxfp8"]
ActivationSlotPolicy = Literal["auto", "microbatch", "stage_microbatch"]


@dataclass(kw_only=True, slots=True)
class DistMoeBackendConfig:
    """Configure Dist-MoE compute, routing capacity, and activation storage.

    Args:
        dtype: Expert-compute format.
        max_routing_imbalance_factor: Routing imbalance retained in device HBM.
        device_memory_budget_bytes: Optional total device arena budget.
        activation_slot_policy: Pipeline activation lifetime granularity. ``auto``
            compares the two exact schedule-derived plans.
        num_activation_slots: Optional explicit lower bound on slot count.
        vmm_host_scratch_imbalance_factor: Total imbalance covered by device and
            host-backed VMM scratch, or ``None`` to disable VMM.
        prefetch_vmm: Allocate VMM storage asynchronously during model setup.
            Requires VMM to be enabled.
        num_sms: Optional SM count used by each CuTe launch.
        bf16_kernel_config: Optional named BF16 grouped-GEMM schedule.
        mxfp8_pipeline: Staged or Mega asynchronous MXFP8 execution.
        mxfp8_fast_math: Use the approximate sigmoid in MXFP8 SwiGLU.
        mxfp8_kernel_config: Optional expert block-scaled CuTe schedule.
        wgrad_dtype: Weight-gradient output dtype.
        inplace_wgrad_accum: Accumulate WGRAD directly into parameter gradients.
    """

    dtype: DistMoeDType = "bf16"
    max_routing_imbalance_factor: float = 1.0
    device_memory_budget_bytes: int | None = None
    activation_slot_policy: ActivationSlotPolicy = "auto"
    num_activation_slots: int | None = None
    vmm_host_scratch_imbalance_factor: float | None = None
    prefetch_vmm: bool = False
    num_sms: int | None = None
    bf16_kernel_config: str | None = None
    mxfp8_pipeline: Literal["staged", "mega"] = "staged"
    mxfp8_fast_math: bool = False
    mxfp8_kernel_config: DistMoeBlockScaledKernelConfig | None = None
    wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    inplace_wgrad_accum: bool = False

    def __post_init__(self) -> None:
        """Validate values that do not depend on the runtime topology."""
        if self.dtype not in ("bf16", "mxfp8"):
            raise ValueError("dtype must be 'bf16' or 'mxfp8'")
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
        if self.dtype == "bf16" and self.mxfp8_kernel_config is not None:
            raise ValueError("mxfp8_kernel_config requires dtype='mxfp8'")
        if self.dtype == "mxfp8" and self.bf16_kernel_config is not None:
            raise ValueError("bf16_kernel_config requires dtype='bf16'")


class DistMoeConverter(ModelConfigConverter):
    """Replace stock routed experts with the fused Dist-MoE backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        backend: DistMoeBackendConfig = field(default_factory=DistMoeBackendConfig)

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        """Convert every stock routed-experts config in ``model_config``."""
        targets = list(model_config.traverse(RoutedExperts.Config))
        for _fqn, config, parent, attr in targets:
            if isinstance(config, DistMoeRoutedExperts.Config):
                continue
            if type(config) is not RoutedExperts.Config:
                raise TypeError(
                    "Dist-MoE cannot convert a specialized routed-experts config; "
                    "express model-specific behavior through the common routed-"
                    "expert contract"
                )
            if (
                type(config.w13) is not GroupedLinear.Config
                or type(config.w2) is not GroupedLinear.Config
                or type(config.activation_fn) is not SwiGLU.Config
            ):
                raise TypeError(
                    "Dist-MoE requires the stock grouped-linear projections and "
                    "SwiGLU activation"
                )
            if not isinstance(config.token_dispatcher, AllToAllTokenDispatcher.Config):
                raise ValueError(
                    "Dist-MoE owns expert communication and cannot be combined "
                    "with a non-standard token dispatcher"
                )
            replacement = DistMoeRoutedExperts.Config(
                w13=config.w13,
                w2=config.w2,
                activation_fn=config.activation_fn,
                token_dispatcher=config.token_dispatcher,
                backend=self.config.backend,
                sharding_config=config.sharding_config,
            )
            if parent is None:
                model_config = replacement
            elif isinstance(parent, list):
                parent[attr] = replacement
            else:
                setattr(parent, attr, replacement)
        logger.info("Converted %d routed-experts modules to Dist-MoE", len(targets))
        return model_config


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
    ) -> None:
        """Select one slot before a pipeline stage enters checkpointed layers."""
        del module, args
        stage_index = kwargs.get("pipeline_stage_index")
        microbatch_index = kwargs.get("pipeline_microbatch_index")
        if stage_index is None and microbatch_index is None:
            return
        if not isinstance(stage_index, int) or not isinstance(microbatch_index, int):
            raise RuntimeError(
                "Dist-MoE pipeline forwards require integer stage and microbatch IDs"
            )
        self.select(stage_index, microbatch_index)

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


class DistMoeRoutedExperts(RoutedExperts):
    """Routed-experts backend fusing dispatch, expert compute, and combine."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        supports_cuda_graphs: ClassVar[bool] = True
        backend: DistMoeBackendConfig = field(default_factory=DistMoeBackendConfig)

    def __init__(self, config: Config):
        super().__init__(config)
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.top_k = config.token_dispatcher.top_k
        if config.backend.dtype == "mxfp8":
            self.w13.weight = torch.nn.Parameter(
                _DistMoeW13ShardedTensor(self.w13.weight.data),
                requires_grad=self.w13.weight.requires_grad,
            )
            self.w2.weight = torch.nn.Parameter(
                _DistMoeW2ShardedTensor(self.w2.weight.data),
                requires_grad=self.w2.weight.requires_grad,
            )
        self._backend_config = config.backend
        self._runtime: _DistMoeRuntime | None = None
        self._ep_group: dist.ProcessGroup | None = None
        self._sp_size = 1

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Apply declared sharding and retain the expert-parallel group."""
        # Dist-MoE bypasses the stock dispatcher at runtime but keeps the common
        # module hierarchy and applies each child's declared parameter sharding.
        Module.parallelize(self, parallel_dims)
        ep_mesh = parallel_dims.get_optional_mesh("ep", include_singleton_axes=True)
        self._ep_group = ep_mesh.get_group() if ep_mesh is not None else None
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        self._sp_size = tp_mesh.size() if tp_mesh is not None else 1

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self._runtime is None:
            if self.w13.weight.device.type == "cpu" and buffer_device is None:
                return
            raise RuntimeError("Dist-MoE runtime was not configured")
        self._runtime.initialize(buffer_device or self.w13.weight.device)

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        expert_output_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the annex using routing decisions produced by TorchTitan.

        Args:
            x_TD: Local input tokens.
            topk_scores_TK: Router scores for selected experts.
            topk_expert_ids_TK: Global IDs of selected experts.
            num_local_tokens_per_expert_E: Counts maintained by TorchTitan for
                load-balancing state. DistMoE derives its own dispatch metadata.
            expert_output_postprocess: Optional route-wise transformation applied
                after peer combine and before score-weighted reduction.

        Returns:
            Combined local expert output.
        """
        del num_local_tokens_per_expert_E
        if self._runtime is None or self._runtime.context is None:
            raise RuntimeError("Dist-MoE context is not initialized")

        w13 = (
            self.w13.weight.to_local()
            if isinstance(self.w13.weight, DTensor)
            else self.w13.weight
        )
        w2 = (
            self.w2.weight.to_local()
            if isinstance(self.w2.weight, DTensor)
            else self.w2.weight
        )
        if self._backend_config.dtype == "mxfp8":
            from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor

            if isinstance(w13, _UnshardedFSDPTensor):
                w13_arg = w13.operands.prepared(w13.flatten(1, 2))
            else:
                w13_arg = _dynamic_prepared_weight(w13, gate_up=True)
            if isinstance(w2, _UnshardedFSDPTensor):
                w2_arg = w2.operands.prepared(w2)
            else:
                w2_arg = _dynamic_prepared_weight(w2, gate_up=False)
        else:
            w13_arg = w13.flatten(1, 2)
            w2_arg = w2

        postprocess = None
        callback = expert_output_postprocess
        if callback is not None:
            factory = getattr(type(callback), "to_dist_moe_postprocess", None)
            if factory is not None:
                postprocess = factory(callback)
                if not isinstance(
                    postprocess,
                    (DistMoeExpertPostprocess, DistMoeInputScaledRMSNorm),
                ):
                    raise TypeError(
                        "to_dist_moe_postprocess() must return a DistMoE "
                        "postprocess configuration"
                    )
            else:

                def postprocess_with_sparse_mesh(value: torch.Tensor) -> torch.Tensor:
                    """Run the caller-owned transform under the expert mesh context."""
                    assert callback is not None
                    with maybe_set_sparse_mesh():
                        return callback(value)

                postprocess = DistMoeExpertPostprocess(postprocess_with_sparse_mesh)

        options = DistMoeExecutionOptions(
            inplace_wgrad_accum=self._backend_config.inplace_wgrad_accum,
            wgrad_parameter_owners=(self.w13.weight, self.w2.weight)
            if self._backend_config.inplace_wgrad_accum
            else None,
            experts_output_postprocess=postprocess,
        )
        return run_dist_moe(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_arg,
            w2_arg,
            self._runtime.context,
            options=options,
        )


def _annex_config(
    module: DistMoeRoutedExperts,
    *,
    max_num_tokens: int,
    num_moe_layers: int,
    num_activation_slots: int,
):
    """Translate one TorchTitan backend policy to the annex configuration."""
    policy = module._backend_config
    blockscaled = (
        DistMoeBlockScaledConfig(
            format=BlockScaledFormat.MXFP8_E4M3,
            fast_math=policy.mxfp8_fast_math,
            pipeline=policy.mxfp8_pipeline,
            kernel=policy.mxfp8_kernel_config,
        )
        if policy.dtype == "mxfp8"
        else None
    )
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
        kernel_config=policy.bf16_kernel_config,
        blockscaled=blockscaled,
        wgrad_dtype=(
            torch.bfloat16 if policy.wgrad_dtype == "bfloat16" else torch.float32
        ),
    )


def _pipeline_stage_modules(
    schedule: PipelineScheduleMulti,
    model_parts: list[torch.nn.Module],
) -> dict[int, list[DistMoeRoutedExperts]]:
    """Map each local logical stage to its Dist-MoE modules."""
    stages = schedule._stages
    if len(stages) != len(model_parts):
        raise RuntimeError("pipeline schedule and model parts disagree")
    result: dict[int, list[DistMoeRoutedExperts]] = {}
    for stage, part in zip(stages, model_parts, strict=True):
        modules = [
            module
            for module in part.modules()
            if isinstance(module, DistMoeRoutedExperts)
        ]
        if modules:
            result[stage.stage_index] = modules
    return result


def _pipeline_slots(
    schedule: PipelineScheduleMulti,
    *,
    pp_rank: int,
    stage_modules: dict[int, list[DistMoeRoutedExperts]],
    policy: DistMoeBackendConfig,
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


def setup_dist_moe(
    *,
    config: Trainer.Config,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
    pp_schedule: object | None,
) -> None:
    """Create one annex context and bind every local Dist-MoE layer to it."""
    modules = [
        module
        for part in model_parts
        for module in part.modules()
        if isinstance(module, DistMoeRoutedExperts)
    ]
    modules = list({id(module): module for module in modules}.values())
    if not modules or config.checkpoint.create_seed_checkpoint:
        return
    if config.training.mixed_precision_param != "bfloat16":
        raise ValueError("Dist-MoE requires mixed_precision_param='bfloat16'")
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("Dist-MoE requires an SM100-or-newer CUDA device")
    policy = modules[0]._backend_config
    if any(module._backend_config != policy for module in modules[1:]):
        raise ValueError("All local Dist-MoE layers must share one backend policy")

    group = modules[0]._ep_group
    if group is None or any(module._ep_group is not group for module in modules):
        raise RuntimeError("Dist-MoE layers must share one expert-parallel group")
    local_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    token_shards = parallel_dims.cp * modules[0]._sp_size
    if local_tokens % token_shards:
        raise ValueError("Dist-MoE input tokens must divide evenly across CP and SP")
    max_num_tokens = local_tokens // token_shards

    num_slots = 1
    layer_depth = len(modules)
    assignments: dict[tuple[int, int], tuple[int, int]] = {}
    stage_modules: dict[int, list[DistMoeRoutedExperts]] = {}
    granularity = "none"
    if parallel_dims.pp_enabled:
        if not isinstance(pp_schedule, PipelineScheduleMulti):
            raise ValueError(
                "Dist-MoE pipeline activation planning requires a multi-stage schedule"
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
    elif policy.num_activation_slots is not None:
        num_slots = policy.num_activation_slots

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
        raise ValueError("All local Dist-MoE layers must resolve one annex config")

    ep_size = dist.get_world_size(group)
    memory_plan = plan_dist_moe_memory(
        annex_config,
        ep_size=ep_size,
        device=device,
    )
    prefetch = (
        prefetch_dist_moe_vmm(
            config=annex_config,
            ep_size=ep_size,
            device=device,
        )
        if memory_plan.uses_host_scratch and policy.prefetch_vmm
        else None
    )
    runtime = _DistMoeRuntime(annex_config, group, prefetch, assignments)
    for module in modules:
        module._runtime = runtime
    if assignments:
        assert isinstance(pp_schedule, PipelineScheduleMulti)
        for stage, part in zip(pp_schedule._stages, model_parts, strict=True):
            if stage.stage_index not in stage_modules:
                continue
            runtime._pipeline_hooks.append(
                part.register_forward_pre_hook(
                    runtime.select_from_stage_forward,
                    with_kwargs=True,
                )
            )
    logger.info("%s", memory_plan.explain())
    if assignments:
        logger.info(
            "Dist-MoE pipeline activation slots: policy=%s slots=%d depth=%d",
            granularity,
            num_slots,
            layer_depth,
        )


def cleanup_dist_moe(model_parts: list[torch.nn.Module]) -> None:
    """Close each distinct Dist-MoE runtime owned by ``model_parts``."""
    runtimes = {
        id(module._runtime): module._runtime
        for part in model_parts
        for module in part.modules()
        if isinstance(module, DistMoeRoutedExperts) and module._runtime is not None
    }
    for runtime in runtimes.values():
        runtime.close()
