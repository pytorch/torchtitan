# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan adapter for the standalone BF16 DistMoE package."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist
from dist_moe import (
    create_context,
    dist_moe as run_dist_moe,
    DistMoeConfig as KernelConfig,
    DistMoeContext,
    DistMoeVmmConfig,
    plan_dist_moe_memory,
    prefetch_dist_moe_vmm,
)
from torch.distributed.tensor import DTensor

from torchtitan.distributed.parallel_dims import ParallelDims, SpmdLayout
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from ._fused_weights import make_fused_gate_up_init

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

__all__ = [
    "cleanup_dist_moe",
    "dist_moe_config",
    "DistMoeBackendConfig",
    "DistMoeRoutedExperts",
    "setup_dist_moe",
]


@dataclass(kw_only=True, slots=True)
class DistMoeBackendConfig:
    """Execution and activation-memory policy for BF16 DistMoE."""

    max_routing_imbalance_factor: float = 1.0
    device_memory_budget_bytes: int | Literal["maximum_useful"] | None = None
    vmm_host_scratch_imbalance_factor: float | Literal["auto"] | None = "auto"
    num_sms: int | None = None
    kernel_config: str | None = None
    wgrad_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    def __post_init__(self) -> None:
        budget = self.device_memory_budget_bytes
        if budget != "maximum_useful" and budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
        ):
            raise ValueError(
                "device_memory_budget_bytes must be 'maximum_useful', None, "
                "or a positive integer"
            )


@dataclass(eq=False)
class _DistMoeRuntime:
    config: KernelConfig
    group: dist.ProcessGroup
    prefetch: Any | None
    context: DistMoeContext | None = None

    def initialize(self, device: torch.device) -> DistMoeContext:
        if self.context is None:
            self.context = create_context(
                group=self.group,
                config=self.config,
                device=device,
                prefetched_vmm=self.prefetch,
            )
            self.prefetch = None
        return self.context

    def close(self) -> None:
        if self.context is not None:
            self.context.close()
            self.context = None


_ACTIVE_RUNTIMES: list[_DistMoeRuntime] = []


class DistMoeRoutedExperts(RoutedExperts):
    """Routed-expert adapter around fused DistMoE dispatch and GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        supports_cuda_graphs: ClassVar[bool] = True
        backend: DistMoeBackendConfig = field(default_factory=DistMoeBackendConfig)

        def build(self, **kwargs) -> DistMoeRoutedExperts:
            config = replace(
                self,
                param_init=_fuse_param_init(self.inner_experts.param_init),
                sharding_config=_fuse_sharding(
                    self.sharding_config,
                    self.inner_experts.sharding_config,
                ),
            )
            return Module.Config.build(config, **kwargs)

    def __init__(self, config: Config):
        Module.__init__(self)
        experts = config.inner_experts
        self.num_experts = experts.num_experts
        self.hidden_dim = experts.dim
        self.intermediate_dim = experts.hidden_dim
        self.top_k = config.token_dispatcher.top_k
        self.w13 = torch.nn.Parameter(
            torch.empty(
                experts.num_experts,
                2,
                experts.hidden_dim,
                experts.dim,
            )
        )
        self.w2_EDF = torch.nn.Parameter(
            torch.empty(experts.num_experts, experts.dim, experts.hidden_dim)
        )
        self._runtime_policy = config.backend
        self._runtime: _DistMoeRuntime | None = None
        self._ep_group: dist.ProcessGroup | None = None
        self._sp_size = 1

        self.register_state_dict_post_hook(type(self)._split_fused_state_on_save)
        self.register_load_state_dict_pre_hook(type(self)._merge_fused_state_on_load)

    @property
    def sp_size(self) -> int:
        return self._sp_size

    def expert_parameters_module(self) -> torch.nn.Module:
        return self

    def synchronize(self) -> None:
        pass

    def parallelize(self, parallel_dims: ParallelDims) -> None:
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
            if self.w13.device.type == "cpu" and buffer_device is None:
                return
            raise RuntimeError("DistMoE backend runtime was not initialized")
        self._runtime.initialize(torch.device(buffer_device or self.w13.device))

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        del num_local_tokens_per_expert_E
        if self._runtime is None or self._runtime.context is None:
            raise RuntimeError("DistMoE context is not initialized")

        w13 = self.w13.to_local() if isinstance(self.w13, DTensor) else self.w13
        w2 = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
        return run_dist_moe(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13.flatten(1, 2),
            w2,
            self._runtime.context,
        )

    @staticmethod
    def _split_fused_state_on_save(module, state_dict, prefix, local_metadata) -> None:
        del module, local_metadata
        w13 = state_dict.pop(f"{prefix}w13")
        w2 = state_dict.pop(f"{prefix}w2_EDF")
        stock_prefix = f"{prefix}inner_experts."
        state_dict[f"{stock_prefix}w1_EFD"] = w13[:, 0].contiguous()
        state_dict[f"{stock_prefix}w2_EDF"] = w2
        state_dict[f"{stock_prefix}w3_EFD"] = w13[:, 1].contiguous()

    @staticmethod
    def _merge_fused_state_on_load(module, state_dict, prefix, *args) -> None:
        del module, args
        stock_prefix = f"{prefix}inner_experts."
        w1_key, w3_key = f"{stock_prefix}w1_EFD", f"{stock_prefix}w3_EFD"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)],
                dim=1,
            )
        w2_key = f"{stock_prefix}w2_EDF"
        if w2_key in state_dict:
            state_dict[f"{prefix}w2_EDF"] = state_dict.pop(w2_key)


def _fuse_param_init(param_init: dict | None) -> dict | None:
    if param_init is None:
        return None
    w1_init = param_init.get("w1_EFD")
    w3_init = param_init.get("w3_EFD")
    fused = {
        key: value
        for key, value in param_init.items()
        if key not in ("w1_EFD", "w3_EFD")
    }
    if w1_init is not None and w3_init is not None:
        fused["w13"] = make_fused_gate_up_init(w1_init, w3_init, gate_up_axis=1)
    return fused or None


def _insert_gate_axis(layout: SpmdLayout) -> SpmdLayout:
    axis_types = {
        axis: spmd.S(axis_type.dim + 1)
        if isinstance(axis_type, spmd.Shard) and axis_type.dim >= 1
        else axis_type
        for axis, axis_type in layout.axis_types.items()
    }
    partition_spec = layout.partition_spec
    if partition_spec is not None:
        partition_spec = (*partition_spec[:1], None, *partition_spec[1:])
    return SpmdLayout(axis_types, partition_spec=partition_spec)


def _fuse_sharding(
    routed: ShardingConfig | None,
    inner: ShardingConfig | None,
) -> ShardingConfig | None:
    if inner is None:
        return routed
    state = dict(inner.state_shardings)
    w1_layout = state.pop("w1_EFD")
    w3_layout = state.pop("w3_EFD")
    if w1_layout != w3_layout:
        raise ValueError("w1_EFD and w3_EFD must use identical sharding")
    state["w13"] = _insert_gate_axis(w1_layout)
    if routed is None:
        return ShardingConfig(state_shardings=state)
    return replace(routed, state_shardings=state)


def _kernel_config(
    module: DistMoeRoutedExperts,
    *,
    num_tokens: int,
    num_moe_layers: int,
) -> KernelConfig:
    policy = module._runtime_policy
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    vmm_factor = policy.vmm_host_scratch_imbalance_factor
    if vmm_factor == "auto":
        vmm_factor = 16.0
    if vmm_factor is not None and not isinstance(vmm_factor, (int, float)):
        raise TypeError(
            "vmm_host_scratch_imbalance_factor must be 'auto', a number, or None"
        )
    vmm = (
        None
        if vmm_factor is None
        else DistMoeVmmConfig(host_scratch_imbalance_factor=float(vmm_factor))
    )
    budget = policy.device_memory_budget_bytes
    return KernelConfig(
        max_num_tokens=num_tokens,
        hidden_dim=module.hidden_dim,
        intermediate_dim=module.intermediate_dim,
        top_k=module.top_k,
        num_experts=module.num_experts,
        num_moe_layers=num_moe_layers,
        max_routing_imbalance_factor=policy.max_routing_imbalance_factor,
        device_memory_budget_bytes=None if budget == "maximum_useful" else budget,
        vmm=vmm,
        num_sms=policy.num_sms,
        kernel_config=policy.kernel_config,
        wgrad_dtype=dtype_by_name[policy.wgrad_dtype],
    )


def setup_dist_moe(
    *,
    config: Trainer.Config,
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
) -> None:
    modules = [
        module
        for part in model_parts
        for module in part.modules()
        if isinstance(module, DistMoeRoutedExperts)
    ]
    if not modules or config.checkpoint.create_seed_checkpoint:
        return
    if parallel_dims.pp > 1:
        raise ValueError("DistMoE pipeline parallelism is not enabled in this adapter")
    if config.training.mixed_precision_param != "bfloat16":
        raise ValueError("DistMoE requires training.mixed_precision_param='bfloat16'")
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        raise ValueError("DistMoE requires a Blackwell SM100 CUDA device")

    group = modules[0]._ep_group
    if group is None:
        ep_mesh = parallel_dims.get_optional_mesh("ep", include_singleton_axes=True)
        if ep_mesh is None:
            raise RuntimeError("DistMoE requires an expert-parallel mesh")
        group = ep_mesh.get_group()
    if any(module._ep_group not in (None, group) for module in modules[1:]):
        raise RuntimeError("DistMoE layers must share one expert-parallel group")

    total_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    shard_degree = parallel_dims.cp * modules[0].sp_size
    if total_tokens % shard_degree:
        raise ValueError("local token count must divide evenly across CP and SP")
    num_tokens = total_tokens // shard_degree
    kernel_config = _kernel_config(
        modules[0],
        num_tokens=num_tokens,
        num_moe_layers=len(modules),
    )
    for module in modules[1:]:
        if _kernel_config(
            module,
            num_tokens=num_tokens,
            num_moe_layers=len(modules),
        ) != kernel_config:
            raise ValueError("All local DistMoE layers must share one configuration")

    requested_budget = modules[0]._runtime_policy.device_memory_budget_bytes
    ep_size = dist.get_world_size(group)
    if requested_budget == "maximum_useful":
        plan = plan_dist_moe_memory(kernel_config, ep_size=ep_size)
        kernel_config = replace(
            kernel_config,
            device_memory_budget_bytes=plan.maximum_useful_device_budget_bytes,
        )
    plan = plan_dist_moe_memory(kernel_config, ep_size=ep_size)
    prefetch = (
        prefetch_dist_moe_vmm(
            config=kernel_config,
            ep_size=ep_size,
            device=device,
        )
        if plan.uses_host_scratch
        else None
    )
    runtime = _DistMoeRuntime(kernel_config, group, prefetch)
    _ACTIVE_RUNTIMES.append(runtime)
    for module in modules:
        module._runtime = runtime


def cleanup_dist_moe() -> None:
    while _ACTIVE_RUNTIMES:
        _ACTIVE_RUNTIMES.pop().close()


def dist_moe_config(
    config: RoutedExperts.Config,
    *,
    backend: DistMoeBackendConfig | None = None,
) -> DistMoeRoutedExperts.Config:
    return DistMoeRoutedExperts.Config(
        inner_experts=config.inner_experts,
        token_dispatcher=config.token_dispatcher,
        backend=backend or DistMoeBackendConfig(),
    )
