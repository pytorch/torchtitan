# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast, Literal, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    ComputeLayout,
    DistMuon as FlexShardDistMuon,
)
from torchtitan.distributed.utils import build_alternative_fsdp_mesh

from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

logger = logging.getLogger(__name__)


__all__ = [
    "Adam",
    "AdamW",
    "BaseOptimizer",
    "DistMuon",
    "OptimizersContainer",
]

MomentDType = Literal["parameter", "bfloat16"]


class BaseOptimizer(Optimizer, Configurable):
    """Base class for configurable TorchTitan optimizers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pattern: str
        """Regex matched against parameter fully qualified names."""


def _validate_moment_dtype(
    *,
    moment_dtype: MomentDType,
    fused: bool | None,
) -> None:
    if moment_dtype not in ("parameter", "bfloat16"):
        raise ValueError(f"Unsupported Adam moment dtype {moment_dtype!r}")
    if moment_dtype == "bfloat16" and fused is not True:
        raise ValueError("bfloat16 Adam moments require fused=True")


def _register_bfloat16_moment_hooks(optimizer: torch.optim.Optimizer) -> None:
    """Create and restore Adam moment tensors in bfloat16."""

    def initialize_moments(
        optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict
    ) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = optimizer.state[param]
                if state:
                    continue
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=param.device)
                    if group.get("capturable") or group.get("fused")
                    else torch.tensor(0.0, dtype=torch.float32)
                )
                state["exp_avg"] = torch.zeros_like(
                    param, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param,
                        dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )

    def restore_moment_dtype(optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key] = state[key].to(dtype=torch.bfloat16)

    optimizer.register_step_pre_hook(initialize_moments)
    optimizer.register_load_state_dict_post_hook(restore_moment_dtype)


class Adam(torch.optim.Adam, BaseOptimizer):
    """Configurable ``torch.optim.Adam``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0.0
        fused: bool | None = True
        moment_dtype: MomentDType = "parameter"

        def __post_init__(self) -> None:
            _validate_moment_dtype(
                moment_dtype=self.moment_dtype,
                fused=self.fused,
            )

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=config.fused,
        )
        if config.moment_dtype == "bfloat16":
            _register_bfloat16_moment_hooks(self)


class AdamW(torch.optim.AdamW, BaseOptimizer):
    """Configurable ``torch.optim.AdamW``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 8e-4
        betas: tuple[float, float] = (0.9, 0.95)
        eps: float = 1e-8
        weight_decay: float = 0.1
        foreach: bool | None = None
        fused: bool | None = True
        moment_dtype: MomentDType = "parameter"

        def __post_init__(self) -> None:
            _validate_moment_dtype(
                moment_dtype=self.moment_dtype,
                fused=self.fused,
            )

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            fused=config.fused,
        )
        if config.moment_dtype == "bfloat16":
            _register_bfloat16_moment_hooks(self)


class DistMuon(FlexShardDistMuon, BaseOptimizer):
    """Configurable TorchTitan wrapper around FlexShard's ``DistMuon``.

    Note:
        The CTA cap is disabled by default because it is a workload-dependent
        memory/performance tradeoff. Enabling it builds explicit redistribution
        meshes backed by resource-independent NCCL child communicators and
        passes them through ``BucketConfig``. Each communicator adds fixed
        memory and may reduce throughput; the cap benefits workloads only when
        the reduction in all-peer connection state outweighs that overhead.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        compute_sharding_by_fqn: Mapping[str, ComputeLayout]
        bucket_configs: Sequence[BucketConfig]
        lr: float = 1e-3
        weight_decay: float = 0.1
        momentum: float = 0.95
        nesterov: bool = True
        ns_coefficients: (
            tuple[float, float, float] | tuple[tuple[float, float, float], ...]
        ) = (3.4445, -4.7750, 2.0315)
        eps: float = 1e-7
        ns_steps: int = 5
        adjust_lr_fn: Literal[
            "original", "match_rms_adamw", "spectral_unclamped"
        ] | None = None
        ns_device: Literal["accelerator"] | None = None
        """Where Newton-Schulz runs. ``None`` keeps the storage device, where each
        parameter's local shard lives (the host under FSDP2 CPU offload).
        ``"accelerator"`` copies each matrix to the current accelerator, runs
        Newton-Schulz there and copies the result back. Momentum, parameters,
        gradients and the exchange between ranks stay where they are, so this only
        changes anything when the parameters live on the host."""
        max_nccl_ctas_for_comms: int = -1  # Cap NCCL CTAs to save memory; <=0 disables.
        """Limit NCCL CTAs for DistMuon redistribution only; ``<= 0`` disables.

        All-to-all opens connection buffers for every peer and channel. A lower
        CTA cap reduces this connection state on dedicated child communicators
        of the ``dp_shard``/``edp_shard`` groups, while FSDP keeps its original
        communicators. Each child adds fixed workspace memory, and fewer CTAs
        may reduce all-to-all bandwidth.

        A positive value requires NCCL on the FSDP shard axes. Other backends
        raise ``ValueError``; fake warns and keeps the storage groups without
        modeling the cap.

        TODO: Replace this temporary all-to-all connection-memory workaround
        with lower-memory redistribution, such as a pull-based symmetric-memory
        implementation.
        """

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        param_groups = tuple(cast(Iterable[dict[str, Any]], params))
        bucket_configs = config.bucket_configs
        if config.max_nccl_ctas_for_comms > 0:
            if len(param_groups) != 1:
                raise ValueError("DistMuon requires exactly one parameter group")
            param_group = param_groups[0]
            group_names = param_group.get("param_names")
            if group_names is None:
                raise ValueError("DistMuon requires param_names")
            named_parameters = tuple(
                zip(group_names, param_group["params"], strict=True)
            )

            resolved_bucket_configs = []
            # OptimizersContainer builds this instance for one local model part;
            # reuse meshes across its buckets without keeping a run-wide cache.
            redistribution_mesh_by_storage_mesh: dict[DeviceMesh, DeviceMesh] = {}
            for bucket_config in bucket_configs:
                matched_parameter = next(
                    (
                        parameter
                        for fqn, parameter in named_parameters
                        if any(
                            fnmatch.fnmatchcase(fqn, pattern)
                            for pattern in bucket_config.patterns
                        )
                    ),
                    None,
                )
                if matched_parameter is None:
                    resolved_bucket_configs.append(bucket_config)
                    continue
                storage_mesh = cast(Any, matched_parameter).device_mesh
                if storage_mesh not in redistribution_mesh_by_storage_mesh:
                    redistribution_mesh_by_storage_mesh[
                        storage_mesh
                    ] = build_alternative_fsdp_mesh(
                        storage_mesh,
                        max_ctas=config.max_nccl_ctas_for_comms,
                    )
                resolved_bucket_configs.append(
                    replace(
                        bucket_config,
                        mesh=redistribution_mesh_by_storage_mesh[storage_mesh],
                    )
                )
            bucket_configs = resolved_bucket_configs

        super().__init__(
            param_groups,
            compute_sharding_by_fqn=config.compute_sharding_by_fqn,
            bucket_configs=bucket_configs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_coefficients=config.ns_coefficients,
            eps=config.eps,
            ns_steps=config.ns_steps,
            adjust_lr_fn=config.adjust_lr_fn,
            ns_device=config.ns_device,
        )


class OptimizersContainer(Optimizer, Stateful, Configurable):
    """A container for multiple optimizers, supporting mixed optimizer types.

    This class wraps multiple optimizers into a single object to simplify the
    training loop. Each configured optimizer selects parameters by regex pattern;
    patterns are checked in order and the first match wins.

    Each model part gets one optimizer instance per matching configuration.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    Args:
        config (Config): Ordered optimizer configurations.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        enable_accelerator_graph: Prepare every optimizer for accelerator graph capture.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizers: list[BaseOptimizer.Config] = field(default_factory=list)
        """Optimizer configurations in first-match-wins pattern order."""

    optimizers: list[Optimizer]
    model_parts: list[nn.Module]

    @staticmethod
    def _build_param_group(
        model: nn.Module,
        optimizer_config: BaseOptimizer.Config,
        claimed: set[str],
    ) -> dict[str, Any]:
        """Build one named PyTorch parameter group from an optimizer config."""
        try:
            pattern = re.compile(optimizer_config.pattern)
        except re.error as error:
            raise ValueError(
                f"Invalid optimizer pattern {optimizer_config.pattern!r}: {error}"
            ) from error

        params: list[nn.Parameter] = []
        param_names: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad and name not in claimed and pattern.search(name):
                params.append(param)
                param_names.append(canonical_fqn(name))
                claimed.add(name)

        if not params:
            raise ValueError(
                f"Optimizer pattern {optimizer_config.pattern!r} matched no parameters"
            )

        return {"params": params, "param_names": param_names}

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        enable_accelerator_graph: bool = False,
    ) -> None:
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(self.model_parts):
            claimed: set[str] = set()
            for optimizer_config in config.optimizers:
                param_group = self._build_param_group(
                    model,
                    optimizer_config,
                    claimed,
                )
                optimizer = optimizer_config.build(params=[param_group])
                if enable_accelerator_graph:
                    if any(
                        "capturable" not in group for group in optimizer.param_groups
                    ):
                        raise ValueError(
                            f"Optimizer {type(optimizer).__name__} does not support "
                            "accelerator graph capture."
                        )
                    for group in optimizer.param_groups:
                        group["initial_lr"] = group["lr"]
                        group["capturable"] = True
                        group["lr"] = torch.tensor(
                            group["lr"],
                            dtype=torch.float32,
                            device=group["params"][0].device,
                        )

                    def _save_host_lr(
                        _optimizer: Optimizer, state_dict: dict[str, Any]
                    ) -> dict[str, Any]:
                        for group in state_dict["param_groups"]:
                            if isinstance(group["lr"], torch.Tensor):
                                group["lr"] = float(group["lr"])
                        return state_dict

                    optimizer.register_state_dict_post_hook(_save_host_lr)
                self.optimizers.append(optimizer)
                self._log_optimizer(
                    optimizer,
                    part_idx,
                    optimizer_config.pattern,
                )
                all_params.extend(param_group["params"])

        self._validate_params(all_params)
        self._post_init(all_params)

    def _log_optimizer(self, optimizer: Optimizer, part_idx: int, pattern: str) -> None:
        """Log one optimizer's parameter assignment."""
        _KEY_KWARGS = {
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "momentum",
            "nesterov",
            "fused",
            "foreach",
        }
        opt_name = type(optimizer).__name__
        for group in optimizer.param_groups:
            num_params = len(group["params"])
            kwargs = {k: v for k, v in group.items() if k in _KEY_KWARGS}
            logger.info(
                f"Optimizer {opt_name} (model_part={part_idx}): "
                f"{num_params} params [{pattern}] {kwargs}"
            )

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Verify every trainable param is assigned to exactly one optimizer."""
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad
        }
        actual = {id(p) for p in all_params}
        if len(all_params) != len(actual):
            raise ValueError(
                "A trainable parameter was assigned to multiple optimizers"
            )
        if expected != actual:
            raise ValueError(
                f"Parameter mismatch: {len(expected)} trainable params in model, "
                f"{len(actual)} in optimizers"
            )

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return a flat, FQN-keyed optimizer state dict for all optimizers.

        Side effect: if an optimizer's state has not been created yet (no training
        step taken), ``init_optim_state`` materializes it with a zero-gradient,
        zero-lr step before reading. The step leaves parameters unchanged, and the
        call is a no-op once state exists.
        """
        result: dict[str, Any] = {}
        for optim in self.optimizers:
            init_optim_state(optim)
            result.update(get_flat_optim_state_dict(optim))
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # init_optim_state must run first: the unflatten step reads each
        # optimizer's live state to learn which state tensors to expect.
        for optim in self.optimizers:
            init_optim_state(optim)
            load_flat_optim_state_dict(optim, state_dict)

    def _post_init(self, all_params: list[nn.Parameter]) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks (e.g. register_step_pre_hook for MoE load balancing).
        Optimizer.__init__(self, all_params, {})

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass
