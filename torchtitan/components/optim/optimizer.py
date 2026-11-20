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
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    ComputeLayout,
    DistMuon as FlexShardDistMuon,
)
from torchtitan.distributed.utils import build_alternative_fsdp_mesh, get_local_tensor

from .offloading import (
    check_pinned_host_rounding,
    cuda_pinned_host_zeros_like,
    pinned_host_bytes,
)
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


def _validate_adam_moments(
    *,
    moment_dtype: MomentDType,
    pinned_host_moment_fraction: float,
    fused: bool | None,
) -> None:
    if moment_dtype not in ("parameter", "bfloat16"):
        raise ValueError(f"Unsupported Adam moment dtype {moment_dtype!r}")
    if moment_dtype == "bfloat16" and fused is not True:
        raise ValueError("bfloat16 Adam moments require fused=True")
    if not 0.0 <= pinned_host_moment_fraction <= 1.0:
        raise ValueError(
            "pinned_host_moment_fraction must be in [0, 1], got "
            f"{pinned_host_moment_fraction}"
        )
    if pinned_host_moment_fraction == 0.0:
        return
    if fused is not True:
        raise ValueError("Pinned host Adam moments require fused=True")
    accelerator = torch.accelerator.current_accelerator()
    # A host without an accelerator only builds configs, e.g. CPU recipe tests.
    if accelerator is not None and (
        accelerator.type != "cuda" or torch.version.hip is not None
    ):
        raise ValueError(
            "Pinned host Adam moments require CUDA. "
            "Set pinned_host_moment_fraction=0."
        )


def _as_dtensor_like(param: Tensor, local: Tensor) -> Tensor:
    """Wrap a local state shard as a DTensor with ``param``'s mesh and placements.

    Returns ``local`` unchanged if ``param`` is not a DTensor. DCP then saves and
    reshards the state like the parameter.
    """
    if not isinstance(param, DTensor):
        return local
    return DTensor.from_local(
        local,
        param.device_mesh,
        param.placements,
        run_check=False,
        shape=param.shape,
        stride=param.stride(),
    )


def _as_cpu_like(moment: Tensor, host: Tensor) -> Tensor:
    """Wrap the CPU tensor ``host`` like ``moment``, without copying it.

    ``DTensor.from_local`` would copy a CPU tensor to the mesh device.
    """
    if not isinstance(moment, DTensor):
        return host
    # TODO: Use a public API once DTensor can wrap a local tensor on another device
    # without a copy (upstream issue not filed yet). FSDP2 CPU offload uses this
    # private constructor the same way.
    return DTensor(host, moment._spec, requires_grad=False)


def _register_cpu_moment_hooks(
    optimizer: torch.optim.Optimizer, cpu_moments: dict[Tensor, dict[str, Tensor]]
) -> None:
    """Save and load pinned host moments through their CPU tensors.

    ``cpu_moments`` maps each parameter with pinned host moments to its CPU
    moments. Other parameters keep the default behavior. ``state_dict()`` returns
    the CPU moments, so checkpointers copy them on the CPU. A CUDA copy between two
    host buffers is synchronous and much slower. ``load_state_dict()`` keeps the
    CUDA moments in the state and copies loaded values into them, unless the values
    were loaded in place into the CPU moments. Without that, it would move the CPU
    tensors to the parameter device.
    """

    def params() -> list[Tensor]:
        return [param for group in optimizer.param_groups for param in group["params"]]

    def use_cpu_moments(
        _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # Checkpointers copy these CPU tensors on the host, outside stream order.
        # Wait for queued optimizer steps that write the moments.
        torch.accelerator.current_stream().synchronize()
        for index, param in enumerate(params()):
            if param in cpu_moments:
                state = state_dict["state"][index]
                state_dict["state"][index] = {**state, **cpu_moments[param]}
        return state_dict

    def load_into_pinned_memory(
        _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        param_list = params()
        state = dict(state_dict["state"])
        for index, saved in state.items():
            param = param_list[index]
            if param not in cpu_moments:
                continue
            saved = dict(saved)
            for key, cpu_moment in cpu_moments[param].items():
                moment = optimizer.state[param][key]
                if saved[key] is not cpu_moment:
                    # Copy on the stream, after pending optimizer steps. A CPU
                    # copy into the pinned memory would race with them.
                    moment.copy_(saved[key])
                saved[key] = moment
            state[index] = saved
        return {**state_dict, "state": state}

    optimizer.register_state_dict_post_hook(use_cpu_moments)
    optimizer.register_load_state_dict_pre_hook(load_into_pinned_memory)


def _init_pinned_host_moments(
    optimizer: torch.optim.Optimizer,
    *,
    fraction: float,
    moment_dtype: MomentDType,
) -> None:
    """Create the Adam state of the first parameters, with moments in pinned memory.

    Parameters are taken in optimizer order until their moments reach ``fraction``
    of all local moment bytes. Each gets the state ``init_optim_state`` would
    create, except that ``exp_avg`` and ``exp_avg_sq`` live in pinned CPU memory
    that CUDA kernels can access. ``init_optim_state`` skips parameters with
    state, so DCP loads into the pinned memory directly. The moment addresses never
    change, so a captured optimizer graph stays valid. The other parameters get
    device moments lazily, in the same fused step.
    """
    params = [
        (name, param)
        for group in optimizer.param_groups
        for name, param in zip(group["param_names"], group["params"], strict=True)
    ]

    def moment_dtype_of(local: Tensor) -> torch.dtype:
        return torch.bfloat16 if moment_dtype == "bfloat16" else local.dtype

    total_bytes = sum(
        2 * local.numel() * moment_dtype_of(local).itemsize
        for local in (get_local_tensor(param) for _, param in params)
    )
    target_bytes = fraction * total_bytes
    offloaded_bytes = 0
    bytes_before = pinned_host_bytes()
    cpu_moments: dict[Tensor, dict[str, Tensor]] = {}
    for name, param in params:
        if offloaded_bytes >= target_bytes:
            break
        local = get_local_tensor(param)
        # An empty shard needs no memory and keeps lazy device moments.
        if local.numel() == 0:
            continue
        if local.device.type != "cuda":
            raise ValueError(
                "Pinned host Adam moments require CUDA parameters, got "
                f"{local.device}. Set pinned_host_moment_fraction=0 with "
                "parallelism.fsdp_enable_cpu_offloading or create_seed_checkpoint."
            )
        if not local.is_contiguous():
            raise ValueError(
                "Pinned host Adam moments require contiguous parameters, got "
                f"strides {local.stride()} for {name} with "
                f"shape {tuple(local.shape)}"
            )
        state = optimizer.state[param]
        state["step"] = torch.zeros((), dtype=torch.float32, device=local.device)
        cpu_moments[param] = {}
        for key in ("exp_avg", "exp_avg_sq"):
            moment, host = cuda_pinned_host_zeros_like(local, moment_dtype_of(local))
            offloaded_bytes += host.nbytes
            state[key] = _as_dtensor_like(param, moment)
            cpu_moments[param][key] = _as_cpu_like(state[key], host)
    check_pinned_host_rounding(
        requested_bytes=offloaded_bytes, bytes_before=bytes_before
    )
    _register_cpu_moment_hooks(optimizer, cpu_moments)


def _register_bfloat16_moment_hooks(optimizer: torch.optim.Optimizer) -> None:
    """Create Adam moment tensors in bfloat16."""

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

    optimizer.register_step_pre_hook(initialize_moments)


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
        pinned_host_moment_fraction: float = 0.0
        """Share of local moment bytes kept in pinned CPU memory, where the fused
        kernel reads and writes them in place. 0 keeps all moments on the device,
        1 offloads all. See ``PINNED_HOST_MOMENTS.md``."""

        def __post_init__(self) -> None:
            _validate_adam_moments(
                moment_dtype=self.moment_dtype,
                pinned_host_moment_fraction=self.pinned_host_moment_fraction,
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
        if config.pinned_host_moment_fraction > 0:
            _init_pinned_host_moments(
                self,
                fraction=config.pinned_host_moment_fraction,
                moment_dtype=config.moment_dtype,
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
        pinned_host_moment_fraction: float = 0.0
        """Share of local moment bytes kept in pinned CPU memory, where the fused
        kernel reads and writes them in place. 0 keeps all moments on the device,
        1 offloads all. See ``PINNED_HOST_MOMENTS.md``."""

        def __post_init__(self) -> None:
            _validate_adam_moments(
                moment_dtype=self.moment_dtype,
                pinned_host_moment_fraction=self.pinned_host_moment_fraction,
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
        if config.pinned_host_moment_fraction > 0:
            _init_pinned_host_moments(
                self,
                fraction=config.pinned_host_moment_fraction,
                moment_dtype=config.moment_dtype,
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
