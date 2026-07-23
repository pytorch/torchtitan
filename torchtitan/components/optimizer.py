# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast, Generic, Literal, overload, Protocol, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import Replicate
from torch.optim import Optimizer
from torchtitan.components.checkpoint_utils import (
    canonical_fqn,
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.muon_adapter import MuonAdapter
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "ParamGroupConfig",
    "default_adamw",
    "register_moe_load_balancing_hook",
]


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with its own optimizer.

    Each entry specifies a regex pattern matching parameter FQNs and a
    self-contained optimizer setup. ``optimizer_name`` and ``optimizer_kwargs``
    fully define the optimizer for matched parameters — no implicit inheritance.

    Patterns are checked in order; first match wins. Place specific patterns
    before broad ones, and use ``r".*"`` as the last entry to catch all
    remaining parameters. Example::

        param_groups=[
            ParamGroupConfig(pattern=r"\\.bias$", ...),   # specific: biases first
            ParamGroupConfig(pattern=r"\\.router\\.", ...),  # specific: routers
            ParamGroupConfig(pattern=r".*", ...),          # catch-all: everything else
        ]
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*', '.*' (catch-all)"""

    optimizer_name: str
    """Optimizer type for this group."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the optimizer constructor.
    Must include all required kwargs (e.g. ``lr``). No implicit defaults."""


T = TypeVar("T", bound=Optimizer)


class _MoELike(Protocol):
    load_balance_coeff: float | None
    tokens_per_expert_E: torch.Tensor  # noqa: N815
    expert_bias_E: torch.Tensor  # noqa: N815


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers, supporting mixed optimizer types.

    This class wraps multiple optimizers into a single object to simplify the
    training loop. Each parameter group is configured via ``ParamGroupConfig``
    with its own optimizer type and kwargs. Parameters are matched to groups
    by regex pattern (first match wins), and groups using the same optimizer
    type are batched into a single optimizer instance for performance.

    Each model part (from pipeline parallelism) may have multiple optimizer
    instances if different parameter groups use different optimizer types.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    Args:
        config (Config): Optimizer configuration with param group definitions.
        model_parts (List[nn.Module]): List of model parts to be optimized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_groups: list[ParamGroupConfig] = field(default_factory=list)
        """Per-parameter-group optimizer configurations. Each entry specifies a
        regex pattern and a self-contained optimizer setup.
        Patterns are checked in order; first match wins."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """
        Optimizer implementation mode applied to optimizer instances that
        support ``fused`` and ``foreach``. Muon uses its built-in implementation
        and ignores this setting. Per-param-group ``optimizer_kwargs`` can
        override the generated implementation kwargs.

        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
          momentum and variance in bfloat16 via a step pre-hook so the fused
          CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
          Only supported for Adam/AdamW. See docs/bf16_optimizer_states.md.
        - more info: https://pytorch.org/docs/stable/optim.html
        """

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "Muon": MuonAdapter,
        }
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_classes[name]

    @staticmethod
    def _build_impl_kwargs(config: Config) -> dict[str, Any]:
        """Build implementation-related kwargs (fused/foreach) from config."""
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "fused": fused,
            "foreach": config.implementation == "foreach",
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        """Build PyTorch param groups from model parameters, partitioned by optimizer.

        Each parameter is assigned to the first matching ParamGroupConfig pattern.

        Returns two dicts keyed by optimizer name and aligned by index: the param
        group dicts to pass to the optimizer constructor, and the regex pattern of
        each group. Patterns are returned separately (not stored on the group) so
        they stay out of the saved optimizer state dict; they are logging-only.

        Each param group dict carries a ``param_names`` list (canonical FQNs
        aligned with ``params``) so PyTorch records the names on the group; the
        checkpoint utilities use those names to build FQN-keyed optimizer state.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        patterns: dict[str, list[str]] = defaultdict(list)
        claimed: set[str] = set()  # first-match-wins

        for pg in param_group_configs:
            pattern = re.compile(pg.pattern)
            params: list[nn.Parameter] = []
            param_names: list[str] = []
            for name, param in model.named_parameters():
                if param.requires_grad and name not in claimed and pattern.search(name):
                    params.append(param)
                    param_names.append(canonical_fqn(name))
                    claimed.add(name)

            if not params:
                raise ValueError(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )

            groups[pg.optimizer_name].append(
                {
                    "params": params,
                    "param_names": param_names,
                    # Muon does not accept fused/foreach options.
                    **(impl_kwargs if pg.optimizer_name != "Muon" else {}),
                    **pg.optimizer_kwargs,
                }
            )
            patterns[pg.optimizer_name].append(pg.pattern)

        return groups, patterns

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        param_group_configs = config.param_groups
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(self.model_parts):
            groups_by_opt_name, patterns_by_opt_name = self._build_param_groups(
                model, param_group_configs, impl_kwargs
            )
            for opt_name, opt_param_groups in groups_by_opt_name.items():
                optimizer = self._resolve_optimizer_cls(opt_name)(opt_param_groups)
                self.optimizers.append(optimizer)
                self._log_optimizer(optimizer, part_idx, patterns_by_opt_name[opt_name])
                for group in opt_param_groups:
                    all_params.extend(group["params"])

        self._validate_params(all_params)

        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)

    def _log_optimizer(
        self, optimizer: Optimizer, part_idx: int, patterns: list[str]
    ) -> None:
        """Log one optimizer's param-group assignments (patterns are logging-only)."""
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
        for group, pattern in zip(optimizer.param_groups, patterns):
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
        assert expected == actual, (
            f"Parameter mismatch: {len(expected)} trainable params in model, "
            f"{len(actual)} in optimizers"
        )

    def __iter__(self) -> Iterator[T]:
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

    def _register_bf16_optimizer_state_hook(self) -> None:
        """Register a step pre-hook to create Adam optimizer states in bfloat16.

        The hook pre-populates optimizer state before Adam's lazy initialization
        runs, so that ``_init_group`` finds non-empty state and skips its own
        fp32 allocation. The fused CUDA kernel then sees the dtype mismatch
        between fp32 params and bf16 states, dispatching to the mixed-precision
        kernel (``FusedAdamMathFunctorMP``).
        """

        def _bf16_state_init_hook(
            optimizer: Optimizer, args: tuple, kwargs: dict
        ) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((), dtype=torch.float32, device=p.device)
                            if group.get("capturable") or group.get("fused")
                            else torch.tensor(0.0, dtype=torch.float32)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        if group.get("amsgrad"):
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p,
                                dtype=torch.bfloat16,
                                memory_format=torch.preserve_format,
                            )

        for optim in self.optimizers:
            if isinstance(optim, (torch.optim.Adam, torch.optim.AdamW)):
                optim.register_step_pre_hook(_bf16_state_init_hook)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


def default_adamw(lr: float = 8e-4, **kwargs: Any) -> OptimizersContainer.Config:
    """Create an OptimizersContainer.Config with a catch-all AdamW param group.

    Use as a convenience for the common case::

        optimizer=default_adamw(lr=3e-4)
    """
    return OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    **kwargs,
                },
            )
        ]
    )


def register_moe_load_balancing_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load balancing.

    This function checks if MoE load balancing is enabled and, if so, registers
    a hook that updates expert biases before each optimizer step.

    Args:
        optimizers: The optimizers container to register the hook on.
        model_parts: List of model parts that may contain MoE layers.
        parallel_dims: Parallel dimensions for distributed communication.
    """

    def _iter_moe_layers(
        model_parts: list[nn.Module],
    ) -> Iterator[tuple[nn.Module, _MoELike]]:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if getattr(transformer_block, "moe_enabled", False):
                    yield transformer_block, cast(_MoELike, transformer_block.moe)

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        moe_layers = list(_iter_moe_layers(model_parts))
        if not moe_layers:
            return False

        load_balance_enabled = moe_layers[0][1].load_balance_coeff is not None
        for _transformer_block, moe in moe_layers[1:]:
            if (moe.load_balance_coeff is not None) != load_balance_enabled:
                raise ValueError(
                    "MoE load_balance_coeff must be configured consistently "
                    "across all MoE layers. Either set it for every MoE layer "
                    "or leave it unset for all MoE layers."
                )
        return load_balance_enabled

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_E_list = []
        for transformer_block, moe in _iter_moe_layers(model_parts):
            tokens_per_expert_E = moe.tokens_per_expert_E
            if _is_recomputation_enabled(transformer_block):
                # TODO: This is a hack, we assume with full AC, the tokens_per_expert_E is counted twice.
                # This does not affect to expert choice, but affects the experts usage metrics.
                # We divide by 2 to correct for this double-counting due to recomputation
                # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                tokens_per_expert_E = tokens_per_expert_E // 2
            tokens_per_expert_E_list.append(tokens_per_expert_E)

        if not tokens_per_expert_E_list:
            return

        tokens_per_expert_E_by_layer = torch.vstack(tokens_per_expert_E_list)

        if parallel_dims.spmd_backend == "full_dtensor":
            # full_dtensor: DTensor mesh includes all axes (DP/CP/TP/EP).
            # redistribute Partial→Replicate covers everything.
            assert isinstance(
                tokens_per_expert_E_by_layer, torch.distributed.tensor.DTensor
            )
            dtensor_mesh = tokens_per_expert_E_by_layer.device_mesh
            # TODO: This incurs multiple sequential all-reduces, one per
            # SPMD mesh axis. We should provide a utility to do a single all-reduce
            # on the flattened global SPMD mesh.
            tokens_per_expert_E_by_layer = tokens_per_expert_E_by_layer.redistribute(
                placements=[Replicate()] * dtensor_mesh.ndim
            )
        else:
            # non-full_dtensor: DTensor mesh only has TP/EP (if enabled).
            # full_tensor() reduces on TP/EP, then all-reduce on loss_mesh
            # covers DP/CP separately.
            is_dtensor = isinstance(
                tokens_per_expert_E_by_layer, torch.distributed.tensor.DTensor
            )
            if is_dtensor:
                dtensor_mesh = tokens_per_expert_E_by_layer.device_mesh
                tokens_per_expert_E_by_layer = (
                    tokens_per_expert_E_by_layer.full_tensor()
                )
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    tokens_per_expert_E_by_layer,
                    group=loss_mesh.get_group(),
                    op=torch.distributed.ReduceOp.SUM,
                )
            if is_dtensor:
                tokens_per_expert_E_by_layer = torch.distributed.tensor.DTensor.from_local(
                    tokens_per_expert_E_by_layer,
                    # pyrefly: ignore [unbound-name]
                    device_mesh=dtensor_mesh,
                    # pyrefly: ignore [unbound-name]
                    placements=[Replicate()] * dtensor_mesh.ndim,
                    run_check=False,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = model_part.get_submodule("layers")
                assert isinstance(layers, nn.ModuleDict)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = cast(_MoELike, transformer_block.moe)
                    load_balance_coeff = moe.load_balance_coeff
                    assert load_balance_coeff is not None

                    tokens_per_expert_E = tokens_per_expert_E_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    expert_bias_delta_E = load_balance_coeff * torch.sign(
                        tokens_per_expert_E.mean() - tokens_per_expert_E
                    )
                    expert_bias_delta_E = (
                        expert_bias_delta_E - expert_bias_delta_E.mean()
                    )
                    moe.expert_bias_E.add_(expert_bias_delta_E)
                    moe.tokens_per_expert_E.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
