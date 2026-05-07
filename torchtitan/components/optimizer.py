# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, overload, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import Replicate
from torch.optim import Optimizer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

__all__ = [
    "OptimizersContainer",
    "OptimizersInBackwardContainer",
    "ParamGroupConfig",
    "register_moe_load_balancing_hook",
]


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with custom optimizer settings.

    Each entry specifies a regex pattern matching parameter FQNs. Parameters
    matching the pattern can override optimizer hyperparameters or switch to a
    different optimizer entirely.

    There are two override modes:

    1. **Same optimizer, different hyperparams** (``optimizer_name`` is None):
       ``optimizer_kwargs`` are merged on top of the global optimizer defaults.
       Unspecified kwargs inherit from the global config.

    2. **Different optimizer** (``optimizer_name`` is set):
       ``optimizer_kwargs`` must be self-contained — no fallbacks from the global
       config. If required kwargs (e.g. ``lr``) are missing, PyTorch's optimizer
       constructor will raise at init time.
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*'"""

    optimizer_name: str | None = None
    """Override the optimizer type for this group. None means use the global
    default. When set, ``optimizer_kwargs`` must provide all required kwargs."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Optimizer-specific keyword arguments.
    When ``optimizer_name`` is None: merged on top of global defaults.
    When ``optimizer_name`` is set: standalone kwargs for the new optimizer."""


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers, supporting mixed optimizer types.

    This class wraps multiple optimizers into a single object to simplify the
    training loop. It supports a two-layer configuration:

    - **Global defaults** (``Config``): the default optimizer type and kwargs.
    - **Per-group overrides** (``ParamGroupConfig``): override kwargs within the
      same optimizer, or switch to a different optimizer entirely.

    Each model part (from pipeline parallelism) may have multiple optimizer
    instances if different parameter groups use different optimizer types.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    Args:
        config (Config): Optimizer configuration including param group overrides.
        model_parts (List[nn.Module]): List of model parts to be optimized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        name: str = "AdamW"
        """Default optimizer to use"""

        lr: float = 8e-4
        """Learning rate to use"""

        optimizer_kwargs: dict[str, Any] = field(
            default_factory=lambda: {
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        )
        """Optimizer-specific keyword arguments passed to the optimizer constructor.
        Defaults are for AdamW. Override entirely when using a different optimizer,
        e.g. ``{"momentum": 0.9}`` for SGD."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """
        Specify which optimizer implementation to use for the default optimizer:
        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
          momentum and variance in bfloat16 via a step pre-hook so the fused
          CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
          Only supported for Adam/AdamW with OptimizersContainer (not
          OptimizersInBackwardContainer). See docs/bf16_optimizer_states.md.
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        param_groups: list[ParamGroupConfig] = field(default_factory=list)
        """Optional per-parameter-group overrides. Each entry specifies a regex
        pattern matching parameter FQNs. Parameters can override optimizer kwargs
        or switch to a different optimizer entirely.
        Parameters not matching any pattern use the global defaults.
        Patterns are checked in order; first match wins."""

        def __post_init__(self):
            if self.implementation == "fused_opt_states_bf16":
                if self.name not in ("Adam", "AdamW"):
                    raise ValueError(
                        "implementation='fused_opt_states_bf16' is only supported "
                        f"for Adam/AdamW, got optimizer '{self.name}'"
                    )

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_classes[name]

    @staticmethod
    def _build_optimizer_kwargs(config: Config) -> dict[str, Any]:
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "lr": config.lr,
            "fused": fused,
            "foreach": config.implementation == "foreach",
            **config.optimizer_kwargs,
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        config: Config,
        default_kwargs: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build PyTorch param groups from model parameters, partitioned by optimizer.

        Each parameter is assigned to the first matching ParamGroupConfig pattern,
        or to the default group if no pattern matches. Returns a dict mapping
        optimizer name to a list of param group dicts for that optimizer.

        Each param group dict includes a ``_label`` key with a sanitized pattern
        string (or ``"default"`` for unmatched params) for logging.

        Override modes:
        - ``optimizer_name`` is None: ``optimizer_kwargs`` merge on top of defaults.
        - ``optimizer_name`` is set: ``optimizer_kwargs`` are standalone.
        """
        if not config.param_groups:
            params = [p for p in model.parameters() if p.requires_grad]
            return {
                config.name: [{"params": params, "_label": "default", **default_kwargs}]
            }

        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        claimed: set[
            str
        ] = set()  # first-match-wins: a param is assigned to the first matching pattern

        for pg in config.param_groups:
            pattern = re.compile(pg.pattern)
            matched = []
            for name, param in model.named_parameters():
                if param.requires_grad and name not in claimed and pattern.search(name):
                    matched.append(param)
                    claimed.add(name)

            if not matched:
                logger.warning(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )
                continue

            label = re.sub(r"[^a-zA-Z0-9._]", "", pg.pattern.replace("|", "_or_"))
            opt_name = pg.optimizer_name or config.name
            base_kwargs = {} if pg.optimizer_name else default_kwargs
            # optimizer_kwargs comes last so user overrides take precedence
            result[opt_name].append(
                {
                    "params": matched,
                    "_label": label,
                    **base_kwargs,
                    **pg.optimizer_kwargs,
                }
            )

        # Default group: unclaimed params
        default_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and name not in claimed
        ]
        if default_params:
            result[config.name].insert(
                0, {"params": default_params, "_label": "default", **default_kwargs}
            )

        return dict(result)

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        default_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        # Maps each optimizer to its model part (for state_dict/load_state_dict)
        self._model_part_indices: list[int] = []

        for part_idx, model in enumerate(self.model_parts):
            groups_by_opt = self._build_param_groups(model, config, default_kwargs)
            for opt_name, param_groups in groups_by_opt.items():
                opt_cls = self._resolve_optimizer_cls(opt_name)
                self.optimizers.append(opt_cls(param_groups))
                self._model_part_indices.append(part_idx)
                for group in param_groups:
                    all_params.extend(group["params"])

        self._validate_params(all_params)

        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params, default_kwargs)
        self._log_summary()

    def _log_summary(self) -> None:
        """Log a summary of optimizer assignments."""
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
        for i, opt in enumerate(self.optimizers):
            opt_name = type(opt).__name__
            part_idx = self._model_part_indices[i]
            for group in opt.param_groups:
                num_params = len(group["params"])
                kwargs = {k: v for k, v in group.items() if k in _KEY_KWARGS}
                label = group["_label"]
                logger.info(
                    f"Optimizer {opt_name} (model_part={part_idx}): "
                    f"{num_params} params [{label}] {kwargs}"
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
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        result = {}
        for opt, part_idx in zip(self.optimizers, self._model_part_indices):
            sd = func(self.model_parts[part_idx], opt)
            result.update(sd)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for opt, part_idx in zip(self.optimizers, self._model_part_indices):
            func(self.model_parts[part_idx], opt)

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

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


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        def __post_init__(self) -> None:
            if self.implementation == "fused_opt_states_bf16":
                raise ValueError(
                    "implementation='fused_opt_states_bf16' is not supported with "
                    "OptimizersInBackwardContainer"
                )
            OptimizersContainer.Config.__post_init__(self)

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        default_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.model_parts = model_parts
        # Maps each optimizer to its model part (for state_dict/load_state_dict)
        self._model_part_indices: list[int] = []

        optim_dict: dict[nn.Parameter, Optimizer] = {}
        for part_idx, model in enumerate(self.model_parts):
            groups_by_opt = self._build_param_groups(model, config, default_kwargs)
            for opt_name, param_groups in groups_by_opt.items():
                opt_cls = self._resolve_optimizer_cls(opt_name)
                for group in param_groups:
                    label = group["_label"]
                    kwargs = {
                        k: v for k, v in group.items() if k not in ("params", "_label")
                    }
                    for p in group["params"]:
                        optim_dict[p] = opt_cls(
                            [{"params": [p], "_label": label, **kwargs}]
                        )
                        self._model_part_indices.append(part_idx)
            all_params.extend(p for p in model.parameters())

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())
        self._validate_params(all_params)
        self._post_init(all_params, default_kwargs)
        self._log_summary()

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass


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

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

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
        tokens_per_expert_list = []
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if loss_mesh is not None:
            if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                # Perform single all-reduce to get global statistics across all processes
                pg = loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = model_part.get_submodule("layers")
                assert isinstance(layers, nn.ModuleDict)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    # pyrefly: ignore [missing-attribute]
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    # pyrefly: ignore [missing-attribute]
                    moe.expert_bias.add_(expert_bias_delta)
                    # pyrefly: ignore [missing-attribute]
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
