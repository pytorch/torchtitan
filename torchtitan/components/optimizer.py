# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

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
    "register_moe_load_balancing_hook",
]


T = TypeVar("T", bound=Optimizer)


class BaseOptimizer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    optimizer_cls: type[Optimizer]
    config: Config

    def __init__(self, config: Config) -> None:
        self.config = config

    def __or__(self, other: "BaseOptimizer") -> "BaseOptimizer":
        """Merge configs: other overrides self. other.optimizer_cls always wins."""
        # Start from self's values, overlay other's non-None values
        merged_config = {
            **self.config.to_dict(),
            **{k: v for k, v in other.config.to_dict().items() if v is not None},
        }
        # Always use other's optimizer_cls (even if self had a different one)
        return type(other)(replace(other.config, **merged_config))

    def get_optimizer_kwargs(self) -> dict[str, Any]:
        """Get the optimizer kwargs from the config dataclass, excluding the optimizer_cls."""
        return {k: v for k, v in self.config.to_dict().items() if v is not None}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseOptimizer):
            return False
        # Must also check optimizer_cls, not just hyperparams
        return (
            self.optimizer_cls == other.optimizer_cls
            and self.config.to_dict() == other.config.to_dict()
        )

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.config.to_dict().items())))

    def __str__(self) -> str:
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.to_dict().items())
        return f"{self.optimizer_cls.__name__}({config_str})"

    def __repr__(self) -> str:
        return self.__str__()


class AdamW(BaseOptimizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 8e-4
        eps: float = 1e-8
        beta1: float = 0.9
        beta2: float = 0.95
        weight_decay: float = 0.1
        fused: bool = True
        foreach: bool = False

    optimizer_cls: type[Optimizer] = torch.optim.AdamW
    # pyrefly: ignore [bad-override]
    config: Config

    def get_optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.config.lr,
            "eps": self.config.eps,
            "betas": (self.config.beta1, self.config.beta2)
            if self.config.beta1 is not None and self.config.beta2 is not None
            else None,
            "fused": self.config.fused,
            "foreach": self.config.foreach,
            "weight_decay": self.config.weight_decay,
        }


class Adam(BaseOptimizer):
    @dataclass(kw_only=True, slots=True)
    class Config(AdamW.Config):
        weight_decay: float = 0.0

    optimizer_cls: type[Optimizer] = torch.optim.Adam
    # pyrefly: ignore [bad-override]
    config: Config

    def get_optimizer_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.config.lr,
            "eps": self.config.eps,
            "betas": (self.config.beta1, self.config.beta2),
            "fused": self.config.fused,
            "foreach": self.config.foreach,
            "weight_decay": self.config.weight_decay,
        }


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        default: BaseOptimizer.Config = field(default_factory=AdamW.Config)
        """Optimizer parameters"""

        overrides: dict[str, BaseOptimizer.Config] = field(default_factory=dict)
        """
        Parameter group overrides to use. For example
        {'model.tok_embeddings.weight' : AdamW.Config(lr=1e-4, weight_decay=0.0)}
        will override the weight decay and learning rate for the parameter group
        containing "model.tok_embeddings.weight" in its name.
        """

    optimizers: list[list[Optimizer]]
    model_parts: list[nn.Module]

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        all_params = []
        self.optimizers = [[]] * len(model_parts)
        self.model_parts = model_parts
        default_optimizer = config.default.build()
        overrid_optimizers = {k: v.build() for k, v in config.overrides.items()}

        # all empty parameter groups
        parts_group_params = [
            {
                default_optimizer | override_params: []
                for override_params in set(overrid_optimizers.values())
                | {default_optimizer}
            }
            for _ in range(len(model_parts))
        ]

        # populate parameter groups based on overrides,
        # if a parameter name matches multiple override keys, the first match will be used
        for model, group_params in zip(self.model_parts, parts_group_params):
            for name, param in [
                (n, p) for n, p in model.named_parameters() if p.requires_grad
            ]:
                all_params.append(param)
                if name in config.overrides:
                    override_params = overrid_optimizers[name]
                    group_params[default_optimizer | override_params].append(
                        (param, name)
                    )
                else:
                    group_params[default_optimizer].append((param, name))

        # assert all groups are not empty
        for group_params in parts_group_params:
            for config_params, params in group_params.items():
                if not params:
                    logger.warning(
                        f"Optimizer group {config_params} has no parameters assigned to it."
                    )

        # create optimizers for each group of parameters with the same optimizer class
        for i, group_params in enumerate(parts_group_params):
            for optimizer_cls in set(params.optimizer_cls for params in group_params):
                optimizer_groups = [
                    {
                        **optim_config.get_optimizer_kwargs(),
                        "params": [param[0] for param in named_params],
                        "param_names": [param[1] for param in named_params],
                    }
                    for optim_config, named_params in group_params.items()
                    if optim_config.optimizer_cls == optimizer_cls
                ]
                self.optimizers[i].append(
                    optimizer_cls(
                        optimizer_groups, **default_optimizer.get_optimizer_kwargs()
                    )
                )

        # log optimizer conguration
        for part in range(len(self.model_parts)):
            logger.info(f"Model part {part} optimizers:")
            for optimizer in self.optimizers[part]:
                logger.info(
                    f"Optimizer {optimizer.__class__.__name__} with param groups:"
                )
                for i, param_group in enumerate(optimizer.param_groups):
                    log_group = {
                        k: v
                        for k, v in param_group.items()
                        if k not in {"params", "param_names"}
                    }
                    logger.info(f"\tGroup {log_group} parameters")
                    for name in param_group.get("param_names", []):
                        logger.info(f"\t\t{name}")

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, default_optimizer.get_optimizer_kwargs())

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(
            optimizer
            for part_optimizers in self.optimizers
            for optimizer in part_optimizers
        )

    def __len__(self) -> int:
        return sum(len(part_optimizers) for part_optimizers in self.optimizers)

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        for part in range(len(self.model_parts)):
            for optimizer in self.optimizers[part]:
                optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for part in range(len(self.model_parts)):
            for optimizer in self.optimizers[part]:
                optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for model_part, optimizers in zip(self.model_parts, self.optimizers)
            for k, v in func(model_part, optimizers).items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for model_part, optimizers in zip(self.model_parts, self.optimizers):
            for optimizer in optimizers:
                func(model_part, optimizer)

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

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
        pass

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_cls = config.default.build().optimizer_cls
        optimizer_kwargs = config.default.build().get_optimizer_kwargs()
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = [[opt] for opt in optim_dict.values()]

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def _validate_length(self, expected_length: int) -> None:
        total = sum(len(part) for part in self.optimizers)
        assert (
            expected_length == total
        ), "Must pass one optimizer per param when using OptimizersInBackwardContainer."

    # pyrefly: ignore [bad-override]
    def step(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def zero_grad(self) -> None:
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
