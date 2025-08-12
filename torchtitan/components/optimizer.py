# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

# Dion optimizer availability will be checked lazily when needed
DION_AVAILABLE = None
MUON_AVAILABLE = None


def _check_dion_availability():
    """Lazy check for Dion optimizer availability."""
    global DION_AVAILABLE
    if DION_AVAILABLE is None:
        try:
            from torchtitan.experiments.dion_optimizer.dion import (
                Dion,
                DionMixedPrecisionConfig,
            )
            from torchtitan.experiments.dion_optimizer.titan_dion import (
                DionOptimizersContainer,
            )

            DION_AVAILABLE = True
        except ImportError:
            DION_AVAILABLE = False
    return DION_AVAILABLE


def _check_muon_availability():
    """Lazy check for Muon optimizer availability."""
    global MUON_AVAILABLE
    if MUON_AVAILABLE is None:
        try:
            from torchtitan.experiments.dion_optimizer.muon import Muon
            from torchtitan.experiments.dion_optimizer.titan_muon import (
                MuonOptimizersContainer,
            )

            MUON_AVAILABLE = True
        except ImportError:
            MUON_AVAILABLE = False
    return MUON_AVAILABLE


__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

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


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
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

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
        use_ft_optimizer: bool = True,
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer``,
    ``OptimizersInBackwardContainer``, and ``DionOptimizersContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
    name = optimizer_config.name

    # Handle Dion optimizer
    if name == "Dion":
        if not _check_dion_availability():
            raise ImportError(
                "Dion optimizer is not available. Please ensure the dion optimizer files are present in "
                "torchtitan/experiments/dion_optimizer/"
            )

        if optim_in_bwd:
            raise NotImplementedError(
                "Dion optimizer does not support early step in backward."
            )

        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not yet supported with Dion optimizer."
            )

        # Import the DionOptimizerConfig and DionOptimizersContainer from titan_dion
        from torchtitan.experiments.dion_optimizer.titan_dion import (
            DionOptimizerConfig,
            DionOptimizersContainer,
        )

        # Create DionOptimizerConfig from optimizer_config
        dion_config = DionOptimizerConfig(
            name="dion",
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            mu=optimizer_config.mu,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            epsilon=optimizer_config.eps,
            rank_fraction=optimizer_config.rank_fraction,
            rank_multiple_of=optimizer_config.rank_multiple_of,
            power_iters=optimizer_config.power_iters,
            qr_method=optimizer_config.qr_method,
            cqr_warmup_steps=optimizer_config.cqr_warmup_steps,
            rcqr_oversample=optimizer_config.rcqr_oversample,
            algorithm=optimizer_config.algorithm,
            replicate_mesh_grad_sync=optimizer_config.replicate_mesh_grad_sync,
            # Parameter-specific optimizer selection
            scalar_optimizer=getattr(optimizer_config, "scalar_optimizer", "adamw"),
            embedding_optimizer=getattr(
                optimizer_config, "embedding_optimizer", "adamw"
            ),
            head_optimizer=getattr(optimizer_config, "head_optimizer", "adamw"),
            routing_optimizer=getattr(optimizer_config, "routing_optimizer", "adamw"),
            expert_optimizer=getattr(optimizer_config, "expert_optimizer", None),
            # Additional optimizer options
            head_lr_scaling=getattr(optimizer_config, "head_lr_scaling", True),
            # Learning rate scaling factors
            scalar_lr_factor=getattr(optimizer_config, "scalar_lr_factor", 1.0),
            embedding_lr_factor=getattr(optimizer_config, "embedding_lr_factor", 1.0),
            head_lr_factor=getattr(optimizer_config, "head_lr_factor", 1.0),
            routing_lr_factor=getattr(optimizer_config, "routing_lr_factor", 1.0),
            expert_lr_factor=getattr(optimizer_config, "expert_lr_factor", 1.0),
        )

        # Set mixed precision dtypes if specified
        if optimizer_config.momentum_dtype:
            dion_config.momentum_dtype = TORCH_DTYPE_MAP[
                optimizer_config.momentum_dtype
            ]
        if optimizer_config.Q_dtype:
            dion_config.Q_dtype = TORCH_DTYPE_MAP[optimizer_config.Q_dtype]
        if optimizer_config.variance_dtype:
            dion_config.variance_dtype = TORCH_DTYPE_MAP[
                optimizer_config.variance_dtype
            ]

        return DionOptimizersContainer(
            model_parts=model_parts,
            dion_config=dion_config,
            parallel_dims=parallel_dims,
        )

    # Handle Muon optimizer
    if name == "Muon":
        if not _check_muon_availability():
            raise ImportError(
                "Muon optimizer is not available. Please ensure the muon optimizer files are present in "
                "torchtitan/experiments/dion_optimizer/"
            )

        if optim_in_bwd:
            raise NotImplementedError(
                "Muon optimizer does not support early step in backward."
            )

        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not yet supported with Muon optimizer."
            )

        # Import the MuonOptimizerConfig and MuonOptimizersContainer from titan_muon
        from torchtitan.experiments.dion_optimizer.titan_muon import (
            MuonOptimizerConfig,
            MuonOptimizersContainer,
        )

        # Create MuonOptimizerConfig from optimizer_config
        muon_config = MuonOptimizerConfig(
            name="muon",
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            mu=optimizer_config.mu,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            epsilon=optimizer_config.eps,
            nesterov=getattr(optimizer_config, "nesterov", False),
            adjust_lr=getattr(optimizer_config, "adjust_lr", "spectral_norm"),
            flatten=getattr(optimizer_config, "flatten", False),
            use_triton=getattr(optimizer_config, "use_triton", False),
            algorithm=optimizer_config.algorithm,
            # Parameter-specific optimizer selection
            scalar_optimizer=getattr(optimizer_config, "scalar_optimizer", "adamw"),
            embedding_optimizer=getattr(
                optimizer_config, "embedding_optimizer", "adamw"
            ),
            head_optimizer=getattr(optimizer_config, "head_optimizer", "adamw"),
            routing_optimizer=getattr(optimizer_config, "routing_optimizer", "adamw"),
            expert_optimizer=getattr(optimizer_config, "expert_optimizer", None),
            # Additional optimizer options
            head_lr_scaling=getattr(optimizer_config, "head_lr_scaling", True),
            # Learning rate scaling factors
            scalar_lr_factor=getattr(optimizer_config, "scalar_lr_factor", 1.0),
            embedding_lr_factor=getattr(optimizer_config, "embedding_lr_factor", 1.0),
            head_lr_factor=getattr(optimizer_config, "head_lr_factor", 1.0),
            routing_lr_factor=getattr(optimizer_config, "routing_lr_factor", 1.0),
            expert_lr_factor=getattr(optimizer_config, "expert_lr_factor", 1.0),
        )

        return MuonOptimizersContainer(
            model_parts=model_parts,
            muon_config=muon_config,
            parallel_dims=parallel_dims,
        )

    # Handle standard optimizers (Adam, AdamW)
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
        )

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
