# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Generic, Iterator, TypeVar

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

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

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
    "build_optimizers_with_moe_load_balancing",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


def preinit_optimizer_states_bf16(optimizers_container: "OptimizersContainer") -> None:
    """
    Pre-initialize optimizer states (exp_avg, exp_avg_sq) directly in bfloat16.
    This MUST be called BEFORE the first optimizer.step() to avoid fp32 allocation spike.

    This reduces optimizer state memory by ~50% (from fp32 to bf16).
    States are allocated in bf16 from the start, avoiding the memory spike from fp32 allocation.
    """
    total_params = 0
    total_bytes = 0
    dtype_device_samples = []
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    for opt_idx, optimizer in enumerate(optimizers_container.optimizers):
        for pg_idx, param_group in enumerate(optimizer.param_groups):
            for p_idx, p in enumerate(param_group["params"]):
                if p.requires_grad:
                    if total_params < 5:
                        dtype_device_samples.append(
                            f"param[{opt_idx}][{pg_idx}][{p_idx}]: dtype={p.dtype}, device={p.device}, shape={list(p.shape)}"
                        )

                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0, dtype=torch.float32, device=p.device)
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=p.dtype, device=p.device
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=p.dtype, device=p.device
                        )
                        total_params += 1
                        bytes_per_element = 2 if p.dtype == torch.bfloat16 else 4
                        total_bytes += p.numel() * 2 * bytes_per_element

                        if total_params <= 3:
                            logger.info(
                                f"[Rank {rank}] State init sample: param dtype={p.dtype}, device={p.device}, "
                                f"exp_avg dtype={state['exp_avg'].dtype}, device={state['exp_avg'].device}"
                            )

    for sample in dtype_device_samples:
        logger.info(f"[Rank {rank}] {sample}")

    logger.info(
        f"[Rank {rank}] Pre-initialized {total_params} optimizer states matching param dtype, "
        f"this rank: {total_bytes / 1e9:.2f} GB"
    )


class BF16StateOptimizersContainer(Generic[T]):
    """
    Wrapper that pre-initializes optimizer states in bfloat16 BEFORE first step.
    This prevents the memory spike from fp32 state allocation.

    IMPORTANT: Call init_bf16_states() BEFORE the first step() to avoid
    rank skew during state allocation. This should be called after model
    setup but before training starts, ideally with a barrier afterwards.
    """

    def __init__(
        self,
        base_container: "OptimizersContainer",
        state_dtype: torch.dtype = torch.bfloat16,
    ):
        self._base = base_container
        self._state_dtype = state_dtype
        self._states_initialized = False

    def init_bf16_states(self):
        """
        Pre-initialize optimizer states in bf16.
        Call this BEFORE training starts, then call a distributed barrier.
        This avoids rank skew during the first optimizer.step().
        """
        if not self._states_initialized:
            logger.info("Pre-initializing optimizer states in bfloat16...")
            preinit_optimizer_states_bf16(self._base)
            self._states_initialized = True
            logger.info("BF16 optimizer state pre-initialization complete.")

    def step(self, *args, **kwargs) -> None:
        if not self._states_initialized:
            logger.warning(
                "BF16 optimizer states not pre-initialized! "
                "Call init_bf16_states() before training to avoid rank skew."
            )
            self.init_bf16_states()
        self._base.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        self._base.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return self._base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._base.load_state_dict(state_dict)

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)


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

    # pyrefly: ignore [bad-override]
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

    # pyrefly: ignore [bad-override]
    def step(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
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
            routing_optimizer=getattr(optimizer_config, "routing_optimizer", None),
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
            state_dtype=TORCH_DTYPE_MAP[optimizer_config.state_dtype],
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
            routing_optimizer=getattr(optimizer_config, "routing_optimizer", None),
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

    container = OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)

    # Wrap with BF16 state container if configured
    state_dtype = getattr(optimizer_config, "state_dtype", "float32")
    if state_dtype == "bfloat16":
        logger.info("Using bfloat16 optimizer states (will pre-init before first step)")
        return BF16StateOptimizersContainer(container, torch.bfloat16)

    return container


def build_optimizers_with_moe_load_balancing(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    optimizers = build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )

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

    return optimizers
