# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DeviceMesh

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims

# Import the Dion optimizer (assuming it's available)
from .dion import Dion, DionMixedPrecisionConfig

__all__ = [
    "DionOptimizersContainer",
    "build_dion_optimizers",
    "DionOptimizerConfig",
]


@dataclass
class DionOptimizerConfig:
    """Extended optimizer config for Dion-specific parameters."""

    # Standard optimizer parameters
    name: str = "dion"
    lr: float = 0.01
    weight_decay: float = 0.01

    # Dion-specific parameters
    mu: float = 0.95  # Momentum for Dion
    betas: tuple[float, float] = (0.9, 0.95)  # Betas for AdamW and Lion
    epsilon: float = 1e-8
    rank_fraction: float = 1.0
    rank_multiple_of: int = 1
    power_iters: int = 1
    qr_method: str = "rcqr"
    cqr_warmup_steps: int = 150
    rcqr_oversample: float = 1.25

    # Algorithm selection per parameter group
    # Can be "dion", "adamw", or "lion"
    algorithm: str = "dion"

    # Mixed precision config
    momentum_dtype: Optional[torch.dtype] = None
    Q_dtype: Optional[torch.dtype] = None
    variance_dtype: Optional[torch.dtype] = None

    # Gradient synchronization
    replicate_mesh_grad_sync: bool = True


class DionOptimizersContainer(OptimizersContainer):
    """A container for Dion optimizers compatible with TorchTitan interface.

    This class wraps the Dion optimizer to make it compatible with the
    TorchTitan OptimizersContainer interface while preserving Dion's
    distributed training capabilities.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        dion_config (DionOptimizerConfig): Configuration for Dion optimizer.
        parallel_dims (ParallelDims): Parallel dimensions configuration.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        dion_config: DionOptimizerConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        self.model_parts = model_parts
        self.dion_config = dion_config
        self.parallel_dims = parallel_dims

        # Setup device meshes from parallel dimensions
        replicate_mesh, outer_shard_mesh, inner_shard_mesh = self._setup_device_meshes(
            parallel_dims
        )

        # Create mixed precision config
        mixed_precision_config = DionMixedPrecisionConfig(
            momentum_dtype=dion_config.momentum_dtype,
            Q_dtype=dion_config.Q_dtype,
            variance_dtype=dion_config.variance_dtype,
        )

        # Collect all parameters from model parts
        all_params = []
        param_groups = []

        for model in model_parts:
            model_params = [p for p in model.parameters() if p.requires_grad]
            all_params.extend(model_params)

            # Create parameter group for this model part
            # You can customize this to assign different algorithms to different layers
            param_group = {
                "params": model_params,
                "algorithm": dion_config.algorithm,
                "rank_fraction": dion_config.rank_fraction,
                "rank_multiple_of": dion_config.rank_multiple_of,
                "lr": dion_config.lr,
                "mu": dion_config.mu,
                "beta1": dion_config.betas[0],
                "beta2": dion_config.betas[1],
                "weight_decay": dion_config.weight_decay,
                "epsilon": dion_config.epsilon,
            }
            param_groups.append(param_group)

        # Create the Dion optimizer
        self.dion_optimizer = Dion(
            param_groups,
            replicate_mesh=replicate_mesh,
            outer_shard_mesh=outer_shard_mesh,
            inner_shard_mesh=inner_shard_mesh,
            replicate_mesh_grad_sync=dion_config.replicate_mesh_grad_sync,
            rank_fraction=dion_config.rank_fraction,
            rank_multiple_of=dion_config.rank_multiple_of,
            lr=dion_config.lr,
            mu=dion_config.mu,
            betas=dion_config.betas,
            weight_decay=dion_config.weight_decay,
            epsilon=dion_config.epsilon,
            power_iters=dion_config.power_iters,
            qr_method=dion_config.qr_method,
            cqr_warmup_steps=dion_config.cqr_warmup_steps,
            rcqr_oversample=dion_config.rcqr_oversample,
            mixed_precision_config=mixed_precision_config,
        )

        # For compatibility with OptimizersContainer interface
        self.optimizers = [self.dion_optimizer]

        # Initialize parent class with dummy optimizer kwargs
        # This ensures hooks and other functionality work
        super().__init__(
            model_parts=model_parts,
            optimizer_cls=torch.optim.SGD,  # Dummy, not used
            optimizer_kwargs={"lr": dion_config.lr},  # Dummy, not used
        )

    def _setup_device_meshes(self, parallel_dims: ParallelDims) -> tuple[
        Optional[Union[DeviceMesh, ProcessGroup]],
        Optional[DeviceMesh],
        Optional[DeviceMesh],
    ]:
        """Setup device meshes based on parallel dimensions."""

        replicate_mesh = None
        outer_shard_mesh = None
        inner_shard_mesh = None

        # Setup replicate mesh for data parallelism
        if parallel_dims.dp_enabled:
            if hasattr(parallel_dims, "dp_mesh"):
                replicate_mesh = parallel_dims.dp_mesh
            elif hasattr(parallel_dims, "dp_pg"):
                replicate_mesh = parallel_dims.dp_pg

        # Setup outer shard mesh (typically FSDP)
        if parallel_dims.pp_enabled:
            # For pipeline parallelism, we might want to use it as outer sharding
            if hasattr(parallel_dims, "pp_mesh"):
                outer_shard_mesh = parallel_dims.pp_mesh

        # Setup inner shard mesh (typically tensor parallelism)
        if parallel_dims.tp_enabled:
            if hasattr(parallel_dims, "tp_mesh"):
                inner_shard_mesh = parallel_dims.tp_mesh

        return replicate_mesh, outer_shard_mesh, inner_shard_mesh

    def __iter__(self):
        """Iterate over optimizers for compatibility."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers."""
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        """Perform optimization step."""
        self.dion_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients."""
        self.dion_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict using distributed checkpoint utilities."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(
                func, self.model_parts, [self.dion_optimizer] * len(self.model_parts)
            )
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict using distributed checkpoint utilities."""
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, [self.dion_optimizer] * len(self.model_parts)))

    def synchronize_for_checkpoint(self) -> None:
        """Synchronize optimizer states for checkpointing."""
        if hasattr(self.dion_optimizer, "synchronize_for_checkpoint"):
            self.dion_optimizer.synchronize_for_checkpoint()


def build_dion_optimizers(
    model_parts: List[nn.Module],
    dion_config: DionOptimizerConfig,
    parallel_dims: ParallelDims,
) -> DionOptimizersContainer:
    """Create a DionOptimizersContainer for the given model parts and config.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        dion_config (DionOptimizerConfig): Dion optimizer configuration.
        parallel_dims (ParallelDims): Parallel dimensions for the model.

    Returns:
        DionOptimizersContainer: Container with Dion optimizer.
    """
    return DionOptimizersContainer(
        model_parts=model_parts,
        dion_config=dion_config,
        parallel_dims=parallel_dims,
    )


def build_optimizers_with_dion_support(
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    dion_config: Optional[DionOptimizerConfig] = None,
) -> OptimizersContainer:
    """Extended build_optimizers function with Dion support.

    This is a drop-in replacement for the original build_optimizers function
    that adds support for the Dion optimizer.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Standard optimizer config.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
        dion_config (Optional[DionOptimizerConfig]): Dion-specific config.
            If provided, will use Dion optimizer instead of standard optimizers.

    Returns:
        OptimizersContainer: Container with appropriate optimizer(s).
    """
    # If Dion config is provided, use Dion optimizer
    if dion_config is not None:
        return build_dion_optimizers(model_parts, dion_config, parallel_dims)

    # Otherwise, fall back to original build_optimizers logic
    from torchtitan.components.optimizer import build_optimizers

    return build_optimizers(model_parts, optimizer_config, parallel_dims)


# Example usage and parameter group configuration utilities
class DionParameterGroupManager:
    """Utility class to manage different algorithms for different parameter groups."""

    @staticmethod
    def create_mixed_param_groups(
        model_parts: List[nn.Module],
        dion_config: DionOptimizerConfig,
        layer_algorithm_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with different algorithms for different layers.

        Args:
            model_parts: List of model parts
            dion_config: Base configuration
            layer_algorithm_map: Mapping from layer name patterns to algorithms
                                Example: {"attention": "dion", "mlp": "adamw", "embed": "lion"}

        Returns:
            List of parameter group dictionaries
        """
        if layer_algorithm_map is None:
            layer_algorithm_map = {"": "dion"}  # Default to dion for all

        param_groups = []

        for model in model_parts:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine algorithm based on layer name
                algorithm = "dion"  # default
                for pattern, algo in layer_algorithm_map.items():
                    if pattern in name:
                        algorithm = algo
                        break

                # Create parameter group
                param_group = {
                    "params": [param],
                    "algorithm": algorithm,
                    "rank_fraction": dion_config.rank_fraction,
                    "rank_multiple_of": dion_config.rank_multiple_of,
                    "lr": dion_config.lr,
                    "mu": dion_config.mu,
                    "beta1": dion_config.betas[0],
                    "beta2": dion_config.betas[1],
                    "weight_decay": dion_config.weight_decay,
                    "epsilon": dion_config.epsilon,
                }
                param_groups.append(param_group)

        return param_groups


# Example configuration for different model architectures
def get_llama_dion_config() -> DionOptimizerConfig:
    """Example Dion configuration optimized for LLaMA-style models."""
    return DionOptimizerConfig(
        name="dion",
        lr=3e-4,
        weight_decay=0.1,
        mu=0.95,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        rank_fraction=0.5,  # Use 50% rank for memory efficiency
        rank_multiple_of=128,  # Align with tensor cores
        algorithm="dion",
        momentum_dtype=torch.float32,  # Higher precision for momentum
        Q_dtype=torch.bfloat16,  # Lower precision for Q matrix
        variance_dtype=torch.float32,  # Higher precision for variance (AdamW)
    )


def get_mixed_algorithm_config() -> tuple[DionOptimizerConfig, Dict[str, str]]:
    """Example configuration using different algorithms for different layers."""
    config = DionOptimizerConfig(
        name="mixed",
        lr=3e-4,
        weight_decay=0.1,
        rank_fraction=0.75,
    )

    # Use Dion for attention layers, AdamW for embeddings, Lion for MLP
    algorithm_map = {
        "attention": "dion",
        "embed": "adamw",
        "mlp": "lion",
    }

    return config, algorithm_map
