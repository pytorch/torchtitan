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

# Import the Muon optimizer (assuming it's available)
from .muon import Muon
from .parameter_classification import create_parameter_groups

__all__ = [
    "MuonOptimizersContainer",
    "build_muon_optimizers",
    "MuonOptimizerConfig",
]


@dataclass
class MuonOptimizerConfig:
    """Extended optimizer config for Muon-specific parameters."""

    # Standard optimizer parameters
    name: str = "muon"
    lr: float = 0.01
    weight_decay: float = 0.01

    # Muon-specific parameters
    mu: float = 0.95  # Momentum for Muon
    betas: tuple[float, float] = (0.9, 0.95)  # Betas for AdamW and Lion
    epsilon: float = 1e-8
    nesterov: bool = False  # Whether to use Nesterov momentum
    adjust_lr: Optional[str] = "spectral_norm"  # "spectral_norm", "rms_norm", or None
    flatten: bool = False  # Whether to flatten 3D+ tensors to 2D
    use_triton: bool = False  # Whether to use Triton kernel for Newton-Schulz

    # Algorithm selection per parameter group
    # Can be "muon", "adamw", or "lion"
    algorithm: str = "muon"

    # Parameter-specific optimizer selection
    scalar_optimizer: str = "adamw"  # For 1D parameters (biases, layer norms)
    embedding_optimizer: str = "adamw"  # For embedding layers
    head_optimizer: str = "adamw"  # For model head/output layers
    routing_optimizer: Optional[str] = None  # For routing layers (DeepSeek MoE)
    expert_optimizer: Optional[str] = None  # For expert weights (MoE experts)

    # Additional optimizer options
    head_lr_scaling: bool = True  # Apply 1/sqrt(dim) scaling to head layers

    # Learning rate scaling factors
    scalar_lr_factor: float = 1.0  # LR multiplier for scalar parameters
    embedding_lr_factor: float = 1.0  # LR multiplier for embedding parameters
    head_lr_factor: float = (
        1.0  # LR multiplier for head parameters (after head_lr_scaling)
    )
    routing_lr_factor: float = 1.0  # LR multiplier for routing parameters
    expert_lr_factor: float = 1.0  # LR multiplier for expert parameters

    # Gradient synchronization
    replicate_mesh_grad_sync: bool = True


class MuonOptimizersContainer(OptimizersContainer):
    """A container for Muon optimizers compatible with TorchTitan interface.

    This class wraps the Muon optimizer to make it compatible with the
    TorchTitan OptimizersContainer interface while preserving Muon's
    distributed training capabilities.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        muon_config (MuonOptimizerConfig): Configuration for Muon optimizer.
        parallel_dims (ParallelDims): Parallel dimensions configuration.
    """

    def __init__(
        self,
        model_parts: List[nn.Module],
        muon_config: MuonOptimizerConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        self.model_parts = model_parts
        self.muon_config = muon_config
        self.parallel_dims = parallel_dims

        # Setup device meshes from parallel dimensions
        distributed_mesh = self._setup_device_mesh(parallel_dims)

        # Classify parameters and create appropriate parameter groups
        param_groups = create_parameter_groups(model_parts, muon_config)

        # Create the Muon optimizer
        self.muon_optimizer = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=muon_config.lr,
            mu=muon_config.mu,
            betas=muon_config.betas,
            weight_decay=muon_config.weight_decay,
            epsilon=muon_config.epsilon,
            nesterov=muon_config.nesterov,
            adjust_lr=muon_config.adjust_lr,
            flatten=muon_config.flatten,
            use_triton=muon_config.use_triton,
        )

        # Initialize parent class with dummy optimizer kwargs
        # This ensures hooks and other functionality work
        super().__init__(
            model_parts=model_parts,
            optimizer_cls=torch.optim.SGD,  # Dummy, not used
            optimizer_kwargs={"lr": muon_config.lr},  # Dummy, not used
        )

        # For compatibility with OptimizersContainer interface
        self.optimizers = [self.muon_optimizer]

    def _setup_device_mesh(
        self, parallel_dims: ParallelDims
    ) -> Optional[Union[DeviceMesh, ProcessGroup]]:
        """Setup device mesh based on parallel dimensions.

        For Muon, we use the dp_shard mesh for distributed communication.
        """
        distributed_mesh = None

        # Get the world mesh from parallel_dims
        world_mesh = parallel_dims.world_mesh

        # For Muon, we primarily use the dp_shard mesh for distributed operations
        if parallel_dims.dp_shard_enabled:
            # Extract the dp_shard submesh
            if "dp_shard" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_shard"]
            elif "dp_shard_cp" in world_mesh.mesh_dim_names:
                # If context parallel is enabled, use dp_shard_cp mesh
                distributed_mesh = world_mesh["dp_shard_cp"]
        elif parallel_dims.dp_replicate_enabled:
            # If no dp_shard but dp_replicate is enabled, use that
            if "dp_replicate" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp_replicate"]
            elif "dp" in world_mesh.mesh_dim_names:
                distributed_mesh = world_mesh["dp"]

        return distributed_mesh

    def __iter__(self):
        """Iterate over optimizers for compatibility."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers."""
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        """Perform optimization step."""
        self.muon_optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients for all optimizers."""
        # Call parent class method to ensure all optimizers in self.optimizers are handled
        super().zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict using distributed checkpoint utilities."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(
                func, self.model_parts, [self.muon_optimizer] * len(self.model_parts)
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
        list(map(func, self.model_parts, [self.muon_optimizer] * len(self.model_parts)))


def build_muon_optimizers(
    model_parts: List[nn.Module],
    muon_config: MuonOptimizerConfig,
    parallel_dims: ParallelDims,
) -> MuonOptimizersContainer:
    """Create a MuonOptimizersContainer for the given model parts and config.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        muon_config (MuonOptimizerConfig): Muon optimizer configuration.
        parallel_dims (ParallelDims): Parallel dimensions for the model.

    Returns:
        MuonOptimizersContainer: Container with Muon optimizer.
    """
    return MuonOptimizersContainer(
        model_parts=model_parts,
        muon_config=muon_config,
        parallel_dims=parallel_dims,
    )


def build_optimizers_with_muon_support(
    model_parts: List[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    muon_config: Optional[MuonOptimizerConfig] = None,
) -> OptimizersContainer:
    """Extended build_optimizers function with Muon support.

    This is a drop-in replacement for the original build_optimizers function
    that adds support for the Muon optimizer.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Standard optimizer config.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
        muon_config (Optional[MuonOptimizerConfig]): Muon-specific config.
            If provided, will use Muon optimizer instead of standard optimizers.

    Returns:
        OptimizersContainer: Container with appropriate optimizer(s).
    """
    # If Muon config is provided, use Muon optimizer
    if muon_config is not None:
        return build_muon_optimizers(model_parts, muon_config, parallel_dims)

    # Otherwise, fall back to original build_optimizers logic
    from torchtitan.components.optimizer import build_optimizers

    return build_optimizers(model_parts, optimizer_config, parallel_dims)


# Example usage and parameter group configuration utilities
class MuonParameterGroupManager:
    """Utility class to manage different algorithms for different parameter groups."""

    @staticmethod
    def create_mixed_param_groups(
        model_parts: List[nn.Module],
        muon_config: MuonOptimizerConfig,
        layer_algorithm_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with different algorithms for different layers.

        Args:
            model_parts: List of model parts
            muon_config: Base configuration
            layer_algorithm_map: Mapping from layer name patterns to algorithms
                                Example: {"attention": "muon", "mlp": "adamw", "embed": "lion"}

        Returns:
            List of parameter group dictionaries
        """
        if layer_algorithm_map is None:
            layer_algorithm_map = {"": "muon"}  # Default to muon for all

        param_groups = []

        for model in model_parts:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Determine algorithm based on layer name
                algorithm = "muon"  # default
                for pattern, algo in layer_algorithm_map.items():
                    if pattern in name:
                        algorithm = algo
                        break

                # Create parameter group
                param_group = {
                    "params": [param],
                    "algorithm": algorithm,
                    "lr": muon_config.lr,
                    "mu": muon_config.mu,
                    "beta1": muon_config.betas[0],
                    "beta2": muon_config.betas[1],
                    "weight_decay": muon_config.weight_decay,
                    "epsilon": muon_config.epsilon,
                    "nesterov": muon_config.nesterov,
                    "adjust_lr": muon_config.adjust_lr,
                    "flatten": muon_config.flatten,
                }
                param_groups.append(param_group)

        return param_groups


# Example configuration for different model architectures
def get_llama_muon_config() -> MuonOptimizerConfig:
    """Example Muon configuration optimized for LLaMA-style models."""
    return MuonOptimizerConfig(
        name="muon",
        lr=3e-4,
        weight_decay=0.1,
        mu=0.95,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        nesterov=False,
        adjust_lr="spectral_norm",  # For learning rate transfer across model scale
        flatten=False,  # Keep False for transformer attention layers
        use_triton=False,  # Conservative default
        algorithm="muon",
    )


def get_mixed_algorithm_config() -> tuple[MuonOptimizerConfig, Dict[str, str]]:
    """Example configuration using different algorithms for different layers."""
    config = MuonOptimizerConfig(
        name="mixed",
        lr=3e-4,
        weight_decay=0.1,
    )

    # Use Muon for attention layers, AdamW for embeddings, Lion for MLP
    algorithm_map = {
        "attention": "muon",
        "embed": "adamw",
        "mlp": "lion",
    }

    return config, algorithm_map
