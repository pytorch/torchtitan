"""
Dion: Distributed Orthonormalized Updates Optimizer

A PyTorch implementation of the Dion optimizer from the paper:
"Dion: Distributed Orthonormalized Updates" by Ahn et al.

This implementation supports:
- Centralized and distributed training
- 3D parallelism (DP, FSDP2, TP)
- Automatic parameter type detection
- Integration with scalar optimizers
- Memory-efficient low-rank approximations
- Decoupled weight decay
"""

import math
import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP2

    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False
    FSDP2 = None

try:
    from torch.optim.lion import Lion

    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    Lion = None


class ShardingStrategy(Enum):
    """Sharding strategies for distributed Dion."""

    NO_SHARD = "no_shard"
    SHARD_GRAD_OP = "shard_grad_op"  # Standard FSDP
    FULL_SHARD = "full_shard"  # FSDP2


class DionOptimizer(Optimizer):
    """
    Dion (DIstributed OrthoNormalization) Optimizer

    Implements the Dion optimizer with support for both centralized and distributed training.
    Uses orthonormalized updates for matrix parameters and configurable scalar optimizers
    for non-matrix parameters.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor μ (default: 0.95)
        rank_factor: Rank fraction r/d for low-rank approximation (default: 1.0, full rank)
        scalar_optimizer: Optimizer class for non-matrix parameters ('adam', 'adamw', 'lion')
        scalar_lr: Learning rate for scalar optimizer (default: None, uses lr)
        weight_decay: Weight decay coefficient (default: 0.01)
        scalar_weight_decay: Weight decay for scalar parameters (default: 0.0)
        eps: Small constant for numerical stability (default: 1e-8)
        oversampling_factor: Factor for randomized QR oversampling (default: 1.25)
        distributed: Whether to use distributed implementation (default: auto-detect)
        matrix_threshold: Minimum size for treating parameter as matrix (default: 32)
        process_group: Process group for distributed training (default: None)
        sharding_strategy: Sharding strategy for FSDP2 (default: FULL_SHARD)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        rank_factor: float = 1.0,
        scalar_optimizer: str = "adamw",
        scalar_lr: Optional[float] = None,
        weight_decay: float = 0.01,
        scalar_weight_decay: float = 0.0,
        eps: float = 1e-8,
        oversampling_factor: float = 1.25,
        distributed: Optional[bool] = None,
        matrix_threshold: int = 32,
        process_group: Optional[dist.ProcessGroup] = None,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        **scalar_kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 < rank_factor <= 1.0:
            raise ValueError(f"Invalid rank factor: {rank_factor}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        self.rank_factor = rank_factor
        self.oversampling_factor = oversampling_factor
        self.matrix_threshold = matrix_threshold
        self.process_group = process_group
        self.sharding_strategy = sharding_strategy

        # Auto-detect distributed mode
        if distributed is None:
            distributed = dist.is_available() and dist.is_initialized()
        self.distributed = distributed

        # Get world size and rank
        if self.distributed:
            self.world_size = dist.get_world_size(self.process_group)
            self.rank = dist.get_rank(self.process_group)
        else:
            self.world_size = 1
            self.rank = 0

        # Setup scalar optimizer
        self.scalar_lr = scalar_lr if scalar_lr is not None else lr
        self.scalar_weight_decay = scalar_weight_decay
        self.scalar_optimizer_class = self._get_scalar_optimizer_class(scalar_optimizer)
        self.scalar_kwargs = scalar_kwargs

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super(DionOptimizer, self).__init__(params, defaults)

        # Initialize parameter classification and scalar optimizers
        self._classify_parameters()
        self._init_scalar_optimizers()

    def _get_scalar_optimizer_class(self, optimizer_name: str):
        """Get scalar optimizer class by name."""
        optimizer_map = {
            "adam": Adam,
            "adamw": AdamW,
        }

        if LION_AVAILABLE:
            optimizer_map["lion"] = Lion
        elif optimizer_name == "lion":
            raise ImportError(
                "Lion optimizer not available. Install torch>=2.0 or use 'adam'/'adamw'"
            )

        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported scalar optimizer: {optimizer_name}")

        return optimizer_map[optimizer_name]

    def _classify_parameters(self):
        """Classify parameters into matrix and scalar types."""
        self.matrix_params = []
        self.scalar_params = []

        for group in self.param_groups:
            for p in group["params"]:
                if self._is_matrix_param(p):
                    self.matrix_params.append(p)
                else:
                    self.scalar_params.append(p)

    def _is_matrix_param(self, param: torch.Tensor) -> bool:
        """Determine if parameter should be treated as a matrix."""
        # Handle DTensor and FSDP wrapped parameters
        if hasattr(param, "_local_tensor"):
            local_param = param._local_tensor
        else:
            local_param = param

        # Must be 2D and both dimensions >= threshold
        return (
            local_param.dim() == 2
            and local_param.size(0) >= self.matrix_threshold
            and local_param.size(1) >= self.matrix_threshold
        )

    def _init_scalar_optimizers(self):
        """Initialize scalar optimizers for non-matrix parameters."""
        if not self.scalar_params:
            self.scalar_optimizer = None
            return

        # Create parameter groups with appropriate scaling
        scalar_groups = []
        for param in self.scalar_params:
            lr_scale = self._get_lr_scale(param)
            param_group = {
                "params": [param],
                "lr": self.scalar_lr * lr_scale,
                "weight_decay": self._get_weight_decay(param),
            }
            param_group.update(self.scalar_kwargs)
            scalar_groups.append(param_group)

        self.scalar_optimizer = self.scalar_optimizer_class(scalar_groups)

    def _get_lr_scale(self, param: torch.Tensor) -> float:
        """Get learning rate scaling factor based on parameter type and shape."""
        if param.dim() == 0:  # Scalar (normalization)
            return 1.0
        elif param.dim() == 1:  # Vector (bias, embedding, etc.)
            if param.numel() > 1000:  # Likely unembedding
                return 1.0 / math.sqrt(param.size(0))
            else:  # Likely bias
                return 1.0
        elif param.dim() == 2:  # Matrix (but below threshold)
            d_out, d_in = param.shape
            return math.sqrt(d_out / d_in)
        else:
            return 1.0

    def _get_weight_decay(self, param: torch.Tensor) -> float:
        """Get weight decay for parameter type."""
        # Only apply weight decay to matrix parameters (not bias/normalization)
        if param.dim() >= 2:
            return self.scalar_weight_decay
        return 0.0

    def _get_param_info(self, param: torch.Tensor) -> Dict[str, Any]:
        """Get sharding information for a parameter."""
        info = {
            "is_dtensor": isinstance(param, DTensor),
            "is_fsdp": False,
            "shard_dims": [],
            "local_shape": param.shape,
            "global_shape": param.shape,
        }

        if info["is_dtensor"]:
            info["local_shape"] = param._local_tensor.shape
            info["global_shape"] = param.shape
            placements = param.placements
            for i, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    info["shard_dims"].append(placement.dim)

        # Check if parameter is part of FSDP module
        if hasattr(param, "_fsdp_wrapped") or hasattr(param, "_is_fsdp"):
            info["is_fsdp"] = True

        return info

    def _init_matrix_state(self, param: torch.Tensor, group: Dict) -> Dict:
        """Initialize state for matrix parameter."""
        state = {}
        param_info = self._get_param_info(param)

        # Get the actual tensor to work with (local tensor for DTensor)
        if isinstance(param, torch.distributed._tensor.DTensor):
            actual_param = param._local_tensor
        else:
            actual_param = param

        # Use local shape
        m, n = actual_param.shape
        rank = min(m, n, max(1, int(min(m, n) * self.rank_factor)))

        # Create momentum buffer with same shape as actual parameter
        state["momentum_buffer"] = torch.zeros_like(actual_param)

        # Create right factor for warm-starting power iteration
        state["right_factor"] = torch.randn(
            n, rank, device=actual_param.device, dtype=actual_param.dtype
        )
        state["right_factor"] = self._normalize_columns(state["right_factor"])

        # Step counter and metadata
        state["step"] = 0
        state["rank"] = rank
        state["param_info"] = param_info

        return state

    def _normalize_columns(self, matrix: torch.Tensor) -> torch.Tensor:
        """Normalize columns of matrix to unit norm."""
        norms = torch.norm(matrix, dim=0, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        return matrix / norms

    def _power_iteration(
        self, B: torch.Tensor, Q_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step of power iteration for low-rank approximation."""
        # P = BQ (left factor)
        P = torch.mm(B, Q_prev)

        # Orthogonalize P using QR decomposition
        P = self._orthogonalize_matrix(P)

        # R = B^T P (right factor)
        R = torch.mm(B.t(), P)

        return P, R

    def _distributed_power_iteration(
        self, B: torch.Tensor, Q_prev: torch.Tensor, param_info: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distributed power iteration (Algorithm 3, lines 5-6)."""
        # In distributed mode, we need to handle sharding
        if param_info["is_dtensor"] or param_info["is_fsdp"]:
            # P = BQ with distributed matrix multiply
            P_local = torch.mm(B, Q_prev)

            # All-reduce across shards
            if self.distributed:
                dist.all_reduce(P_local, op=dist.ReduceOp.SUM, group=self.process_group)
                P_local = P_local / self.world_size

            # Orthogonalize
            P = self._distributed_orthogonalize(P_local)

            # R = B^T P with distributed matrix multiply
            R_local = torch.mm(B.t(), P)

            # All-reduce across shards
            if self.distributed:
                dist.all_reduce(R_local, op=dist.ReduceOp.SUM, group=self.process_group)
                R_local = R_local / self.world_size

            return P, R_local
        else:
            # Fallback to centralized version
            return self._power_iteration(B, Q_prev)

    def _orthogonalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix using QR decomposition."""
        if self.distributed:
            return self._distributed_orthogonalize(matrix)
        else:
            Q, _ = torch.linalg.qr(matrix)
            return Q

    def _distributed_orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Distributed orthogonalization using randomized Cholesky QR (Algorithm 2)."""
        m, r = matrix.shape
        k = max(r, int(self.oversampling_factor * r))

        # Random sketching matrix
        S = torch.randn(k, m, device=matrix.device, dtype=matrix.dtype) / math.sqrt(k)

        # First iteration: randomized QR
        G = torch.mm(S, matrix)

        # Aggregate across distributed dimensions if needed
        if dist.is_initialized() and self.process_group is not None:
            dist.all_reduce(G, op=dist.ReduceOp.SUM, group=self.process_group)

        # QR decomposition (only need R)
        try:
            _, R1 = torch.linalg.qr(G)
        except:
            # Fallback for numerical issues
            _, R1 = torch.linalg.qr(G + 1e-8 * torch.eye(k, r, device=G.device))

        # Solve for B
        B = torch.linalg.solve_triangular(R1.t(), matrix.t(), upper=False).t()

        # Second iteration: Cholesky QR
        H = torch.mm(B.t(), B)

        # Aggregate across distributed dimensions if needed
        if dist.is_initialized() and self.process_group is not None:
            dist.all_reduce(H, op=dist.ReduceOp.SUM, group=self.process_group)

        # Cholesky decomposition with numerical stability
        try:
            R2 = torch.linalg.cholesky(H)
        except:
            # Add small diagonal for numerical stability
            H_stable = H + torch.eye(r, device=H.device, dtype=H.dtype) * 1e-8
            R2 = torch.linalg.cholesky(H_stable)

        # Final orthogonalized result
        result = torch.linalg.solve_triangular(R2.t(), B.t(), upper=False).t()

        return result

    def _distributed_column_normalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Distributed column normalization (Algorithm 3, function at bottom)."""
        # Compute local squared norms
        local_norms_squared = torch.sum(matrix**2, dim=0)

        # All-reduce to get global norms
        if self.distributed:
            dist.all_reduce(
                local_norms_squared, op=dist.ReduceOp.SUM, group=self.process_group
            )

        # Compute global norms
        global_norms = torch.sqrt(local_norms_squared + 1e-8)

        # Normalize columns
        return matrix / global_norms

    def _step_matrix_param(
        self, param: torch.Tensor, grad: torch.Tensor, group: Dict, state: Dict
    ):
        """Perform Dion update step for matrix parameter."""
        momentum = group["momentum"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]

        # Get state variables
        momentum_buffer = state["momentum_buffer"]
        right_factor = state["right_factor"]
        rank = state["rank"]
        param_info = state["param_info"]

        # Handle DTensor case - ensure both tensors are of the same type
        if isinstance(param, torch.distributed._tensor.DTensor):
            # For DTensor parameters, we need to ensure all operations use the local tensors
            if isinstance(momentum_buffer, torch.distributed._tensor.DTensor):
                momentum_buffer = momentum_buffer._local_tensor

            if isinstance(grad, torch.distributed._tensor.DTensor):
                grad = grad._local_tensor

            # Work with local tensors for all operations
            buffer = momentum_buffer + grad

            # Power iteration with local tensors
            P, R = self._power_iteration(buffer, right_factor)

            # Approximation
            approx = torch.mm(P, R.t())

            # Error feedback: M_t = B_t - (1-μ)P_t R_t^T
            momentum_buffer.copy_(buffer - (1 - momentum) * approx)

            # Update right factor with column normalization
            right_factor.copy_(self._normalize_columns(R))

            # Compute scaled orthonormal update
            m, n = param._local_tensor.shape
            scale = math.sqrt(m / n)

            # Normalize columns of R to get Q
            Q = self._normalize_columns(R)

            update = torch.mm(P, Q.t())

            # Apply decoupled weight decay
            if weight_decay != 0:
                param._local_tensor.mul_(1 - lr * weight_decay)

            # Update parameters: X_t = X_{t-1} - η * scale * P_t Q_t^T
            param._local_tensor.add_(update, alpha=-lr * scale)

            state["step"] += 1
            return

        # Standard case for non-DTensor parameters
        # Form buffer B_t = M_{t-1} + G_t (no weight decay in gradient)
        buffer = momentum_buffer + grad

        # Power iteration: approximate B_t ≈ P_t R_t^T
        if self.distributed and (param_info["is_dtensor"] or param_info["is_fsdp"]):
            P, R = self._distributed_power_iteration(buffer, right_factor, param_info)
        else:
            P, R = self._power_iteration(buffer, right_factor)

        # Approximation
        approx = torch.mm(P, R.t())

        # Error feedback: M_t = B_t - (1-μ)P_t R_t^T
        momentum_buffer.copy_(buffer - (1 - momentum) * approx)

        # Update right factor with column normalization
        if self.distributed and (param_info["is_dtensor"] or param_info["is_fsdp"]):
            right_factor.copy_(self._distributed_column_normalize(R))
        else:
            right_factor.copy_(self._normalize_columns(R))

        # Compute scaled orthonormal update
        m, n = param.shape
        scale = math.sqrt(m / n)

        # Normalize columns of R to get Q
        if self.distributed and (param_info["is_dtensor"] or param_info["is_fsdp"]):
            Q = self._distributed_column_normalize(R)
        else:
            Q = self._normalize_columns(R)

        update = torch.mm(P, Q.t())

        # Apply decoupled weight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        # Update parameters: X_t = X_{t-1} - η * scale * P_t Q_t^T
        param.add_(update, alpha=-lr * scale)

        state["step"] += 1

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update matrix parameters with Dion
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                # Handle FSDP wrapped parameters
                grad = param.grad
                if hasattr(param, "_local_tensor"):
                    param_to_update = param._local_tensor
                    grad = (
                        param.grad._local_tensor
                        if hasattr(param.grad, "_local_tensor")
                        else param.grad
                    )
                else:
                    param_to_update = param

                if self._is_matrix_param(param):
                    # Initialize state if needed
                    if param not in self.state:
                        self.state[param] = self._init_matrix_state(param, group)

                    self._step_matrix_param(
                        param_to_update, grad, group, self.state[param]
                    )

        # Update scalar parameters with scalar optimizer
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients for all parameters."""
        super().zero_grad(set_to_none)
        if self.scalar_optimizer is not None:
            self.scalar_optimizer.zero_grad(set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        state_dict = super().state_dict()
        if self.scalar_optimizer is not None:
            state_dict["scalar_optimizer"] = self.scalar_optimizer.state_dict()
        state_dict["distributed_config"] = {
            "distributed": self.distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "sharding_strategy": self.sharding_strategy.value,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        scalar_state = state_dict.pop("scalar_optimizer", None)
        distributed_config = state_dict.pop("distributed_config", None)

        super().load_state_dict(state_dict)

        if scalar_state is not None and self.scalar_optimizer is not None:
            self.scalar_optimizer.load_state_dict(scalar_state)

        # Verify distributed configuration matches
        if distributed_config is not None:
            if distributed_config["distributed"] != self.distributed:
                warnings.warn(
                    f"Distributed mode mismatch: checkpoint has distributed={distributed_config['distributed']}, "
                    f"current has distributed={self.distributed}"
                )

    def add_param_group(self, param_group: Dict):
        """Add a parameter group to the optimizer."""
        super().add_param_group(param_group)
        # Re-classify parameters and reinitialize scalar optimizers
        self._classify_parameters()
        self._init_scalar_optimizers()


# Utility functions for easy integration


def create_dion_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    rank_factor: float = 0.25,
    scalar_optimizer: str = "adamw",
    **kwargs,
) -> DionOptimizer:
    """
    Create a Dion optimizer with recommended settings.

    Args:
        model: PyTorch model
        lr: Learning rate
        rank_factor: Rank fraction for low-rank approximation
        scalar_optimizer: Optimizer for non-matrix parameters
        **kwargs: Additional arguments for DionOptimizer

    Returns:
        Configured DionOptimizer instance
    """
    return DionOptimizer(
        model.parameters(),
        lr=lr,
        rank_factor=rank_factor,
        scalar_optimizer=scalar_optimizer,
        **kwargs,
    )


def create_dion_optimizer_fsdp2(
    model: nn.Module,
    lr: float = 0.01,
    rank_factor: float = 0.25,
    scalar_optimizer: str = "adamw",
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    process_group: Optional[dist.ProcessGroup] = None,
    **kwargs,
) -> DionOptimizer:
    """
    Create a Dion optimizer configured for FSDP2 training.

    Args:
        model: PyTorch model (should be wrapped with FSDP2)
        lr: Learning rate
        rank_factor: Rank fraction for low-rank approximation
        scalar_optimizer: Optimizer for non-matrix parameters
        sharding_strategy: FSDP2 sharding strategy
        process_group: Process group for distributed training
        **kwargs: Additional arguments for DionOptimizer

    Returns:
        Configured DionOptimizer instance for FSDP2
    """
    if not FSDP2_AVAILABLE:
        raise RuntimeError("FSDP2 is not available in this PyTorch version")

    return DionOptimizer(
        model.parameters(),
        lr=lr,
        rank_factor=rank_factor,
        scalar_optimizer=scalar_optimizer,
        distributed=True,
        sharding_strategy=sharding_strategy,
        process_group=process_group,
        **kwargs,
    )


def get_parameter_info(model: nn.Module, matrix_threshold: int = 32) -> Dict[str, Any]:
    """
    Analyze model parameters for Dion optimization.

    Args:
        model: PyTorch model
        matrix_threshold: Minimum size for matrix classification

    Returns:
        Dictionary with parameter counts and memory usage
    """
    matrix_params = 0
    scalar_params = 0
    matrix_memory = 0
    scalar_memory = 0
    distributed_params = 0

    for param in model.parameters():
        param_memory = param.numel() * param.element_size()

        # Check if distributed
        if isinstance(param, DTensor) or hasattr(param, "_fsdp_wrapped"):
            distributed_params += param.numel()

        if (
            param.dim() == 2
            and param.size(0) >= matrix_threshold
            and param.size(1) >= matrix_threshold
        ):
            matrix_params += param.numel()
            matrix_memory += param_memory
        else:
            scalar_params += param.numel()
            scalar_memory += param_memory

    return {
        "matrix_params": matrix_params,
        "scalar_params": scalar_params,
        "matrix_memory_mb": matrix_memory / 1024**2,
        "scalar_memory_mb": scalar_memory / 1024**2,
        "total_params": matrix_params + scalar_params,
        "distributed_params": distributed_params,
        "matrix_fraction": (
            matrix_params / (matrix_params + scalar_params)
            if (matrix_params + scalar_params) > 0
            else 0
        ),
    }


# Example usage with FSDP2
if __name__ == "__main__":
    import os

    from torch.nn.parallel import DistributedDataParallel as DDP

    # Example model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, d_ff=2048, vocab_size=50000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.ff1 = nn.Linear(d_model, d_ff, bias=False)
            self.ff2 = nn.Linear(d_ff, d_model, bias=False)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x):
            # Simplified forward pass
            x = self.embedding(x)
            return self.lm_head(x)

    # Single GPU example
    print("=== Single GPU Example ===")
    model = SimpleTransformer()

    # Analyze parameters
    param_info = get_parameter_info(model)
    print("Parameter Analysis:")
    for key, value in param_info.items():
        print(f"  {key}: {value}")

    # Create optimizer
    optimizer = create_dion_optimizer(
        model,
        lr=0.01,
        rank_factor=0.25,
        scalar_optimizer="adamw",
        weight_decay=0.01,
    )

    print(f"\nOptimizer created successfully!")
    print(f"Matrix parameters: {len(optimizer.matrix_params)}")
    print(f"Scalar parameters: {len(optimizer.scalar_params)}")
    print(f"Distributed mode: {optimizer.distributed}")

    # Test optimization step
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), input_ids.view(-1)
    )

    # Backward pass
    loss.backward()

    # Optimization step
    optimizer.step()
    optimizer.zero_grad()

    print(f"\nTest optimization step completed successfully!")
    print(f"Loss: {loss.item():.4f}")

    # Distributed example (requires proper setup)
    if dist.is_available() and os.environ.get("RANK") is not None:
        print("\n=== Distributed Example ===")

        # Initialize process group
        dist.init_process_group(backend="nccl")

        # Create model
        model = SimpleTransformer()

        # Wrap with FSDP2 if available
        if FSDP2_AVAILABLE:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

            model = FSDP(
                model,
                cpu_offload=CPUOffload(offload_params=False),
                use_orig_params=True,  # Important for Dion
            )

            # Create FSDP2-aware optimizer
            optimizer = create_dion_optimizer_fsdp2(
                model,
                lr=0.01,
                rank_factor=0.25,
                scalar_optimizer="adamw",
                weight_decay=0.01,
            )

            print(f"FSDP2 model and Dion optimizer created successfully!")
            print(f"World size: {optimizer.world_size}")
            print(f"Rank: {optimizer.rank}")

        # Cleanup
        dist.destroy_process_group()
