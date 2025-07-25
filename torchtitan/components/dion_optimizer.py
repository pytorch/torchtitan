# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dion in PyTorch: Distributed Orthonormalized Updates Optimizer

An *unofficial* PyTorch implementation of the Dion optimizer from the paper:
"Dion: Distributed Orthonormalized Updates" by Ahn et al.
https://arxiv.org/abs/2504.05295


Supports:
- Single gpu and distributed training
- 1D parallelisms (DP, FSDP2)
- TP will need more testing
- Model head exclusion from matrix param treatment
"""

import math
import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.adamw import AdamW

from .gemm_utils import CutlassGemmManager


# optional
torch.backends.cuda.matmul.allow_tf32 = True

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP2

    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False
    FSDP2 = None

# not recommended, just adding b/c paper mentioned it
try:
    from torch.optim.lion import Lion

    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    Lion = None


def create_fqn_based_exclusion_fn(
    model: nn.Module,
    head_names: Optional[List[str]] = None,
    head_suffixes: Optional[List[str]] = None,
) -> Callable[[torch.Tensor], bool]:
    """
    Create a function to exclude parameters based on their fully qualified names (FQNs).

    Args:
        model: PyTorch model to analyze
        head_names: List of exact parameter names to exclude (e.g., ['output.weight', 'lm_head.weight'])
        head_suffixes: List of parameter name suffixes to exclude (e.g., ['output', 'lm_head', 'classifier'])

    Returns:
        Function that returns True if parameter should be excluded from matrix treatment
    """
    if head_names is None:
        head_names = []

    if head_suffixes is None:
        head_suffixes = [
            "output",
            "lm_head",
            "classifier",
            "head",
            "prediction_head",
            "decoder",
            "output_projection",
        ]

    # Create mapping from parameter tensor to its FQN
    param_to_name = {}
    for name, param in model.named_parameters():
        param_to_name[param] = name

    def should_exclude(param: torch.Tensor) -> bool:
        if param.dim() != 2:
            return False

        param_name = param_to_name.get(param, "")

        # Check exact name matches
        if param_name in head_names:
            return True

        # Check if parameter name ends with any of the head suffixes
        for suffix in head_suffixes:
            if param_name.endswith(f".{suffix}.weight") or param_name.endswith(
                f".{suffix}"
            ):
                return True
            # Also check if the parameter name is exactly the suffix (for top-level parameters)
            if param_name == f"{suffix}.weight" or param_name == suffix:
                return True

        return False

    return should_exclude


def create_combined_exclusion_fn(
    model: nn.Module,
    vocab_size: Optional[int] = None,
    head_names: Optional[List[str]] = None,
    head_suffixes: Optional[List[str]] = None,
    excluded_shapes: Optional[List[Tuple[int, int]]] = None,
    prefer_fqn: bool = True,
) -> Callable[[torch.Tensor], bool]:
    """
    Create a combined exclusion function that uses multiple strategies.

    Args:
        model: PyTorch model to analyze
        vocab_size: Vocabulary size for shape-based detection
        head_names: List of exact parameter names to exclude
        head_suffixes: List of parameter name suffixes to exclude
        excluded_shapes: List of exact shapes to exclude
        prefer_fqn: If True, prioritize FQN-based detection over shape-based

    Returns:
        Function that returns True if parameter should be excluded from matrix treatment
    """
    fqn_fn = create_fqn_based_exclusion_fn(model, head_names, head_suffixes)

    shape_fn = None
    if vocab_size is not None:
        shape_fn = create_model_head_exclusion_fn(vocab_size)

    exact_shape_fn = None
    if excluded_shapes:
        exact_shape_fn = create_shape_based_exclusion_fn(excluded_shapes)

    def should_exclude(param: torch.Tensor) -> bool:
        if param.dim() != 2:
            return False

        # Always check FQN first if prefer_fqn is True
        if prefer_fqn and fqn_fn(param):
            return True

        # Check exact shapes
        if exact_shape_fn and exact_shape_fn(param):
            return True

        # Check vocab-based shape detection
        if shape_fn and shape_fn(param):
            return True

        # If not prefer_fqn, check FQN last as fallback
        if not prefer_fqn and fqn_fn(param):
            return True

        return False

    return should_exclude


class DionOptimizer(Optimizer):
    """
    Dion (DIstributed OrthoNormalization) Optimizer

    Unofficial - Implements the Dion optimizer with support for both centralized and distributed training.
    Orthonormalized updates for matrix parameters via low rank decomposition, and configurable scalar optimizers
    (ala AdamW) for non-matrix parameters.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor μ (default: 0.95)
        rank_factor: Rank fraction r/d for low-rank approximation (default: .75, full rank = 1.0)
        scalar_optimizer: Optimizer class for non-matrix parameters ('adamw', 'lion')

        scalar_lr: Learning rate for scalar optimizer (default: None, uses lr)
        weight_decay: Weight decay coefficient (default: 0.01)
        scalar_weight_decay: Weight decay for scalar parameters (default: 0.0)

        eps: Small constant for numerical stability (default: 1e-8)
        oversampling_factor: Factor for randomized QR oversampling (default: 1.25), based on paper settings
        distributed: Whether to use distributed implementation (default: auto-detect)

        matrix_threshold: Minimum size for treating parameter as matrix (default: 32)
        exclude_from_matrix: Optional callable to exclude specific parameters from matrix treatment.
                           Takes a parameter tensor and returns True if it should be excluded.
                           If None and auto_exclude_heads=True, will be automatically created.

        model: Optional model reference for automatic head detection (default: None)

        TODO: this might be a bit verbose...not sure we want to expose all these options but lets see how well autodetect works
        ------ head detection options ------
        auto_exclude_heads: Whether to automatically exclude model heads from matrix treatment (default: True)
        head_names: List of exact parameter names to exclude (e.g., ['output.weight', 'lm_head.weight'])
        head_suffixes: List of parameter name suffixes to exclude (default: ['output', 'lm_head', 'classifier', etc.])
        vocab_size: Vocabulary size for shape-based model head detection (default: auto-detect from model)
        excluded_shapes: List of exact (height, width) shapes to exclude from matrix treatment
        use_fqn_detection: Whether to use FQN-based detection over shape-based (default: True)
        debug_classification: Whether to print parameter classification details (default: False)
        report_detected_heads: Whether to report detected model heads to user (default: True)
        ------ end head detection options ------

        process_group: Process group for distributed training (default: None)
        max_norm: Maximum gradient norm for clipping (default: 10.0)
        use_randomized_cholesky_qr: Whether to use randomized Cholesky QR for orthogonalization (default: True)
        **scalar_kwargs: Additional arguments for scalar optimizer

    Examples:
        # Basic usage with auto head detection
        optimizer = DionOptimizer(model.parameters(), model=model, lr=0.01)
        # Output:
        # Dion: Detected 1 model head(s) → using AdamW optimization:
        #   output.weight: (32000, 4096) (131,072,000 params)
        #   Total excluded: 131,072,000 parameters
        #   Remaining matrix parameters: 24 → using Dion optimization

        # Manual head exclusion
        optimizer = DionOptimizer(
            model.parameters(),
            model=model,
            head_suffixes=['output', 'lm_head'],
            debug_classification=True
        )

        # Disable reporting
        optimizer = DionOptimizer(
            model.parameters(),
            model=model,
            report_detected_heads=False
        )

        # Disable auto detection and use manual exclusion
        def my_exclusion_fn(param):
            return param.shape == (vocab_size, hidden_size)
        optimizer = DionOptimizer(
            model.parameters(),
            auto_exclude_heads=False,
            exclude_from_matrix=my_exclusion_fn
        )

    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        rank_factor: float = 0.55,
        scalar_optimizer: str = "adamw",
        scalar_lr: Optional[float] = None,
        weight_decay: float = 0.01,
        scalar_weight_decay: float = 0.0,
        eps: float = 1e-8,
        oversampling_factor: float = 1.25,
        distributed: Optional[bool] = None,
        matrix_threshold: int = 32,
        exclude_from_matrix: Optional[Callable[[torch.Tensor], bool]] = None,
        model: Optional[nn.Module] = None,
        auto_exclude_heads: bool = True,
        head_names: Optional[List[str]] = None,
        head_suffixes: Optional[List[str]] = None,
        debug_classification: bool = False,
        report_detected_heads: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        max_norm: float = 10.0,
        use_randomized_cholesky_qr: bool = True,
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
        self.max_norm = max_norm
        self.use_randomized_cholesky_qr = use_randomized_cholesky_qr
        self.debug_classification = debug_classification
        self.report_detected_heads = report_detected_heads

        # Auto head detection setup - only use FQN-based detection
        if exclude_from_matrix is None and auto_exclude_heads:
            if model is not None:
                # Use FQN-based detection for reliable head identification
                exclude_from_matrix = create_fqn_based_exclusion_fn(
                    model=model,
                    head_names=head_names,
                    head_suffixes=head_suffixes,
                )
            else:
                warnings.warn(
                    "auto_exclude_heads=True but no model provided. "
                    "Cannot perform automatic head detection. Set auto_exclude_heads=False to disable this warning."
                )

        self.exclude_from_matrix = exclude_from_matrix

        # Store model reference for later use
        self.model = model

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

        # Create parameter name mapping for reporting (after param_groups is available)
        self._param_groups_with_names = []
        if model is not None:
            # Create mapping from parameter tensor to its name for better reporting
            param_to_name = {param: name for name, param in model.named_parameters()}
            for group in self.param_groups:
                for param in group["params"]:
                    name = param_to_name.get(param, f"unknown_param_{id(param)}")
                    self._param_groups_with_names.append((name, param))

        # Initialize CUTLASS GEMM manager for optimized matrix operations
        try:
            self.cutlass_gemm_manager = CutlassGemmManager()
        except Exception as e:
            warnings.warn(
                f"Failed to initialize CutlassGemmManager: {e}. Falling back to torch.mm"
            )
            self.cutlass_gemm_manager = None

        # Initialize parameter classification and scalar optimizers
        self._classify_parameters()
        self._init_scalar_optimizers()

        # Report detected heads to user
        self._report_detected_heads()

    def _get_scalar_optimizer_class(self, optimizer_name: str):
        """Get scalar optimizer class by name."""
        optimizer_map = {"adamw": AdamW}

        if LION_AVAILABLE:
            optimizer_map["lion"] = Lion
        elif optimizer_name == "lion":
            raise ImportError("Lion optimizer not available. Use 'adamw' instead")

        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported scalar optimizer: {optimizer_name}")

        return optimizer_map[optimizer_name]

    def _is_matrix_param(self, param: torch.Tensor) -> bool:
        """Determine if parameter should be treated as a matrix."""

        # First check if this parameter should be explicitly excluded
        if self.exclude_from_matrix is not None and self.exclude_from_matrix(param):
            return False

        # Must be 2D and both dimensions >= threshold
        return (
            param.dim() == 2
            and param.size(0) >= self.matrix_threshold
            and param.size(1) >= self.matrix_threshold
        )

    def _classify_parameters(self):
        """Classify parameters into matrix and scalar types."""
        self.matrix_params = []
        self.scalar_params = []
        self.excluded_params_info = []  # Store info about excluded parameters

        # Create parameter name mapping for excluded parameter tracking
        param_to_name = {}
        if hasattr(self, "_param_groups_with_names"):
            # If we have named parameters, use them
            for name, param in self._param_groups_with_names:
                param_to_name[param] = name

        for group in self.param_groups:
            for p in group["params"]:
                param_name = param_to_name.get(p, f"param_shape_{p.shape}")

                if self._is_matrix_param(p):
                    self.matrix_params.append(p)
                else:
                    self.scalar_params.append(p)

                    # Check if this was excluded from matrix treatment
                    if (
                        p.dim() == 2
                        and p.size(0) >= self.matrix_threshold
                        and p.size(1) >= self.matrix_threshold
                        and self.exclude_from_matrix is not None
                        and self.exclude_from_matrix(p)
                    ):
                        self.excluded_params_info.append(
                            (param_name, p.shape, p.numel())
                        )

        # Log classification for debugging
        if self.debug_classification:
            print(f"=== Dion Optimizer Internal Classification ===")

            # Show matrix parameters with names if available
            print(f"Matrix parameters (Dion): {len(self.matrix_params)}")
            matrix_param_names = []
            for p in self.matrix_params:
                param_name = param_to_name.get(p, f"param_shape_{p.shape}")
                matrix_param_names.append((param_name, p.shape, p.numel()))

            for name, shape, numel in matrix_param_names:
                print(f"  {name}: {shape} ({numel:,} params)")

            # Show excluded parameters
            if self.excluded_params_info:
                print(
                    f"\nExcluded from matrix treatment (AdamW): {len(self.excluded_params_info)}"
                )
                for name, shape, numel in self.excluded_params_info:
                    print(f"  {name}: {shape} ({numel:,} params)")

            # Show remaining scalar parameters
            remaining_scalar_count = len(self.scalar_params) - len(
                self.excluded_params_info
            )
            if remaining_scalar_count > 0:
                print(f"\nOther scalar parameters (AdamW): {remaining_scalar_count}")
                for p in self.scalar_params:
                    param_name = param_to_name.get(p, f"param_shape_{p.shape}")
                    # Only show if not already shown in excluded
                    if not any(
                        name == param_name for name, _, _ in self.excluded_params_info
                    ):
                        print(f"  {param_name}: {p.shape} ({p.numel():,} params)")

            print("=" * 48)

    def _report_detected_heads(self):
        """Report detected model heads to the user."""
        if (
            self.excluded_params_info
            and self.exclude_from_matrix is not None
            and self.report_detected_heads
        ):

            head_names = [name for name, _, _ in self.excluded_params_info]
            total_head_params = sum(numel for _, _, numel in self.excluded_params_info)

            if (
                not self.debug_classification
            ):  # Only print if not already shown in debug
                print(f"Dion: Detected {len(head_names)} model head(s) → using AdamW:")
                for name, shape, numel in self.excluded_params_info:
                    print(f"  {name}: {shape} ({numel:,} params)")

                print(
                    f"  Transformer layers using Dion: {len(self.matrix_params)} matrix parameters"
                )
                print()

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
        """Get learning rate scaling factor based on parameter type."""
        if hasattr(param, "_local_tensor"):
            actual_param = param._local_tensor
        else:
            actual_param = param

        if actual_param.dim() == 0:  # Scalar (normalization)
            return 1.0
        elif actual_param.dim() == 1:  # Vector (bias, embedding, etc.)
            if actual_param.numel() > 1000:  # Likely unembedding
                return 1.0 / math.sqrt(actual_param.size(0))
            else:  # Likely bias
                return 1.0
        elif actual_param.dim() == 2:  # Matrix (but excluded from matrix treatment)
            d_out, d_in = actual_param.shape
            # For model heads, use standard scaling
            if self.exclude_from_matrix is not None and self.exclude_from_matrix(
                actual_param
            ):
                return 1.0 / math.sqrt(d_in)  # Standard initialization scaling
            return math.sqrt(d_out / d_in)
        else:
            return 1.0

    def _get_weight_decay(self, param: torch.Tensor) -> float:
        """Get weight decay for parameter type."""

        # Apply scalar weight decay to all scalar parameters
        # For excluded 2D parameters (like model heads), still apply weight decay
        if param.dim() >= 2:
            return self.scalar_weight_decay
        return 0.0

    def _init_matrix_state(self, param: torch.Tensor, group: Dict) -> Dict:
        """Initialize state for matrix parameter."""
        state = {}

        m, n = param.shape
        rank = min(m, n, max(1, int(min(m, n) * self.rank_factor)))

        # Create momentum buffer with same shape as actual parameter
        state["momentum_buffer"] = torch.zeros_like(param)

        # Create right factor for warm-starting power iteration
        state["right_factor"] = torch.randn(
            n, rank, device=param.device, dtype=param.dtype
        )
        state["right_factor"] = self._normalize_columns(state["right_factor"])

        # Step counter and metadata
        state["step"] = 0
        state["rank"] = rank
        state["local_m"] = m
        state["local_n"] = n

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
        if self.use_randomized_cholesky_qr:
            P = self._distributed_orthogonalize_matrix(P)
        else:
            P = self._orthogonalize_matrix(P)

        # R = B^T P (right factor)
        R = torch.mm(B.t(), P)

        return P, R

    def _distributed_power_iteration(
        self, B: torch.Tensor, Q_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Distributed power iteration for FSDP2.

        Key additions:
        1. All-reduce P after local computation
        2. All-reduce R after local computation
        """
        # Step 1: Local computation P_hat = B_local * Q
        P_local = torch.mm(B, Q_prev)

        # Step 2: Synchronize P across all ranks (FSDP shards + DP ranks)
        # This implements E_DP[Σ_FS[P̂]] from Algorithm 3
        P_sync = P_local.clone()
        if self.distributed:
            dist.all_reduce(P_sync, op=dist.ReduceOp.SUM, group=self.process_group)
            P_sync = P_sync / self.world_size

        # Step 3: Orthogonalize the synchronized P
        if self.use_randomized_cholesky_qr:
            P = self._distributed_orthogonalize_matrix(P_sync)
        else:
            P = self._orthogonalize_matrix(P_sync)

        # Step 4: Local computation R_hat = B_local^T * P
        R_local = torch.mm(B.t(), P)

        # Step 5: Synchronize R across all ranks
        # This implements E_DP[R̂] from Algorithm 3
        R = R_local.clone()
        if self.distributed:
            dist.all_reduce(R, op=dist.ReduceOp.SUM, group=self.process_group)
            R = R / self.world_size

        return P, R

    def _orthogonalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix using QR decomposition (proven stable version)."""
        try:
            Q, _ = torch.linalg.qr(matrix)
            return Q
        except:
            # Fallback for numerical issues
            matrix_stabilized = matrix + 1e-8 * torch.randn_like(matrix)
            Q, _ = torch.linalg.qr(matrix_stabilized)
            return Q

    def _distributed_orthogonalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Distributed orthogonalization using randomized Cholesky QR (Algorithm 2 in the paper)."""
        m, r = matrix.shape
        k = max(r, int(self.oversampling_factor * r))

        # Random sketching matrix
        S = torch.randn(k, m, device=matrix.device, dtype=matrix.dtype) / math.sqrt(k)

        # First iteration: randomized QR
        G = torch.mm(S, matrix)

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

        # Cholesky decomposition with numerical stability
        for jitter in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
            try:
                H_stable = H + torch.eye(r, device=H.device, dtype=H.dtype) * jitter
                R2 = torch.linalg.cholesky(H_stable)
                break
            except:
                if jitter == 1e-1:
                    # Fallback to QR if Cholesky fails
                    Q, _ = torch.linalg.qr(matrix)
                    return Q

        # Final orthogonalized result
        result = torch.linalg.solve_triangular(R2.t(), B.t(), upper=False).t()
        return result

    def _distributed_column_normalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Distributed column normalization for FSDP2."""
        # Compute local column norms squared
        local_norms_sq = torch.sum(matrix * matrix, dim=0)

        # All-reduce sum to get global column norms
        if self.distributed:
            dist.all_reduce(
                local_norms_sq, op=dist.ReduceOp.SUM, group=self.process_group
            )

        # Take square root and normalize
        global_norms = torch.sqrt(torch.clamp(local_norms_sq, min=1e-16))
        return matrix / torch.clamp(global_norms, min=1e-8)

    def _step_matrix_param(
        self, param: torch.Tensor, grad: torch.Tensor, group: Dict, state: Dict
    ):
        """Perform Dion update step for matrix parameter."""
        momentum = group["momentum"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]

        # Check for NaN or Inf in gradient
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            warnings.warn("NaN or Inf detected in gradient. Skipping update.")
            return

        # Clip gradient to prevent extreme values
        grad_norm = torch.norm(grad)
        if grad_norm > self.max_norm:
            grad = grad * (self.max_norm / (grad_norm + eps))

        # Get state variables
        momentum_buffer = state["momentum_buffer"]
        right_factor = state["right_factor"]
        rank = state["rank"]

        # Ensure momentum buffer has correct shape
        if momentum_buffer.shape != grad.shape:
            momentum_buffer = torch.zeros_like(grad)
            state["momentum_buffer"] = momentum_buffer

        # Ensure right factor has correct shape
        expected_n = grad.shape[1]
        if right_factor.shape[0] != expected_n:
            right_factor = torch.randn(
                expected_n, rank, device=grad.device, dtype=grad.dtype
            )
            right_factor = self._normalize_columns(right_factor)
            state["right_factor"] = right_factor

        # Form buffer B_t = M_{t-1} + G_t (Algorithm 1, line 3)
        buffer = momentum_buffer + grad

        # Power iteration: approximate B_t ≈ P_t R_t^T (Algorithm 1, line 4)
        try:
            if self.distributed:
                P, R = self._distributed_power_iteration(buffer, right_factor)
            else:
                P, R = self._power_iteration(buffer, right_factor)
        except RuntimeError as e:
            warnings.warn(f"Power iteration failed: {e}. Skipping update.")
            return

        # Approximation for error feedback
        approx = torch.mm(P, R.t())

        # Error feedback: M_t = B_t - (1-μ)P_t R_t^T (Algorithm 1, line 6)
        momentum_buffer.copy_(buffer - (1 - momentum) * approx)

        # Update right factor with column normalization (Algorithm 1, line 8)
        if self.distributed:
            right_factor.copy_(self._distributed_column_normalize(R))
        else:
            right_factor.copy_(self._normalize_columns(R))

        # Compute scaled orthonormal update (Algorithm 1, line 9)
        m, n = param.shape
        scale = math.sqrt(m / n)

        # Normalize columns of R to get Q
        if self.distributed:
            Q = self._distributed_column_normalize(R)
        else:
            Q = self._normalize_columns(R)

        # Use CUTLASS GEMM for the critical update computation (good candidate for epilogue fusion)
        if self.cutlass_gemm_manager is not None:
            try:
                update = self.cutlass_gemm_manager.gemm(P, Q, transpose_B=True)
            except Exception as e:
                warnings.warn(f"CUTLASS GEMM failed: {e}. Falling back to torch.mm")
                update = torch.mm(P, Q.t())
        else:
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

                if self._is_matrix_param(param):
                    # Initialize state if needed
                    if param not in self.state:
                        self.state[param] = self._init_matrix_state(param, group)

                    self._step_matrix_param(param, param.grad, group, self.state[param])

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
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        scalar_state = state_dict.pop("scalar_optimizer", None)
        distributed_config = state_dict.pop("distributed_config", None)

        super().load_state_dict(state_dict)

        if scalar_state is not None and self.scalar_optimizer is not None:
            self.scalar_optimizer.load_state_dict(scalar_state)

        if distributed_config is not None:
            if distributed_config["distributed"] != self.distributed:
                warnings.warn(
                    f"Distributed mode mismatch: checkpoint has distributed="
                    f"{distributed_config['distributed']}, current has distributed={self.distributed}"
                )

    def add_param_group(self, param_group: Dict):
        """Add a parameter group to the optimizer."""
        super().add_param_group(param_group)
        self._classify_parameters()
        self._init_scalar_optimizers()


# Utility functions


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
        **kwargs: Additional arguments for DionOptimizer (see DionOptimizer.__init__ for full list)

    Returns:
        Configured DionOptimizer
    """
    return DionOptimizer(
        model.parameters(),
        model=model,
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
    process_group: Optional[dist.ProcessGroup] = None,
    **kwargs,
) -> DionOptimizer:
    """Create a Dion optimizer configured for FSDP2 training."""
    return DionOptimizer(
        model.parameters(),
        model=model,
        lr=lr,
        rank_factor=rank_factor,
        scalar_optimizer=scalar_optimizer,
        distributed=True,
        process_group=process_group,
        **kwargs,
    )


def get_parameter_info(
    model: nn.Module,
    matrix_threshold: int = 32,
    exclude_from_matrix: Optional[Callable[[torch.Tensor], bool]] = None,
    show_details: bool = False,
) -> Dict[str, Any]:
    """
    Analyze model parameters for Dion optimization.

    Args:
        model: PyTorch model
        matrix_threshold: Minimum size for matrix classification
        exclude_from_matrix: Optional exclusion function
        show_details: Whether to print detailed parameter breakdown

    Returns:
        Dictionary with parameter counts and memory usage
    """
    matrix_params = 0
    scalar_params = 0
    matrix_memory = 0
    scalar_memory = 0
    distributed_params = 0
    excluded_matrix_params = 0

    matrix_param_details = []
    scalar_param_details = []
    excluded_param_details = []

    for name, param in model.named_parameters():
        param_memory = param.numel() * param.element_size()

        # Check if distributed
        if hasattr(param, "_local_tensor") or hasattr(param, "_fsdp_wrapped"):
            distributed_params += param.numel()

        # Check if would be matrix but is excluded
        would_be_matrix = (
            param.dim() == 2
            and param.size(0) >= matrix_threshold
            and param.size(1) >= matrix_threshold
        )

        is_excluded = exclude_from_matrix is not None and exclude_from_matrix(param)

        if would_be_matrix and not is_excluded:
            matrix_params += param.numel()
            matrix_memory += param_memory
            matrix_param_details.append((name, param.shape, param.numel()))
        else:
            scalar_params += param.numel()
            scalar_memory += param_memory
            if would_be_matrix and is_excluded:
                excluded_matrix_params += param.numel()
                excluded_param_details.append((name, param.shape, param.numel()))
            else:
                scalar_param_details.append((name, param.shape, param.numel()))

    result = {
        "matrix_params": matrix_params,
        "scalar_params": scalar_params,
        "excluded_matrix_params": excluded_matrix_params,
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

    if show_details:
        print("=== Parameter Analysis Details ===")
        print(f"Matrix parameters (Dion optimization): {len(matrix_param_details)}")
        for name, shape, numel in matrix_param_details:
            print(f"  {name}: {shape} ({numel:,} params)")

        print(f"\nExcluded matrix parameters (AdamW): {len(excluded_param_details)}")
        for name, shape, numel in excluded_param_details:
            print(f"  {name}: {shape} ({numel:,} params)")

        print(f"\nScalar parameters (AdamW): {len(scalar_param_details)}")
        for name, shape, numel in scalar_param_details:
            print(f"  {name}: {shape} ({numel:,} params)")

        print(f"\nSummary:")
        print(f"  Total parameters: {result['total_params']:,}")
        print(
            f"  Matrix parameters: {result['matrix_params']:,} ({result['matrix_fraction']:.1%})"
        )
        print(f"  Excluded matrix parameters: {result['excluded_matrix_params']:,}")
        print(f"  Scalar parameters: {result['scalar_params']:,}")
        print(f"  Matrix memory: {result['matrix_memory_mb']:.1f} MB")
        print(f"  Scalar memory: {result['scalar_memory_mb']:.1f} MB")
        print("=" * 35)

    return result
