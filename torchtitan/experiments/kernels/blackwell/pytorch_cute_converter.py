import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .group_gemm_base import GroupGEMMStrategy


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm_kernel import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
    print("✓ CUTLASS and strategies imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ CUTLASS import failed: {e}")
    print("CUTLASSGroupedGemmStrategy will not be available")


logger = logging.getLogger(__name__)

__all__ = ["PyTorchToCuteConverter", "ExpertOperationMetadata"]


class PyTorchToCuteConverter:
    """
    Standalone converter for PyTorch tensors to CUTE tensors.

    """

    # Data type mappings
    DTYPE_MAP = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int8: cutlass.Int8,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }

    def __init__(self, alignment: int = 16, acc_dtype=cutlass.Float32):
        """
        Initialize the converter.

        Args:
            alignment: Memory alignment requirement for CUTE tensors
            acc_dtype: Accumulation data type for CUTLASS operations
        """
        self.alignment = alignment
        self.acc_dtype = acc_dtype

    def get_cutlass_dtype(self, torch_dtype: torch.dtype):
        """Convert PyTorch dtype to CUTLASS dtype with validation."""
        if torch_dtype not in self.DTYPE_MAP:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return self.DTYPE_MAP[torch_dtype]

    def convert_tensor_to_cute(
        self,
        tensor: torch.Tensor,
        make_dynamic: bool = True,
        dynamic_leading_dim: int = 1,
    ) -> "cute.Tensor":
        """
        Convert PyTorch tensor to CUTE tensor with validation.

        Args:
            tensor: Input PyTorch tensor
            make_dynamic: Whether to mark layout as dynamic
            dynamic_leading_dim: Which dimension to make dynamic

        Returns:
            CUTE tensor ready for CUTLASS operations
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Convert to MNKL format if needed
        if len(tensor.shape) == 2:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
        else:
            mnkl_tensor = tensor

        # Create CUTE tensor
        cute_tensor = from_dlpack(mnkl_tensor, assumed_align=self.alignment)
        cute_tensor.element_type = self.get_cutlass_dtype(tensor.dtype)

        if make_dynamic:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=dynamic_leading_dim
            )

        return cute_tensor

    def create_metadata_tensors(
        self,
        problem_sizes: List[List[int]],
        strides_abc: List[List[List[int]]],
        ptrs_abc: List[List[int]],
        device: torch.device,
    ) -> Tuple:
        """
        Create CUTE tensors for grouped GEMM metadata with validation.

        Args:
            problem_sizes: List of [M, N, K, L] for each problem
            strides_abc: List of stride information for A, B, C tensors
            ptrs_abc: List of data pointers for A, B, C tensors
            device: Target device

        Returns:
            Tuple of (problem_sizes_cute, strides_cute, ptrs_cute)
        """
        if not problem_sizes:
            raise ValueError("problem_sizes cannot be empty")

        if not (len(problem_sizes) == len(strides_abc) == len(ptrs_abc)):
            raise ValueError("All metadata lists must have the same length")

        # Convert to PyTorch tensors with validation
        try:
            problem_sizes_tensor = torch.tensor(
                problem_sizes, dtype=torch.int32, device=device
            )
            strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
            ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)
        except Exception as e:
            raise ValueError(f"Failed to create metadata tensors: {e}")

        # Convert to CUTE tensors
        return (
            from_dlpack(problem_sizes_tensor, assumed_align=self.alignment),
            from_dlpack(strides_tensor, assumed_align=self.alignment),
            from_dlpack(ptrs_tensor, assumed_align=self.alignment),
        )

    def create_initial_tensors(
        self,
        problem_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> List:
        """
        Create initial CUTE tensors for kernel compilation with validation.

        Args:
            problem_shape: (M, N, K, L) shape tuple
            device: Target device
            dtype: PyTorch data type

        Returns:
            List of CUTE tensors for kernel compilation
        """
        M, N, K, L = problem_shape

        if any(dim <= 0 for dim in [M, N, K, L]):
            raise ValueError(f"Invalid problem shape: {problem_shape}")

        # Create PyTorch tensors
        tensors = [
            torch.randn(M, K, dtype=dtype, device=device),  # A
            torch.randn(N, K, dtype=dtype, device=device),  # B
            torch.zeros(M, N, dtype=dtype, device=device),  # C
        ]

        # Convert to CUTE tensors
        cute_tensors = []
        for tensor in tensors:
            cute_tensor = self.convert_tensor_to_cute(tensor)
            cute_tensors.append(cute_tensor)

        return cute_tensors

    def create_tensormap_buffer(
        self,
        device: torch.device,
        sm_count: int,
        tensormap_count: int = 3,
        tensormap_bytes: int = 128,
    ):
        """
        Create tensormap buffer for CUTLASS kernel with validation.

        Args:
            device: Target device
            sm_count: Number of streaming multiprocessors
            tensormap_count: Number of tensormap entries
            tensormap_bytes: Bytes per tensormap entry

        Returns:
            CUTE tensor for tensormap buffer
        """
        if sm_count <= 0:
            raise ValueError(f"Invalid sm_count: {sm_count}")

        if tensormap_bytes % 8 != 0:
            raise ValueError(
                f"tensormap_bytes must be divisible by 8: {tensormap_bytes}"
            )

        tensormap_tensor = torch.zeros(
            (sm_count, tensormap_count, tensormap_bytes // 8),
            dtype=torch.int64,
            device=device,
        )

        return from_dlpack(tensormap_tensor, assumed_align=self.alignment)


class ExpertOperationMetadata:
    """Helper class to manage metadata for individual expert operations."""

    def __init__(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ):
        self.input_tensor = input_tensor.contiguous()
        self.weight_tensor = weight_tensor.contiguous()
        self.output_tensor = output_tensor.contiguous()

        # Validate dimensions
        self._validate_dimensions()

        # Extract shapes
        self.M, self.K = self.input_tensor.shape
        self.N = self.weight_tensor.shape[0]  # Assuming [out_features, in_features]
        self.L = 1

    def _validate_dimensions(self):
        """Validate tensor dimensions for matrix multiplication."""
        if len(self.input_tensor.shape) != 2:
            raise ValueError(
                f"Input tensor must be 2D, got shape: {self.input_tensor.shape}"
            )

        if len(self.weight_tensor.shape) != 2:
            raise ValueError(
                f"Weight tensor must be 2D, got shape: {self.weight_tensor.shape}"
            )

        if len(self.output_tensor.shape) != 2:
            raise ValueError(
                f"Output tensor must be 2D, got shape: {self.output_tensor.shape}"
            )

        input_k = self.input_tensor.shape[1]
        weight_k = self.weight_tensor.shape[1]

        if input_k != weight_k:
            raise ValueError(
                f"Matrix multiplication dimension mismatch: "
                f"input K={input_k} vs weight K={weight_k}"
            )

    def get_problem_size(self) -> List[int]:
        """Get problem size in MNKL format."""
        return [self.M, self.N, self.K, self.L]

    def get_strides(self) -> List[List[int]]:
        """Get stride information for all tensors."""
        # Convert to MNKL format for stride extraction
        input_mnkl = self.input_tensor.unsqueeze(-1)
        weight_mnkl = self.weight_tensor.unsqueeze(-1)
        output_mnkl = self.output_tensor.unsqueeze(-1)

        return [
            list(input_mnkl.stride()[:2]),
            list(weight_mnkl.stride()[:2]),
            list(output_mnkl.stride()[:2]),
        ]

    def get_pointers(self) -> List[int]:
        """Get data pointers for all tensors."""
        return [
            self.input_tensor.data_ptr(),
            self.weight_tensor.data_ptr(),
            self.output_tensor.data_ptr(),
        ]
