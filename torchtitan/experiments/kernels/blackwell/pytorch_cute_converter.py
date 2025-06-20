"""
Standalone PyTorch to CUTE tensor converter for CUTLASS Group GEMM operations.

This module provides utilities to convert PyTorch tensors to CUTE tensors
with proper layout, alignment, and data type handling for CUTLASS kernels.
"""

from typing import List, Optional, Tuple, Union

import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
except ImportError as e:
    HAS_CUTLASS = False
    print(f"❌ CUTLASS import failed: {e}")


class PyTorchToCuteConverter:
    """
    Converter class for PyTorch tensors to CUTE tensors for CUTLASS Group GEMM.

    Handles data type mapping, memory layout, alignment, and stride manipulation
    for optimal CUTLASS kernel performance.
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

    def __init__(self, default_alignment: int = 16, default_acc_dtype=cutlass.Float32):
        """
        Initialize the converter.

        Args:
            default_alignment: Memory alignment requirement for CUTE tensors
            default_acc_dtype: Default accumulation data type for CUTLASS
        """
        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        self.default_alignment = default_alignment
        self.default_acc_dtype = default_acc_dtype

    def get_cutlass_dtype(self, torch_dtype: torch.dtype):
        """Convert PyTorch dtype to CUTLASS dtype."""
        if torch_dtype not in self.DTYPE_MAP:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return self.DTYPE_MAP[torch_dtype]

    def torch_to_cute_tensor(
        self,
        tensor: torch.Tensor,
        alignment: Optional[int] = None,
        make_dynamic: bool = True,
        dynamic_leading_dim: int = 1,
    ) -> "cute.Tensor":
        """
        Convert a single PyTorch tensor to CUTE tensor.

        Args:
            tensor: Input PyTorch tensor
            alignment: Memory alignment (uses default if None)
            make_dynamic: Whether to mark layout as dynamic
            dynamic_leading_dim: Which dimension to make dynamic

        Returns:
            CUTE tensor ready for CUTLASS operations
        """
        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        # Ensure tensor is contiguous
        tensor = tensor.contiguous()

        # Convert to MNKL format (add batch dimension if needed)
        if len(tensor.shape) == 2:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
        else:
            mnkl_tensor = tensor

        # Get alignment
        align = alignment or self.default_alignment

        # Convert to CUTE tensor
        cute_tensor = from_dlpack(mnkl_tensor, assumed_align=align)

        # Set element type
        cutlass_dtype = self.get_cutlass_dtype(tensor.dtype)
        cute_tensor.element_type = cutlass_dtype

        # Make layout dynamic if requested
        if make_dynamic:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=dynamic_leading_dim
            )

        return cute_tensor

    def create_grouped_gemm_tensors(
        self,
        A_tensors: List[torch.Tensor],
        B_tensors: List[torch.Tensor],
        C_tensors: List[torch.Tensor],
        alignment: Optional[int] = None,
    ) -> Tuple[List, List, List]:
        """
        Convert lists of PyTorch tensors to CUTE tensors for grouped GEMM.

        Args:
            A_tensors: List of A matrices (input tensors)
            B_tensors: List of B matrices (weight tensors)
            C_tensors: List of C matrices (output tensors)
            alignment: Memory alignment

        Returns:
            Tuple of (cute_A_tensors, cute_B_tensors, cute_C_tensors)
        """
        cute_A = [self.torch_to_cute_tensor(A, alignment) for A in A_tensors]
        cute_B = [self.torch_to_cute_tensor(B, alignment) for B in B_tensors]
        cute_C = [self.torch_to_cute_tensor(C, alignment) for C in C_tensors]

        return cute_A, cute_B, cute_C

    def create_metadata_tensors(
        self,
        problem_sizes: List[List[int]],
        strides_abc: List[List[List[int]]],
        ptrs_abc: List[List[int]],
        device: torch.device,
        alignment: Optional[int] = None,
    ) -> Tuple:
        """
        Create CUTE tensors for grouped GEMM metadata.

        Args:
            problem_sizes: List of [M, N, K, L] for each problem
            strides_abc: List of stride information for A, B, C tensors
            ptrs_abc: List of data pointers for A, B, C tensors
            device: Target device
            alignment: Memory alignment

        Returns:
            Tuple of (problem_sizes_cute, strides_cute, ptrs_cute)
        """
        align = alignment or self.default_alignment

        # Convert to PyTorch tensors first
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(problem_sizes_tensor, assumed_align=align)
        strides_cute = from_dlpack(strides_tensor, assumed_align=align)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=align)

        return problem_sizes_cute, strides_cute, ptrs_cute

    def create_initial_compilation_tensors(
        self,
        problem_shape: Tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        alignment: Optional[int] = None,
    ) -> List:
        """
        Create initial tensors needed for CUTLASS kernel compilation.

        Args:
            problem_shape: (M, N, K, L) shape tuple
            device: Target device
            dtype: PyTorch data type
            alignment: Memory alignment

        Returns:
            List of CUTE tensors for kernel compilation
        """
        M, N, K, L = problem_shape
        align = alignment or self.default_alignment

        # Create PyTorch tensors
        tensors = [
            torch.randn(M, K, dtype=dtype, device=device),  # A
            torch.randn(N, K, dtype=dtype, device=device),  # B
            torch.zeros(M, N, dtype=dtype, device=device),  # C
        ]

        # Convert to CUTE tensors
        cute_tensors = []
        for tensor in tensors:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
            cute_tensor = from_dlpack(mnkl_tensor, assumed_align=align)
            cute_tensor.element_type = self.get_cutlass_dtype(dtype)
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)
            cute_tensors.append(cute_tensor)

        return cute_tensors

    def create_tensormap_buffer(
        self,
        device: torch.device,
        sm_count: int,
        tensormap_count: int = 3,
        tensormap_bytes: int = 128,
        alignment: Optional[int] = None,
    ):
        """
        Create tensormap buffer for CUTLASS kernel.

        Args:
            device: Target device
            sm_count: Number of streaming multiprocessors
            tensormap_count: Number of tensormap entries
            tensormap_bytes: Bytes per tensormap entry
            alignment: Memory alignment

        Returns:
            CUTE tensor for tensormap buffer
        """
        align = alignment or self.default_alignment

        tensormap_tensor = torch.zeros(
            (sm_count, tensormap_count, tensormap_bytes // 8),
            dtype=torch.int64,
            device=device,
        )

        return from_dlpack(tensormap_tensor, assumed_align=align)


class GroupedGemmTensorManager:
    """
    High-level manager for grouped GEMM tensor operations.

    Provides simplified interface for common grouped GEMM tensor conversion patterns.
    """

    def __init__(self, alignment: int = 16, dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the tensor manager.

        Args:
            alignment: Memory alignment for CUTE tensors
            dtype: Default PyTorch data type
        """
        self.converter = PyTorchToCuteConverter(default_alignment=alignment)
        self.dtype = dtype
        self.alignment = alignment

    def prepare_expert_operation(
        self,
        input_tokens: torch.Tensor,
        expert_weights: torch.Tensor,
        output_tensor: torch.Tensor,
        transpose_weight: bool = True,
    ) -> Tuple:
        """
        Prepare tensors for a single expert operation (e.g., one GEMM in grouped GEMM).

        Args:
            input_tokens: Input tensor [M, K]
            expert_weights: Weight tensor [N, K] or [K, N]
            output_tensor: Output tensor [M, N]
            transpose_weight: Whether weight needs transposition

        Returns:
            Tuple of (cute_input, cute_weight, cute_output, problem_size, strides, ptrs)
        """
        # Ensure tensors are contiguous
        input_tokens = input_tokens.contiguous()
        expert_weights = expert_weights.contiguous()
        output_tensor = output_tensor.contiguous()

        # Handle weight transposition
        if transpose_weight and len(expert_weights.shape) == 2:
            # For CUTLASS, we often need weights in specific layout
            # This handles the common case where PyTorch weights are [out_features, in_features]
            # but CUTLASS expects [in_features, out_features] or specific stride pattern
            print(
                f"Warning: weight transposition not supported...recommend using strides for this case"
            )
            pass  # Keep original - handle via strides in CUTLASS

        # Convert to CUTE tensors
        cute_input = self.converter.torch_to_cute_tensor(input_tokens)
        cute_weight = self.converter.torch_to_cute_tensor(expert_weights)
        cute_output = self.converter.torch_to_cute_tensor(output_tensor)

        # Prepare metadata
        M, K = input_tokens.shape
        if transpose_weight:
            N = expert_weights.shape[0]  # Weight is [N, K]
        else:
            N = expert_weights.shape[1]  # Weight is [K, N]
        L = 1

        problem_size = [M, N, K, L]

        # Get strides (handle MNKL format)
        input_mnkl = input_tokens.unsqueeze(-1)
        weight_mnkl = expert_weights.unsqueeze(-1)
        output_mnkl = output_tensor.unsqueeze(-1)

        strides = [
            list(input_mnkl.stride()[:2]),
            list(weight_mnkl.stride()[:2]),
            list(output_mnkl.stride()[:2]),
        ]

        ptrs = [
            input_tokens.data_ptr(),
            expert_weights.data_ptr(),
            output_tensor.data_ptr(),
        ]

        return cute_input, cute_weight, cute_output, problem_size, strides, ptrs

    def prepare_grouped_operation(
        self,
        input_list: List[torch.Tensor],
        weight_list: List[torch.Tensor],
        output_list: List[torch.Tensor],
        device: torch.device,
        transpose_weights: bool = True,
    ) -> Tuple:
        """
        Prepare tensors for grouped GEMM operation.

        Args:
            input_list: List of input tensors
            weight_list: List of weight tensors
            output_list: List of output tensors
            device: Target device
            transpose_weights: Whether weights need transposition

        Returns:
            Tuple of (initial_tensors, problem_sizes_cute, strides_cute, ptrs_cute)
        """
        if not (len(input_list) == len(weight_list) == len(output_list)):
            raise ValueError("All lists must have the same length")

        # Collect metadata for all operations
        all_problem_sizes = []
        all_strides = []
        all_ptrs = []

        for inp, weight, out in zip(input_list, weight_list, output_list):
            _, _, _, problem_size, strides, ptrs = self.prepare_expert_operation(
                inp, weight, out, transpose_weights
            )
            all_problem_sizes.append(problem_size)
            all_strides.append(strides)
            all_ptrs.append(ptrs)

        # Create metadata tensors
        problem_sizes_cute, strides_cute, ptrs_cute = (
            self.converter.create_metadata_tensors(
                all_problem_sizes, all_strides, all_ptrs, device
            )
        )

        # Create initial tensors for compilation (use first problem size as template)
        initial_tensors = self.converter.create_initial_compilation_tensors(
            tuple(all_problem_sizes[0]), device, self.dtype
        )

        return initial_tensors, problem_sizes_cute, strides_cute, ptrs_cute


# Convenience functions for common use cases
def pytorch_to_cute_tensor(tensor: torch.Tensor, alignment: int = 16) -> "cute.Tensor":
    """
    Simple conversion function for single PyTorch tensor to CUTE tensor.

    Args:
        tensor: PyTorch tensor to convert
        alignment: Memory alignment requirement

    Returns:
        CUTE tensor
    """
    converter = PyTorchToCuteConverter(alignment)
    return converter.torch_to_cute_tensor(tensor)


def prepare_moe_expert_batch(
    input_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    m_sizes: torch.Tensor,
    m_offsets: torch.Tensor,
    transpose_weights: bool = True,
) -> Tuple:
    """
    Prepare batch of expert operations for MoE grouped GEMM.

    Args:
        input_tokens: All input tokens [total_tokens, in_features]
        expert_weights: Stacked expert weights [num_experts, out_features, in_features]
        m_sizes: Number of tokens per expert [num_experts]
        m_offsets: Token offsets per expert [num_experts + 1]
        transpose_weights: Whether to transpose weights

    Returns:
        Prepared tensors and metadata for grouped GEMM
    """
    manager = GroupedGemmTensorManager()
    device = input_tokens.device

    # Prepare individual expert operations
    input_list = []
    weight_list = []
    output_list = []

    # Convert sizes and offsets to CPU for iteration
    sizes_cpu = m_sizes.cpu().tolist()
    offsets_cpu = m_offsets.cpu().tolist()

    for expert_idx, size in enumerate(sizes_cpu):
        if size > 0:
            offset = offsets_cpu[expert_idx]

            # Get expert data
            expert_input = input_tokens[offset : offset + size].contiguous()
            expert_weight = expert_weights[expert_idx].contiguous()

            # Create output tensor
            M, K = expert_input.shape
            N = expert_weight.shape[0] if transpose_weights else expert_weight.shape[1]
            expert_output = torch.empty(M, N, dtype=input_tokens.dtype, device=device)

            input_list.append(expert_input)
            weight_list.append(expert_weight)
            output_list.append(expert_output)

    return manager.prepare_grouped_operation(
        input_list, weight_list, output_list, device, transpose_weights
    )


# Example usage and testing
def test_converter():
    """Test the PyTorch to CUTE tensor converter."""
    if not HAS_CUTLASS:
        print("❌ CUTLASS not available for testing")
        return False

    print("🧪 Testing PyTorch to CUTE Tensor Converter")
    print("=" * 50)

    try:
        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Create test data
        M, N, K = 128, 256, 512
        input_tensor = torch.randn(M, K, dtype=dtype, device=device)
        weight_tensor = torch.randn(N, K, dtype=dtype, device=device)
        output_tensor = torch.zeros(M, N, dtype=dtype, device=device)

        # Test single tensor conversion
        converter = PyTorchToCuteConverter()
        cute_input = converter.torch_to_cute_tensor(input_tensor)
        print(f"✅ Single tensor conversion successful")
        print(f"   Input shape: {input_tensor.shape} -> CUTE tensor created")

        # Test tensor manager
        manager = GroupedGemmTensorManager()
        cute_inp, cute_weight, cute_out, problem_size, strides, ptrs = (
            manager.prepare_expert_operation(input_tensor, weight_tensor, output_tensor)
        )
        print(f"✅ Expert operation preparation successful")
        print(f"   Problem size: {problem_size}")

        # Test grouped operation preparation
        input_list = [input_tensor, input_tensor[:64]]
        weight_list = [weight_tensor, weight_tensor]
        output_list = [output_tensor, output_tensor[:64]]

        initial_tensors, prob_cute, strides_cute, ptrs_cute = (
            manager.prepare_grouped_operation(
                input_list, weight_list, output_list, device
            )
        )
        print(f"✅ Grouped operation preparation successful")
        print(f"   Number of operations: {len(input_list)}")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_converter()
