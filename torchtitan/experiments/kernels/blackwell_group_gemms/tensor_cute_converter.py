#!/usr/bin/env python3
"""
PyTorch to CUTLASS CUTE Tensor Converter

A utility class for converting between PyTorch tensors and CUTLASS CUTE tensors
with support for GEMM operations and grouped GEMM workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# CUTLASS imports with error handling
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
except ImportError as e:
    HAS_CUTLASS = False
    cutlass = None
    cute = None
    from_dlpack = None


class TensorConversionError(Exception):
    """Custom exception for tensor conversion errors."""

    pass


class PyTorchCuteConverter:
    """
    Utility class for converting between PyTorch tensors and CUTLASS CUTE tensors.

    Handles the conversion process including:
    - Format transformation (2D GEMM -> 3D MNKL)
    - Memory layout management
    - Data type mapping
    - Stride extraction
    - Dynamic layout marking
    """

    # Data type mappings
    DTYPE_MAPPINGS = {
        torch.float16: cutlass.Float16 if HAS_CUTLASS else None,
        torch.float32: cutlass.Float32 if HAS_CUTLASS else None,
        torch.bfloat16: cutlass.BFloat16 if HAS_CUTLASS else None,
        torch.int8: cutlass.Int8 if HAS_CUTLASS else None,
        torch.int32: cutlass.Int32 if HAS_CUTLASS else None,
    }

    def __init__(self, device: Optional[torch.device] = None, alignment: int = 16):
        """
        Initialize the converter.

        Args:
            device: CUDA device to use (defaults to current device)
            alignment: Memory alignment assumption for CUTE tensors
        """
        if not HAS_CUTLASS:
            raise ImportError("CUTLASS is not available. Please install CUTLASS.")

        self.device = device or torch.device("cuda")
        self.alignment = alignment

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    def _get_cutlass_dtype(self, torch_dtype: torch.dtype):
        """Convert PyTorch dtype to CUTLASS dtype."""
        if torch_dtype not in self.DTYPE_MAPPINGS:
            raise TensorConversionError(f"Unsupported dtype: {torch_dtype}")
        return self.DTYPE_MAPPINGS[torch_dtype]

    def _validate_tensor(self, tensor: torch.Tensor, expected_dims: int = 2):
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            raise TensorConversionError("Input must be a PyTorch tensor")

        if not tensor.is_cuda:
            raise TensorConversionError("Tensor must be on CUDA device")

        if tensor.dim() != expected_dims:
            raise TensorConversionError(
                f"Expected {expected_dims}D tensor, got {tensor.dim()}D"
            )

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor


class GemmTensorConverter(PyTorchCuteConverter):
    """Specialized converter for GEMM operations (A @ B = C)."""

    def pytorch_to_cute_gemm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        transpose_B: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert PyTorch GEMM tensors (A @ B = C) to CUTE format.

        Args:
            A: Input tensor A of shape (M, K)
            B: Input tensor B of shape (K, N)
            C: Output tensor C of shape (M, N)

        Returns:
            Dictionary containing CUTE tensors, strides, and metadata
        """
        # Validate inputs
        A = self._validate_tensor(A, 2)
        B = self._validate_tensor(B, 2)
        C = self._validate_tensor(C, 2)

        M, K1 = A.shape

        if transpose_B:
            K2, N = B.shape
        else:
            N, K2 = B.shape

        M2, N2 = C.shape

        if K1 != K2:
            raise TensorConversionError(
                f"Matrix dimension mismatch: A.shape[1]={K1} != B.shape[0]={K2}"
            )
        if M != M2 or N != N2:
            raise TensorConversionError(
                f"Output tensor shape mismatch: expected ({M}, {N}), got ({M2}, {N2})"
            )

        # Convert to MNKL format
        A_mnkl = A.unsqueeze(-1).contiguous()  # (M, K) -> (M, K, 1)
        if transpose_B:
            B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()  # (K, N) -> (N, K, 1)
        else:
            B_mnkl = B.unsqueeze(-1).contiguous()  # (K, N) -> (1, K, N)
        C_mnkl = C.unsqueeze(-1).contiguous()  # (M, N) -> (M, N, 1)

        # Create CUTE tensors
        A_cute = from_dlpack(A_mnkl, assumed_align=self.alignment)
        B_cute = from_dlpack(B_mnkl, assumed_align=self.alignment)
        C_cute = from_dlpack(C_mnkl, assumed_align=self.alignment)

        # Set data types
        cutlass_dtype = self._get_cutlass_dtype(A.dtype)
        A_cute.element_type = cutlass_dtype
        B_cute.element_type = cutlass_dtype
        C_cute.element_type = cutlass_dtype

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        # Extract metadata
        A_strides = A_mnkl.stride()[:2]
        B_strides = B_mnkl.stride()[:2]
        C_strides = C_mnkl.stride()[:2]

        return {
            "cute_tensors": {"A": A_cute, "B": B_cute, "C": C_cute},
            "mnkl_tensors": {"A": A_mnkl, "B": B_mnkl, "C": C_mnkl},
            "original_tensors": {"A": A, "B": B, "C": C},
            "strides": {"A": A_strides, "B": B_strides, "C": C_strides},
            "pointers": {
                "A": A_mnkl.data_ptr(),
                "B": B_mnkl.data_ptr(),
                "C": C_mnkl.data_ptr(),
            },
            "problem_size": (M, N, K1, 1),  # (M, N, K, L) format
        }

    def cute_to_pytorch_gemm(
        self, cute_result: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract PyTorch tensors from CUTE conversion result.

        Args:
            cute_result: Result from pytorch_to_cute_gemm()

        Returns:
            Tuple of (A, B, C) PyTorch tensors
        """
        original = cute_result["original_tensors"]
        return original["A"], original["B"], original["C"]

    def create_gemm_tensors(
        self,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.float16,
        fill_random: bool = True,
    ) -> Dict[str, Any]:
        """
        Create and convert GEMM tensors in one step.

        Args:
            M, N, K: GEMM dimensions
            dtype: Tensor data type
            fill_random: Whether to fill with random values

        Returns:
            CUTE conversion result dictionary
        """
        if fill_random:
            A = torch.randn(M, K, dtype=dtype, device=self.device)
            B = torch.randn(K, N, dtype=dtype, device=self.device)
        else:
            A = torch.zeros(M, K, dtype=dtype, device=self.device)
            B = torch.zeros(K, N, dtype=dtype, device=self.device)

        C = torch.zeros(M, N, dtype=dtype, device=self.device)

        return self.pytorch_to_cute_gemm(A, B, C)


class GroupedGemmConverter(PyTorchCuteConverter):
    """Specialized converter for grouped GEMM operations."""

    def pytorch_to_cute_grouped(
        self, tensor_groups: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Convert multiple GEMM problems to grouped CUTE format.

        Args:
            tensor_groups: List of (A, B, C) tensor tuples

        Returns:
            Dictionary containing grouped CUTE tensors and metadata
        """
        if not tensor_groups:
            raise TensorConversionError("Empty tensor group list")

        cute_tensors = []
        mnkl_tensors = []
        original_tensors = []
        strides = []
        pointers = []
        problem_sizes = []

        gemm_converter = GemmTensorConverter(self.device, self.alignment)

        # Convert each group
        for i, (A, B, C) in enumerate(tensor_groups):
            try:
                result = gemm_converter.pytorch_to_cute_gemm(A, B, C)

                cute_tensors.append(result["cute_tensors"])
                mnkl_tensors.append(result["mnkl_tensors"])
                original_tensors.append(result["original_tensors"])
                strides.append(
                    [
                        result["strides"]["A"],
                        result["strides"]["B"],
                        result["strides"]["C"],
                    ]
                )
                pointers.append(
                    [
                        result["pointers"]["A"],
                        result["pointers"]["B"],
                        result["pointers"]["C"],
                    ]
                )
                problem_sizes.append(result["problem_size"])

            except Exception as e:
                raise TensorConversionError(f"Failed to convert group {i}: {e}")

        # Create metadata tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=self.device
        )
        strides_tensor = torch.tensor(strides, dtype=torch.int32, device=self.device)
        pointers_tensor = torch.tensor(pointers, dtype=torch.int64, device=self.device)

        # Create CUTE metadata tensors
        problem_sizes_cute = from_dlpack(
            problem_sizes_tensor, assumed_align=self.alignment
        )
        strides_cute = from_dlpack(strides_tensor, assumed_align=self.alignment)
        pointers_cute = from_dlpack(pointers_tensor, assumed_align=self.alignment)

        return {
            "cute_tensors": cute_tensors,
            "mnkl_tensors": mnkl_tensors,
            "original_tensors": original_tensors,
            "metadata": {
                "problem_sizes": problem_sizes_cute,
                "strides": strides_cute,
                "pointers": pointers_cute,
            },
            "metadata_tensors": {
                "problem_sizes": problem_sizes_tensor,
                "strides": strides_tensor,
                "pointers": pointers_tensor,
            },
            "num_groups": len(tensor_groups),
            "problem_sizes_list": problem_sizes,
        }

    def create_grouped_tensors(
        self,
        problem_sizes: List[Tuple[int, int, int]],
        dtype: torch.dtype = torch.float16,
        fill_random: bool = True,
    ) -> Dict[str, Any]:
        """
        Create grouped GEMM tensors from problem size specifications.

        Args:
            problem_sizes: List of (M, N, K) tuples
            dtype: Tensor data type
            fill_random: Whether to fill with random values

        Returns:
            Grouped CUTE conversion result
        """
        tensor_groups = []

        for M, N, K in problem_sizes:
            if fill_random:
                A = torch.randn(M, K, dtype=dtype, device=self.device)
                B = torch.randn(K, N, dtype=dtype, device=self.device)
            else:
                A = torch.zeros(M, K, dtype=dtype, device=self.device)
                B = torch.zeros(K, N, dtype=dtype, device=self.device)

            C = torch.zeros(M, N, dtype=dtype, device=self.device)
            tensor_groups.append((A, B, C))

        return self.pytorch_to_cute_grouped(tensor_groups)


class PersistentGemmConverter(PyTorchCuteConverter):
    """Specialized converter for persistent GEMM operations like the Blackwell persistent dense GEMM."""

    def __init__(self, device: Optional[torch.device] = None, alignment: int = 16):
        """
        Initialize the persistent GEMM converter.

        Args:
            device: CUDA device to use (defaults to current device)
            alignment: Memory alignment assumption for CUTE tensors
        """
        super().__init__(device, alignment)

        # Import hardware info for max active clusters calculation
        try:
            import cutlass.utils as utils

            self.hardware_info = utils.HardwareInfo()
        except ImportError:
            self.hardware_info = None

    def create_and_permute_tensor(
        self,
        l: int,
        mode0: int,
        mode1: int,
        is_mode0_major: bool,
        dtype: torch.dtype,
        cutlass_dtype,
        is_dynamic_layout: bool = True,
        fill_random: bool = True,
    ) -> Dict[str, Any]:
        """
        Create and permute tensor following the persistent GEMM pattern.

        This follows the same logic as the persistent GEMM's create_and_permute_tensor function.

        Args:
            l: Batch dimension
            mode0: First mode dimension
            mode1: Second mode dimension
            is_mode0_major: Whether mode0 is the major (contiguous) dimension
            dtype: PyTorch data type
            cutlass_dtype: CUTLASS data type
            is_dynamic_layout: Whether to mark layout as dynamic
            fill_random: Whether to fill with random values

        Returns:
            Dictionary containing reference tensor, CUTE tensor, and PyTorch tensor
        """
        # Determine tensor shape and permutation order
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)

        # Handle special fp8 types that PyTorch doesn't support natively
        torch_dtype = dtype
        if hasattr(cutlass_dtype, "width") and cutlass_dtype in {
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            torch_dtype = torch.uint8

        # Create tensor on CPU first
        if fill_random:
            is_unsigned = cutlass_dtype in {cutlass.Uint8} if HAS_CUTLASS else False
            min_val = 0 if is_unsigned else -2
            max_val = 4 if is_unsigned else 2

            if torch_dtype == torch.uint8:
                torch_tensor_cpu = torch.randint(
                    min_val, max_val, shape, dtype=torch_dtype
                )
            else:
                torch_tensor_cpu = torch.randn(shape, dtype=torch_dtype) * 2 + 1
                torch_tensor_cpu = torch.clamp(torch_tensor_cpu, min_val, max_val)
        else:
            torch_tensor_cpu = torch.zeros(shape, dtype=torch_dtype)

        # Permute to final layout
        torch_tensor_cpu = torch_tensor_cpu.permute(permute_order).contiguous()

        # Move to GPU
        torch_tensor = torch_tensor_cpu.cuda()

        # Create reference tensor in float32
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create CUTE tensor
        cute_tensor = from_dlpack(torch_tensor, assumed_align=self.alignment)
        cute_tensor.element_type = cutlass_dtype

        if is_dynamic_layout:
            leading_dim = 0 if is_mode0_major else 1
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        return {
            "reference": f32_torch_tensor,
            "cute": cute_tensor,
            "torch": torch_tensor,
            "shape": torch_tensor.shape,
            "permute_order": permute_order,
            "is_mode0_major": is_mode0_major,
        }

    def create_persistent_gemm_tensors(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype,
        c_dtype,
        a_major: str = "k",
        b_major: str = "k",
        c_major: str = "n",
        fill_random: bool = True,
    ) -> Dict[str, Any]:
        """
        Create A, B, C tensors for persistent GEMM following the Blackwell pattern.

        Args:
            m, n, k, l: GEMM dimensions (l is batch dimension)
            ab_dtype: CUTLASS data type for A and B tensors
            c_dtype: CUTLASS data type for C tensor
            a_major: Major dimension for A ("m" or "k")
            b_major: Major dimension for B ("n" or "k")
            c_major: Major dimension for C ("m" or "n")
            fill_random: Whether to fill with random values

        Returns:
            Dictionary containing tensor conversion results for A, B, C
        """
        # Determine PyTorch dtypes
        ab_torch_dtype = torch.float16  # Default fallback
        c_torch_dtype = torch.float32  # Default fallback

        if HAS_CUTLASS:
            try:
                import cutlass.torch as cutlass_torch

                if ab_dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
                    ab_torch_dtype = cutlass_torch.dtype(ab_dtype)
                if c_dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
                    c_torch_dtype = cutlass_torch.dtype(c_dtype)
            except (ImportError, AttributeError):
                pass

        # Create tensor A
        a_result = self.create_and_permute_tensor(
            l,
            m,
            k,
            a_major == "m",
            ab_torch_dtype,
            ab_dtype,
            is_dynamic_layout=True,
            fill_random=fill_random,
        )

        # Create tensor B
        b_result = self.create_and_permute_tensor(
            l,
            n,
            k,
            b_major == "n",
            ab_torch_dtype,
            ab_dtype,
            is_dynamic_layout=True,
            fill_random=fill_random,
        )

        # Create tensor C
        c_result = self.create_and_permute_tensor(
            l,
            m,
            n,
            c_major == "m",
            c_torch_dtype,
            c_dtype,
            is_dynamic_layout=True,
            fill_random=False,  # C is typically zero-initialized
        )

        return {
            "A": a_result,
            "B": b_result,
            "C": c_result,
            "problem_size": (m, n, k, l),
            "major_modes": {"a": a_major, "b": b_major, "c": c_major},
        }

    def get_stream_and_max_clusters(
        self, cluster_shape_mn: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Get CUDA stream and maximum active clusters for persistent GEMM.

        Args:
            cluster_shape_mn: Cluster shape (M, N)

        Returns:
            Dictionary containing stream and cluster information
        """
        # Import CUDA bindings
        try:
            import cuda.bindings.driver as cuda
        except ImportError:
            raise ImportError("CUDA bindings not available")

        # Get current PyTorch stream and convert to CUstream
        torch_stream = torch.cuda.current_stream()
        cuda_stream = cuda.CUstream(torch_stream.cuda_stream)

        # Calculate max active clusters
        max_active_clusters = 1  # Default fallback
        if self.hardware_info is not None:
            cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
            max_active_clusters = self.hardware_info.get_max_active_clusters(
                cluster_size
            )

        return {
            "torch_stream": torch_stream,
            "cuda_stream": cuda_stream,
            "max_active_clusters": max_active_clusters,
            "cluster_size": cluster_shape_mn[0] * cluster_shape_mn[1],
        }

    def compile_persistent_gemm(
        self, gemm_kernel, tensors: Dict[str, Any], cluster_shape_mn: Tuple[int, int]
    ) -> Any:
        """
        Compile persistent GEMM kernel with proper stream and cluster configuration.

        Args:
            gemm_kernel: The persistent GEMM kernel to compile
            tensors: Result from create_persistent_gemm_tensors()
            cluster_shape_mn: Cluster shape (M, N)

        Returns:
            Compiled GEMM kernel
        """
        stream_info = self.get_stream_and_max_clusters(cluster_shape_mn)

        # Extract CUTE tensors
        a_tensor = tensors["A"]["cute"]
        b_tensor = tensors["B"]["cute"]
        c_tensor = tensors["C"]["cute"]

        # Compile the kernel
        compiled_kernel = cute.compile(
            gemm_kernel,
            a_tensor,
            b_tensor,
            c_tensor,
            stream_info["max_active_clusters"],
            stream_info["cuda_stream"],
        )

        return {
            "kernel": compiled_kernel,
            "stream_info": stream_info,
            "tensors": tensors,
        }


class TensorMapConverter(PyTorchCuteConverter):
    """Utility for creating tensor maps needed by CUTLASS kernels."""

    def create_tensormap_buffer(
        self, sm_count: int, tensor_count: int = 3, element_size: int = 8
    ) -> Dict[str, Any]:
        """
        Create a tensor map buffer for CUTLASS kernels.

        Args:
            sm_count: Number of streaming multiprocessors
            tensor_count: Number of tensors (typically 3 for A, B, C)
            element_size: Size of tensormap elements in bytes

        Returns:
            Dictionary with tensormap buffer and CUTE tensor
        """
        buffer_size = 128 // element_size
        tensormap_tensor = torch.zeros(
            (sm_count, tensor_count, buffer_size), dtype=torch.int64, device=self.device
        )

        tensormap_cute = from_dlpack(tensormap_tensor, assumed_align=self.alignment)

        return {
            "buffer": tensormap_tensor,
            "cute": tensormap_cute,
            "shape": tensormap_tensor.shape,
        }


# Example usage and testing functions
def example_single_gemm():
    """Example of converting a single GEMM operation."""
    print("=== Single GEMM Conversion Example ===")

    converter = GemmTensorConverter()

    # Create test tensors
    M, N, K = 512, 256, 128
    result = converter.create_gemm_tensors(M, N, K, dtype=torch.float16)

    print(f"Problem size: {result['problem_size']}")
    print(f"A strides: {result['strides']['A']}")
    print(f"B strides: {result['strides']['B']}")
    print(f"C strides: {result['strides']['C']}")

    # Extract original tensors
    A, B, C = converter.cute_to_pytorch_gemm(result)
    print(f"Original tensor shapes: A{A.shape}, B{B.shape}, C{C.shape}")

    return result


def example_grouped_gemm():
    """Example of converting grouped GEMM operations."""
    print("\n=== Grouped GEMM Conversion Example ===")

    converter = GroupedGemmConverter()

    # Define problem sizes
    problem_sizes = [(256, 256, 128), (512, 128, 256), (128, 512, 256)]

    result = converter.create_grouped_tensors(problem_sizes, dtype=torch.float16)

    print(f"Number of groups: {result['num_groups']}")
    print(f"Problem sizes: {result['problem_sizes_list']}")
    print(f"Metadata tensor shapes:")
    print(f"  Problem sizes: {result['metadata_tensors']['problem_sizes'].shape}")
    print(f"  Strides: {result['metadata_tensors']['strides'].shape}")
    print(f"  Pointers: {result['metadata_tensors']['pointers'].shape}")

    return result


def example_persistent_gemm():
    """Example of converting persistent GEMM operations."""
    print("\n=== Persistent GEMM Conversion Example ===")

    converter = PersistentGemmConverter()

    # Problem dimensions
    M, N, K, L = 8192, 8192, 8192, 1

    # Mock CUTLASS data types (would be real cutlass types in actual usage)
    if HAS_CUTLASS:
        ab_dtype = cutlass.Float16
        c_dtype = cutlass.Float16
    else:
        ab_dtype = torch.float16  # Fallback for demo
        c_dtype = torch.float16  # Fallback for demo

    # Create tensors
    result = converter.create_persistent_gemm_tensors(
        M,
        N,
        K,
        L,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        a_major="k",  # A is K-major (row-major)
        b_major="k",  # B is K-major
        c_major="n",  # C is N-major (column-major)
    )

    print(f"Problem size (M,N,K,L): {result['problem_size']}")
    print(f"Major modes: {result['major_modes']}")
    print(f"Tensor shapes:")
    print(f"  A: {result['A']['shape']}")
    print(f"  B: {result['B']['shape']}")
    print(f"  C: {result['C']['shape']}")

    # Get stream and cluster info
    cluster_shape_mn = (2, 1)
    stream_info = converter.get_stream_and_max_clusters(cluster_shape_mn)
    print(f"Max active clusters: {stream_info['max_active_clusters']}")
    print(f"Cluster size: {stream_info['cluster_size']}")

    return result


if __name__ == "__main__":
    if not HAS_CUTLASS:
        print("CUTLASS not available - running limited examples")

    try:
        # Run examples
        single_result = example_single_gemm()
        grouped_result = example_grouped_gemm()
        persistent_result = example_persistent_gemm()

        print("\n✓ All conversions completed successfully!")

    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        exit(1)
