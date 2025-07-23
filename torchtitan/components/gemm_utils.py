""" Cutlass GEMM manager class for handling PyTorch-CUTE / Cutlass GEMM operations. """

import math
import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


# CUTLASS imports
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.cute.dense_gemm import (  # CUTLASS GEMM kernel
        DenseGemmKernel,
    )

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    warnings.warn("CUTLASS not available. Only torch.mm backend will be supported.")


def convert_to_mnkl_format(A, B, C):
    """Convert PyTorch tensors to MNKL format for CUTLASS CUTE."""
    # A: (M, K) -> (M, K, 1)
    A_mnkl = A.unsqueeze(-1).contiguous()

    # B: (K, N) -> (N, K, 1)
    B_mnkl = B.transpose(0, 1).unsqueeze(-1).contiguous()

    # C: (M, N) -> (M, N, 1)
    C_mnkl = C.unsqueeze(-1).contiguous()

    return A_mnkl, B_mnkl, C_mnkl


def create_cute_tensors(A_mnkl, B_mnkl, C_mnkl, dtype):
    """Convert PyTorch tensors to CUTE tensors with proper setup."""
    # Convert to CUTE tensors using DLPack
    A_cute = from_dlpack(A_mnkl, assumed_align=16)
    B_cute = from_dlpack(B_mnkl, assumed_align=16)
    C_cute = from_dlpack(C_mnkl, assumed_align=16)

    # Set CUTLASS data types
    A_cute.element_type = dtype
    B_cute.element_type = dtype
    C_cute.element_type = dtype

    # Mark layouts as dynamic with correct leading dimensions
    # For A (M,K,1), the K dimension has stride 1
    A_cute = A_cute.mark_layout_dynamic(leading_dim=1)

    # For B (N,K,1), the K dimension has stride 1
    B_cute = B_cute.mark_layout_dynamic(leading_dim=1)

    # For C (M,N,1), the N dimension has stride 1
    C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

    return A_cute, B_cute, C_cute


class CutlassGemmManager:
    """Manager for CUTLASS GEMM operations with proper tensor conversion and kernel caching."""

    def __init__(self, dtype=torch.float32, device="cuda"):
        if not CUTLASS_AVAILABLE:
            raise RuntimeError(
                "CUTLASS not available but CutlassGemmManager was requested"
            )

        self.dtype = dtype
        self.device = device
        self.stream = torch.cuda.Stream()
        self.cuda_stream = cuda.CUstream(self.stream.cuda_stream)

        # Kernel cache for different configurations - use compiled kernels
        self._compiled_kernel_cache = {}
        self._kernel_cache = {}

        # Use a simple, stable configuration that works reliably
        self.default_config = {
            "acc_dtype": cutlass.Float32,
            "use_2cta_instrs": False,  # Use simpler 1CTA for stability
            "mma_tiler_mn": (64, 64),  # Conservative tile size
            "cluster_shape_mn": (1, 1),  # Single cluster for simplicity
            "use_tma_store": False,  # Disable TMA for stability
        }

    def _get_cutlass_dtype(self, torch_dtype):
        """Convert PyTorch dtype to CUTLASS dtype."""
        dtype_map = {
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
            torch.int8: cutlass.Int8,
            torch.uint8: cutlass.Uint8,
        }
        return dtype_map.get(torch_dtype, cutlass.Float32)

    def _get_or_create_compiled_kernel(self, A_cute, B_cute, C_cute):
        """Get cached compiled kernel or create new one."""
        # Create a cache key based on tensor shapes and types
        key = (
            tuple(A_cute.shape),
            tuple(B_cute.shape),
            tuple(C_cute.shape),
            str(A_cute.element_type),
            str(B_cute.element_type),
            str(C_cute.element_type),
        )

        if key not in self._compiled_kernel_cache:
            # Create a simple kernel that should work for most cases
            kernel = DenseGemmKernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=False,  # Use simpler 1CTA
                mma_tiler_mn=(64, 64),  # Conservative tile size
                cluster_shape_mn=(1, 1),  # Single cluster
                use_tma_store=False,  # Disable TMA for stability
            )

            # Compile the kernel once and cache it
            try:
                compiled_kernel = cute.compile(
                    kernel, A_cute, B_cute, C_cute, self.cuda_stream
                )
                self._compiled_kernel_cache[key] = compiled_kernel
                print(
                    f"Compiled and cached CUTLASS kernel for shapes: A{A_cute.shape}, B{B_cute.shape}, C{C_cute.shape}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compile CUTLASS kernel: {e}")

        return self._compiled_kernel_cache[key]

    def _get_or_create_kernel(self, m, n, k, dtype):
        """Get cached kernel or create new one with default layouts."""
        key = (m, n, k, dtype)
        if key not in self._kernel_cache:
            cutlass_dtype = self._get_cutlass_dtype(dtype)

            # Use a simple, working configuration similar to the working example
            # Start with the most basic configuration that should work
            try:
                kernel = DenseGemmKernel(
                    acc_dtype=cutlass.Float32,
                    use_2cta_instrs=False,
                    mma_tiler_mn=(64, 64),
                    cluster_shape_mn=(1, 1),
                    use_tma_store=False,
                )
                print(f"Created basic CUTLASS kernel for {m}x{k} @ {k}x{n}")
            except Exception as e:
                raise RuntimeError(f"Failed to create basic CUTLASS kernel: {e}")

            # Cache the kernel
            self._kernel_cache[key] = kernel

        return self._kernel_cache[key]

    def gemm(self, A, B, transpose_A=False, transpose_B=False):
        """Perform GEMM using CUTLASS with proper kernel caching."""
        try:
            # Handle transpositions by actually transposing the tensors
            if transpose_A:
                A = A.t().contiguous()
            if transpose_B:
                B = B.t().contiguous()

            # Get dimensions for the GEMM operation
            m, k = A.shape
            k2, n = B.shape
            assert k == k2, f"Inner dimensions must match: {k} != {k2}"

            # Create output tensor
            C = torch.zeros((m, n), dtype=A.dtype, device=A.device)

            # Convert to MNKL format using the working approach
            A_mnkl, B_mnkl, C_mnkl = convert_to_mnkl_format(A, B, C)

            # Create CUTE tensors using the working approach
            cutlass_dtype = self._get_cutlass_dtype(A.dtype)
            A_cute, B_cute, C_cute = create_cute_tensors(
                A_mnkl, B_mnkl, C_mnkl, cutlass_dtype
            )

            # Get or create compiled kernel (this caches the compiled kernel)
            compiled_kernel = self._get_or_create_compiled_kernel(
                A_cute, B_cute, C_cute
            )

            # Execute the kernel
            compiled_kernel(A_cute, B_cute, C_cute, self.cuda_stream)
            torch.cuda.synchronize()

            # Return the result (remove the batch dimension)
            return C_mnkl.squeeze(-1)

        except Exception as e:
            # Fallback to PyTorch implementation if CUTLASS fails
            warnings.warn(
                f"CUTLASS GEMM failed with error: {e}. Falling back to torch.mm"
            )
            if transpose_A and transpose_B:
                return torch.mm(A.t(), B.t())
            elif transpose_A:
                return torch.mm(A.t(), B)
            elif transpose_B:
                return torch.mm(A, B.t())
            else:
                return torch.mm(A, B)
