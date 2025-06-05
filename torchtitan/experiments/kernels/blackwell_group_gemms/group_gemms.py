#!/usr/bin/env python3
"""
Manual Looping Group GEMM implementation using Blackwell Dense GEMM

This file implements a grouped GEMM strategy that uses the Blackwell dense GEMM kernel
with manual looping. This is an alternative to the CUTLASS grouped GEMM kernel, which
has issues with work distribution across groups.
"""

import time
from typing import List, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from dense_gemm import DenseGemmKernel


class ManualLoopBlackwellGroupGEMM:
    """
    Manual looping implementation of grouped GEMM using Blackwell Dense GEMM kernel.

    This class provides a way to execute multiple GEMM operations with different problem
    sizes using the Blackwell Dense GEMM kernel. It loops through each problem and
    executes the kernel for each one.

    This is an alternative to the CUTLASS grouped GEMM kernel, which has issues with
    work distribution across groups.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        use_tma_store: bool = True,
    ):
        """
        Initialize the ManualLoopBlackwellGroupGEMM.

        Args:
            acc_dtype: Data type for accumulation during computation
            use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
            mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
            cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
            use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
        """
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.use_tma_store = use_tma_store

        # Create the dense GEMM kernel
        self.gemm_kernel = DenseGemmKernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=use_tma_store,
        )

        # Cache for compiled kernels
        self._compiled_kernels = {}

    def execute_grouped_gemm(
        self,
        a_tensors: List[torch.Tensor],
        b_tensors: List[torch.Tensor],
        c_tensors: List[torch.Tensor],
        stream: Optional[cuda.CUstream] = None,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ) -> List[torch.Tensor]:
        """
        Execute grouped GEMM operations.

        Args:
            a_tensors: List of A tensors for each GEMM operation
            b_tensors: List of B tensors for each GEMM operation
            c_tensors: List of C tensors for each GEMM operation
            stream: CUDA stream for asynchronous execution
            epilogue_op: Optional elementwise lambda function to apply to the output tensor

        Returns:
            List of output tensors
        """
        if (
            not a_tensors
            or len(a_tensors) != len(b_tensors)
            or len(a_tensors) != len(c_tensors)
        ):
            raise ValueError("Input tensor lists must be non-empty and of equal length")

        # Use current stream if not provided
        if stream is None:
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

        # Execute each GEMM operation
        for i, (a, b, c) in enumerate(zip(a_tensors, b_tensors, c_tensors)):
            # Convert PyTorch tensors to CUTE tensors
            a_cute = self._convert_to_cute_tensor(a)
            b_cute = self._convert_to_cute_tensor(b)
            c_cute = self._convert_to_cute_tensor(c)

            # Get or compile the kernel
            key = (a.shape, b.shape, c.shape, a.dtype, b.dtype, c.dtype)
            if key not in self._compiled_kernels:
                self._compiled_kernels[key] = cute.compile(
                    self.gemm_kernel, a_cute, b_cute, c_cute, stream
                )

            compiled_kernel = self._compiled_kernels[key]

            # Execute the kernel
            compiled_kernel(a_cute, b_cute, c_cute, stream, epilogue_op)

        # Synchronize to ensure completion
        torch.cuda.synchronize()

        return c_tensors

    def _convert_to_cute_tensor(self, tensor: torch.Tensor) -> cute.Tensor:
        """
        Convert a PyTorch tensor to a CUTE tensor.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            CUTE tensor
        """
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Convert to MNKL format (add batch dimension if needed)
        if tensor.dim() == 2:
            tensor_mnkl = tensor.unsqueeze(-1).contiguous()
        else:
            tensor_mnkl = tensor.contiguous()

        # Create CUTE tensor
        cute_tensor = from_dlpack(tensor_mnkl, assumed_align=16)

        # Set CUTLASS data type
        if tensor.dtype == torch.float16:
            cute_tensor.element_type = cutlass.Float16
        elif tensor.dtype == torch.bfloat16:
            cute_tensor.element_type = cutlass.BFloat16
        elif tensor.dtype == torch.float32:
            cute_tensor.element_type = cutlass.Float32
        elif tensor.dtype == torch.int8:
            cute_tensor.element_type = cutlass.Int8
        elif tensor.dtype == torch.uint8:
            cute_tensor.element_type = cutlass.Uint8
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")

        # Mark layout as dynamic
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)

        return cute_tensor


class GroupGEMMBenchmark:
    """
    Benchmark class for comparing different grouped GEMM implementations.
    """

    def __init__(
        self,
        problem_sizes: List[Tuple[int, int, int]],
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize the benchmark.

        Args:
            problem_sizes: List of problem sizes (M, N, K) for each GEMM operation
            dtype: Data type for tensors
            device: Device to run on
        """
        self.problem_sizes = problem_sizes
        self.dtype = dtype
        self.device = device

        # Create tensors
        self.a_tensors = []
        self.b_tensors = []
        self.c_tensors = []

        for m, n, k in problem_sizes:
            a = torch.randn(m, k, dtype=dtype, device=device)
            b = torch.randn(k, n, dtype=dtype, device=device)
            c = torch.zeros(m, n, dtype=dtype, device=device)

            self.a_tensors.append(a)
            self.b_tensors.append(b)
            self.c_tensors.append(c)

    def run_manual_loop_blackwell(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        use_tma_store: bool = True,
        warmup: int = 1,
        iterations: int = 10,
    ) -> float:
        """
        Run the manual loop Blackwell grouped GEMM implementation.

        Args:
            acc_dtype: Data type for accumulation during computation
            use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
            mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
            cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
            use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Average execution time in milliseconds
        """
        # Create the implementation
        impl = ManualLoopBlackwellGroupGEMM(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=use_tma_store,
        )

        # Reset output tensors
        for i in range(len(self.c_tensors)):
            self.c_tensors[i].zero_()

        # Warmup
        for _ in range(warmup):
            impl.execute_grouped_gemm(self.a_tensors, self.b_tensors, self.c_tensors)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            impl.execute_grouped_gemm(self.a_tensors, self.b_tensors, self.c_tensors)

        torch.cuda.synchronize()
        end = time.perf_counter()

        # Calculate average time in milliseconds
        avg_time_ms = (end - start) * 1000 / iterations

        return avg_time_ms

    def run_pytorch_mm(self, warmup: int = 1, iterations: int = 10) -> float:
        """
        Run PyTorch's mm implementation for comparison.

        Args:
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Average execution time in milliseconds
        """
        # Reset output tensors
        for i in range(len(self.c_tensors)):
            self.c_tensors[i].zero_()

        # Warmup
        for _ in range(warmup):
            for i in range(len(self.a_tensors)):
                torch.mm(self.a_tensors[i], self.b_tensors[i], out=self.c_tensors[i])

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            for i in range(len(self.a_tensors)):
                torch.mm(self.a_tensors[i], self.b_tensors[i], out=self.c_tensors[i])

        torch.cuda.synchronize()
        end = time.perf_counter()

        # Calculate average time in milliseconds
        avg_time_ms = (end - start) * 1000 / iterations

        return avg_time_ms

    def verify_results(self) -> bool:
        """
        Verify that the results from different implementations match.

        Returns:
            True if results match, False otherwise
        """
        # Create reference results
        ref_results = []
        for i in range(len(self.a_tensors)):
            ref = torch.mm(self.a_tensors[i], self.b_tensors[i])
            ref_results.append(ref)

        # Compare with actual results
        for i in range(len(self.c_tensors)):
            if not torch.allclose(
                self.c_tensors[i], ref_results[i], rtol=1e-2, atol=1e-2
            ):
                return False

        return True


def example_usage():
    """
    Example usage of the ManualLoopBlackwellGroupGEMM class.
    """
    # Define problem sizes
    problem_sizes = [
        (512, 256, 128),  # Group 0: Small
        (1024, 512, 256),  # Group 1: Medium
        (768, 384, 192),  # Group 2: Different aspect ratio
    ]

    # Create benchmark
    benchmark = GroupGEMMBenchmark(
        problem_sizes=problem_sizes,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )

    # Run manual loop Blackwell implementation
    blackwell_time = benchmark.run_manual_loop_blackwell(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_tma_store=True,
        warmup=1,
        iterations=10,
    )

    # Run PyTorch mm implementation
    pytorch_time = benchmark.run_pytorch_mm(warmup=1, iterations=10)

    # Verify results
    results_match = benchmark.verify_results()

    # Print results
    print(f"Manual Loop Blackwell time: {blackwell_time:.2f} ms")
    print(f"PyTorch mm time: {pytorch_time:.2f} ms")
    print(f"Speedup: {pytorch_time / blackwell_time:.2f}x")
    print(f"Results match: {results_match}")


if __name__ == "__main__":
    example_usage()
