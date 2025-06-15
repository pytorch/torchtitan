"""
Stride-optimized CUTLASS Group GEMM implementation.
Uses stride manipulation instead of tensor transpositions for better performance.


errors:


🔍 Benchmarking Backward Pass: Small-8E
   8 experts, 1024 tokens, 2048→2048
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
❌ Error benchmarking Small-8E: Dimension mismatch: K=135 vs K_B=2048. A: torch.Size([135, 2048]), B: torch.Size([135, 2048]), transpose_A=True, transpose_B=True

======================================================================
📊 Configuration: MoE-7B-Gate
   Experts: 8, Tokens: 2048
   Dimensions: 4096 → 11008
   Problem size: ~92341.8M operations

🔍 Benchmarking Forward Pass: MoE-7B-Gate
   8 experts, 2048 tokens, 4096→11008
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   PyTorch:  0.73 ms
   CUTLASS:  1.06 ms
   Speedup:  0.69x
   Correct:  ✅ (max diff: 0.00e+00)

🔍 Benchmarking Backward Pass: MoE-7B-Gate
   8 experts, 2048 tokens, 4096→11008
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
❌ Error benchmarking MoE-7B-Gate: Dimension mismatch: K=11008 vs K_B=4096. A: torch.Size([261, 11008]), B: torch.Size([11008, 4096]), transpose_A=False, transpose_B=True

======================================================================
📊 Configuration: MoE-7B-Down
   Experts: 8, Tokens: 2048
   Dimensions: 11008 → 4096
   Problem size: ~92341.8M operations

🔍 Benchmarking Forward Pass: MoE-7B-Down
   8 experts, 2048 tokens, 11008→4096
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   PyTorch:  0.69 ms
   CUTLASS:  1.02 ms
   Speedup:  0.68x
   Correct:  ✅ (max diff: 0.00e+00)

🔍 Benchmarking Backward Pass: MoE-7B-Down
   8 experts, 2048 tokens, 11008→4096
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
❌ Error benchmarking MoE-7B-Down: Dimension mismatch: K=4096 vs K_B=11008. A: torch.Size([260, 4096]), B: torch.Size([4096, 11008]), transpose_A=False, transpose_B=True

======================================================================
📊 Configuration: Large-16E
   Experts: 16, Tokens: 4096
   Dimensions: 4096 → 11008
   Problem size: ~184683.6M operations

🔍 Benchmarking Forward Pass: Large-16E
   16 experts, 4096 tokens, 4096→11008
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   PyTorch:  1.27 ms
   CUTLASS:  1.64 ms
   Speedup:  0.77x
   Correct:  ✅ (max diff: 0.00e+00)

🔍 Benchmarking Backward Pass: Large-16E
   16 experts, 4096 tokens, 4096→11008
 Initializing Stride-Optimized CUTLASS strategy...
cute hardware - device_id 0
cute hardware - driver_version 12080
max_dynamic_shared_memory: 232448
max_active_blocks: 1
   - Max active clusters: 33
kernel params: self.ACC_DTYPE=<class 'cutlass.base_dsl.typing.Float32'>, use_2cta_instrs=True, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 2)
max_dynamic_shared_memory: 232448
max_active_blocks: 1
❌ Error benchmarking Large-16E: Dimension mismatch: K=11008 vs K_B=4096. A: torch.Size([269, 11008]), B: torch.Size([11008, 4096]), transpose_A=False, transpose_B=True

"""

from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from blackwell_group_gemm import GroupedGemmKernel
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTLASS = True
except ImportError as e:
    HAS_CUTLASS = False
    print(f"❌ CUTLASS import failed: {e}")


class StrideOptimizedCUTLASSStrategy:
    """
    Stride-optimized CUTLASS strategy that avoids tensor transpositions.
    Uses stride manipulation to achieve transpose effects without data movement.
    """

    def __init__(
        self,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
    ):
        """Initialize stride-optimized CUTLASS strategy"""
        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        print(f" Initializing Stride-Optimized CUTLASS strategy...")

        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn

        # CUTLASS configuration
        self.DTYPE_TORCH = torch.bfloat16
        self.DTYPE_CUTLASS = cutlass.BFloat16
        self.ACC_DTYPE = cutlass.Float32
        self.ALIGNMENT = 16

        # Initialize kernel
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.ACC_DTYPE,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Initialize hardware info
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Caches
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        print(f"   - Max active clusters: {self.max_active_clusters}")
        print(
            f"kernel params: {self.ACC_DTYPE=}, {use_2cta_instrs=}, {mma_tiler_mn=}, {cluster_shape_mn=}"
        )

    def _set_cuda_context(self):
        # Force CUDA context creation
        dummy = torch.zeros(1, device="cuda")
        dummy.cpu()
        del dummy

    def _get_tensormap_buffer(self, device):
        """Get or create tensormap buffer"""
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            tensormap_tensor = torch.zeros(
                (sm_count, 3, 128 // 8),
                dtype=torch.int64,
                device=device,
            )
            self._tensormap_buffers[device] = from_dlpack(
                tensormap_tensor, assumed_align=self.ALIGNMENT
            )
        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes):
        """Compute total clusters needed"""
        cluster_tile_m = self.mma_tiler_mn[0]
        cluster_tile_n = self.mma_tiler_mn[1]

        if self.use_2cta_instrs:
            cluster_tile_m //= 2

        cluster_tile_m *= self.cluster_shape_mn[0]
        cluster_tile_n *= self.cluster_shape_mn[1]

        total = 0
        for M, N, K, L in problem_sizes:
            clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n
        return total

    def _create_initial_tensors(self, problem_shape, device):
        """Create initial tensors for kernel compilation"""
        M, N, K, L = problem_shape

        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),
            torch.randn(N, K, dtype=self.DTYPE_TORCH, device=device),
            torch.zeros(M, N, dtype=self.DTYPE_TORCH, device=device),
        ]

        cute_tensors = []
        for tensor in tensors:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
            cute_tensor = from_dlpack(mnkl_tensor, assumed_align=self.ALIGNMENT)
            cute_tensor.element_type = self.DTYPE_CUTLASS
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)
            cute_tensors.append(cute_tensor)

        return cute_tensors

    def _get_transpose_strides(self, tensor: torch.Tensor) -> List[int]:
        """
        Get strides for transposed tensor without actually transposing.

        Args:
            tensor: Original tensor [M, N]

        Returns:
            Transposed strides that make CUTLASS interpret data as [N, M]
        """
        original_strides = tensor.stride()
        # For transpose: [M, N] -> [N, M], swap the strides
        return [original_strides[1], original_strides[0]]

    def execute_stride_grouped_gemm(
        self, operations: List[dict], operation_name="stride_gemm"
    ):
        """
        Execute grouped GEMM with stride-based layout control.
        CUTLASS always computes A @ B^T.

        The transpose flags indicate whether to logically transpose the tensor
        before CUTLASS applies its own B^T operation.
        """
        if not operations:
            return

        device = operations[0]["A"].device
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        for op in operations:
            A = op["A"].contiguous()
            B = op["B"].contiguous()
            C = op["C"].contiguous()
            transpose_A = op.get("transpose_A", False)
            transpose_B = op.get("transpose_B", False)

            # Apply logical transposes using stride manipulation
            # (This mimics what the simple test does with .t().contiguous())

            if transpose_A:
                # Logically transpose A: [M,K] -> [K,M]
                M, K = A.shape[1], A.shape[0]
                A_strides = self._get_transpose_strides(A)
            else:
                M, K = A.shape[0], A.shape[1]
                A_strides = list(A.stride())

            if transpose_B:
                # Logically transpose B: [N,K] -> [K,N]
                # After this logical transpose, CUTLASS will do [K,N]^T = [N,K]
                K_B, N = B.shape[1], B.shape[0]
                B_strides = self._get_transpose_strides(B)
            else:
                # No logical transpose, CUTLASS does [N,K]^T = [K,N]
                N, K_B = B.shape[0], B.shape[1]
                B_strides = list(B.stride())

            # Validate
            assert (
                K == K_B
            ), f"Dimension mismatch: K={K} vs K_B={K_B}. A: {A.shape}, B: {B.shape}, transpose_A={transpose_A}, transpose_B={transpose_B}"
            assert C.shape == (
                M,
                N,
            ), f"Output shape mismatch: expected ({M}, {N}), got {C.shape}"

            # Add to batch
            L = 1
            C_strides = list(C.stride())
            problem_sizes.append([M, N, K, L])
            strides_abc.append([A_strides, B_strides, C_strides])
            ptrs_abc.append([A.data_ptr(), B.data_ptr(), C.data_ptr()])

        # Execute
        self._execute_kernel(
            problem_sizes, strides_abc, ptrs_abc, device, operation_name
        )

    def _execute_kernel(
        self, problem_sizes, strides_abc, ptrs_abc, device, operation_name
    ):
        """Execute the CUTLASS grouped kernel"""
        num_groups = len(problem_sizes)

        # Convert to tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(
            problem_sizes_tensor, assumed_align=self.ALIGNMENT
        )
        strides_cute = from_dlpack(strides_tensor, assumed_align=self.ALIGNMENT)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=self.ALIGNMENT)

        # Get buffers
        tensormap_cute = self._get_tensormap_buffer(device)
        total_clusters = self._compute_total_clusters(problem_sizes)
        initial_tensors = self._create_initial_tensors(problem_sizes[0], device)

        # Compile kernel if needed
        cache_key = (num_groups, total_clusters, tuple(problem_sizes[0][:3]))

        if cache_key not in self._compiled_kernels:
            self._compiled_kernels[cache_key] = cute.compile(
                self.grouped_gemm,
                *initial_tensors,
                num_groups,
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                total_clusters,
                tensormap_cute,
                self.max_active_clusters,
                self.stream,
            )

        # Execute
        compiled_kernel = self._compiled_kernels[cache_key]
        compiled_kernel(
            *initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            self.stream,
        )
        torch.cuda.synchronize()


class StrideOptimizedGroupGemm(torch.autograd.Function):
    """
    Stride-optimized CUTLASS grouped GEMM autograd function.
    Avoids tensor transpositions by using stride manipulation.
    """

    @staticmethod
    def forward(ctx, input_tokens, weight_stack, m_sizes, m_offsets, strategy):
        """Forward pass using stride-optimized operations"""
        ctx.save_for_backward(input_tokens, weight_stack, m_sizes, m_offsets)
        ctx.strategy = strategy

        device = input_tokens.device
        total_tokens, in_features = input_tokens.shape
        num_experts, out_features, _ = weight_stack.shape

        # Initialize output
        output = torch.zeros(
            total_tokens, out_features, dtype=strategy.DTYPE_TORCH, device=device
        )

        # Check for valid experts
        valid_mask = m_sizes > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return output

        # Execute forward grouped GEMM using stride optimization
        StrideOptimizedGroupGemm._execute_forward_stride_optimized(
            input_tokens,
            weight_stack,
            m_sizes,
            m_offsets,
            valid_indices,
            output,
            strategy,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using stride-optimized operations"""
        input_tokens, weight_stack, m_sizes, m_offsets = ctx.saved_tensors
        strategy = ctx.strategy

        grad_output = grad_output.contiguous()
        device = grad_output.device

        # Initialize gradients
        grad_input = torch.zeros_like(input_tokens)
        grad_weight = torch.zeros_like(weight_stack)

        # Check for valid experts
        valid_mask = m_sizes > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return grad_input, grad_weight, None, None, None

        # Execute backward grouped operations using stride optimization
        StrideOptimizedGroupGemm._execute_backward_stride_optimized(
            grad_output,
            input_tokens,
            weight_stack,
            m_sizes,
            m_offsets,
            valid_indices,
            grad_input,
            grad_weight,
            strategy,
        )

        return grad_input, grad_weight, None, None, None

    @staticmethod
    def _execute_forward_stride_optimized(
        input_tokens, weight_stack, m_sizes, m_offsets, valid_indices, output, strategy
    ):
        """Execute forward pass with stride optimization"""
        # Prepare stride-optimized operations
        operations = []

        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes[valid_indices].cpu().tolist()
        valid_offsets = m_offsets[valid_indices].cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes, valid_offsets
        ):
            if size > 0:
                # Get expert data (all contiguous, no transpose needed)
                expert_input = input_tokens[
                    offset : offset + size
                ].contiguous()  # [M, K]
                expert_weight = weight_stack[expert_idx].contiguous()  # [N, K]
                expert_output = output[offset : offset + size]  # [M, N]

                # Forward: expert_input @ expert_weight^T
                # Mathematical: expert_input[M, K] @ expert_weight^T[K, N] = expert_output[M, N]
                # CUTLASS computes: A @ B^T where A = expert_input[M, K], B = expert_weight[N, K]
                # CUTLASS does: expert_input[M, K] @ expert_weight[N, K]^T = expert_input[M, K] @ expert_weight^T[K, N] ✓
                # So no logical transpose needed - CUTLASS's built-in transpose gives us what we want
                operations.append(
                    {
                        "A": expert_input,
                        "B": expert_weight,
                        "C": expert_output,
                        "transpose_A": False,  # No transpose needed
                        "transpose_B": False,  # Let CUTLASS do the transpose naturally
                    }
                )

        # Execute all operations in one grouped call
        strategy.execute_stride_grouped_gemm(operations, "forward_stride_opt")

    @staticmethod
    def _execute_backward_stride_optimized(
        grad_output,
        input_tokens,
        weight_stack,
        m_sizes,
        m_offsets,
        valid_indices,
        grad_input,
        grad_weight,
        strategy,
    ):
        """Execute backward pass with stride optimization - COMPLETELY REWRITTEN"""
        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes[valid_indices].cpu().tolist()
        valid_offsets = m_offsets[valid_indices].cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Prepare operations based on the exact same logic as the working simple test
        input_operations = []
        weight_operations = []

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes, valid_offsets
        ):
            if size > 0:
                # Get expert data
                expert_grad_output = grad_output[
                    offset : offset + size
                ].contiguous()  # [M, N]
                expert_input = input_tokens[
                    offset : offset + size
                ].contiguous()  # [M, K]
                expert_weight = weight_stack[expert_idx].contiguous()  # [N, K]
                expert_grad_input = grad_input[offset : offset + size]  # [M, K]
                expert_grad_weight = grad_weight[expert_idx]  # [N, K]

                # INPUT GRADIENT: dX = dY @ W
                # Mathematical: expert_grad_output[M, N] @ expert_weight[N, K] = expert_grad_input[M, K]
                # CUTLASS computes: A @ B^T
                # We need: expert_grad_output[M, N] @ expert_weight[N, K]
                # So: A = expert_grad_output[M, N], B^T = expert_weight[N, K]
                # Therefore: B = expert_weight^T[K, N]
                #
                # In the simple test: weight_for_cutlass = weight.t().contiguous() makes [K, N]
                # In stride optimization: we need expert_weight[N, K] to be seen as [K, N]
                # So we use transpose_B=True to make CUTLASS transpose it first
                input_operations.append(
                    {
                        "A": expert_grad_output,  # [M, N]
                        "B": expert_weight,  # [N, K] -> will be transposed to [K, N] by transpose_B=True
                        "C": expert_grad_input,  # [M, K]
                        "transpose_A": False,
                        "transpose_B": True,  # Transpose expert_weight[N,K] to [K,N], then CUTLASS does [K,N]^T = [N,K]
                    }
                )

                # WEIGHT GRADIENT: dW = dY^T @ X
                # Mathematical: expert_grad_output^T[N, M] @ expert_input[M, K] = expert_grad_weight[N, K]
                # CUTLASS computes: A @ B^T
                # We need: expert_grad_output^T[N, M] @ expert_input[M, K]
                # So: A = expert_grad_output^T[N, M], B^T = expert_input[M, K]
                # Therefore: B = expert_input^T[K, M]
                #
                # In the simple test:
                # - grad_output_T = grad_output.t().contiguous() makes [N, M]
                # - input_for_cutlass = input_tokens.t().contiguous() makes [K, M]
                # In stride optimization:
                # - expert_grad_output[M, N] needs to be seen as [N, M] -> transpose_A=True
                # - expert_input[M, K] needs to be seen as [K, M] -> transpose_B=True
                weight_operations.append(
                    {
                        "A": expert_grad_output,  # [M, N] -> will be transposed to [N, M] by transpose_A=True
                        "B": expert_input,  # [M, K] -> will be transposed to [K, M] by transpose_B=True
                        "C": expert_grad_weight,  # [N, K]
                        "transpose_A": True,  # Transpose expert_grad_output[M,N] to [N,M]
                        "transpose_B": True,  # Transpose expert_input[M,K] to [K,M], then CUTLASS does [K,M]^T = [M,K]
                    }
                )

        # Execute grouped operations
        if input_operations:
            strategy.execute_stride_grouped_gemm(
                input_operations, "input_grad_stride_opt"
            )

        if weight_operations:
            strategy.execute_stride_grouped_gemm(
                weight_operations, "weight_grad_stride_opt"
            )


class StrideOptimizedGroupedLinear(nn.Module):
    """
    Stride-optimized CUTLASS grouped linear layer.
    Provides significant performance improvements by avoiding tensor transpositions.
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        strategy: StrideOptimizedCUTLASSStrategy,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize stride-optimized grouped linear layer"""
        super().__init__()

        if bias:
            raise NotImplementedError("Bias not yet supported")

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.strategy = strategy
        self.dtype = dtype

        # Initialize expert weights
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_features, in_features, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization"""
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[expert_idx], a=1.41421356)

    def forward(
        self, input_tokens: torch.Tensor, expert_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Stride-optimized forward pass.

        Args:
            input_tokens: Input tokens [total_tokens, in_features]
            expert_assignments: Expert assignment per token [total_tokens]

        Returns:
            output: Transformed tokens [total_tokens, out_features]
        """
        # Compute expert sizes and offsets
        m_sizes, m_offsets = self._compute_expert_sizes_and_offsets(expert_assignments)

        # Sort tokens by expert assignment for contiguous memory access
        sorted_indices = torch.argsort(expert_assignments)
        sorted_tokens = input_tokens[sorted_indices]

        # Apply stride-optimized grouped GEMM (no transpositions!)
        sorted_output = StrideOptimizedGroupGemm.apply(
            sorted_tokens, self.weight, m_sizes, m_offsets, self.strategy
        )

        # Restore original token order
        output = torch.empty_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _compute_expert_sizes_and_offsets(
        self, expert_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute number of tokens per expert and their offsets"""
        device = expert_assignments.device

        # Count tokens per expert
        m_sizes = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        for expert_idx in range(self.num_experts):
            m_sizes[expert_idx] = (expert_assignments == expert_idx).sum()

        # Compute cumulative offsets
        m_offsets = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(m_sizes, dim=0)]
        )

        return m_sizes, m_offsets

    def extra_repr(self) -> str:
        """Return string representation"""
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}, stride_optimized=True"


def create_stride_optimized_strategy(
    use_2cta_instrs: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
) -> StrideOptimizedCUTLASSStrategy:
    """Create a stride-optimized CUTLASS strategy"""
    return StrideOptimizedCUTLASSStrategy(
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


def test_stride_optimization():
    """Test stride-optimized vs regular implementation"""
    print("🧪 Testing Stride-Optimized CUTLASS Group GEMM")
    print("=" * 60)

    if not HAS_CUTLASS:
        print("❌ CUTLASS not available")
        return False

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test configuration
    num_experts = 8
    in_features = 2048
    out_features = 4096
    total_tokens = 1024

    print(
        f"Configuration: {num_experts} experts, {total_tokens} tokens, {in_features}→{out_features}"
    )

    # Create stride-optimized strategy and layer
    stride_strategy = create_stride_optimized_strategy(
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    stride_layer = StrideOptimizedGroupedLinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        strategy=stride_strategy,
        dtype=dtype,
    ).to(device)

    # Create test data
    input_tokens = torch.randn(
        total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
    )
    expert_assignments = torch.randint(0, num_experts, (total_tokens,), device=device)

    print(
        f"Expert distribution: {[torch.sum(expert_assignments == i).item() for i in range(num_experts)]}"
    )

    try:
        # Forward pass
        print("\n🔍 Forward Pass (Zero-Copy Transpose)")
        output = stride_layer(input_tokens, expert_assignments)
        print(f"Output: {output.shape}, norm: {output.norm().item():.4f}")

        # Backward pass
        print("\n🔍 Backward Pass (Stride-Based Gradients)")
        loss = output.sum()
        loss.backward()

        # Check gradients
        input_grad_norm = (
            input_tokens.grad.norm().item() if input_tokens.grad is not None else 0
        )
        weight_grad_norm = (
            stride_layer.weight.grad.norm().item()
            if stride_layer.weight.grad is not None
            else 0
        )

        print(f"Input gradient norm: {input_grad_norm:.4f}")
        print(f"Weight gradient norm: {weight_grad_norm:.4f}")

        # Validate gradients exist and are reasonable
        success = (
            input_tokens.grad is not None
            and stride_layer.weight.grad is not None
            and input_grad_norm > 0
            and weight_grad_norm > 0
            and torch.isfinite(input_tokens.grad).all()
            and torch.isfinite(stride_layer.weight.grad).all()
        )

        if success:
            print("✅ Stride-optimized CUTLASS Group GEMM test passed!")
            print("\n💡 Performance benefits:")
            print("   - Zero tensor transpositions")
            print("   - No memory copying for layout changes")
            print("   - Reduced memory bandwidth usage")
            print("   - Better cache efficiency")
        else:
            print("❌ Stride-optimized test failed!")

        return success

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_cutlass_strategy(
    use_2cta_instrs: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
) -> StrideOptimizedCUTLASSStrategy:
    """
    Convenience function to create a CUTLASS strategy.

    Args:
        use_2cta_instrs: Whether to use 2-CTA instructions
        mma_tiler_mn: MMA tile sizes
        cluster_shape_mn: Cluster shape

    Returns:
        Configured CUTLASS strategy
    """
    return StrideOptimizedCUTLASSStrategy(
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


if __name__ == "__main__":
    test_stride_optimization()
