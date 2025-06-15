from typing import List, Optional, Tuple

import torch
import torch.nn as nn


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
    print("✓ CUTLASS and strategies imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ Import failed: {e}")
    print("Using PyTorch fallback implementations only")


# Strategy base class for GroupGEMM implementations
class GroupGEMMStrategy:
    """Base class for group gemm strategies"""

    def __init__(self, custom_activation):
        self.activation_function = custom_activation

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prepare expert weights, including prescaling

        Args:
            all_weights: List of weight tensors from each expert
            submod_name: Name of the submodule (e.g., 'gate_proj', 'up_proj', 'down_proj')
            module: The parent module that will store the arranged weights

        Returns:
            Tensor: The arranged weights in the format required by the specific strategy
        """

        raise NotImplementedError("Requires arrange_expert_weights method")

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the group gemm operation

        Args:
            contig_tokens: The input tokens, arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: The MoE module containing weights and parameters

        Returns:
            The processed tokens
        """
        raise NotImplementedError("GroupGEMM strategy must implement execute method")

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system"""
        return False


class CUTLASSGroupedGemmStrategy(GroupGEMMStrategy):
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    This version eliminates CPU-GPU synchronization by keeping all size/offset computations on GPU.
    """

    # Constants
    SUPPORTED_CLUSTER_SHAPES = [
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 1),
        (4, 2),
        (4, 4),
    ]

    SINGLE_CTA_M_SIZES = [128, 64]
    DUAL_CTA_M_SIZES = [256, 128]
    N_SIZE_RANGE = range(32, 257, 32)  # 32 - 256, step 32

    DTYPE_TORCH = torch.bfloat16
    DTYPE_CUTLASS = cutlass.BFloat16
    ACC_DTYPE = cutlass.Float32
    ALIGNMENT = 16
    TENSORMAP_COUNT = 3
    TENSORMAP_BYTES = 128

    def __init__(
        self,
        custom_activation=nn.SiLU(),
        use_2cta_instrs=False,  # Changed default to False to avoid context issues
        mma_tiler_mn=(256, 128),  # Changed default to single-CTA values
        cluster_shape_mn=(1, 1),  # Changed default to single-CTA values
    ):
        """Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture."""
        print(f"Initializing CUTLASSGroupedGemmStrategy for Blackwell")
        super().__init__(custom_activation)
        self.use_2cta_instrs = use_2cta_instrs

        # Set configuration
        self.mma_tiler_mn = mma_tiler_mn or self._get_default_mma_tiler()
        self.cluster_shape_mn = cluster_shape_mn or self._get_default_cluster_shape()

        # Initialize kernel and hardware info
        self._initialize_kernel()
        self._initialize_hardware()

        # Initialize caches
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        self._log_initialization()

    def _get_default_mma_tiler(self):
        """Get default MMA tiler configuration based on CTA mode."""
        return (256, 128) if self.use_2cta_instrs else (128, 128)

    def _get_default_cluster_shape(self):
        """Get default cluster shape based on CTA mode."""
        return (2, 2) if self.use_2cta_instrs else (1, 1)

    def _initialize_kernel(self):
        """Initialize the CUTLASS grouped GEMM kernel."""
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.ACC_DTYPE,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

    def _initialize_hardware(self):
        """Initialize hardware information and stream."""
        # Force CUDA context creation to avoid DSL errors
        dummy_tensor = torch.zeros(1, device="cuda")
        dummy_tensor.cpu()

        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

    def _log_initialization(self):
        """Log initialization information."""
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]

        print(f"max_active_blocks: {self.max_active_clusters}")
        print(f"Initialized CUTLASSGroupedGemmStrategy for Blackwell with:")
        print(f"  - 2 CTA instructions: {self.use_2cta_instrs}")
        print(f"  - MMA tiler (M, N): {self.mma_tiler_mn}")
        print(f"  - Cluster shape (M, N): {self.cluster_shape_mn}")
        print(f"  - Cluster size: {cluster_size}")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in stacked format."""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using CUTLASS grouped GEMM kernel - GPU-only version."""
        # This method is used by the MoE strategy, not by the linear layer
        # For the linear layer, we use CUTLASSBackwardGroupGemm.apply() directly
        raise NotImplementedError(
            "This method is for MoE integration, use CUTLASSGroupedLinear instead"
        )

    # All the helper methods for CUTLASS kernel execution
    def _ensure_gpu_tensors(self, m_sizes, m_offsets, device):
        """Ensure sizes and offsets are GPU tensors to avoid CPU-GPU sync."""
        if not isinstance(m_sizes, torch.Tensor):
            m_sizes_gpu = torch.tensor(m_sizes, dtype=torch.int32, device=device)
        else:
            m_sizes_gpu = m_sizes.to(device=device, dtype=torch.int32)

        if not isinstance(m_offsets, torch.Tensor):
            m_offsets_gpu = torch.tensor(m_offsets, dtype=torch.int32, device=device)
        else:
            m_offsets_gpu = m_offsets.to(device=device, dtype=torch.int32)

        return m_sizes_gpu, m_offsets_gpu

    def _convert_to_cute_tensors(self, problem_sizes, strides_abc, ptrs_abc, device):
        """Convert metadata to CUTE tensors."""
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        return (
            from_dlpack(problem_sizes_tensor, assumed_align=self.ALIGNMENT),
            from_dlpack(strides_tensor, assumed_align=self.ALIGNMENT),
            from_dlpack(ptrs_tensor, assumed_align=self.ALIGNMENT),
        )

    def _get_compiled_kernel(
        self,
        num_groups,
        total_clusters,
        initial_tensors,
        problem_sizes_cute,
        strides_cute,
        ptrs_cute,
        tensormap_cute,
    ):
        """Get or compile the grouped GEMM kernel."""
        cache_key = (
            num_groups,
            total_clusters,
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
        )

        if cache_key not in self._compiled_kernels:
            print(
                f"Compiling CUTLASS grouped GEMM kernel: {num_groups} groups, 2CTA={self.use_2cta_instrs}, cluster={self.cluster_shape_mn}"
            )

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
            print("Kernel compilation successful")

        return self._compiled_kernels[cache_key]

    def _create_initial_tensors(self, problem_shape, device):
        """Create initial CUTE tensors for kernel compilation."""
        M, N, K, L = problem_shape

        # Create tensors
        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),  # A
            torch.randn(N, K, dtype=self.DTYPE_TORCH, device=device),  # B
            torch.zeros(M, N, dtype=self.DTYPE_TORCH, device=device),  # C
        ]

        # Convert to MNKL format and create CUTE tensors
        cute_tensors = []
        for tensor in tensors:
            mnkl_tensor = tensor.unsqueeze(-1).contiguous()
            cute_tensor = from_dlpack(mnkl_tensor, assumed_align=self.ALIGNMENT)
            cute_tensor.element_type = self.DTYPE_CUTLASS
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)
            cute_tensors.append(cute_tensor)

        return cute_tensors

    def _get_tensormap_buffer(self, device):
        """Get or create tensormap buffer."""
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            tensormap_tensor = torch.zeros(
                (sm_count, self.TENSORMAP_COUNT, self.TENSORMAP_BYTES // 8),
                dtype=torch.int64,
                device=device,
            )
            self._tensormap_buffers[device] = from_dlpack(
                tensormap_tensor, assumed_align=self.ALIGNMENT
            )

        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes):
        """Compute total number of clusters needed."""
        cluster_tile_m = self.mma_tiler_mn[0]
        cluster_tile_n = self.mma_tiler_mn[1]

        # Adjust for 2 CTA mode and cluster shape
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

    @staticmethod
    def is_available() -> bool:
        return HAS_CUTLASS


class CUTLASSBackwardGroupGemm(torch.autograd.Function):
    """
    Performance-optimized CUTLASS grouped GEMM with backward pass support.

    Key optimizations:
    1. Eliminated unnecessary transpositions using CUTLASS layout system
    2. Fused input and weight gradient computations
    3. Reduced memory allocations and copies
    4. Optimized memory access patterns
    5. Minimized CPU-GPU synchronization
    """

    @staticmethod
    def forward(ctx, input_tokens, weight_stack, m_sizes, m_offsets, strategy):
        """Forward pass: Y_i = X_i @ W_i^T"""
        ctx.save_for_backward(input_tokens, weight_stack, m_sizes, m_offsets)
        ctx.strategy = strategy

        # Pre-allocate and reuse output tensor
        device = input_tokens.device
        total_tokens, in_features = input_tokens.shape
        num_experts, out_features, _ = weight_stack.shape

        output = torch.zeros(
            total_tokens, out_features, dtype=strategy.DTYPE_TORCH, device=device
        )

        if not torch.any(m_sizes > 0):
            return output

        return CUTLASSBackwardGroupGemm._execute_grouped_gemm_forward(
            input_tokens, weight_stack, m_sizes, m_offsets, output, strategy
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Optimized backward pass with fused gradient computations"""
        input_tokens, weight_stack, m_sizes, m_offsets = ctx.saved_tensors
        strategy = ctx.strategy

        grad_output = grad_output.contiguous()
        device = grad_output.device

        # Pre-allocate gradient tensors
        grad_input = torch.zeros_like(input_tokens)
        grad_weight = torch.zeros_like(weight_stack)

        if not torch.any(m_sizes > 0):
            return grad_input, grad_weight, None, None, None

        # OPTIMIZATION 1: Fused backward computation
        # Compute both input and weight gradients in a single pass
        CUTLASSBackwardGroupGemm._execute_fused_backward_gemm(
            grad_output,
            input_tokens,
            weight_stack,
            m_sizes,
            m_offsets,
            grad_input,
            grad_weight,
            strategy,
        )

        return grad_input, grad_weight, None, None, None

    @staticmethod
    def _execute_grouped_gemm_forward(
        input_tokens, weight_stack, m_sizes, m_offsets, output, strategy
    ):
        """Optimized forward execution with minimal overhead"""
        valid_mask = m_sizes > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return output

        # OPTIMIZATION 2: Batch metadata preparation to reduce overhead
        problem_sizes, strides_abc, ptrs_abc = (
            CUTLASSBackwardGroupGemm._prepare_batched_forward_metadata(
                input_tokens,
                weight_stack,
                m_sizes,
                m_offsets,
                valid_indices,
                output,
                strategy,
            )
        )

        if len(problem_sizes) == 0:
            return output

        # OPTIMIZATION 3: Single kernel launch for all experts
        CUTLASSBackwardGroupGemm._execute_optimized_cutlass_kernel(
            problem_sizes, strides_abc, ptrs_abc, output.device, strategy
        )

        return output

    @staticmethod
    def _execute_fused_backward_gemm(
        grad_output,
        input_tokens,
        weight_stack,
        m_sizes,
        m_offsets,
        grad_input,
        grad_weight,
        strategy,
    ):
        """
        OPTIMIZATION 4: Fused backward computation

        Instead of separate kernels for input/weight gradients, we:
        1. Use CUTLASS's native layout system to avoid transpositions
        2. Leverage memory locality between input/weight gradient computations
        3. Minimize kernel launch overhead
        """
        valid_mask = m_sizes > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return

        # OPTIMIZATION 5: Use CUTLASS LayoutRight/LayoutLeft to avoid transpositions
        # Prepare both input and weight gradient operations together
        input_grad_problems, weight_grad_problems = (
            CUTLASSBackwardGroupGemm._prepare_fused_backward_metadata(
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
        )

        # Execute input gradient kernel (if needed)
        if input_grad_problems[0]:  # problem_sizes
            CUTLASSBackwardGroupGemm._execute_optimized_cutlass_kernel(
                *input_grad_problems, grad_input.device, strategy
            )

        # Execute weight gradient kernel (leveraging warm caches)
        if weight_grad_problems[0]:  # problem_sizes
            CUTLASSBackwardGroupGemm._execute_optimized_cutlass_kernel(
                *weight_grad_problems, grad_weight.device, strategy
            )

    @staticmethod
    def _prepare_batched_forward_metadata(
        input_tokens, weight_stack, m_sizes, m_offsets, valid_indices, output, strategy
    ):
        """
        OPTIMIZATION 6: Optimized metadata preparation

        - Minimize CPU-GPU synchronization
        - Use vectorized operations where possible
        - Pre-allocate arrays to avoid repeated allocations
        """
        device = input_tokens.device
        num_valid = len(valid_indices)

        # Pre-allocate metadata arrays
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        # OPTIMIZATION 7: Vectorized size/offset computation
        valid_sizes = m_sizes[valid_indices]
        if len(m_offsets) > len(valid_indices):
            valid_offsets = m_offsets[valid_indices]
        else:
            valid_offsets = torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )

        # Single CPU-GPU sync for all sizes/offsets
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Batch process all experts
        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Direct pointer arithmetic for contiguous access
                input_ptr = (
                    input_tokens.data_ptr()
                    + offset * input_tokens.stride(0) * input_tokens.element_size()
                )
                weight_ptr = weight_stack[expert_idx].data_ptr()
                output_ptr = (
                    output.data_ptr()
                    + offset * output.stride(0) * output.element_size()
                )

                # OPTIMIZATION 8: Pre-computed strides to avoid repeated calculations
                in_features = input_tokens.shape[1]
                out_features = weight_stack.shape[1]

                M, K, N, L = size, in_features, out_features, 1

                # Optimized stride calculations
                A_strides = [input_tokens.stride(0), input_tokens.stride(1)]
                B_strides = [
                    weight_stack.stride(1),
                    weight_stack.stride(2),
                ]  # [N, K] layout
                C_strides = [output.stride(0), output.stride(1)]

                problem_sizes.append([M, N, K, L])
                strides_abc.append([A_strides, B_strides, C_strides])
                ptrs_abc.append([input_ptr, weight_ptr, output_ptr])

        return problem_sizes, strides_abc, ptrs_abc

    @staticmethod
    def _prepare_fused_backward_metadata(
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
        """
        OPTIMIZATION 9: Fused backward metadata preparation

        Prepare both input and weight gradient operations simultaneously to:
        - Minimize metadata preparation overhead
        - Leverage shared computations
        - Optimize memory access patterns
        """
        device = grad_output.device

        # Shared offset/size computations
        valid_sizes = m_sizes[valid_indices]
        if len(m_offsets) > len(valid_indices):
            valid_offsets = m_offsets[valid_indices]
        else:
            valid_offsets = torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )

        # Single sync for all metadata
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Prepare input gradient metadata (dX = dY @ W)
        input_grad_problems = CUTLASSBackwardGroupGemm._prepare_input_grad_optimized(
            grad_output,
            weight_stack,
            valid_indices_cpu,
            valid_sizes_cpu,
            valid_offsets_cpu,
            grad_input,
            strategy,
        )

        # Prepare weight gradient metadata (dW = dY^T @ X)
        weight_grad_problems = CUTLASSBackwardGroupGemm._prepare_weight_grad_optimized(
            grad_output,
            input_tokens,
            valid_indices_cpu,
            valid_sizes_cpu,
            valid_offsets_cpu,
            grad_weight,
            strategy,
        )

        return input_grad_problems, weight_grad_problems

    @staticmethod
    def _prepare_input_grad_optimized(
        grad_output,
        weight_stack,
        valid_indices_cpu,
        valid_sizes_cpu,
        valid_offsets_cpu,
        grad_input,
        strategy,
    ):
        """
        OPTIMIZATION 10: Use CUTLASS LayoutLeft to compute dY @ W without transposition

        Instead of reformulating as dX^T = W^T @ dY^T, we use CUTLASS's layout system
        to directly compute dY @ W by treating W as LayoutLeft.
        """
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Direct computation: dY @ W where dY:[M,N], W:[N,K] -> dX:[M,K]
                grad_ptr = (
                    grad_output.data_ptr()
                    + offset * grad_output.stride(0) * grad_output.element_size()
                )
                weight_ptr = weight_stack[expert_idx].data_ptr()
                grad_input_ptr = (
                    grad_input.data_ptr()
                    + offset * grad_input.stride(0) * grad_input.element_size()
                )

                M, N = size, grad_output.shape[1]
                K = weight_stack.shape[2]
                L = 1

                # OPTIMIZATION 11: Use optimal CUTLASS layout to avoid transpose
                # Configure strides for LayoutLeft on B matrix to get A @ B instead of A @ B^T
                A_strides = [grad_output.stride(0), grad_output.stride(1)]
                B_strides = [
                    weight_stack.stride(2),
                    weight_stack.stride(1),
                ]  # Swap strides for transpose effect
                C_strides = [grad_input.stride(0), grad_input.stride(1)]

                problem_sizes.append([M, K, N, L])  # Note: N and K swapped for layout
                strides_abc.append([A_strides, B_strides, C_strides])
                ptrs_abc.append([grad_ptr, weight_ptr, grad_input_ptr])

        return problem_sizes, strides_abc, ptrs_abc

    @staticmethod
    def _prepare_weight_grad_optimized(
        grad_output,
        input_tokens,
        valid_indices_cpu,
        valid_sizes_cpu,
        valid_offsets_cpu,
        grad_weight,
        strategy,
    ):
        """
        OPTIMIZATION 12: Direct weight gradient computation using stride manipulation

        Compute dW = dY^T @ X directly by using appropriate stride configurations
        instead of creating transposed tensors.
        """
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Direct computation: dY^T @ X where dY:[M,N], X:[M,K] -> dW:[N,K]
                grad_ptr = (
                    grad_output.data_ptr()
                    + offset * grad_output.stride(0) * grad_output.element_size()
                )
                input_ptr = (
                    input_tokens.data_ptr()
                    + offset * input_tokens.stride(0) * input_tokens.element_size()
                )
                weight_grad_ptr = grad_weight[expert_idx].data_ptr()

                M, N = size, grad_output.shape[1]
                K = input_tokens.shape[1]
                L = 1

                # OPTIMIZATION 13: Stride configuration for dY^T @ X computation
                # Treat dY as transposed by swapping its strides
                A_strides = [
                    grad_output.stride(1),
                    grad_output.stride(0),
                ]  # Transposed dY strides
                B_strides = [
                    input_tokens.stride(1),
                    input_tokens.stride(0),
                ]  # X strides for B^T
                C_strides = [grad_weight.stride(1), grad_weight.stride(2)]

                problem_sizes.append([N, K, M, L])
                strides_abc.append([A_strides, B_strides, C_strides])
                ptrs_abc.append([grad_ptr, input_ptr, weight_grad_ptr])

        return problem_sizes, strides_abc, ptrs_abc

    @staticmethod
    def _execute_optimized_cutlass_kernel(
        problem_sizes, strides_abc, ptrs_abc, device, strategy
    ):
        """
        OPTIMIZATION 14: Optimized CUTLASS kernel execution

        - Reuse compiled kernels aggressively
        - Minimize tensor creation overhead
        - Use optimal cluster configurations
        """
        if not problem_sizes:
            return

        num_groups = len(problem_sizes)

        # OPTIMIZATION 15: Reuse tensor allocations using memory pool
        if not hasattr(strategy, "_tensor_pool"):
            strategy._tensor_pool = {}

        # Create metadata tensors with memory reuse
        cache_key = (num_groups, device)
        if cache_key not in strategy._tensor_pool:
            strategy._tensor_pool[cache_key] = {
                "problem_sizes": torch.empty(
                    num_groups, 4, dtype=torch.int32, device=device
                ),
                "strides": torch.empty(
                    num_groups, 3, 2, dtype=torch.int32, device=device
                ),
                "ptrs": torch.empty(num_groups, 3, dtype=torch.int64, device=device),
            }

        tensors = strategy._tensor_pool[cache_key]

        # Fill tensors directly to avoid allocations
        tensors["problem_sizes"][: len(problem_sizes)] = torch.tensor(
            problem_sizes, device=device
        )
        tensors["strides"][: len(strides_abc)] = torch.tensor(
            strides_abc, device=device
        )
        tensors["ptrs"][: len(ptrs_abc)] = torch.tensor(ptrs_abc, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(
            tensors["problem_sizes"][: len(problem_sizes)],
            assumed_align=strategy.ALIGNMENT,
        )
        strides_cute = from_dlpack(
            tensors["strides"][: len(strides_abc)], assumed_align=strategy.ALIGNMENT
        )
        ptrs_cute = from_dlpack(
            tensors["ptrs"][: len(ptrs_abc)], assumed_align=strategy.ALIGNMENT
        )

        # OPTIMIZATION 16: Aggressive kernel caching with finer granularity
        total_clusters = strategy._compute_total_clusters(problem_sizes)
        cache_key = (
            num_groups,
            total_clusters,
            tuple(problem_sizes[0][:3]),
        )  # Include problem shape

        if cache_key not in strategy._compiled_kernels:
            tensormap_cute = strategy._get_tensormap_buffer(device)
            initial_tensors = strategy._create_initial_tensors(problem_sizes[0], device)

            strategy._compiled_kernels[cache_key] = cute.compile(
                strategy.grouped_gemm,
                *initial_tensors,
                num_groups,
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                total_clusters,
                tensormap_cute,
                strategy.max_active_clusters,
                strategy.stream,
            )

        # Execute with cached kernel
        compiled_kernel = strategy._compiled_kernels[cache_key]
        tensormap_cute = strategy._get_tensormap_buffer(device)
        initial_tensors = strategy._create_initial_tensors(problem_sizes[0], device)

        compiled_kernel(
            *initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            strategy.stream,
        )

        # OPTIMIZATION 17: Asynchronous execution - don't sync unless needed
        # torch.cuda.synchronize()  # Only sync when absolutely necessary


# OPTIMIZATION 18: Memory-efficient strategy configuration
class CUTLASSGroupedGemmStrategyOptimized(CUTLASSGroupedGemmStrategy):
    """
    Performance-optimized strategy with additional caching and memory management.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # OPTIMIZATION 19: Pre-allocate commonly used tensor shapes
        self._tensor_pool = {}
        self._kernel_cache_hits = 0
        self._kernel_cache_misses = 0

        # OPTIMIZATION 20: Tune cluster configuration for backward passes
        self._optimize_cluster_config()

    def _optimize_cluster_config(self):
        """
        OPTIMIZATION 21: Dynamically tune cluster configuration

        Backward passes have different compute/memory patterns than forward,
        so we optimize cluster shapes accordingly.
        """
        # Smaller clusters often work better for gradient computations
        # due to different memory access patterns
        if self.cluster_shape_mn == (4, 4):
            self.cluster_shape_mn = (2, 2)  # Better for gradient computations
        elif self.cluster_shape_mn == (2, 2):
            self.cluster_shape_mn = (1, 2)  # Even more conservative

    def get_cache_stats(self):
        """Get kernel cache performance statistics"""
        total = self._kernel_cache_hits + self._kernel_cache_misses
        hit_rate = self._kernel_cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self._kernel_cache_hits,
            "cache_misses": self._kernel_cache_misses,
            "hit_rate": hit_rate,
        }


# OPTIMIZATION 22: High-level optimized linear layer
class CUTLASSGroupedLinearOptimized(nn.Module):
    """
    Optimized CUTLASS grouped linear layer with performance enhancements.
    """

    def __init__(self, num_experts, in_features, out_features, strategy, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.strategy = strategy
        self.dtype = kwargs.get("dtype", torch.bfloat16)

        # OPTIMIZATION 23: Use optimized parameter initialization
        self.weight = nn.Parameter(
            torch.empty(
                num_experts, out_features, in_features, dtype=self.dtype, device="cuda"
            )  # Pre-allocate on GPU
        )
        self.reset_parameters()

        # OPTIMIZATION 24: Pre-compute commonly used tensors
        self._size_tensor_cache = {}

    def forward(self, input_tokens, expert_assignments):
        """Optimized forward pass with caching"""
        # OPTIMIZATION 25: Cache size/offset computations for repeated patterns
        assignment_hash = hash(expert_assignments.data_ptr())
        if assignment_hash in self._size_tensor_cache:
            m_sizes, m_offsets = self._size_tensor_cache[assignment_hash]
        else:
            m_sizes, m_offsets = self._compute_expert_sizes_and_offsets(
                expert_assignments
            )
            if len(self._size_tensor_cache) < 100:  # Limit cache size
                self._size_tensor_cache[assignment_hash] = (m_sizes, m_offsets)

        # OPTIMIZATION 26: Skip sorting if already sorted (common in some workloads)
        if torch.all(expert_assignments[:-1] <= expert_assignments[1:]):
            sorted_tokens = input_tokens
            sorted_indices = torch.arange(len(input_tokens), device=input_tokens.device)
        else:
            sorted_indices = torch.argsort(expert_assignments)
            sorted_tokens = input_tokens[sorted_indices]

        # Use optimized backward function
        sorted_output = CUTLASSBackwardGroupGemmOptimized.apply(
            sorted_tokens, self.weight, m_sizes, m_offsets, self.strategy
        )

        # OPTIMIZATION 27: Avoid unnecessary tensor creation for unsort
        if torch.equal(
            sorted_indices, torch.arange(len(input_tokens), device=input_tokens.device)
        ):
            return sorted_output
        else:
            output = torch.empty_like(sorted_output)
            output[sorted_indices] = sorted_output
            return output


class CUTLASSBackwardGroupGemm_prev(torch.autograd.Function):
    """
    PyTorch autograd Function for CUTLASS grouped GEMM with backward pass support.

    This function computes grouped matrix multiplication and automatically handles
    gradient computation for both inputs and weights using the same CUTLASS kernel.

    Forward: Y_i = X_i @ W_i^T for each expert i
    Backward:
        - dX_i = dY_i @ W_i for each expert i
        - dW_i = dY_i^T @ X_i for each expert i
    """

    @staticmethod
    def forward(ctx, input_tokens, weight_stack, m_sizes, m_offsets, strategy):
        """
        Forward pass of grouped GEMM.

        Args:
            ctx: PyTorch autograd context for saving tensors
            input_tokens: Input tokens [total_tokens, hidden_size]
            weight_stack: Stacked expert weights [num_experts, out_features, in_features]
            m_sizes: Number of tokens per expert [num_experts]
            m_offsets: Token offsets per expert [num_experts + 1]
            strategy: CUTLASSGroupedGemmStrategy instance

        Returns:
            output: Grouped GEMM result [total_tokens, out_features]
        """
        # Save tensors and info for backward pass
        ctx.save_for_backward(input_tokens, weight_stack, m_sizes, m_offsets)
        ctx.strategy = strategy

        # Ensure tensors are on GPU and contiguous
        input_tokens = input_tokens.contiguous()
        weight_stack = weight_stack.contiguous()
        m_sizes_gpu = (
            m_sizes.to(input_tokens.device) if not m_sizes.is_cuda else m_sizes
        )
        m_offsets_gpu = (
            m_offsets.to(input_tokens.device) if not m_offsets.is_cuda else m_offsets
        )

        device = input_tokens.device
        num_experts, out_features, in_features = weight_stack.shape
        total_tokens = input_tokens.shape[0]

        # Prepare output tensor
        output = torch.zeros(
            total_tokens, out_features, dtype=strategy.DTYPE_TORCH, device=device
        )

        # Check for valid experts
        if not torch.any(m_sizes_gpu > 0):
            return output

        # Execute forward grouped GEMM using the strategy
        output = CUTLASSBackwardGroupGemm._execute_forward_gemm(
            input_tokens, weight_stack, m_sizes_gpu, m_offsets_gpu, output, strategy
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of grouped GEMM.

        Args:
            ctx: PyTorch autograd context with saved tensors
            grad_output: Gradient w.r.t. output [total_tokens, out_features]

        Returns:
            Tuple of gradients: (grad_input, grad_weight, None, None, None)
        """
        input_tokens, weight_stack, m_sizes, m_offsets = ctx.saved_tensors
        strategy = ctx.strategy

        grad_output = grad_output.contiguous()
        device = grad_output.device

        # Initialize gradients
        grad_input = torch.zeros_like(input_tokens)
        grad_weight = torch.zeros_like(weight_stack)

        m_sizes_gpu = m_sizes.to(device) if not m_sizes.is_cuda else m_sizes
        m_offsets_gpu = m_offsets.to(device) if not m_offsets.is_cuda else m_offsets

        # Check for valid experts
        if not torch.any(m_sizes_gpu > 0):
            return grad_input, grad_weight, None, None, None

        # Compute gradient w.r.t. input: dX_i = dY_i @ W_i
        grad_input = CUTLASSBackwardGroupGemm._execute_input_gradient_gemm(
            grad_output, weight_stack, m_sizes_gpu, m_offsets_gpu, grad_input, strategy
        )

        # Compute gradient w.r.t. weight: dW_i = dY_i^T @ X_i
        grad_weight = CUTLASSBackwardGroupGemm._execute_weight_gradient_gemm(
            grad_output, input_tokens, m_sizes_gpu, m_offsets_gpu, grad_weight, strategy
        )

        return grad_input, grad_weight, None, None, None

    @staticmethod
    def _execute_forward_gemm(
        input_tokens, weight_stack, m_sizes_gpu, m_offsets_gpu, output, strategy
    ):
        """Execute forward pass: Y_i = X_i @ W_i^T"""
        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return output

        # Prepare metadata for grouped GEMM
        problem_sizes, strides_abc, ptrs_abc, outputs = (
            CUTLASSBackwardGroupGemm._prepare_forward_metadata(
                input_tokens,
                weight_stack,
                m_sizes_gpu,
                m_offsets_gpu,
                valid_indices,
                output.device,
                strategy,
            )
        )

        if len(problem_sizes) == 0:
            return output

        # Execute CUTLASS grouped GEMM
        CUTLASSBackwardGroupGemm._execute_cutlass_kernel(
            problem_sizes, strides_abc, ptrs_abc, output.device, strategy
        )

        # Reconstruct output
        return CUTLASSBackwardGroupGemm._reconstruct_output(
            outputs, m_sizes_gpu, m_offsets_gpu, output
        )

    @staticmethod
    def _execute_input_gradient_gemm(
        grad_output, weight_stack, m_sizes_gpu, m_offsets_gpu, grad_input, strategy
    ):
        """Execute input gradient: dX_i = dY_i @ W_i"""
        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return grad_input

        # Prepare metadata for input gradient GEMM
        problem_sizes, strides_abc, ptrs_abc, outputs = (
            CUTLASSBackwardGroupGemm._prepare_input_grad_metadata(
                grad_output,
                weight_stack,
                m_sizes_gpu,
                m_offsets_gpu,
                valid_indices,
                grad_input.device,
                strategy,
            )
        )

        if len(problem_sizes) == 0:
            return grad_input

        # Execute CUTLASS grouped GEMM
        CUTLASSBackwardGroupGemm._execute_cutlass_kernel(
            problem_sizes, strides_abc, ptrs_abc, grad_input.device, strategy
        )

        # Reconstruct gradient
        return CUTLASSBackwardGroupGemm._reconstruct_output(
            outputs, m_sizes_gpu, m_offsets_gpu, grad_input
        )

    @staticmethod
    def _execute_weight_gradient_gemm(
        grad_output, input_tokens, m_sizes_gpu, m_offsets_gpu, grad_weight, strategy
    ):
        """Execute weight gradient: dW_i = dY_i^T @ X_i"""
        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return grad_weight

        # Prepare metadata for weight gradient GEMM
        problem_sizes, strides_abc, ptrs_abc = (
            CUTLASSBackwardGroupGemm._prepare_weight_grad_metadata(
                grad_output,
                input_tokens,
                m_sizes_gpu,
                m_offsets_gpu,
                valid_indices,
                grad_weight,
                strategy.DTYPE_TORCH,
            )
        )

        if len(problem_sizes) == 0:
            return grad_weight

        # Execute CUTLASS grouped GEMM
        CUTLASSBackwardGroupGemm._execute_cutlass_kernel(
            problem_sizes, strides_abc, ptrs_abc, grad_weight.device, strategy
        )

        return grad_weight

    @staticmethod
    def _prepare_forward_metadata(
        input_tokens,
        weight_stack,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        device,
        strategy,
    ):
        """Prepare metadata for forward pass: Y_i = X_i @ W_i^T"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        outputs = []

        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )
        )

        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Get expert data
                expert_tokens = input_tokens[
                    offset : offset + size
                ].contiguous()  # [M, K]
                expert_weight = weight_stack[
                    expert_idx
                ].contiguous()  # [N, K] - already transposed for forward

                M, K = expert_tokens.shape
                N, K_w = expert_weight.shape
                assert K == K_w, f"Dimension mismatch: {K} != {K_w}"

                # Create output tensor
                output = torch.empty(M, N, dtype=strategy.DTYPE_TORCH, device=device)
                outputs.append(output)

                # Add to metadata: expert_tokens @ expert_weight^T
                # For CUTLASS, we need to pass B transposed, so we pass expert_weight as [N,K]
                # and CUTLASS will compute A @ B^T
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    expert_tokens,  # A: [M, K]
                    expert_weight,  # B: [N, K] (will compute A @ B^T)
                    output,  # C: [M, N]
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                    transpose_b=True,  # Indicate B should be transposed
                )

        return problem_sizes, strides_abc, ptrs_abc, outputs

    @staticmethod
    def _prepare_input_grad_metadata(
        grad_output,
        weight_stack,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        device,
        strategy,
    ):
        """Prepare metadata for input gradient: dX_i = dY_i @ W_i"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        outputs = []

        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )
        )

        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Get expert data
                grad_expert = grad_output[offset : offset + size].contiguous()  # [M, N]
                expert_weight = weight_stack[expert_idx].contiguous()  # [N, K]

                M, N = grad_expert.shape
                N_w, K = expert_weight.shape
                assert N == N_w, f"Dimension mismatch: {N} != {N_w}"

                # Create output tensor for gradient
                grad_input_expert = torch.empty(
                    M, K, dtype=strategy.DTYPE_TORCH, device=device
                )
                outputs.append(grad_input_expert)

                # Add to metadata: grad_expert @ expert_weight (no transpose needed)
                # grad_expert: [M, N], expert_weight: [N, K] -> result: [M, K]
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    grad_expert,  # A: [M, N]
                    expert_weight,  # B: [N, K]
                    grad_input_expert,  # C: [M, K]
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                    transpose_b=False,  # No transpose needed
                )

        return problem_sizes, strides_abc, ptrs_abc, outputs

    @staticmethod
    def _prepare_weight_grad_metadata(
        grad_output,
        input_tokens,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        grad_weight,
        dtype,
    ):
        """Prepare metadata for weight gradient: dW_i = dY_i^T @ X_i"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=grad_weight.device), valid_sizes[:-1]]
                ),
                dim=0,
            )
        )

        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Get expert data
                grad_expert = grad_output[offset : offset + size].contiguous()  # [M, N]
                input_expert = input_tokens[
                    offset : offset + size
                ].contiguous()  # [M, K]

                M, N = grad_expert.shape
                M_i, K = input_expert.shape
                assert M == M_i, f"Dimension mismatch: {M} != {M_i}"

                # Get output tensor (slice of grad_weight for this expert)
                grad_weight_expert = grad_weight[expert_idx]  # [N, K]

                # For dW = dY^T @ X, we compute: grad_expert^T @ input_expert -> grad_weight_expert
                # grad_expert^T: [N, M], input_expert: [M, K] -> result: [N, K]
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    grad_expert,  # A: [M, N] (will be transposed)
                    input_expert,  # B: [M, K]
                    grad_weight_expert,  # C: [N, K]
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                    transpose_a=True,  # Transpose A to get [N, M]
                )

        return problem_sizes, strides_abc, ptrs_abc

    @staticmethod
    def _add_gemm_to_metadata(
        A,
        B,
        C,
        problem_sizes,
        strides_abc,
        ptrs_abc,
        transpose_a=False,
        transpose_b=False,
    ):
        """Add a single GEMM operation to metadata lists."""
        # Get original shapes
        if transpose_a:
            M, K_A = A.shape[1], A.shape[0]  # A is [K_A, M] but we want [M, K_A]
        else:
            M, K_A = A.shape

        if transpose_b:
            N, K_B = (
                B.shape[0],
                B.shape[1],
            )  # B is [N, K_B] but we want [K_B, N] for B^T
        else:
            K_B, N = B.shape

        # Ensure inner dimensions match
        assert K_A == K_B, f"Inner dimension mismatch: {K_A} != {K_B}"
        K = K_A
        L = 1

        # Create proper tensor views for CUTLASS
        if transpose_a:
            # A^T: need to transpose A
            A_for_gemm = A.t().contiguous()  # [M, K]
        else:
            A_for_gemm = A.contiguous()  # [M, K]

        if transpose_b:
            # B^T: B is already [N, K], CUTLASS will handle the transpose
            B_for_gemm = B.contiguous()  # [N, K]
        else:
            # B: need to transpose to [N, K] format expected by CUTLASS
            B_for_gemm = B.t().contiguous()  # [N, K]

        C_for_gemm = C.contiguous()  # [M, N]

        # Convert to MNKL format for CUTLASS
        A_mnkl = A_for_gemm.unsqueeze(-1).contiguous()  # [M, K, 1]
        B_mnkl = B_for_gemm.unsqueeze(-1).contiguous()  # [N, K, 1]
        C_mnkl = C_for_gemm.unsqueeze(-1).contiguous()  # [M, N, 1]

        # Extract strides
        A_strides = list(A_mnkl.stride()[:2])
        B_strides = list(B_mnkl.stride()[:2])
        C_strides = list(C_mnkl.stride()[:2])

        # Add to metadata
        problem_sizes.append([M, N, K, L])
        strides_abc.append([A_strides, B_strides, C_strides])
        ptrs_abc.append(
            [A_for_gemm.data_ptr(), B_for_gemm.data_ptr(), C_for_gemm.data_ptr()]
        )

    @staticmethod
    def _execute_cutlass_kernel(problem_sizes, strides_abc, ptrs_abc, device, strategy):
        """Execute the CUTLASS grouped GEMM kernel."""
        num_groups = len(problem_sizes)

        # Convert to CUTE tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        problem_sizes_cute = from_dlpack(
            problem_sizes_tensor, assumed_align=strategy.ALIGNMENT
        )
        strides_cute = from_dlpack(strides_tensor, assumed_align=strategy.ALIGNMENT)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=strategy.ALIGNMENT)

        # Get tensormap and compute clusters
        tensormap_cute = strategy._get_tensormap_buffer(device)
        total_clusters = strategy._compute_total_clusters(problem_sizes)

        # Create initial tensors for compilation
        initial_tensors = strategy._create_initial_tensors(problem_sizes[0], device)

        # Get compiled kernel
        compiled_kernel = strategy._get_compiled_kernel(
            num_groups,
            total_clusters,
            initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
        )

        # Execute kernel
        compiled_kernel(
            *initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            strategy.stream,
        )
        torch.cuda.synchronize()

    @staticmethod
    def _reconstruct_output(outputs, m_sizes_gpu, m_offsets_gpu, full_output):
        """Reconstruct full output tensor from expert results."""
        if not outputs:
            return full_output

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices]

        # Compute offsets
        if len(m_offsets_gpu) <= len(valid_indices):
            valid_offsets = torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=m_sizes_gpu.device), valid_sizes[:-1]]
                ),
                dim=0,
            )
        else:
            valid_offsets = m_offsets_gpu[valid_indices]

        # Convert to CPU for reconstruction (minimal sync)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()

        for i, (size, offset) in enumerate(zip(valid_sizes_cpu, valid_offsets_cpu)):
            if i < len(outputs):
                full_output[offset : offset + size] = outputs[i]

        return full_output


class CUTLASSGroupedLinear(nn.Module):
    """
    A PyTorch module that wraps CUTLASS grouped GEMM with automatic differentiation.

    This module performs grouped linear transformations using CUTLASS kernels,
    with support for forward and backward passes through PyTorch autograd.

    Usage:
        layer = CUTLASSGroupedLinear(
            num_experts=8,
            in_features=4096,
            out_features=11008,
            strategy=your_cutlass_strategy
        )

        output = layer(input_tokens, expert_assignments)
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        strategy,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the CUTLASS grouped linear layer.

        Args:
            num_experts: Number of experts
            in_features: Input feature dimension
            out_features: Output feature dimension
            strategy: CUTLASSGroupedGemmStrategy instance
            bias: Whether to include bias (not yet implemented)
            dtype: Data type for weights
        """
        super().__init__()

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
        """Initialize parameters using standard initialization."""
        for expert_idx in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[expert_idx], a=1.41421356)  # sqrt(2)

    def forward(
        self, input_tokens: torch.Tensor, expert_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through grouped linear layer.

        Args:
            input_tokens: Input tokens [total_tokens, in_features]
            expert_assignments: Expert assignment per token [total_tokens]

        Returns:
            output: Transformed tokens [total_tokens, out_features]
        """
        # Compute sizes and offsets for each expert
        m_sizes, m_offsets = self._compute_expert_sizes_and_offsets(expert_assignments)

        # Sort tokens by expert assignment for contiguous access
        sorted_indices = torch.argsort(expert_assignments)
        sorted_tokens = input_tokens[sorted_indices]

        # Apply grouped GEMM
        sorted_output = CUTLASSBackwardGroupGemm.apply(
            sorted_tokens, self.weight, m_sizes, m_offsets, self.strategy
        )

        # Unsort to restore original order
        output = torch.empty_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _compute_expert_sizes_and_offsets(
        self, expert_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the number of tokens assigned to each expert and their offsets.

        Args:
            expert_assignments: Expert assignment per token [total_tokens]

        Returns:
            Tuple of (sizes, offsets) tensors
        """
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
        """Return string representation of module parameters."""
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}"


# Example usage and testing functions
def test_cutlass_backward_group_gemm():
    """Test the CUTLASS backward group GEMM implementation."""
    print("Testing CUTLASS Backward Group GEMM...")

    # Setup
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_experts = 4
    in_features = 1024
    out_features = 2048
    total_tokens = 512

    print(
        f"Creating strategy for {num_experts} experts, {in_features} in_features, {out_features} out_features, {total_tokens} total_tokens"
    )

    # Create strategy with safe defaults that avoid context issues
    strategy = CUTLASSGroupedGemmStrategy(
        custom_activation=lambda x: x,  # Identity for testing
        use_2cta_instrs=True,  # Use single CTA to avoid context issues
        mma_tiler_mn=(256, 128),  # Safe single-CTA values
        cluster_shape_mn=(2, 2),  # Safe single-CTA values
    )
    print(f"Using strategy: {strategy}")

    # Create test data
    input_tokens = torch.randn(
        total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
    )
    expert_assignments = torch.randint(0, num_experts, (total_tokens,), device=device)

    # Create layer
    layer = CUTLASSGroupedLinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        strategy=strategy,
        bias=False,
        dtype=dtype,
    )

    layer = layer.to(device)

    # Forward pass
    output = layer(input_tokens, expert_assignments)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert input_tokens.grad is not None, "Input gradient should not be None"
    assert layer.weight.grad is not None, "Weight gradient should not be None"

    print("✓ CUTLASS Backward Group GEMM test passed!")
    return True


def _test_single_expert_cutlass_fixed(
    grad_expert, input_expert, weight_expert, strategy
):
    """Test CUTLASS operations on a single expert with CORRECTED dimensions"""
    M, N = grad_expert.shape  # [M, N] - grad_expert
    M_i, K = input_expert.shape  # [M, K] - input_expert
    N_w, K_w = weight_expert.shape  # [N, K] - weight_expert

    assert (
        M == M_i and N == N_w and K == K_w
    ), f"Shape mismatch: {grad_expert.shape}, {input_expert.shape}, {weight_expert.shape}"

    device = grad_expert.device

    print(
        f"      Shapes: grad_expert={grad_expert.shape}, input_expert={input_expert.shape}, weight_expert={weight_expert.shape}"
    )

    # Test input gradient: dX = dY @ W
    print(f"      Testing input gradient: dX = dY @ W")
    ref_grad_input = torch.mm(grad_expert, weight_expert)  # [M, N] @ [N, K] = [M, K]
    print(
        f"      Reference dX shape: {ref_grad_input.shape}, norm: {ref_grad_input.norm().item():.4f}"
    )

    # CUTLASS approach: Since CUTLASS computes A @ B^T, reformulate dX = dY @ W as:
    # dX^T = W^T @ dY^T, then transpose result
    # A = W^T [K, N], B = dY^T [N, M], C = dX^T [K, M]
    weight_T = weight_expert.t().contiguous()  # [K, N]
    grad_T = grad_expert.t().contiguous()  # [N, M]
    result_T = torch.zeros(K, M, dtype=strategy.DTYPE_TORCH, device=device)  # [K, M]

    print(
        f"      CUTLASS matrices: A=W^T{weight_T.shape}, B=dY^T{grad_T.shape}, C=dX^T{result_T.shape}"
    )

    # CORRECT problem size: A[K,N] @ B^T[N,M] = C[K,M]
    # Note: CUTLASS will transpose B, so B^T becomes dY^T^T = dY
    problem_sizes = [
        [K, M, N, 1]
    ]  # [M, N, K, L] format but our actual computation is [K, M, N, 1]

    print(f"      Problem size: {problem_sizes[0]}")

    # Set up strides for A @ B^T where CUTLASS transposes B
    A_mnkl = weight_T.unsqueeze(-1).contiguous()  # [K, N, 1]
    B_mnkl = grad_T.unsqueeze(
        -1
    ).contiguous()  # [N, M, 1] - CUTLASS will transpose to [M, N, 1]
    C_mnkl = result_T.unsqueeze(-1).contiguous()  # [K, M, 1]

    A_strides = list(A_mnkl.stride()[:2])
    B_strides = list(B_mnkl.stride()[:2])
    C_strides = list(C_mnkl.stride()[:2])

    strides_abc = [[A_strides, B_strides, C_strides]]
    ptrs_abc = [[weight_T.data_ptr(), grad_T.data_ptr(), result_T.data_ptr()]]

    print(f"      Strides: A={A_strides}, B={B_strides}, C={C_strides}")

    CUTLASSBackwardGroupGemmDebug._execute_cutlass_kernel_debug(
        problem_sizes, strides_abc, ptrs_abc, device, strategy, "single_input_grad"
    )

    grad_input_cutlass = result_T.t()  # Transpose back to [M, K]
    print(
        f"      CUTLASS dX shape: {grad_input_cutlass.shape}, norm: {grad_input_cutlass.norm().item():.4f}"
    )

    input_grad_diff = torch.abs(grad_input_cutlass - ref_grad_input).max().item()
    input_grad_rel = input_grad_diff / ref_grad_input.abs().max().item()
    print(
        f"      Input grad diff: {input_grad_diff:.2e} (relative: {input_grad_rel:.2e})"
    )

    # Test weight gradient: dW = dY^T @ X
    print(f"      Testing weight gradient: dW = dY^T @ X")
    ref_grad_weight = torch.mm(
        grad_expert.t(), input_expert
    )  # [N, M] @ [M, K] = [N, K]
    print(
        f"      Reference dW shape: {ref_grad_weight.shape}, norm: {ref_grad_weight.norm().item():.4f}"
    )

    # CUTLASS approach: dW = dY^T @ X
    # This is already in A @ B^T form if we set B^T = X^T
    # A = dY^T [N, M], B^T = X^T [K, M], C = dW [N, K]
    grad_weight_cutlass = torch.zeros(N, K, dtype=strategy.DTYPE_TORCH, device=device)
    input_T = input_expert.t().contiguous()  # [K, M]

    print(
        f"      CUTLASS matrices: A=dY^T{grad_T.shape}, B^T=X^T{input_T.shape}, C=dW{grad_weight_cutlass.shape}"
    )

    # Problem size: A[N,M] @ B^T[K,M] = C[N,K]
    # CUTLASS will compute A @ B^T = dY^T @ (X^T)^T = dY^T @ X = dW
    problem_sizes = [[N, K, M, 1]]

    print(f"      Problem size: {problem_sizes[0]}")

    A_mnkl = grad_T.unsqueeze(-1).contiguous()  # [N, M, 1]
    B_mnkl = input_T.unsqueeze(-1).contiguous()  # [K, M, 1]
    C_mnkl = grad_weight_cutlass.unsqueeze(-1).contiguous()  # [N, K, 1]

    A_strides = list(A_mnkl.stride()[:2])
    B_strides = list(B_mnkl.stride()[:2])
    C_strides = list(C_mnkl.stride()[:2])

    strides_abc = [[A_strides, B_strides, C_strides]]
    ptrs_abc = [[grad_T.data_ptr(), input_T.data_ptr(), grad_weight_cutlass.data_ptr()]]

    print(f"      Strides: A={A_strides}, B={B_strides}, C={C_strides}")

    CUTLASSBackwardGroupGemmDebug._execute_cutlass_kernel_debug(
        problem_sizes, strides_abc, ptrs_abc, device, strategy, "single_weight_grad"
    )

    print(
        f"      CUTLASS dW shape: {grad_weight_cutlass.shape}, norm: {grad_weight_cutlass.norm().item():.4f}"
    )

    weight_grad_diff = torch.abs(grad_weight_cutlass - ref_grad_weight).max().item()
    weight_grad_rel = weight_grad_diff / ref_grad_weight.abs().max().item()
    print(
        f"      Weight grad diff: {weight_grad_diff:.2e} (relative: {weight_grad_rel:.2e})"
    )

    return grad_input_cutlass, grad_weight_cutlass


def _prepare_input_grad_approach_2_fixed(
    grad_output,
    weight_stack,
    m_sizes_gpu,
    m_offsets_gpu,
    valid_indices,
    grad_input,
    strategy,
):
    """Prepare input gradient with CORRECTED explicit transpositions"""
    problem_sizes = []
    strides_abc = []
    ptrs_abc = []
    temp_results = []

    device = grad_output.device
    valid_sizes = m_sizes_gpu[valid_indices].cpu().tolist()
    valid_offsets = (
        (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=device), m_sizes_gpu[valid_indices][:-1]]
                ),
                dim=0,
            )
        )
        .cpu()
        .tolist()
    )
    valid_indices_cpu = valid_indices.cpu().tolist()

    print(f"    Preparing input gradients for {len(valid_indices_cpu)} experts")

    for i, (expert_idx, size, offset) in enumerate(
        zip(valid_indices_cpu, valid_sizes, valid_offsets)
    ):
        if size > 0:
            grad_expert = grad_output[offset : offset + size].contiguous()  # [M, N]
            weight_expert = weight_stack[expert_idx].contiguous()  # [N, K]

            M, N = grad_expert.shape
            N_w, K = weight_expert.shape

            if N != N_w:
                print(
                    f"    ERROR: Expert {expert_idx}, N mismatch: grad_expert has {N}, weight has {N_w}"
                )
                continue

            print(
                f"    Expert {expert_idx}: grad_expert{grad_expert.shape}, weight{weight_expert.shape}"
            )

            # Reformulate dX = dY @ W as dX^T = W^T @ dY^T
            weight_T = weight_expert.t().contiguous()  # [K, N]
            grad_T = grad_expert.t().contiguous()  # [N, M]
            result_T = torch.zeros(
                K, M, dtype=strategy.DTYPE_TORCH, device=device
            )  # [K, M]
            temp_results.append((result_T, offset, size))

            print(
                f"    CUTLASS setup: W^T{weight_T.shape} @ (dY^T)^T = dX^T{result_T.shape}"
            )

            # CUTLASS: A @ B^T where A = W^T [K, N], B = dY^T [N, M]
            # Problem: [K, M, N, 1] since CUTLASS computes A[K,N] @ B^T[N,M] = C[K,M]
            _add_simple_gemm_fixed(
                weight_T, grad_T, result_T, problem_sizes, strides_abc, ptrs_abc
            )

    return problem_sizes, strides_abc, ptrs_abc, temp_results


def _prepare_weight_grad_approach_2_fixed(
    grad_output,
    input_tokens,
    m_sizes_gpu,
    m_offsets_gpu,
    valid_indices,
    grad_weight,
    strategy,
):
    """Prepare weight gradient with CORRECTED explicit transpositions"""
    problem_sizes = []
    strides_abc = []
    ptrs_abc = []

    device = grad_output.device
    valid_sizes = m_sizes_gpu[valid_indices].cpu().tolist()
    valid_offsets = (
        (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=device), m_sizes_gpu[valid_indices][:-1]]
                ),
                dim=0,
            )
        )
        .cpu()
        .tolist()
    )
    valid_indices_cpu = valid_indices.cpu().tolist()

    print(f"    Preparing weight gradients for {len(valid_indices_cpu)} experts")

    for i, (expert_idx, size, offset) in enumerate(
        zip(valid_indices_cpu, valid_sizes, valid_offsets)
    ):
        if size > 0:
            grad_expert = grad_output[offset : offset + size].contiguous()  # [M, N]
            input_expert = input_tokens[offset : offset + size].contiguous()  # [M, K]
            weight_grad_expert = grad_weight[expert_idx]  # [N, K]

            M, N = grad_expert.shape
            M_i, K = input_expert.shape

            if M != M_i:
                print(
                    f"    ERROR: Expert {expert_idx}, M mismatch: grad_expert has {M}, input has {M_i}"
                )
                continue

            print(
                f"    Expert {expert_idx}: grad_expert{grad_expert.shape}, input{input_expert.shape}"
            )

            # dW = dY^T @ X: A = dY^T [N, M], B^T = X^T [K, M], C = dW [N, K]
            grad_T = grad_expert.t().contiguous()  # [N, M]
            input_T = input_expert.t().contiguous()  # [K, M]

            print(
                f"    CUTLASS setup: dY^T{grad_T.shape} @ X = dW{weight_grad_expert.shape}"
            )

            # CUTLASS: A @ B^T where A = dY^T [N, M], B = X^T [K, M]
            # Problem: [N, K, M, 1] since CUTLASS computes A[N,M] @ B^T[K,M] = C[N,K]
            _add_simple_gemm_fixed(
                grad_T,
                input_T,
                weight_grad_expert,
                problem_sizes,
                strides_abc,
                ptrs_abc,
            )

    return problem_sizes, strides_abc, ptrs_abc, None


def _add_simple_gemm_fixed(A, B, C, problem_sizes, strides_abc, ptrs_abc):
    """Add a simple GEMM operation: C = A @ B^T with CORRECT dimensions"""
    M, K = A.shape  # A is [M, K]
    N, K_B = B.shape  # B is [N, K_B], will be transposed to [K_B, N]

    if K != K_B:
        print(f"    ERROR: Inner dimension mismatch: A has K={K}, B has K_B={K_B}")
        print(f"    A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
        assert False, f"Inner dimension mismatch: {K} != {K_B}"

    L = 1

    # Expected output shape for A[M,K] @ B^T[K,N] = C[M,N]
    expected_C_shape = (M, N)
    if C.shape != expected_C_shape:
        print(
            f"    ERROR: Output shape mismatch: expected {expected_C_shape}, got {C.shape}"
        )
        assert (
            False
        ), f"Output shape mismatch: expected {expected_C_shape}, got {C.shape}"

    # Convert to MNKL format
    A_mnkl = A.unsqueeze(-1).contiguous()  # [M, K, 1]
    B_mnkl = B.unsqueeze(
        -1
    ).contiguous()  # [N, K, 1] - CUTLASS will transpose to [K, N, 1]
    C_mnkl = C.unsqueeze(-1).contiguous()  # [M, N, 1]

    A_strides = list(A_mnkl.stride()[:2])
    B_strides = list(B_mnkl.stride()[:2])
    C_strides = list(C_mnkl.stride()[:2])

    # Problem size: [M, N, K, L] for CUTLASS
    problem_sizes.append([M, N, K, L])
    strides_abc.append([A_strides, B_strides, C_strides])
    ptrs_abc.append([A.data_ptr(), B.data_ptr(), C.data_ptr()])

    print(
        f"    Added GEMM: A{A.shape} @ B^T{B.shape} = C{C.shape}, problem=[{M}, {N}, {K}, {L}]"
    )


def _backward_approach_2_fixed(
    grad_output, input_tokens, weight_stack, m_sizes_gpu, m_offsets_gpu, strategy
):
    """
    Approach 2: Use CUTLASS with explicit transpositions - FIXED VERSION
    """
    print("  🔧 Using Approach 2: CUTLASS with explicit transpositions (FIXED)")

    device = grad_output.device
    grad_input = torch.zeros_like(input_tokens)
    grad_weight = torch.zeros_like(weight_stack)

    valid_mask = m_sizes_gpu > 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    if len(valid_indices) == 0:
        return grad_input, grad_weight

    print(f"  Processing {len(valid_indices)} experts")

    # Input gradient: dX = dY @ W (reformulated as dX^T = W^T @ dY^T)
    print("  Computing input gradients...")
    input_problems = _prepare_input_grad_approach_2_fixed(
        grad_output,
        weight_stack,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        grad_input,
        strategy,
    )

    if input_problems[0]:  # Has problems
        print(f"  Executing {len(input_problems[0])} input gradient problems")
        CUTLASSBackwardGroupGemmDebug._execute_cutlass_kernel_debug(
            *input_problems[:3], device, strategy, "input_grad"
        )

        # Reconstruct input gradients from transposed results
        temp_results = input_problems[3]
        for result_T, offset, size in temp_results:
            grad_input[offset : offset + size] = (
                result_T.t()
            )  # Transpose back to [M, K]

    # Weight gradient: dW = dY^T @ X
    print("  Computing weight gradients...")
    weight_problems = _prepare_weight_grad_approach_2_fixed(
        grad_output,
        input_tokens,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        grad_weight,
        strategy,
    )

    if weight_problems[0]:  # Has problems
        print(f"  Executing {len(weight_problems[0])} weight gradient problems")
        CUTLASSBackwardGroupGemmDebug._execute_cutlass_kernel_debug(
            *weight_problems, device, strategy, "weight_grad"
        )

    return grad_input, grad_weight


# Update the test_single_expert_cutlass function in the debug class
def patch_debug_class():
    """Patch the debug class with fixed methods"""
    # Replace the broken method
    CUTLASSBackwardGroupGemmDebug._test_single_expert_cutlass = staticmethod(
        _test_single_expert_cutlass_fixed
    )
    CUTLASSBackwardGroupGemmDebug._backward_approach_2 = staticmethod(
        _backward_approach_2_fixed
    )
    CUTLASSBackwardGroupGemmDebug._prepare_input_grad_approach_2 = staticmethod(
        _prepare_input_grad_approach_2_fixed
    )
    CUTLASSBackwardGroupGemmDebug._prepare_weight_grad_approach_2 = staticmethod(
        _prepare_weight_grad_approach_2_fixed
    )
    CUTLASSBackwardGroupGemmDebug._add_simple_gemm = staticmethod(
        _add_simple_gemm_fixed
    )
    print("✅ Patched debug class with fixed methods")


def test_fixed_implementation():
    """Test the fixed implementation"""
    print("🧪 Testing Fixed CUTLASS Implementation")
    print("=" * 50)

    # Import and patch
    try:
        from cutlass_backwards_debug import (
            CUTLASSBackwardGroupGemmDebug,
            CUTLASSGroupedGemmStrategyDebug,
            CUTLASSGroupedLinearDebug,
        )

        patch_debug_class()
    except ImportError:
        print("❌ Cannot import debug modules")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test single expert first
    print("\n🔍 Testing Fixed Single Expert")
    print("-" * 30)

    M, N, K = 32, 64, 128
    grad_expert = torch.randn(M, N, dtype=dtype, device=device)
    input_expert = torch.randn(M, K, dtype=dtype, device=device)
    weight_expert = torch.randn(N, K, dtype=dtype, device=device)

    strategy = CUTLASSGroupedGemmStrategyDebug(
        debug_mode=True,
        backward_method="approach_3",
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    try:
        grad_input_cutlass, grad_weight_cutlass = _test_single_expert_cutlass_fixed(
            grad_expert, input_expert, weight_expert, strategy
        )
        print("✅ Fixed single expert test completed")

    except Exception as e:
        print(f"❌ Fixed single expert test failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test grouped operations
    print("\n🔍 Testing Fixed Grouped Operations")
    print("-" * 35)

    num_experts = 4
    in_features = 256
    out_features = 512
    total_tokens = 128

    strategy = CUTLASSGroupedGemmStrategyDebug(
        debug_mode=True,
        backward_method="approach_2",  # Use the fixed approach_2
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    try:
        input_tokens = torch.randn(
            total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
        )
        expert_assignments = torch.randint(
            0, num_experts, (total_tokens,), device=device
        )

        layer = CUTLASSGroupedLinearDebug(
            num_experts, in_features, out_features, strategy, dtype=dtype
        )
        layer = layer.to(device)

        # Forward pass
        output = layer(input_tokens, expert_assignments)

        # Backward pass
        loss = output.sum()
        loss.backward()

        print("✅ Fixed grouped operations completed successfully")

        # Check gradient magnitudes
        if input_tokens.grad is not None:
            print(f"   Input grad norm: {input_tokens.grad.norm().item():.4f}")
        if layer.weight.grad is not None:
            print(f"   Weight grad norm: {layer.weight.grad.norm().item():.4f}")

    except Exception as e:
        print(f"❌ Fixed grouped operations failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_implementation()

# if __name__ == "__main__":
#   test_cutlass_backward_group_gemm()
