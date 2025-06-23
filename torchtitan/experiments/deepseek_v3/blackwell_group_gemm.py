"""
Complete Blackwell CUTLASS Group GEMM (Cute DSL) with autograd support.

"""

from typing import Tuple

import torch
import torch.nn as nn

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
except ImportError as e:
    HAS_CUTLASS = False
    print(f"❌ CUTLASS import failed: {e}")


class CUTLASSGroupGemmStrategy:
    """
    Production CUTLASS strategy for grouped GEMM operations.
    Handles both forward and backward passes with proper dimension management.
    """

    def __init__(
        self,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
    ):
        """
        Initialize CUTLASS strategy.

        Args:
            use_2cta_instrs: Whether to use 2-CTA instructions
            mma_tiler_mn: MMA tile sizes (M, N)
            cluster_shape_mn: Cluster shape (M, N)
        """
        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        print(f"🔧 Initializing CUTLASS GroupGemm strategy...")

        # Force CUDA context creation
        dummy = torch.zeros(1, device="cuda")
        dummy.cpu()

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

        print(f"✅ Strategy initialized:")
        print(f"   - 2CTA: {use_2cta_instrs}")
        print(f"   - MMA tiler: {mma_tiler_mn}")
        print(f"   - Cluster shape: {cluster_shape_mn}")
        print(f"   - Max active clusters: {self.max_active_clusters}")

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

    def execute_grouped_gemm(
        self, A_list, B_list, C_list, operation_name="grouped_gemm"
    ):
        """
        Execute grouped GEMM operations: C_i = A_i @ B_i^T for each i

        Args:
            A_list: List of A matrices
            B_list: List of B matrices
            C_list: List of output C matrices
            operation_name: Name for debugging
        """
        if not A_list or len(A_list) != len(B_list) or len(A_list) != len(C_list):
            return

        device = A_list[0].device

        # Prepare metadata for all operations
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []

        for A, B, C in zip(A_list, B_list, C_list):
            M, K = A.shape
            N, K_B = B.shape

            assert K == K_B, f"Inner dimension mismatch: {K} != {K_B}"
            assert C.shape == (
                M,
                N,
            ), f"Output shape mismatch: expected ({M}, {N}), got {C.shape}"

            # Ensure contiguous
            A = A.contiguous()
            B = B.contiguous()
            C = C.contiguous()

            L = 1

            # Convert to MNKL format
            A_mnkl = A.unsqueeze(-1).contiguous()
            B_mnkl = B.unsqueeze(-1).contiguous()
            C_mnkl = C.unsqueeze(-1).contiguous()

            # Add to batch
            problem_sizes.append([M, N, K, L])
            strides_abc.append(
                [
                    list(A_mnkl.stride()[:2]),
                    list(B_mnkl.stride()[:2]),
                    list(C_mnkl.stride()[:2]),
                ]
            )
            ptrs_abc.append([A.data_ptr(), B.data_ptr(), C.data_ptr()])

        # Execute grouped kernel
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


class CUTLASSGroupGemm(torch.autograd.Function):
    """
    PyTorch autograd Function for CUTLASS grouped GEMM.

    Forward: Y_i = X_i @ W_i^T for each expert i
    Backward:
        - dX_i = dY_i @ W_i for each expert i
        - dW_i = dY_i^T @ X_i for each expert i
    """

    @staticmethod
    def forward(ctx, input_tokens, weight_stack, m_sizes, m_offsets, strategy):
        """
        Forward pass: Y_i = X_i @ W_i^T

        Args:
            ctx: Autograd context
            input_tokens: Sorted input tokens [total_tokens, in_features]
            weight_stack: Expert weights [num_experts, out_features, in_features]
            m_sizes: Tokens per expert [num_experts]
            m_offsets: Token offsets [num_experts + 1]
            strategy: CUTLASSGroupGemmStrategy instance
        """
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

        # Execute forward grouped GEMM
        CUTLASSGroupGemm._execute_forward_grouped(
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
        """
        Backward pass: compute dX and dW

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Upstream gradient [total_tokens, out_features]
        """
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

        # Execute backward grouped operations
        CUTLASSGroupGemm._execute_backward_grouped(
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
    def _execute_forward_grouped(
        input_tokens, weight_stack, m_sizes, m_offsets, valid_indices, output, strategy
    ):
        """Execute grouped forward pass"""
        # Prepare expert operations
        A_list = []  # Input matrices
        B_list = []  # Weight matrices (will be transposed by CUTLASS)
        C_list = []  # Output matrices

        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes[valid_indices].cpu().tolist()
        valid_offsets = (
            (
                m_offsets[valid_indices]
                if len(m_offsets) > len(valid_indices)
                else torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=input_tokens.device),
                            m_sizes[valid_indices][:-1],
                        ]
                    ),
                    dim=0,
                )
            )
            .cpu()
            .tolist()
        )
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes, valid_offsets
        ):
            if size > 0:
                # Get expert data
                expert_input = input_tokens[
                    offset : offset + size
                ].contiguous()  # [M, K]
                expert_weight = weight_stack[expert_idx].contiguous()  # [N, K]
                expert_output = output[offset : offset + size]  # [M, N]

                A_list.append(expert_input)
                B_list.append(expert_weight)
                C_list.append(expert_output)

        # Execute grouped GEMM: expert_input @ expert_weight^T
        strategy.execute_grouped_gemm(A_list, B_list, C_list, "forward")

    @staticmethod
    def _execute_backward_grouped(
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
        """Execute grouped backward pass"""
        # Convert to CPU for iteration (minimal sync)
        valid_sizes = m_sizes[valid_indices].cpu().tolist()
        valid_offsets = (
            (
                m_offsets[valid_indices]
                if len(m_offsets) > len(valid_indices)
                else torch.cumsum(
                    torch.cat(
                        [
                            torch.tensor([0], device=grad_output.device),
                            m_sizes[valid_indices][:-1],
                        ]
                    ),
                    dim=0,
                )
            )
            .cpu()
            .tolist()
        )
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Prepare input gradient operations: dX_i = dY_i @ W_i
        input_A_list = []  # grad_output matrices
        input_B_list = []  # weight matrices (transposed for CUTLASS)
        input_C_list = []  # grad_input matrices

        # Prepare weight gradient operations: dW_i = dY_i^T @ X_i
        weight_A_list = []  # grad_output^T matrices
        weight_B_list = []  # input matrices (transposed for CUTLASS)
        weight_C_list = []  # grad_weight matrices

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

                # Input gradient: dX = dY @ W
                # CUTLASS: dY @ (W^T)^T where W^T is passed as B
                weight_for_input = expert_weight.t().contiguous()  # [K, N]
                input_A_list.append(expert_grad_output)
                input_B_list.append(weight_for_input)
                input_C_list.append(expert_grad_input)

                # Weight gradient: dW = dY^T @ X
                # CUTLASS: dY^T @ (X^T)^T where X^T is passed as B
                grad_output_T = expert_grad_output.t().contiguous()  # [N, M]
                input_for_weight = expert_input.t().contiguous()  # [K, M]
                weight_A_list.append(grad_output_T)
                weight_B_list.append(input_for_weight)
                weight_C_list.append(expert_grad_weight)

        # Execute grouped operations
        if input_A_list:
            strategy.execute_grouped_gemm(
                input_A_list, input_B_list, input_C_list, "input_gradient"
            )

        if weight_A_list:
            strategy.execute_grouped_gemm(
                weight_A_list, weight_B_list, weight_C_list, "weight_gradient"
            )


class CUTLASSGroupedLinear(nn.Module):
    """
    CUTLASS-accelerated grouped linear layer for expert-based models.

    Performs grouped linear transformations Y_i = X_i @ W_i^T for each expert i,
    with automatic differentiation support for both forward and backward passes.

    Usage:
        layer = CUTLASSGroupedLinear(
            num_experts=8,
            in_features=4096,
            out_features=11008,
            strategy=CUTLASSGroupGemmStrategy()
        )

        output = layer(input_tokens, expert_assignments)
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        strategy: CUTLASSGroupGemmStrategy,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize CUTLASS grouped linear layer.

        Args:
            num_experts: Number of experts
            in_features: Input feature dimension
            out_features: Output feature dimension
            strategy: CUTLASS strategy instance
            bias: Whether to include bias (not supported yet)
            dtype: Parameter data type
        """
        super().__init__()

        if bias:
            raise NotImplementedError("Bias not yet supported")

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.strategy = strategy
        self.dtype = dtype

        # Initialize expert weights [num_experts, out_features, in_features]
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
        Forward pass through grouped linear layer.

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

        # Apply grouped GEMM
        sorted_output = CUTLASSGroupGemm.apply(
            sorted_tokens, self.weight, m_sizes, m_offsets, self.strategy
        )

        # Restore original token order
        output = torch.empty_like(sorted_output)
        output[sorted_indices] = sorted_output

        return output

    def _compute_expert_sizes_and_offsets(
        self, expert_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute number of tokens per expert and their offsets.

        Args:
            expert_assignments: Expert assignment per token [total_tokens]

        Returns:
            m_sizes: Tokens per expert [num_experts]
            m_offsets: Cumulative token offsets [num_experts + 1]
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
        """Return string representation of module parameters"""
        return f"num_experts={self.num_experts}, in_features={self.in_features}, out_features={self.out_features}"


def test_cutlass_group_gemm():
    """Test the complete CUTLASS group GEMM implementation"""
    print("🧪 Testing Complete CUTLASS Group GEMM")
    print("=" * 50)

    if not HAS_CUTLASS:
        print("❌ CUTLASS not available")
        return False

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test configuration
    num_experts = 4
    in_features = 512
    out_features = 1024
    total_tokens = 256

    print(
        f"Configuration: {num_experts} experts, {total_tokens} tokens, {in_features}→{out_features}"
    )

    # Create strategy and layer
    strategy = CUTLASSGroupGemmStrategy(
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )

    layer = CUTLASSGroupedLinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        strategy=strategy,
        dtype=dtype,
    ).to(device)

    # Create test data
    input_tokens = torch.randn(
        total_tokens, in_features, dtype=dtype, device=device, requires_grad=True
    )
    expert_assignments = torch.randint(0, num_experts, (total_tokens,), device=device)

    print(f"Input: {input_tokens.shape}")
    print(f"Expert assignments: {expert_assignments.shape}")
    print(
        f"Expert distribution: {[torch.sum(expert_assignments == i).item() for i in range(num_experts)]}"
    )

    try:
        # Forward pass
        print("\n🔍 Forward Pass")
        output = layer(input_tokens, expert_assignments)
        print(f"Output: {output.shape}, norm: {output.norm().item():.4f}")

        # Backward pass
        print("\n🔍 Backward Pass")
        loss = output.sum()
        loss.backward()

        # Check gradients
        input_grad_norm = (
            input_tokens.grad.norm().item() if input_tokens.grad is not None else 0
        )
        weight_grad_norm = (
            layer.weight.grad.norm().item() if layer.weight.grad is not None else 0
        )

        print(f"Input gradient norm: {input_grad_norm:.4f}")
        print(f"Weight gradient norm: {weight_grad_norm:.4f}")

        # Validate gradients exist and are reasonable
        success = (
            input_tokens.grad is not None
            and layer.weight.grad is not None
            and input_grad_norm > 0
            and weight_grad_norm > 0
            and torch.isfinite(input_tokens.grad).all()
            and torch.isfinite(layer.weight.grad).all()
        )

        if success:
            print("✅ CUTLASS Group GEMM test passed!")
        else:
            print("❌ CUTLASS Group GEMM test failed!")

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
) -> CUTLASSGroupGemmStrategy:
    """
    Convenience function to create a CUTLASS strategy.

    Args:
        use_2cta_instrs: Whether to use 2-CTA instructions
        mma_tiler_mn: MMA tile sizes
        cluster_shape_mn: Cluster shape

    Returns:
        Configured CUTLASS strategy
    """
    return CUTLASSGroupGemmStrategy(
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


if __name__ == "__main__":
    test_cutlass_group_gemm()
