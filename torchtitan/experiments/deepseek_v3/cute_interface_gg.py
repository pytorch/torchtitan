"""
"Improved" CUTLASS Group GEMM Strategy using the PyTorch to CUTE converter.

This version leverages the standalone converter classes to simplify tensor conversion
and metadata preparation, making the code more maintainable and less error-prone.
"""

from typing import Any, Dict, List, Tuple

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


import logging

from torchtitan.experiments.kernels.blackwell.pytorch_cute_converter import (
    GroupedGemmTensorManager,
    PyTorchToCuteConverter,
)

logger = logging.getLogger(__name__)


class ImprovedCUTLASSGroupedGemmStrategy:
    """
    Improved CUTLASS grouped GEMM strategy using converter classes.

    This version provides cleaner code with better separation of concerns:
    - Tensor conversion is handled by dedicated converter classes
    - Reduced boilerplate and manual tensor manipulation
    - Better error handling and validation
    - More maintainable codebase
    """

    # Configuration constants
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
    N_SIZE_RANGE = range(32, 257, 32)

    DTYPE_TORCH = torch.bfloat16
    DTYPE_CUTLASS = cutlass.BFloat16
    ACC_DTYPE = cutlass.Float32
    ALIGNMENT = 16

    def __init__(
        self,
        custom_activation,
        use_2cta_instrs: bool = True,
        mma_tiler_mn: Tuple[int, int] = (256, 128),
        cluster_shape_mn: Tuple[int, int] = (4, 4),
    ):
        """
        Initialize the improved CUTLASS grouped GEMM strategy.

        Args:
            custom_activation: Activation function (e.g., SiLU)
            use_2cta_instrs: Whether to use 2-CTA instructions
            mma_tiler_mn: MMA tile sizes (M, N)
            cluster_shape_mn: Cluster shape (M, N)
        """
        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        self.activation_function = custom_activation
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn or self._get_default_mma_tiler()
        self.cluster_shape_mn = cluster_shape_mn or self._get_default_cluster_shape()

        # Initialize converter and tensor manager
        self.converter = PyTorchToCuteConverter(
            default_alignment=self.ALIGNMENT, default_acc_dtype=self.ACC_DTYPE
        )
        self.tensor_manager = GroupedGemmTensorManager(
            alignment=self.ALIGNMENT, dtype=self.DTYPE_TORCH
        )

        # Initialize CUTLASS components
        self._initialize_kernel()
        self._initialize_hardware()

        # Caches
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        self._log_initialization()

    def _get_default_mma_tiler(self) -> Tuple[int, int]:
        """Get default MMA tiler based on CTA mode."""
        return (256, 128) if self.use_2cta_instrs else (128, 128)

    def _get_default_cluster_shape(self) -> Tuple[int, int]:
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
        """Initialize hardware information and CUDA stream."""
        # TODO - if we do not have a cuda context, this will fail...
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

    def _log_initialization(self):
        """Log initialization details."""
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        print(f"✅ Improved CUTLASS Strategy initialized:")
        print(f"   - 2 CTA mode: {self.use_2cta_instrs}")
        print(f"   - MMA tiler: {self.mma_tiler_mn}")
        print(f"   - Cluster shape: {self.cluster_shape_mn}")
        print(f"   - Max active clusters: {self.max_active_clusters}")

    def arrange_expert_weights(
        self, all_weights: List[torch.Tensor], submod_name: str, module
    ) -> torch.Tensor:
        """Store weights in stacked format."""
        # TODO - let's pre-transsose...
        return torch.stack(all_weights)

    def execute(
        self,
        contig_tokens: torch.Tensor,
        m_sizes: torch.Tensor,
        m_offsets: torch.Tensor,
        module,
    ) -> torch.Tensor:
        """
        Execute grouped GEMM operation using improved tensor management.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Expert sizes tensor
            m_offsets: Expert offsets tensor
            module: MoE module containing weights

        Returns:
            Processed output tokens
        """
        # Ensure GPU tensors to avoid CPU-GPU sync
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        # Get weights and validate
        weights = self._get_and_validate_weights(module)
        device = contig_tokens.device

        # Early exit if no valid experts
        if not self._has_valid_experts(m_sizes_gpu):
            return torch.zeros(
                contig_tokens.shape[0],
                weights["gate"].shape[2],
                dtype=self.DTYPE_TORCH,
                device=device,
            )

        # Execute three-stage MoE computation
        return self._execute_moe_computation(
            contig_tokens, weights, m_sizes_gpu, m_offsets_gpu, device
        )

    def _ensure_gpu_tensors(
        self, m_sizes, m_offsets, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure sizes and offsets are GPU tensors."""
        if not isinstance(m_sizes, torch.Tensor):
            m_sizes_gpu = torch.tensor(m_sizes, dtype=torch.int32, device=device)
        else:
            m_sizes_gpu = m_sizes.to(device=device, dtype=torch.int32)

        if not isinstance(m_offsets, torch.Tensor):
            m_offsets_gpu = torch.tensor(m_offsets, dtype=torch.int32, device=device)
        else:
            m_offsets_gpu = m_offsets.to(device=device, dtype=torch.int32)

        return m_sizes_gpu, m_offsets_gpu

    def _get_and_validate_weights(self, module) -> Dict[str, torch.Tensor]:
        """Extract and validate weight tensors."""
        required_weights = ["gate_proj_weight", "up_proj_weight", "down_proj_weight"]
        weights = {}

        for weight_name in required_weights:
            if not hasattr(module, weight_name):
                raise ValueError(f"Module missing required weight: {weight_name}")
            weights[weight_name.split("_")[0]] = module.get_parameter(weight_name)

        return weights

    def _has_valid_experts(self, m_sizes_gpu: torch.Tensor) -> bool:
        """Check if any experts have tokens (single sync point)."""
        return torch.any(m_sizes_gpu > 0).item()

    def _execute_moe_computation(
        self,
        contig_tokens: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Execute the complete MoE computation pipeline."""
        print(f"⚙️  Executing MoE computation on {device}")
        print(f"Stage 1: Gate and Up projections")
        if m_sizes_gpu.requires_grad:
            m_sizes_gpu = m_sizes_gpu.detach()
            m_offsets_gpu = m_offsets_gpu.detach()

        # Stage 1: Gate and Up projections
        gate_outputs, up_outputs = self._execute_gate_up_projections(
            contig_tokens,
            weights["gate"].detach(),
            weights["up"].detach(),
            m_sizes_gpu,
            m_offsets_gpu,
            device,
        )

        print(f"Stage 2: Apply activation and combine")
        # Stage 2: Apply activation and combine
        hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

        # Stage 3: Down projection
        print(f"Stage 3: Down projection")
        down_outputs = self._execute_down_projection(
            hidden_states, weights["down"].detach(), m_sizes_gpu, device
        )

        # Stage 4: Reconstruct output
        print(f"Stage 4: Reconstruct output")
        return self._reconstruct_output(
            down_outputs, contig_tokens, m_sizes_gpu, m_offsets_gpu
        )

    def _execute_gate_up_projections(
        self,
        input_tokens: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        device: torch.device,
    ) -> Tuple[List, List]:
        """Execute gate and up projections using the tensor manager."""

        # Get valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return [], []

        # Prepare expert operations using tensor manager
        gate_ops, up_ops = self._prepare_gate_up_operations(
            input_tokens,
            gate_weights,
            up_weights,
            m_sizes_gpu,
            m_offsets_gpu,
            valid_indices,
            device,
        )

        # Execute grouped GEMMs
        if gate_ops["inputs"]:
            self._execute_grouped_gemm_operations(gate_ops, device, "gate_up")

        return gate_ops["outputs"], up_ops["outputs"]

    def _prepare_gate_up_operations(
        self,
        input_tokens: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        valid_indices: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Dict, Dict]:
        """Prepare gate and up operations using the tensor manager."""

        # Convert indices for iteration (minimal sync)
        valid_indices_cpu = valid_indices.cpu().tolist()
        valid_sizes = m_sizes_gpu[valid_indices].cpu().tolist()
        valid_offsets = (
            self._compute_valid_offsets(
                m_sizes_gpu, m_offsets_gpu, valid_indices, device
            )
            .cpu()
            .tolist()
        )

        # Prepare operation lists
        gate_ops = {"inputs": [], "weights": [], "outputs": [], "metadata": []}
        up_ops = {"inputs": [], "weights": [], "outputs": [], "metadata": []}

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes, valid_offsets
        ):
            if size > 0:
                # Get expert data
                expert_input = input_tokens[offset : offset + size].contiguous()
                gate_weight = gate_weights[expert_idx].contiguous()
                up_weight = up_weights[expert_idx].contiguous()

                # Create output tensors
                M, K = expert_input.shape
                N = gate_weight.shape[0]  # Assuming [out_features, in_features]

                gate_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                up_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Use tensor manager to prepare operations
                for ops, weight, output in [
                    (gate_ops, gate_weight, gate_output),
                    (up_ops, up_weight, up_output),
                ]:

                    (
                        cute_input,
                        cute_weight,
                        cute_output,
                        problem_size,
                        strides,
                        ptrs,
                    ) = self.tensor_manager.prepare_expert_operation(
                        expert_input, weight, output, transpose_weight=True
                    )

                    ops["inputs"].append(expert_input)
                    ops["weights"].append(weight)
                    ops["outputs"].append(output)
                    ops["metadata"].append((problem_size, strides, ptrs))

        return gate_ops, up_ops

    def _compute_valid_offsets(
        self,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        valid_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute valid offsets for expert operations."""
        valid_sizes = m_sizes_gpu[valid_indices]

        if len(m_offsets_gpu) > len(valid_indices):
            return m_offsets_gpu[valid_indices]
        else:
            return torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )

    def _execute_down_projection(
        self,
        hidden_states: List[torch.Tensor],
        down_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Execute down projection using tensor manager."""

        if not hidden_states:
            return []

        # Get valid expert indices
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Prepare down projection operations
        down_ops = {"inputs": [], "weights": [], "outputs": [], "metadata": []}

        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < len(hidden_states):
                hidden = hidden_states[i]
                down_weight = down_weights[expert_idx].contiguous()

                M, K = hidden.shape
                N = down_weight.shape[0]
                down_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Use tensor manager
                cute_input, cute_weight, cute_output, problem_size, strides, ptrs = (
                    self.tensor_manager.prepare_expert_operation(
                        hidden, down_weight, down_output, transpose_weight=True
                    )
                )

                down_ops["inputs"].append(hidden)
                down_ops["weights"].append(down_weight)
                down_ops["outputs"].append(down_output)
                down_ops["metadata"].append((problem_size, strides, ptrs))

        # Execute grouped GEMM
        if down_ops["inputs"]:
            self._execute_grouped_gemm_operations(down_ops, device, "down")

        return down_ops["outputs"]

    def _execute_grouped_gemm_operations(
        self, operations: Dict, device: torch.device, stage_name: str
    ):
        """Execute grouped GEMM operations using converter."""

        if not operations["metadata"]:
            return

        # Extract metadata
        all_problem_sizes = []
        all_strides = []
        all_ptrs = []

        for problem_size, strides, ptrs in operations["metadata"]:
            all_problem_sizes.append(problem_size)
            all_strides.append(strides)
            all_ptrs.append(ptrs)

        # Create CUTE metadata tensors using converter
        problem_sizes_cute, strides_cute, ptrs_cute = (
            self.converter.create_metadata_tensors(
                all_problem_sizes, all_strides, all_ptrs, device
            )
        )

        # Get other required tensors
        num_groups = len(all_problem_sizes)
        total_clusters = self._compute_total_clusters(all_problem_sizes)
        tensormap_cute = self._get_tensormap_buffer(device)

        # Create initial tensors for compilation using converter
        initial_tensors = self.converter.create_initial_compilation_tensors(
            tuple(all_problem_sizes[0]), device, self.DTYPE_TORCH
        )

        # Get or Compile kernel
        compiled_kernel = self._get_or_compile_kernel(
            num_groups,
            total_clusters,
            initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
        )

        # Execute
        compiled_kernel(
            *initial_tensors,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            self.stream,
        )
        torch.cuda.synchronize()

    def _get_or_compile_kernel(
        self,
        num_groups: int,
        total_clusters: int,
        initial_tensors: List,
        problem_sizes_cute,
        strides_cute,
        ptrs_cute,
        tensormap_cute,
    ):
        """Get compiled kernel from cache or compile new one."""

        cache_key = (
            num_groups,
            total_clusters,
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
        )

        if cache_key not in self._compiled_kernels:
            print(f"Compiling kernel: {num_groups} groups, 2CTA={self.use_2cta_instrs}")

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
            print("✅ Kernel compilation successful")

        return self._compiled_kernels[cache_key]

    def _get_tensormap_buffer(self, device: torch.device):
        """Get tensormap buffer using converter."""
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            self._tensormap_buffers[device] = self.converter.create_tensormap_buffer(
                device, sm_count, tensormap_count=3, tensormap_bytes=128
            )
        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes: List[List[int]]) -> int:
        """Compute total clusters needed for all problems."""
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

    def _apply_activation_and_combine(
        self, gate_outputs: List[torch.Tensor], up_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply activation function and combine gate/up outputs."""
        if not gate_outputs or not up_outputs:
            return []

        return [
            self.activation_function(gate_out) * up_out
            for gate_out, up_out in zip(gate_outputs, up_outputs)
        ]

    def _reconstruct_output(
        self,
        down_outputs: List[torch.Tensor],
        contig_tokens: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the full output tensor."""

        # Initialize output
        output = torch.zeros(
            contig_tokens.shape[0],
            down_outputs[0].shape[1] if down_outputs else contig_tokens.shape[1],
            dtype=self.DTYPE_TORCH,
            device=contig_tokens.device,
        )

        if not down_outputs:
            return output

        # Get valid expert information
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices].cpu().tolist()
        valid_offsets = (
            self._compute_valid_offsets(
                m_sizes_gpu, m_offsets_gpu, valid_indices, contig_tokens.device
            )
            .cpu()
            .tolist()
        )

        # Copy results back
        for i, (size, offset) in enumerate(zip(valid_sizes, valid_offsets)):
            if i < len(down_outputs) and size > 0:
                output[offset : offset + size] = down_outputs[i]

        return output

    @staticmethod
    def is_available() -> bool:
        """Check if CUTLASS is available."""
        return HAS_CUTLASS


# Factory function for easy creation
def create_improved_cutlass_strategy(
    custom_activation,
    use_2cta_instrs: bool = True,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (4, 4),
) -> ImprovedCUTLASSGroupedGemmStrategy:
    """
    Factory function to create improved CUTLASS strategy.

    Args:
        custom_activation: Activation function
        use_2cta_instrs: Use 2-CTA instructions
        mma_tiler_mn: MMA tile sizes
        cluster_shape_mn: Cluster shape

    Returns:
        Configured strategy instance
    """
    return ImprovedCUTLASSGroupedGemmStrategy(
        custom_activation=custom_activation,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )


# Test function
def test_improved_strategy():
    """Test the improved CUTLASS strategy."""
    if not HAS_CUTLASS:
        print("❌ CUTLASS not available for testing")
        return False

    print("Testing Improved CUTLASS Strategy")
    print("=" * 50)

    # note - we have to make a pytorch cuda context or this will fail
    dummy_tensor = torch.randn(1, 1, device="cuda")
    a = dummy_tensor.to("cpu").item()

    try:
        import torch.nn.functional as F

        device = torch.device("cuda")
        dtype = torch.bfloat16

        # Test parameters
        num_experts = 8
        in_features = 2048
        out_features = 4096
        intermediate_size = 8192
        total_tokens = 1024

        # Create strategy
        strategy = create_improved_cutlass_strategy(
            custom_activation=F.silu,
            use_2cta_instrs=True,
            mma_tiler_mn=(256, 128),
            cluster_shape_mn=(2, 2),
        )

        # Create mock module with weights
        class MockModule:
            def __init__(self):
                self.gate_proj_weight = torch.randn(
                    num_experts,
                    intermediate_size,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
                self.up_proj_weight = torch.randn(
                    num_experts,
                    intermediate_size,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
                self.down_proj_weight = torch.randn(
                    num_experts,
                    out_features,
                    intermediate_size,
                    dtype=dtype,
                    device=device,
                )

            def get_parameter(self, name):
                return getattr(self, name)

        module = MockModule()

        # Create test data
        contig_tokens = torch.randn(
            total_tokens, in_features, dtype=dtype, device=device
        )
        expert_assignments = torch.randint(
            0, num_experts, (total_tokens,), device=device
        )

        # Compute expert sizes and offsets
        m_sizes = torch.zeros(num_experts, dtype=torch.int32, device=device)
        for expert_idx in range(num_experts):
            m_sizes[expert_idx] = (expert_assignments == expert_idx).sum()

        m_offsets = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(m_sizes, dim=0)]
        )

        print(f"Expert sizes: {m_sizes.cpu().tolist()}")

        # Execute strategy
        print("Executing improved CUTLASS strategy...")
        output = strategy.execute(contig_tokens, m_sizes, m_offsets, module)

        print(f"✅ Execution successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output norm: {output.norm().item():.4f}")
        print(f"   Output dtype: {output.dtype}")

        # Validate output
        assert output.shape == (total_tokens, out_features)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()

        print("✅ All validations passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_improved_strategy()
