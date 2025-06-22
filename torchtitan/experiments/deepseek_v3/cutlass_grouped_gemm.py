"""
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    Optimized version with pre-transposed weights to eliminate runtime transpose operations.
"""

"""

"""

# Disable file caching while keeping in-memory cache available, defaults to False.
# export CUTE_DSL_DISABLE_FILE_CACHING=True

# Maximum number of cache files allowed, defaults to 1000.
# export CUTE_DSL_FILE_CACHING_CAPACITY=1000

import logging
from typing import Any, Dict, List, Tuple

import torch


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
    print(f"✗ CUTLASS import failed: {e}")
    print("CUTLASSGroupedGemmStrategy will not be available")

from torchtitan.experiments.kernels.blackwell.pytorch_cute_converter import (
    GroupedGemmTensorManager,
    PyTorchToCuteConverter,
)

# Import base class - adjust path as needed based on your project structure
from .group_gemms import GroupGEMMStrategy

logger = logging.getLogger(__name__)


# ================== current working version ==================


class CUTLASSGroupedGemmStrategy(GroupGEMMStrategy):
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    This version eliminates CPU-GPU synchronization by keeping all size/offset computations on GPU.
    """

    # Constants (same as before)
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
        custom_activation,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(4, 4),
    ):
        """Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture."""
        super().__init__(custom_activation)
        self.use_2cta_instrs = use_2cta_instrs

        # Set configuration
        self.mma_tiler_mn = mma_tiler_mn or self._get_default_mma_tiler()
        self.cluster_shape_mn = cluster_shape_mn or self._get_default_cluster_shape()

        # Validate configurations
        # self._validate_configurations()

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
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

    def _validate_configurations(self):
        """Validate configurations for Blackwell."""
        self._validate_mma_tiler()
        self._validate_cluster_shape()
        self._validate_2cta_constraints()

    def _validate_mma_tiler(self):
        """Validate MMA tiler configuration."""
        m_size, n_size = self.mma_tiler_mn

        valid_m_sizes = (
            self.DUAL_CTA_M_SIZES if self.use_2cta_instrs else self.SINGLE_CTA_M_SIZES
        )
        mode_name = "2 CTA" if self.use_2cta_instrs else "single CTA"

        if m_size not in valid_m_sizes:
            raise ValueError(
                f"For {mode_name} mode on Blackwell, MMA tiler M must be in {valid_m_sizes}, got {m_size}"
            )

        if n_size not in self.N_SIZE_RANGE:
            raise ValueError(
                f"MMA tiler N must be in range [32, 256] with step 32, got {n_size}"
            )

    def _validate_cluster_shape(self):
        """Validate cluster shape configuration."""
        if self.cluster_shape_mn not in self.SUPPORTED_CLUSTER_SHAPES:
            raise ValueError(
                f"Cluster shape {self.cluster_shape_mn} not supported on Blackwell. "
                f"Valid cluster shapes are: {self.SUPPORTED_CLUSTER_SHAPES}"
            )

    def _validate_2cta_constraints(self):
        """Validate 2 CTA specific constraints."""
        if self.use_2cta_instrs and self.cluster_shape_mn[0] % 2 != 0:
            valid_2cta_shapes = [
                shape for shape in self.SUPPORTED_CLUSTER_SHAPES if shape[0] % 2 == 0
            ]
            raise ValueError(
                f"For 2 CTA mode, cluster shape M must be even, got {self.cluster_shape_mn[0]}. "
                f"Valid 2 CTA cluster shapes: {valid_2cta_shapes}"
            )

    def _log_initialization(self):
        """Log initialization information."""
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        print(f"Initialized CUTLASSGroupedGemmStrategy for Blackwell with:")
        print(f"  - 2 CTA instructions: {self.use_2cta_instrs}")
        print(f"  - MMA tiler (M, N): {self.mma_tiler_mn}")
        print(f"  - Cluster shape (M, N): {self.cluster_shape_mn}")
        print(f"  - Cluster size: {cluster_size}")
        if cluster_size > 1:
            print(f"  - Using multi-CTA parallelism")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in stacked format."""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute using CUTLASS grouped GEMM kernel - GPU-only version.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing weights
        """
        # Convert to GPU tensors if needed (avoid CPU-GPU sync)
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        # Validate inputs
        # self._validate_inputs(contig_tokens, m_sizes_gpu, module)

        # Get weights and device
        weights = self._get_weights(module)
        device = contig_tokens.device

        # Prepare output tensor
        output = torch.zeros(
            contig_tokens.shape[0],
            weights["gate"].shape[2],
            dtype=self.DTYPE_TORCH,
            device=device,
        )

        # Check for valid experts using GPU operations (no sync)
        if not self._has_valid_experts_gpu(m_sizes_gpu):
            return output

        # Execute the three-stage computation using GPU-only operations
        gate_outputs, up_outputs = self._execute_projections_gpu(
            contig_tokens,
            weights["gate"],
            weights["up"],
            m_sizes_gpu,
            m_offsets_gpu,
            device,
        )

        hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

        final_outputs = self._execute_down_projection_gpu(
            hidden_states, weights["down"], m_sizes_gpu, device
        )

        return self._reconstruct_output_gpu(
            final_outputs, m_sizes_gpu, m_offsets_gpu, output
        )

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

    def _has_valid_experts_gpu(self, m_sizes_gpu):
        """Check if any experts have tokens using GPU operations (no sync)."""
        return torch.any(
            m_sizes_gpu > 0
        ).item()  # Single sync here is unavoidable for control flow

    def _validate_inputs(self, contig_tokens, m_sizes_gpu, module):
        """Validate input parameters."""
        if contig_tokens.dtype != self.DTYPE_TORCH:
            raise ValueError(
                f"Expected input dtype {self.DTYPE_TORCH}, got {contig_tokens.dtype}"
            )

        if len(contig_tokens.shape) != 2:
            raise ValueError(
                f"Expected 2D input tensor, got shape {contig_tokens.shape}"
            )

        required_params = ["gate_proj_weight", "up_proj_weight", "down_proj_weight"]
        for param in required_params:
            if not hasattr(module, param) or module.get_parameter(param) is None:
                raise ValueError(f"Module missing required parameter: {param}")

    def _get_weights(self, module):
        """Extract and return weight tensors from module."""
        return {
            "gate": module.get_parameter("gate_proj_weight"),
            "up": module.get_parameter("up_proj_weight"),
            "down": module.get_parameter("down_proj_weight"),
        }

    def _execute_projections_gpu(
        self, input_tokens, weight1, weight2, m_sizes_gpu, m_offsets_gpu, device
    ):
        """Execute gate and up projections using GPU-only operations."""
        # Find valid experts using GPU operations
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return [], []

        # Prepare metadata in batch using GPU operations
        problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs = (
            self._prepare_gate_up_metadata_gpu(
                input_tokens,
                weight1,
                weight2,
                m_sizes_gpu,
                m_offsets_gpu,
                valid_indices,
                device,
            )
        )

        if len(problem_sizes) == 0:
            return [], []

        # Execute grouped GEMM
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return gate_outputs, up_outputs

    def _prepare_gate_up_metadata_gpu(
        self,
        input_tokens,
        gate_weights,
        up_weights,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        device,
    ):
        """Prepare metadata for gate and up projections"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        gate_outputs = []
        up_outputs = []

        # Extract valid sizes and offsets (minimal sync - only for valid experts)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )
        )

        # Convert to Python for iteration (unavoidable in this test for metadata preparation)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, (expert_idx, size, offset) in enumerate(
            zip(valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu)
        ):
            if size > 0:
                # Get expert data
                expert_tokens = input_tokens[offset : offset + size].contiguous()
                gate_weight = gate_weights[expert_idx].contiguous()
                up_weight = up_weights[expert_idx].contiguous()

                M, K = expert_tokens.shape
                N = gate_weight.shape[0]
                L = 1

                # Create output tensors
                gate_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                up_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Add both projections to metadata
                for weight, output, output_list in [
                    (gate_weight, gate_output, gate_outputs),
                    (up_weight, up_output, up_outputs),
                ]:
                    self._add_projection_to_metadata(
                        expert_tokens,
                        weight,
                        output,
                        problem_sizes,
                        strides_abc,
                        ptrs_abc,
                    )
                    output_list.append(output)

        return problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs

    def _execute_down_projection_gpu(
        self, hidden_states, down_weights, m_sizes_gpu, device
    ):
        """Execute down projection using GPU operations."""
        if not hidden_states:
            return []

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        # Prepare metadata
        problem_sizes, strides_abc, ptrs_abc, down_outputs = (
            self._prepare_down_metadata_gpu(
                hidden_states, down_weights, valid_indices, device
            )
        )

        if len(problem_sizes) == 0:
            return []

        # Execute grouped GEMM
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return down_outputs

    def _prepare_down_metadata_gpu(
        self, hidden_states, down_weights, valid_indices, device
    ):
        """Prepare metadata for down projection using GPU operations."""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        down_outputs = []

        # Convert indices to CPU for iteration (minimal sync)
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < len(hidden_states):
                hidden = hidden_states[i]
                down_weight = down_weights[expert_idx].contiguous()

                M, K = hidden.shape
                N = down_weight.shape[0]

                # Create output tensor
                down_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                down_outputs.append(down_output)

                # Add to metadata
                self._add_projection_to_metadata(
                    hidden,
                    down_weight,
                    down_output,
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                )

        return problem_sizes, strides_abc, ptrs_abc, down_outputs

    def _add_projection_to_metadata(
        self,
        input_tensor,
        weight_tensor,
        output_tensor,
        problem_sizes,
        strides_abc,
        ptrs_abc,
    ):
        """Add a single projection to the metadata lists."""
        M, K = input_tensor.shape
        N = weight_tensor.shape[0]
        L = 1

        # Convert to MNKL format
        input_mnkl = input_tensor.unsqueeze(-1).contiguous()
        weight_mnkl = weight_tensor.unsqueeze(-1).contiguous()
        output_mnkl = output_tensor.unsqueeze(-1).contiguous()

        # Extract strides
        input_strides = list(input_mnkl.stride()[:2])
        weight_strides = list(weight_mnkl.stride()[:2])
        output_strides = list(output_mnkl.stride()[:2])

        # Add to metadata
        problem_sizes.append([M, N, K, L])
        strides_abc.append([input_strides, weight_strides, output_strides])
        ptrs_abc.append(
            [
                input_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                output_tensor.data_ptr(),
            ]
        )

    def _execute_grouped_gemm(self, problem_sizes, strides_abc, ptrs_abc, device):
        """Execute the grouped GEMM kernel."""
        num_groups = len(problem_sizes)

        # Convert to CUTE tensors
        problem_sizes_cute, strides_cute, ptrs_cute = self._convert_to_cute_tensors(
            problem_sizes, strides_abc, ptrs_abc, device
        )

        # Get tensormap and compute clusters
        tensormap_cute = self._get_tensormap_buffer(device)
        total_clusters = self._compute_total_clusters(problem_sizes)

        # Get initial tensors for compilation
        initial_tensors = self._create_initial_tensors(problem_sizes[0], device)

        # Compile or retrieve kernel
        compiled_kernel = self._get_compiled_kernel(
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
            self.stream,
        )
        # torch.cuda.synchronize()

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

    def _apply_activation_and_combine(self, gate_outputs, up_outputs):
        """Apply activation and combine gate/up outputs."""
        return [
            self.activation_function(gate_out) * up_out
            for gate_out, up_out in zip(gate_outputs, up_outputs)
        ]

    def _reconstruct_output_gpu(
        self, final_outputs, m_sizes_gpu, m_offsets_gpu, output
    ):
        """Reconstruct the full output tensor using GPU operations (minimal sync)."""
        if not final_outputs:
            return output

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices]

        # Compute offsets if not provided properly
        if len(m_offsets_gpu) <= len(valid_indices):
            valid_offsets = torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=m_sizes_gpu.device), valid_sizes[:-1]]
                ),
                dim=0,
            )
        else:
            valid_offsets = m_offsets_gpu[valid_indices]

        # Convert to CPU for final reconstruction (minimal sync)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()

        for i, (size, offset) in enumerate(zip(valid_sizes_cpu, valid_offsets_cpu)):
            if i < len(final_outputs):
                output[offset : offset + size] = final_outputs[i]

        return output

    @staticmethod
    def is_available() -> bool:
        return HAS_CUTLASS


# ================== end, current working version ==================


class CUTLASSGroupedGemmStrategy_pre_transpose(GroupGEMMStrategy):
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    Optimized version with pre-transposed weights - based exactly on working "_prev" version
    with ONLY the transpose optimization added.
    """

    # Constants (same as before)
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
        custom_activation,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(4, 4),
    ):
        """Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture."""
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
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

    def _log_initialization(self):
        """Log initialization information."""
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        print(f"Initialized CUTLASSGroupedGemmStrategy for Blackwell with:")
        print(f"  - 2 CTA instructions: {self.use_2cta_instrs}")
        print(f"  - MMA tiler (M, N): {self.mma_tiler_mn}")
        print(f"  - Cluster shape (M, N): {self.cluster_shape_mn}")
        print(f"  - Cluster size: {cluster_size}")
        print(f"  - Weight optimization: Pre-transposed (no runtime transpose)")
        if cluster_size > 1:
            print(f"  - Using multi-CTA parallelism")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in stacked format with pre-transposition optimization."""
        # Pre-transpose weights to eliminate runtime .t() calls
        transposed_weights = []
        for weight in all_weights:
            transposed_weights.append(weight.t().contiguous())
        return torch.stack(transposed_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute using CUTLASS grouped GEMM kernel - GPU-only version.
        EXACT copy of working version except weights are pre-transposed.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing weights
        """
        # Convert to GPU tensors if needed (avoid CPU-GPU sync)
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        # Get weights and device
        weights = self._get_weights(module)
        device = contig_tokens.device

        # Prepare output tensor - adjust for pre-transposed weights
        # Final output size should be hidden_size (2048)
        # In pre-transposed down weights: [num_experts, intermediate_size, hidden_size]
        # So shape[2] gives us hidden_size
        output = torch.zeros(
            contig_tokens.shape[0],
            weights["down"].shape[2],  # hidden_size from pre-transposed down weights
            dtype=self.DTYPE_TORCH,
            device=device,
        )

        # Check for valid experts using GPU operations (no sync)
        if not self._has_valid_experts_gpu(m_sizes_gpu):
            return output

        # Execute the three-stage computation using GPU-only operations
        gate_outputs, up_outputs = self._execute_projections_gpu(
            contig_tokens,
            weights["gate"],
            weights["up"],
            m_sizes_gpu,
            m_offsets_gpu,
            device,
        )

        hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

        final_outputs = self._execute_down_projection_gpu(
            hidden_states, weights["down"], m_sizes_gpu, device
        )

        return self._reconstruct_output_gpu(
            final_outputs, m_sizes_gpu, m_offsets_gpu, output
        )

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

    def _has_valid_experts_gpu(self, m_sizes_gpu):
        """Check if any experts have tokens using GPU operations (no sync)."""
        return torch.any(
            m_sizes_gpu > 0
        ).item()  # Single sync here is unavoidable for control flow

    def _get_weights(self, module):
        """Extract and return weight tensors from module."""
        return {
            "gate": module.get_parameter("gate_proj_weight"),
            "up": module.get_parameter("up_proj_weight"),
            "down": module.get_parameter("down_proj_weight"),
        }

    def _execute_projections_gpu(
        self, input_tokens, weight1, weight2, m_sizes_gpu, m_offsets_gpu, device
    ):
        """Execute gate and up projections using GPU-only operations."""
        # Find valid experts using GPU operations
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return [], []

        # Prepare metadata in batch using GPU operations
        problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs = (
            self._prepare_gate_up_metadata_gpu(
                input_tokens,
                weight1,
                weight2,
                m_sizes_gpu,
                m_offsets_gpu,
                valid_indices,
                device,
            )
        )

        if len(problem_sizes) == 0:
            return [], []

        # Execute grouped GEMM
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return gate_outputs, up_outputs

    def _prepare_gate_up_metadata_gpu(
        self,
        input_tokens,
        gate_weights,
        up_weights,
        m_sizes_gpu,
        m_offsets_gpu,
        valid_indices,
        device,
    ):
        """Prepare metadata for gate and up projections"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        gate_outputs = []
        up_outputs = []

        # Extract valid sizes and offsets (minimal sync - only for valid experts)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )
        )

        # Convert to Python for iteration (unavoidable in this test for metadata preparation)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, (expert_idx, size, offset) in enumerate(
            zip(valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu)
        ):
            if size > 0:
                # Get expert data - weights are now PRE-TRANSPOSED, no .t() needed!
                expert_tokens = input_tokens[offset : offset + size].contiguous()
                gate_weight = gate_weights[
                    expert_idx
                ].contiguous()  # Already transposed
                up_weight = up_weights[expert_idx].contiguous()  # Already transposed

                M, K = expert_tokens.shape
                # Pre-transposed gate/up weights: [hidden_size, intermediate_size]
                # So gate_weight.shape = [hidden_size, intermediate_size] = [2048, 1408]
                # We want N = intermediate_size for the output
                K_weight, N = (
                    gate_weight.shape
                )  # K_weight=2048 (hidden), N=1408 (intermediate)
                L = 1

                # Verify dimensions match for matrix multiplication
                if K != K_weight:
                    raise ValueError(
                        f"Dimension mismatch: expert_tokens {expert_tokens.shape} vs gate_weight {gate_weight.shape}"
                    )

                # Create output tensors with intermediate_size (N=1408)
                gate_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                up_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Add both projections to metadata
                for weight, output, output_list in [
                    (gate_weight, gate_output, gate_outputs),
                    (up_weight, up_output, up_outputs),
                ]:
                    self._add_projection_to_metadata(
                        expert_tokens,
                        weight,
                        output,
                        problem_sizes,
                        strides_abc,
                        ptrs_abc,
                    )
                    output_list.append(output)

        return problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs

    def _execute_down_projection_gpu(
        self, hidden_states, down_weights, m_sizes_gpu, device
    ):
        """Execute down projection using GPU operations."""
        if not hidden_states:
            return []

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        # Prepare metadata
        problem_sizes, strides_abc, ptrs_abc, down_outputs = (
            self._prepare_down_metadata_gpu(
                hidden_states, down_weights, valid_indices, device
            )
        )

        if len(problem_sizes) == 0:
            return []

        # Execute grouped GEMM
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return down_outputs

    def _prepare_down_metadata_gpu(
        self, hidden_states, down_weights, valid_indices, device
    ):
        """Prepare metadata for down projection using GPU operations."""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        down_outputs = []

        # Convert indices to CPU for iteration (minimal sync)
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < len(hidden_states):
                hidden = hidden_states[i]
                down_weight = down_weights[
                    expert_idx
                ].contiguous()  # Already transposed

                M, K = hidden.shape  # [batch, intermediate_size] = [12288, 1408]
                # Pre-transposed down weights: [intermediate_size, hidden_size] = [1408, 2048]
                K_weight, N = (
                    down_weight.shape
                )  # K_weight=1408 (intermediate), N=2048 (hidden)

                # Verify dimensions match for matrix multiplication
                if K != K_weight:
                    raise ValueError(
                        f"Dimension mismatch: hidden {hidden.shape} vs down_weight {down_weight.shape}"
                    )

                # Create output tensor with hidden_size (N=2048)
                down_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                down_outputs.append(down_output)

                # Add to metadata
                self._add_projection_to_metadata(
                    hidden,
                    down_weight,
                    down_output,
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                )

        return problem_sizes, strides_abc, ptrs_abc, down_outputs

    def _add_projection_to_metadata(
        self,
        input_tensor,
        weight_tensor,
        output_tensor,
        problem_sizes,
        strides_abc,
        ptrs_abc,
    ):
        """Add a single projection to the metadata lists with pre-transposed weights."""
        M, K = input_tensor.shape
        # Pre-transposed weights have shape [K, N] where K matches input's last dim
        K_weight, N = weight_tensor.shape
        L = 1

        # Verify dimension compatibility
        if K != K_weight:
            raise ValueError(
                f"Matrix multiplication dimension mismatch: input {input_tensor.shape} vs weight {weight_tensor.shape}"
            )

        # Convert to MNKL format
        input_mnkl = input_tensor.unsqueeze(-1).contiguous()
        weight_mnkl = weight_tensor.unsqueeze(-1).contiguous()
        output_mnkl = output_tensor.unsqueeze(-1).contiguous()

        # Extract strides
        input_strides = list(input_mnkl.stride()[:2])
        weight_strides = list(weight_mnkl.stride()[:2])
        output_strides = list(output_mnkl.stride()[:2])

        # Add to metadata
        problem_sizes.append([M, N, K, L])
        strides_abc.append([input_strides, weight_strides, output_strides])
        ptrs_abc.append(
            [
                input_tensor.data_ptr(),
                weight_tensor.data_ptr(),
                output_tensor.data_ptr(),
            ]
        )

    def _execute_grouped_gemm(self, problem_sizes, strides_abc, ptrs_abc, device):
        """Execute the grouped GEMM kernel - EXACT copy of working version."""
        num_groups = len(problem_sizes)

        # Convert to CUTE tensors
        problem_sizes_cute, strides_cute, ptrs_cute = self._convert_to_cute_tensors(
            problem_sizes, strides_abc, ptrs_abc, device
        )

        # Get tensormap and compute clusters
        tensormap_cute = self._get_tensormap_buffer(device)
        total_clusters = self._compute_total_clusters(problem_sizes)

        # Get initial tensors for compilation
        initial_tensors = self._create_initial_tensors(problem_sizes[0], device)

        # Compile or retrieve kernel
        compiled_kernel = self._get_compiled_kernel(
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
            self.stream,
        )
        torch.cuda.synchronize()

    def _convert_to_cute_tensors(self, problem_sizes, strides_abc, ptrs_abc, device):
        """Convert metadata to CUTE tensors - EXACT copy of working version."""
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
        """Get or compile the grouped GEMM kernel - EXACT copy of working version."""
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
        """Create initial CUTE tensors for kernel compilation - EXACT copy of working version."""
        M, N, K, L = problem_shape

        # Create tensors - SAME AS WORKING VERSION
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
        """Get or create tensormap buffer - EXACT copy of working version."""
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
        """Compute total number of clusters needed - EXACT copy of working version."""
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

    def _apply_activation_and_combine(self, gate_outputs, up_outputs):
        """Apply activation and combine gate/up outputs - EXACT copy of working version."""
        return [
            self.activation_function(gate_out) * up_out
            for gate_out, up_out in zip(gate_outputs, up_outputs)
        ]

    def _reconstruct_output_gpu(
        self, final_outputs, m_sizes_gpu, m_offsets_gpu, output
    ):
        """Reconstruct the full output tensor using GPU operations - EXACT copy of working version."""
        if not final_outputs:
            return output

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices]

        # Compute offsets if not provided properly
        if len(m_offsets_gpu) <= len(valid_indices):
            valid_offsets = torch.cumsum(
                torch.cat(
                    [torch.tensor([0], device=m_sizes_gpu.device), valid_sizes[:-1]]
                ),
                dim=0,
            )
        else:
            valid_offsets = m_offsets_gpu[valid_indices]

        # Convert to CPU for final reconstruction (minimal sync)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()

        for i, (size, offset) in enumerate(zip(valid_sizes_cpu, valid_offsets_cpu)):
            if i < len(final_outputs):
                output[offset : offset + size] = final_outputs[i]

        return output

    @staticmethod
    def is_available() -> bool:
        return HAS_CUTLASS


# =================== prev version ===================


class CUTLASSGroupedGemmStrategy_external_converter(GroupGEMMStrategy):
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
