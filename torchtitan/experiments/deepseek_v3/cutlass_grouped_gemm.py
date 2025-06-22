"""
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    Optimized version with pre-transposed weights to eliminate runtime transpose operations.
"""

"""
Shapes:
Kernel compilation successful
[DEBUG] Down projection expert 0 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 1 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 2 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
2025-06-21 22:58:46,021 - INFO - cuModuleLoadData 1080453344
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 3 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
2025-06-21 22:58:46,021 - INFO - cuModuleGetFunction <CUmodule 0x4040a660> kernel_cutlass_kernel_torchtitanexperimentskernelsblackwellcute_grouped_gemmGroupedGemmKernel_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____MMAAtom_ThrID21_ShapeMNK25612816__0
2025-06-21 22:58:46,021 - INFO - <CUfunction 0x177c2870> <-- cuModuleGetFunction
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 4 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 5 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 6 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])
[DEBUG] Matrix mult (optimized): input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Down projection expert 7 (optimized)
  - hidden: torch.Size([12288, 1408])
  - down_weight (pre-transposed): torch.Size([1408, 2048])


Error:
Error using cutlass strategy: The expanded size of the tensor (1408) must match the existing size (2048) at non-singleton dimension 1.  Target sizes: [12288, 1408].  Tensor sizes: [12288, 2048]

"""

# Disable file caching while keeping in-memory cache available, defaults to False.
# export CUTE_DSL_DISABLE_FILE_CACHING=True

# Maximum number of cache files allowed, defaults to 1000.
# export CUTE_DSL_FILE_CACHING_CAPACITY=1000

import logging

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

# Import base class - adjust path as needed based on your project structure
from .group_gemms import GroupGEMMStrategy

logger = logging.getLogger(__name__)


class CUTLASSGroupedGemmStrategy(GroupGEMMStrategy):
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
        """
        Store weights in stacked format with pre-transposition for optimal GEMM performance.

        This eliminates the need for runtime transpose operations.

        Original PyTorch weight shapes:
        - gate_proj_weight: [intermediate_size, hidden_size]
        - up_proj_weight: [intermediate_size, hidden_size]
        - down_proj_weight: [hidden_size, intermediate_size]

        Pre-transposed shapes for direct GEMM usage:
        - gate_proj_weight: [hidden_size, intermediate_size] (transposed)
        - up_proj_weight: [hidden_size, intermediate_size] (transposed)
        - down_proj_weight: [intermediate_size, hidden_size] (transposed)
        """
        print(f"[arrange_expert_weights] Processing {submod_name}")

        # Determine if this weight needs transposition based on submodule name
        needs_transpose = submod_name in ["gate_proj_weight", "up_proj_weight"]

        transposed_weights = []
        for i, weight in enumerate(all_weights):
            original_shape = weight.shape

            if needs_transpose:
                # Transpose gate/up weights: [intermediate_size, hidden_size] -> [hidden_size, intermediate_size]
                transposed_weight = weight.t().contiguous()
                print(
                    f"[arrange_expert_weights] {submod_name} expert {i}: {original_shape} -> {transposed_weight.shape} (transposed)"
                )
            else:
                # Keep down weights as-is for now, will transpose during stacking
                # down_proj_weight: [hidden_size, intermediate_size] -> [intermediate_size, hidden_size]
                transposed_weight = weight.t().contiguous()
                print(
                    f"[arrange_expert_weights] {submod_name} expert {i}: {original_shape} -> {transposed_weight.shape} (transposed)"
                )

            transposed_weights.append(transposed_weight)

        # Stack all transposed weights
        stacked = torch.stack(transposed_weights)
        print(
            f"[arrange_expert_weights] {submod_name} final stacked shape: {stacked.shape}"
        )

        return stacked

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute using CUTLASS grouped GEMM kernel with pre-transposed weights.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing pre-transposed weights
        """
        # Convert to GPU tensors if needed (avoid CPU-GPU sync)
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        # Get pre-transposed weights and device
        weights = self._get_weights(module)
        device = contig_tokens.device

        # Prepare output tensor using down projection output size
        output = torch.zeros(
            contig_tokens.shape[0],
            weights["down"].shape[1],  # hidden_size (after transpose)
            dtype=self.DTYPE_TORCH,
            device=device,
        )

        # Check for valid experts using GPU operations (minimal sync)
        if not self._has_valid_experts_gpu(m_sizes_gpu):
            return output

        # Execute the three-stage computation with pre-transposed weights
        gate_outputs, up_outputs = self._execute_projections_gpu(
            contig_tokens,
            weights["gate"],  # Already transposed to [hidden_size, intermediate_size]
            weights["up"],  # Already transposed to [hidden_size, intermediate_size]
            m_sizes_gpu,
            m_offsets_gpu,
            device,
        )

        hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

        final_outputs = self._execute_down_projection_gpu(
            hidden_states,
            weights["down"],  # Already transposed to [intermediate_size, hidden_size]
            m_sizes_gpu,
            device,
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
        """Check if any experts have tokens using GPU operations (minimal sync)."""
        return torch.any(m_sizes_gpu > 0).item()

    def _get_weights(self, module):
        """Extract pre-transposed weight tensors from module."""
        return {
            "gate": module.get_parameter(
                "gate_proj_weight"
            ),  # Pre-transposed to [num_experts, hidden_size, intermediate_size]
            "up": module.get_parameter(
                "up_proj_weight"
            ),  # Pre-transposed to [num_experts, hidden_size, intermediate_size]
            "down": module.get_parameter(
                "down_proj_weight"
            ),  # Pre-transposed to [num_experts, intermediate_size, hidden_size]
        }

    def _execute_projections_gpu(
        self, input_tokens, weight1, weight2, m_sizes_gpu, m_offsets_gpu, device
    ):
        """Execute gate and up projections using pre-transposed weights."""
        # Find valid experts using GPU operations
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return [], []

        # Prepare metadata with pre-transposed weights
        problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs = (
            self._prepare_gate_up_metadata_gpu(
                input_tokens,
                weight1,  # Pre-transposed gate weights
                weight2,  # Pre-transposed up weights
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
        """Prepare metadata for gate and up projections using pre-transposed weights."""
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

        # Convert to Python for iteration (unavoidable for metadata preparation)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, (expert_idx, size, offset) in enumerate(
            zip(valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu)
        ):
            if size > 0:
                # Get expert data and PRE-TRANSPOSED weights (no runtime transpose needed!)
                expert_tokens = input_tokens[offset : offset + size].contiguous()
                gate_weight = gate_weights[
                    expert_idx
                ].contiguous()  # Already [hidden_size, intermediate_size]
                up_weight = up_weights[
                    expert_idx
                ].contiguous()  # Already [hidden_size, intermediate_size]

                print(f"[DEBUG] Gate/Up projection expert {expert_idx} (optimized)")
                print(f"  - expert_tokens: {expert_tokens.shape}")
                print(f"  - gate_weight (pre-transposed): {gate_weight.shape}")
                print(f"  - up_weight (pre-transposed): {up_weight.shape}")

                M, K = expert_tokens.shape  # M = batch_size, K = hidden_size
                K_weight, N = (
                    gate_weight.shape
                )  # K_weight = hidden_size, N = intermediate_size

                # Verify dimension compatibility
                if K != K_weight:
                    raise ValueError(
                        f"Dimension mismatch: input has {K} features but weight expects {K_weight}"
                    )

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
        """Execute down projection using pre-transposed weights."""
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
        """Prepare metadata for down projection using pre-transposed weights."""
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
                ].contiguous()  # Already [intermediate_size, hidden_size]

                print(f"[DEBUG] Down projection expert {expert_idx} (optimized)")
                print(f"  - hidden: {hidden.shape}")
                print(f"  - down_weight (pre-transposed): {down_weight.shape}")

                M, K = hidden.shape  # M = batch_size, K = intermediate_size
                K_weight, N = (
                    down_weight.shape
                )  # K_weight = intermediate_size, N = hidden_size

                # Verify dimension compatibility
                if K != K_weight:
                    raise ValueError(
                        f"Dimension mismatch: hidden has {K} features but weight expects {K_weight}"
                    )

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
        """Add a single projection to the metadata lists (weights are pre-transposed)."""
        M, K = input_tensor.shape
        K_weight, N = weight_tensor.shape
        L = 1

        print(
            f"[DEBUG] Matrix mult (optimized): input {input_tensor.shape} @ weight {weight_tensor.shape} -> output {output_tensor.shape}"
        )

        # Verify dimension compatibility
        if K != K_weight:
            raise ValueError(
                f"Matrix multiplication dimension mismatch: {K} != {K_weight}"
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

    # Rest of the methods remain the same...
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
        torch.cuda.synchronize()

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
                f"Compiling CUTLASS grouped GEMM kernel: {num_groups} groups, "
                f"2CTA={self.use_2cta_instrs}, cluster={self.cluster_shape_mn}"
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

        # Create tensors (weights are already in correct transposed format)
        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),  # A
            torch.randn(
                K, N, dtype=self.DTYPE_TORCH, device=device
            ),  # B (pre-transposed)
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
