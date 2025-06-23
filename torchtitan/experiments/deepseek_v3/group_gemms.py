# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

try:
    import deep_gemm

    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_AVAILABLE = False

if DEEPGEMM_AVAILABLE:
    import dsgemm_kernels
    import dsgemm_utils

try:
    # from torchao.float8.config import ScalingGranularity
    # from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

    TORCHAO_FP8_GG_AVAILABLE = False

except ImportError:
    TORCHAO_FP8_GG_AVAILABLE = False
    # raise NotImplementedError("Missing TorchAO")

try:
    from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
        # ALIGN_SIZE_M,
        grouped_gemm_forward,
    )

    TRITON_MG_GROUP_GEMM_AVAILABLE = True
except ImportError:
    TRITON_MG_GROUP_GEMM_AVAILABLE = False

try:
    from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_forward import (
        cg_grouped_gemm_forward,
    )

    TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE = True
except ImportError:
    TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE = False

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    # Import our strategy - UPDATE PATH AS NEEDED

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

import logging


logger = logging.getLogger(__name__)


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


# ========= Implementations ===================

__all__ = [
    "TorchFP8GroupGEMM",
    "DSGroupGEMM",
    "TorchBF16GroupGEMM",
    "TorchAOBF16GroupGEMM",
    "TritonCGBF16GroupGEMM",
    # "CUTLASSGroupedGemmStrategy",
    "ManualLoopGroupGEMM",
    # "ImprovedCUTLASSGroupedGemmStrategy",
]


class ManualLoopGroupGEMM(GroupGEMMStrategy):
    """Manual looping baseline implementation for any arch (esp Blackwell) support"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a stacked format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using manual loops over experts"""
        # Get weights

        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Prepare output tensor
        hidden_size = w_gate.shape[
            2
        ]  # stacked weights shape [num_experts, out_dim, in_dim]
        output = torch.zeros(
            contig_tokens.shape[0],
            hidden_size,
            dtype=contig_tokens.dtype,
            device=contig_tokens.device,
        )

        # Process each expert sequentially
        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                # Get tokens for this expert
                expert_tokens = contig_tokens[offset : offset + size]

                # Get weights for this expert
                gate_weight = w_gate[expert_idx]  # [out_dim, in_dim]
                up_weight = w_up[expert_idx]
                down_weight = w_down[expert_idx]

                # Forward pass: gate and up projections
                gate_out = torch.mm(expert_tokens, gate_weight.t())
                up_out = torch.mm(expert_tokens, up_weight.t())

                # Apply activation and combine
                hidden = self.activation_function(gate_out) * up_out

                # Down projection
                expert_output = torch.mm(hidden, down_weight.t())

                # Store results
                output[offset : offset + size] = expert_output

            offset += size

        return output

    @staticmethod
    def is_available() -> bool:
        return True


class CUTLASSGroupedGemmStrategy_down_not_working(GroupGEMMStrategy):
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    This version pre-transposes weights during arrangement and eliminates CPU-GPU synchronization
    by keeping all size/offset computations on GPU.
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
        validate=True,
    ):
        """Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture."""
        super().__init__(custom_activation)
        self.use_2cta_instrs = use_2cta_instrs
        self.validate = validate

        # Set configuration
        self.mma_tiler_mn = mma_tiler_mn or self._get_default_mma_tiler()
        self.cluster_shape_mn = cluster_shape_mn or self._get_default_cluster_shape()

        # Validate configurations only if validation is enabled
        if self.validate:
            self._validate_configurations()

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
        print(f"  - Pre-transposed weights: Enabled")
        print(f"  - Input validation: {'Enabled' if self.validate else 'Disabled'}")
        if cluster_size > 1:
            print(f"  - Using multi-CTA parallelism")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in pre-transposed stacked format for optimal CUTLASS performance."""
        # Pre-transpose weights from [out_dim, in_dim] to [in_dim, out_dim]
        # This eliminates the need for transpose operations during execution
        transposed_weights = [w.t().contiguous() for w in all_weights]
        print(f"Pre-transposing weights for {submod_name} module")
        return torch.stack(transposed_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute using CUTLASS grouped GEMM kernel with pre-transposed weights.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing pre-transposed weights
        """
        print(f"gpu tensor conversion next")
        # Convert to GPU tensors if needed (avoid CPU-GPU sync)
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        print(f"validate inputs")
        # Validate inputs only if validation is enabled
        if self.validate:
            self._validate_inputs(contig_tokens, m_sizes_gpu, module)

        # Get pre-transposed weights and device
        print(f"get weights")
        weights = self._get_weights(module)
        device = contig_tokens.device

        # Prepare output tensor
        print(f"prepare output tensor")
        output = torch.zeros(
            contig_tokens.shape[0],
            weights["gate"].shape[2],  # output dimension is now last dimension
            dtype=self.DTYPE_TORCH,
            device=device,
        )

        # Check for valid experts using GPU operations (no sync)
        if not self._has_valid_experts_gpu(m_sizes_gpu):
            return output

        # Execute the three-stage computation using GPU-only operations with pre-transposed weights
        print(f"executing gate and up projections for {module} module")
        gate_outputs, up_outputs = self._execute_projections_gpu(
            contig_tokens,
            weights["gate"],
            weights["up"],
            m_sizes_gpu,
            m_offsets_gpu,
            device,
        )

        print(f"hiddent states next")
        hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

        print(f"down projection next")

        final_outputs = self._execute_down_projection_gpu(
            hidden_states, weights["down"], m_sizes_gpu, device
        )
        print(f"final outputs next")
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
        """Extract and return pre-transposed weight tensors from module."""
        return {
            "gate": module.get_parameter(
                "gate_proj_weight"
            ),  # [num_experts, in_dim, out_dim]
            "up": module.get_parameter("up_proj_weight"),
            "down": module.get_parameter("down_proj_weight"),
        }

    def _execute_projections_gpu(
        self, input_tokens, weight1, weight2, m_sizes_gpu, m_offsets_gpu, device
    ):
        """Execute gate and up projections using GPU-only operations with pre-transposed weights."""
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
        """Prepare metadata for gate and up projections with pre-transposed weights"""
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
                # Pre-transposed weights: [in_dim, out_dim] - no transpose needed
                gate_weight = gate_weights[expert_idx].contiguous()
                up_weight = up_weights[expert_idx].contiguous()

                M, K = expert_tokens.shape
                N = gate_weight.shape[1]  # output dimension is now second dimension
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
        """Execute down projection using GPU operations with pre-transposed weights."""
        if not hidden_states:
            return []
        print(f"{hidden_states=}, {down_weights=}")
        assert hidden_states.shape[1] == down_weights.shape[1]
        assert False, "check weights hidden"
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
        """Prepare metadata for down projection using GPU operations with pre-transposed weights."""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        down_outputs = []

        # Convert indices to CPU for iteration (minimal sync)
        valid_indices_cpu = valid_indices.cpu().tolist()

        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < len(hidden_states):
                hidden = hidden_states[i]
                # Pre-transposed weights: [in_dim, out_dim] - no transpose needed
                down_weight = down_weights[expert_idx].contiguous()

                M, K = hidden.shape
                N = down_weight.shape[1]  # output dimension is now second dimension

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
        """Add a single projection to the metadata lists (assumes pre-transposed weights)."""
        M, K = input_tensor.shape
        N = weight_tensor.shape[1]  # output dimension from pre-transposed weight
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
                f"Compiling CUTLASS grouped GEMM kernel: {num_groups} groups, 2CTA={self.use_2cta_instrs}, cluster={self.cluster_shape_mn}, Pre-transposed weights"
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

        # Create tensors with pre-transposed weight layout
        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),  # A (input)
            torch.randn(
                K, N, dtype=self.DTYPE_TORCH, device=device
            ),  # B (pre-transposed weight)
            torch.zeros(M, N, dtype=self.DTYPE_TORCH, device=device),  # C (output)
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


# ========================= end of CUTLASSGroupedGemmStrategy =========================


class TritonCGBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation of Triton Contiguous group Gemm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""

        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)

        gate_proj = cg_grouped_gemm_forward(valid_tokens, w_gate, m_indices)

        up_proj = cg_grouped_gemm_forward(valid_tokens, w_up, m_indices)

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)

        down_proj_out = cg_grouped_gemm_forward(hidden_outputs, w_down, m_indices)

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE


class TorchBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation for PyTorch native BF16  _grouped_mm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        gate_proj = torch._grouped_mm(
            contig_tokens,
            w_gate.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )
        up_proj = torch._grouped_mm(
            contig_tokens,
            w_up.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)
        hidden_outputs = torch._grouped_mm(
            hidden_outputs,
            w_down.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        return hidden_outputs


class TorchAOBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation using TorchAO's grouped_gemm_forward"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        return torch.cat(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        gate_proj = grouped_gemm_forward(contig_tokens, w_gate, m_sizes)
        up_proj = grouped_gemm_forward(contig_tokens, w_up, m_sizes)

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)
        hidden_outputs = grouped_gemm_forward(hidden_outputs, w_down, m_sizes)

        return hidden_outputs

    @staticmethod
    def is_available() -> bool:
        return TRITON_MG_GROUP_GEMM_AVAILABLE


class TorchFP8GroupGEMM(GroupGEMMStrategy):
    """Implementation using TorchAO's _scaled_grouped_mm with FP8 rowwise precision and weight prescaling"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage with prescaling"""
        # Stack weights as in the original implementation
        combined_weights = torch.stack(all_weights)

        # Transpose weights for column-major format
        transposed_weights = combined_weights.transpose(-2, -1)

        # Convert weights to float8 format with appropriate scaling
        weight_scales = tensor_to_scale(
            transposed_weights,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,  # Use -2 for transposed weights
            round_scales_to_power_of_2=True,
        )

        # Scale the weights
        scaled_weights = transposed_weights.to(torch.float32) * weight_scales

        # Convert to FP8
        fp8_weights = to_fp8_saturated(scaled_weights, torch.float8_e4m3fn)

        # Register as module parameters
        module.register_parameter(
            f"{submod_name}_fp8",
            nn.Parameter(
                fp8_weights,
            ),
        )

        module.register_parameter(
            f"{submod_name}_scales",
            nn.Parameter(
                weight_scales,
            ),
        )

        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get prescaled transposed weights and scales
        gate_proj_fp8 = module.get_parameter("gate_proj_fp8")
        gate_proj_scales = module.get_parameter("gate_proj_scales")
        up_proj_fp8 = module.get_parameter("up_proj_fp8")
        up_proj_scales = module.get_parameter("up_proj_scales")
        down_proj_fp8 = module.get_parameter("down_proj_fp8")
        down_proj_scales = module.get_parameter("down_proj_scales")

        # Convert input tokens to FP8 with appropriate scaling
        token_scales = tensor_to_scale(
            contig_tokens,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        scaled_tokens = contig_tokens.to(torch.float32) * token_scales
        fp8_tokens = to_fp8_saturated(scaled_tokens, torch.float8_e4m3fn)

        # Run first two GEMMs (gate and up projections) using prescaled weights
        gate_proj = torch._scaled_grouped_mm(
            fp8_tokens,
            gate_proj_fp8,
            token_scales.squeeze().reciprocal(),
            gate_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        up_proj = torch._scaled_grouped_mm(
            fp8_tokens,
            up_proj_fp8,
            token_scales.squeeze().reciprocal(),
            up_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Convert hidden_outputs to FP8 for the third GEMM
        hidden_scales = tensor_to_scale(
            hidden_outputs,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        scaled_hidden = hidden_outputs.to(torch.float32) * hidden_scales
        fp8_hidden = to_fp8_saturated(scaled_hidden, torch.float8_e4m3fn)

        # Run the third GEMM (down projection)
        result = torch._scaled_grouped_mm(
            fp8_hidden,
            down_proj_fp8,
            hidden_scales.squeeze().reciprocal(),
            down_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        return result

    @staticmethod
    def is_available() -> bool:
        return TORCHAO_FP8_GG_AVAILABLE


class DSGroupGEMM(GroupGEMMStrategy):
    """Implementation using DeepGEMM with FP8 quantization"""

    def __init__(self, custom_activation, use_triton_quant=True):
        self.activation_function = custom_activation
        self.use_triton_quant = use_triton_quant

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        combined_weights = torch.stack(all_weights)

        fp8, scales = dsgemm_utils.prepare_fp8_weight(combined_weights)

        # prescale weights
        # TODO - this creates 2 sets of weights, need to resolve this for traiing aspect.
        module.register_parameter(
            f"{submod_name}_fp8",
            nn.Parameter(
                fp8,
            ),
        )

        module.register_parameter(
            f"{submod_name}_scales",
            nn.Parameter(
                scales,
            ),
        )

        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)

        # Get expert weights for all projections
        gate_proj_weight_fp8 = module.get_parameter("gate_proj_fp8")
        gate_proj_scales = module.get_parameter("gate_proj_scales")
        up_proj_weight_fp8 = module.get_parameter("up_proj_fp8")
        up_proj_scales = module.get_parameter("up_proj_scales")
        down_proj_weight_fp8 = module.get_parameter("down_proj_fp8")
        down_proj_scales = module.get_parameter("down_proj_scales")

        # Get dimensions
        m_actual_tokens = valid_tokens.shape[0]
        intermediate_size = module.get_parameter("gate_proj_weight").shape[1]
        hidden_size = module.get_parameter("down_proj_weight").shape[1]

        # Allocate output buffers
        gate_proj_out = torch.empty(
            (m_actual_tokens, intermediate_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )
        up_proj_out = torch.empty_like(gate_proj_out)

        # Allocate output buffer for down projection
        down_proj_out = torch.empty(
            (m_actual_tokens, hidden_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )

        # Prepare input in FP8 format (shared by gate and up projections)
        if self.use_triton_quant:
            gate_up_input = dsgemm_kernels.groupwise_activation_quant(valid_tokens)
        else:
            gate_up_input = dsgemm_utils.prepare_fp8_input(valid_tokens)

        # Run first GEMM (gate projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (gate_proj_weight_fp8, gate_proj_scales),
            gate_proj_out,
            m_indices,
        )

        # Run second GEMM (up projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (up_proj_weight_fp8, up_proj_scales),
            up_proj_out,
            m_indices,
        )

        # Apply activation
        hidden_states = self.activation_function(gate_proj_out) * up_proj_out

        # Run third GEMM (down projection)
        if self.use_triton_quant:
            hidden_states_quantized = dsgemm_kernels.groupwise_activation_quant(
                hidden_states
            )
        else:
            hidden_states_quantized = dsgemm_utils.prepare_fp8_input(hidden_states)

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            hidden_states_quantized,
            (down_proj_weight_fp8, down_proj_scales),
            down_proj_out,
            m_indices,
        )

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return DEEPGEMM_AVAILABLE
