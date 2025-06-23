# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell.

"""

# Disable file caching while keeping in-memory cache available, defaults to False.
# export CUTE_DSL_DISABLE_FILE_CACHING=True

# Maximum number of cache files allowed, defaults to 1000.
# export CUTE_DSL_FILE_CACHING_CAPACITY=1000

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from .group_gemm_base import GroupGEMMStrategy


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm_kernel import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
    print("✓ CUTLASS and strategies imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ CUTLASS import failed: {e}")
    print("CUTLASSGroupedGemmStrategy will not be available")

from torchtitan.experiments.kernels.blackwell.pytorch_cute_converter import (
    ExpertOperationMetadata,
    PyTorchToCuteConverter,
)


logger = logging.getLogger(__name__)


class CUTLASSGroupedGemmStrategy(GroupGEMMStrategy):
    """
    Improved CUTLASS GroupedGemmKernel strategy with better tensor conversion.

    This version eliminates CPU-GPU synchronization and provides cleaner tensor
    management through dedicated converter classes.
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
    TENSORMAP_COUNT = 3
    TENSORMAP_BYTES = 128

    def __init__(
        self,
        custom_activation,
        use_2cta_instrs: bool = True,
        mma_tiler_mn: Optional[Tuple[int, int]] = None,
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the improved CUTLASS grouped GEMM strategy."""
        super().__init__(custom_activation)

        if not HAS_CUTLASS:
            raise RuntimeError("CUTLASS not available")

        # Set configuration
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn or self._get_default_mma_tiler()
        self.cluster_shape_mn = cluster_shape_mn or self._get_default_cluster_shape()

        # Initialize converter
        self.converter = PyTorchToCuteConverter(
            alignment=self.ALIGNMENT, acc_dtype=self.ACC_DTYPE
        )

        # Initialize kernel and hardware
        self._initialize_components()

        # Initialize caches
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        self._log_initialization()

    def _get_default_mma_tiler(self) -> Tuple[int, int]:
        """Get default MMA tiler configuration based on CTA mode."""
        return (256, 128) if self.use_2cta_instrs else (128, 128)

    def _get_default_cluster_shape(self) -> Tuple[int, int]:
        """Get default cluster shape based on CTA mode."""
        return (4, 2) if self.use_2cta_instrs else (1, 1)

    def _initialize_components(self):
        """Initialize CUTLASS kernel and hardware components."""
        # Initialize kernel
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.ACC_DTYPE,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Initialize hardware info
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        # Initialize CUDA stream
        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

    def _log_initialization(self):
        """Log initialization information."""
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        print(f"✅  CUTLASS Blackwell Group Gemm Strategy initialized:")
        print(f"   - 2 CTA instructions: {self.use_2cta_instrs}")
        print(f"   - MMA tiler (M, N): {self.mma_tiler_mn}")
        print(f"   - Cluster shape (M, N): {self.cluster_shape_mn}")
        print(f"   - Cluster size: {cluster_size}")
        print(f"   - Max active clusters: {self.max_active_clusters}")

    def arrange_expert_weights(
        self, all_weights: List[torch.Tensor], submod_name: str, module
    ) -> torch.Tensor:
        """Store weights in stacked format."""
        return torch.stack(all_weights)

    def execute(
        self,
        contig_tokens: torch.Tensor,
        m_sizes: torch.Tensor,
        m_offsets: torch.Tensor,
        module,
    ) -> torch.Tensor:
        """
        Execute using improved CUTLASS grouped GEMM with better tensor management.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing weights
        """
        try:
            # Ensure GPU tensors and validate inputs
            m_sizes_gpu, m_offsets_gpu = self._prepare_gpu_tensors(
                m_sizes, m_offsets, contig_tokens.device
            )
            self._validate_inputs(contig_tokens, m_sizes_gpu, module)

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

            # Early exit if no valid experts
            if not self._has_valid_experts_gpu(m_sizes_gpu):
                return output

            # Execute three-stage MoE computation
            gate_outputs, up_outputs = self._execute_gate_up_projections(
                contig_tokens,
                weights["gate"],
                weights["up"],
                m_sizes_gpu,
                m_offsets_gpu,
                device,
            )

            hidden_states = self._apply_activation_and_combine(gate_outputs, up_outputs)

            final_outputs = self._execute_down_projection(
                hidden_states, weights["down"], m_sizes_gpu, device
            )

            return self._reconstruct_output_gpu(
                final_outputs, m_sizes_gpu, m_offsets_gpu, output
            )

        except Exception as e:
            logger.error(f"Error in CUTLASS execution: {e}")
            raise

    def _prepare_gpu_tensors(
        self, m_sizes, m_offsets, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure sizes and offsets are GPU tensors with validation."""
        if not isinstance(m_sizes, torch.Tensor):
            m_sizes_gpu = torch.tensor(m_sizes, dtype=torch.int32, device=device)
        else:
            m_sizes_gpu = m_sizes.to(device=device, dtype=torch.int32)

        if not isinstance(m_offsets, torch.Tensor):
            m_offsets_gpu = torch.tensor(m_offsets, dtype=torch.int32, device=device)
        else:
            m_offsets_gpu = m_offsets.to(device=device, dtype=torch.int32)

        return m_sizes_gpu, m_offsets_gpu

    def _validate_inputs(
        self, contig_tokens: torch.Tensor, m_sizes_gpu: torch.Tensor, module
    ):
        """Validate input parameters with comprehensive checks."""
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

    def _has_valid_experts_gpu(self, m_sizes_gpu: torch.Tensor) -> bool:
        """Check if any experts have tokens using GPU operations."""
        return torch.any(m_sizes_gpu > 0).item()

    def _get_weights(self, module) -> Dict[str, torch.Tensor]:
        """Extract and return weight tensors from module."""
        return {
            "gate": module.get_parameter("gate_proj_weight"),
            "up": module.get_parameter("up_proj_weight"),
            "down": module.get_parameter("down_proj_weight"),
        }

    def _execute_gate_up_projections(
        self,
        input_tokens: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        device: torch.device,
    ) -> Tuple[List, List]:
        """Execute gate and up projections using improved tensor management."""
        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return [], []

        # Prepare metadata using improved helper
        operations_metadata = self._prepare_projection_metadata(
            input_tokens,
            gate_weights,
            up_weights,
            m_sizes_gpu,
            m_offsets_gpu,
            valid_indices,
            device,
        )

        if not operations_metadata:
            return [], []

        # Execute grouped GEMM
        self._execute_grouped_gemm_with_metadata(operations_metadata, device)

        # Extract outputs
        gate_outputs = [op["gate_output"] for op in operations_metadata]
        up_outputs = [op["up_output"] for op in operations_metadata]

        return gate_outputs, up_outputs

    def _prepare_projection_metadata(
        self,
        input_tokens: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        valid_indices: torch.Tensor,
        device: torch.device,
    ) -> List[Dict]:
        """Prepare metadata for projections using improved helpers."""
        operations_metadata = []

        # Extract valid information
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = self._compute_valid_offsets(
            m_sizes_gpu, m_offsets_gpu, valid_indices, device
        )

        # Convert to CPU for iteration (minimal sync)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()
        valid_indices_cpu = valid_indices.cpu().tolist()

        for expert_idx, size, offset in zip(
            valid_indices_cpu, valid_sizes_cpu, valid_offsets_cpu
        ):
            if size > 0:
                # Get expert data
                expert_tokens = input_tokens[offset : offset + size].contiguous()
                gate_weight = gate_weights[expert_idx].contiguous()
                up_weight = up_weights[expert_idx].contiguous()

                M, K = expert_tokens.shape
                N = gate_weight.shape[0]

                # Create output tensors
                gate_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)
                up_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Create metadata for both projections using helper class
                gate_metadata = ExpertOperationMetadata(
                    expert_tokens, gate_weight, gate_output
                )
                up_metadata = ExpertOperationMetadata(
                    expert_tokens, up_weight, up_output
                )

                operations_metadata.append(
                    {
                        "gate_metadata": gate_metadata,
                        "up_metadata": up_metadata,
                        "gate_output": gate_output,
                        "up_output": up_output,
                    }
                )

        return operations_metadata

    def _execute_down_projection(
        self,
        hidden_states: List[torch.Tensor],
        down_weights: torch.Tensor,
        m_sizes_gpu: torch.Tensor,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Execute down projection using improved tensor management."""
        if not hidden_states:
            return []

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_indices_cpu = valid_indices.cpu().tolist()

        # Prepare down projection metadata
        down_operations = []
        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < len(hidden_states):
                hidden = hidden_states[i]
                down_weight = down_weights[expert_idx].contiguous()

                M, K = hidden.shape
                N = down_weight.shape[0]
                down_output = torch.empty(M, N, dtype=self.DTYPE_TORCH, device=device)

                # Create metadata using helper class
                down_metadata = ExpertOperationMetadata(
                    hidden, down_weight, down_output
                )
                down_operations.append(
                    {
                        "metadata": down_metadata,
                        "output": down_output,
                    }
                )

        if not down_operations:
            return []

        # Execute grouped GEMM for down projection
        self._execute_grouped_gemm_for_down(down_operations, device)

        return [op["output"] for op in down_operations]

    def _execute_grouped_gemm_with_metadata(
        self, operations_metadata: List[Dict], device: torch.device
    ):
        """Execute grouped GEMM using operations metadata."""
        # Collect all metadata for both gate and up projections
        all_problem_sizes = []
        all_strides = []
        all_ptrs = []

        for op in operations_metadata:
            # Add gate projection
            gate_meta = op["gate_metadata"]
            all_problem_sizes.append(gate_meta.get_problem_size())
            all_strides.append(gate_meta.get_strides())
            all_ptrs.append(gate_meta.get_pointers())

            # Add up projection
            up_meta = op["up_metadata"]
            all_problem_sizes.append(up_meta.get_problem_size())
            all_strides.append(up_meta.get_strides())
            all_ptrs.append(up_meta.get_pointers())

        if not all_problem_sizes:
            return

        # Execute using improved converter
        self._execute_cutlass_kernel(all_problem_sizes, all_strides, all_ptrs, device)

    def _execute_grouped_gemm_for_down(
        self, down_operations: List[Dict], device: torch.device
    ):
        """Execute grouped GEMM for down projection."""
        all_problem_sizes = []
        all_strides = []
        all_ptrs = []

        for op in down_operations:
            metadata = op["metadata"]
            all_problem_sizes.append(metadata.get_problem_size())
            all_strides.append(metadata.get_strides())
            all_ptrs.append(metadata.get_pointers())

        if not all_problem_sizes:
            return

        # Execute using improved converter
        self._execute_cutlass_kernel(all_problem_sizes, all_strides, all_ptrs, device)

    def _execute_cutlass_kernel(
        self,
        problem_sizes: List[List[int]],
        strides_abc: List[List[List[int]]],
        ptrs_abc: List[List[int]],
        device: torch.device,
    ):
        """Execute CUTLASS kernel using improved converter."""
        num_groups = len(problem_sizes)

        # Convert to CUTE tensors using improved converter
        (
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
        ) = self.converter.create_metadata_tensors(
            problem_sizes, strides_abc, ptrs_abc, device
        )

        # Get other required components
        tensormap_cute = self._get_tensormap_buffer(device)
        total_clusters = self._compute_total_clusters(problem_sizes)

        # Get initial tensors for compilation using improved converter
        initial_tensors = self.converter.create_initial_tensors(
            tuple(problem_sizes[0]), device, self.DTYPE_TORCH
        )

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

    def _get_tensormap_buffer(self, device: torch.device):
        """Get or create tensormap buffer using improved converter."""
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            self._tensormap_buffers[device] = self.converter.create_tensormap_buffer(
                device, sm_count, self.TENSORMAP_COUNT, self.TENSORMAP_BYTES
            )
        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes: List[List[int]]) -> int:
        """Compute total number of clusters needed."""
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

    def _get_compiled_kernel(
        self,
        num_groups: int,
        total_clusters: int,
        initial_tensors: List,
        problem_sizes_cute,
        strides_cute,
        ptrs_cute,
        tensormap_cute,
    ):
        """Get or compile the grouped GEMM kernel with caching."""
        cache_key = (
            num_groups,
            total_clusters,
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
        )

        if cache_key not in self._compiled_kernels:
            print(
                f"Compiling CUTLASS kernel: {num_groups} groups, 2CTA={self.use_2cta_instrs}"
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
            print("✅ Kernel compilation successful")

        return self._compiled_kernels[cache_key]

    def _apply_activation_and_combine(
        self, gate_outputs: List[torch.Tensor], up_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply activation and combine gate/up outputs."""
        return [
            self.activation_function(gate_out) * up_out
            for gate_out, up_out in zip(gate_outputs, up_outputs)
        ]

    def _reconstruct_output_gpu(
        self,
        final_outputs: List[torch.Tensor],
        m_sizes_gpu: torch.Tensor,
        m_offsets_gpu: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the full output tensor using GPU operations."""
        if not final_outputs:
            return output

        # Find valid experts
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices]

        # Compute offsets
        valid_offsets = self._compute_valid_offsets(
            m_sizes_gpu, m_offsets_gpu, valid_indices, m_sizes_gpu.device
        )

        # Convert to CPU for final reconstruction (minimal sync)
        valid_sizes_cpu = valid_sizes.cpu().tolist()
        valid_offsets_cpu = valid_offsets.cpu().tolist()

        for i, (size, offset) in enumerate(zip(valid_sizes_cpu, valid_offsets_cpu)):
            if i < len(final_outputs):
                output[offset : offset + size] = final_outputs[i]

        return output

    @staticmethod
    def is_available() -> bool:
        """Check if CUTLASS is available."""
        return HAS_CUTLASS
