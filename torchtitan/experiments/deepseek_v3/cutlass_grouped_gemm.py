# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CUTLASS GroupedGEMM Strategy for Blackwell architecture.

This module contains the CUTLASSGroupedGemmStrategy implementation that uses
CUTLASS GroupedGemmKernel for high-performance group GEMM operations with
minimal CPU-GPU synchronization and optional input validation.
"""

"""
[DEBUG] Gate/Up projection expert 14
  - expert_tokens: torch.Size([12288, 2048])
  - gate_weight (after .t()): torch.Size([2048, 1408])
  - up_weight (after .t()): torch.Size([2048, 1408])
[DEBUG] Matrix mult: input torch.Size([12288, 2048]) @ weight torch.Size([2048, 1408]) -> output torch.Size([12288, 1408])
[DEBUG] Gate/Up projection expert 15[DEBUG] Matrix mult: input torch.Size([12288, 2048]) @ weight torch.Size([2048, 1408]) -> output torch.Size([12288, 1408])

  - expert_tokens: torch.Size([12288, 2048])
  - gate_weight (after .t()): torch.Size([2048, 1408])
  - up_weight (after .t()): torch.Size([2048, 1408])
[DEBUG] Matrix mult: input torch.Size([12288, 2048]) @ weight torch.Size([2048, 1408]) -> output torch.Size([12288, 1408])
[DEBUG] Matrix mult: input torch.Size([12288, 2048]) @ weight torch.Size([2048, 1408]) -> output torch.Size([12288, 1408])
down projection:
[DEBUG] Down projection expert 10[DEBUG] Matrix mult: input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])

[DEBUG] Down projection expert 10
  - hidden: torch.Size([12288, 1408])
  - hidden: torch.Size([12288, 1408])  - down_weight (after .t()): torch.Size([1408, 2048])

  - down_weight (after .t()): torch.Size([1408, 2048])
[DEBUG] Matrix mult: input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])
[DEBUG] Matrix mult: input torch.Size([12288, 1408]) @ weight torch.Size([1408, 2048]) -> output torch.Size([12288, 2048])




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
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations on Blackwell architecture.

    """

    # Constants for Blackwell architecture support
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
        cluster_shape_mn=(2, 2),
        validate=False,
        debug_shapes=False,
    ):
        """
        Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture.

        Args:
            custom_activation: Activation function to use (e.g., SiLU)
            use_2cta_instrs: Whether to use 2 CTA instructions for better performance
            mma_tiler_mn: MMA tiler configuration (M, N)
            cluster_shape_mn: Cluster shape configuration (M, N)
            validate: Whether to validate inputs (disable for performance in production)
            debug_shapes: Whether to log tensor shapes for debugging dimension mismatches
        """
        super().__init__(custom_activation)
        self.use_2cta_instrs = use_2cta_instrs
        self.validate = validate
        self.debug_shapes = debug_shapes

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
        logger.info(f"Initialized CUTLASSGroupedGemmStrategy for Blackwell with:")
        logger.info(f"  - 2 CTA instructions: {self.use_2cta_instrs}")
        logger.info(f"  - MMA tiler (M, N): {self.mma_tiler_mn}")
        logger.info(f"  - Cluster shape (M, N): {self.cluster_shape_mn}")
        logger.info(f"  - Cluster size: {cluster_size}")
        logger.info(f"  - Weight format: Standard PyTorch (runtime transpose)")
        logger.info(
            f"  - Input validation: {'Enabled' if self.validate else 'Disabled'}"
        )
        logger.info(f"  - CPU-GPU sync optimization: Enabled")
        logger.info(
            f"  - Debug shapes: {'Enabled' if self.debug_shapes else 'Disabled'}"
        )
        if cluster_size > 1:
            logger.info(f"  - Using multi-CTA parallelism")

    def _debug_log_shapes(self, message, **tensors):
        """Log tensor shapes for debugging if debug_shapes is enabled"""
        if self.debug_shapes:
            shape_info = []
            for name, tensor in tensors.items():
                if hasattr(tensor, "shape"):
                    shape_info.append(f"{name}: {tensor.shape}")
                else:
                    shape_info.append(f"{name}: {type(tensor)}")
            logger.debug(f"[SHAPE DEBUG] {message} - {', '.join(shape_info)}")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in stacked format (NO transpose - keep original PyTorch format)"""
        # Keep original weight format for compatibility:
        # gate/up: [intermediate_size, hidden_size]
        # down: [hidden_size, intermediate_size]

        # DEBUG: Print shapes to verify no transpose
        print(f"[arrange_expert_weights] Processing {submod_name}")
        for i, w in enumerate(all_weights):
            print(f"[arrange_expert_weights] {submod_name} expert {i}: {w.shape}")

        # NO TRANSPOSE - just stack the original weights
        stacked = torch.stack(all_weights)
        print(
            f"[arrange_expert_weights] {submod_name} final stacked shape: {stacked.shape}"
        )
        return stacked

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute using CUTLASS grouped GEMM kernel with standard PyTorch weight format.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Tensor of expert sizes (GPU tensor to avoid sync)
            m_offsets: Tensor of expert offsets (GPU tensor to avoid sync)
            module: MoE module containing weights in standard PyTorch format
        """
        # Convert to GPU tensors if needed (avoid CPU-GPU sync)
        m_sizes_gpu, m_offsets_gpu = self._ensure_gpu_tensors(
            m_sizes, m_offsets, contig_tokens.device
        )

        # Validate inputs only if validation is enabled
        if self.validate:
            self._validate_inputs(contig_tokens, m_sizes_gpu, module)

        # Get weights and device
        weights = self._get_weights(module)
        device = contig_tokens.device

        # Debug logging - force print for visibility
        if self.debug_shapes:
            print(f"[DEBUG] Input tensors - contig_tokens: {contig_tokens.shape}")
            print(f"[DEBUG] Gate weights: {weights['gate'].shape}")
            print(f"[DEBUG] Up weights: {weights['up'].shape}")
            print(f"[DEBUG] Down weights: {weights['down'].shape}")
            print(f"[DEBUG] m_sizes_gpu: {m_sizes_gpu}")
            print(f"[DEBUG] m_offsets_gpu: {m_offsets_gpu}")

        # Prepare output tensor - use down projection weight shape for final output size
        # Down weights are [num_experts, hidden_size, intermediate_size], so output is hidden_size
        output = torch.zeros(
            contig_tokens.shape[0],
            weights["down"].shape[1],  # hidden_size from down projection
            dtype=self.DTYPE_TORCH,
            device=device,
        )

        # Check for valid experts using GPU operations (defer sync)
        has_valid_experts = self._has_valid_experts_gpu(m_sizes_gpu)

        # Early exit if no valid experts (minimal sync only when needed)
        if not has_valid_experts.item():
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
        """Ensure sizes and offsets are GPU tensors with minimal CPU-GPU sync"""
        # Convert m_sizes
        if not isinstance(m_sizes, torch.Tensor):
            m_sizes_gpu = torch.tensor(m_sizes, dtype=torch.int32, device=device)
        else:
            # Only move if not already on correct device (avoids unnecessary transfer)
            if m_sizes.device != device or m_sizes.dtype != torch.int32:
                m_sizes_gpu = m_sizes.to(device=device, dtype=torch.int32)
            else:
                m_sizes_gpu = m_sizes

        # Convert m_offsets
        if not isinstance(m_offsets, torch.Tensor):
            m_offsets_gpu = torch.tensor(m_offsets, dtype=torch.int32, device=device)
        else:
            # Only move if not already on correct device (avoids unnecessary transfer)
            if m_offsets.device != device or m_offsets.dtype != torch.int32:
                m_offsets_gpu = m_offsets.to(device=device, dtype=torch.int32)
            else:
                m_offsets_gpu = m_offsets

        return m_sizes_gpu, m_offsets_gpu

    def _has_valid_experts_gpu(self, m_sizes_gpu):
        """Check if any experts have tokens using GPU operations (no sync)."""
        # Return the tensor itself - let caller decide when to sync
        return torch.any(m_sizes_gpu > 0)

    def _validate_inputs(self, contig_tokens, m_sizes_gpu, module):
        """Validate input parameters with minimal GPU sync"""
        # Check dtype without sync (comparison is done on device info)
        if contig_tokens.dtype != self.DTYPE_TORCH:
            raise ValueError(
                f"Expected input dtype {self.DTYPE_TORCH}, got {contig_tokens.dtype}"
            )

        # Check tensor dimensionality (no sync needed)
        if len(contig_tokens.shape) != 2:
            raise ValueError(
                f"Expected 2D input tensor, got shape {contig_tokens.shape}"
            )

        # Check parameter existence (no sync needed)
        required_params = ["gate_proj_weight", "up_proj_weight", "down_proj_weight"]
        for param in required_params:
            if not hasattr(module, param) or module.get_parameter(param) is None:
                raise ValueError(f"Module missing required parameter: {param}")

    def _get_weights(self, module):
        """Extract and return weight tensors from module (original format, not transposed)."""
        return {
            "gate": module.get_parameter(
                "gate_proj_weight"
            ),  # [num_experts, intermediate_size, hidden_size]
            "up": module.get_parameter(
                "up_proj_weight"
            ),  # [num_experts, intermediate_size, hidden_size]
            "down": module.get_parameter(
                "down_proj_weight"
            ),  # [num_experts, hidden_size, intermediate_size]
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
        """Prepare metadata for gate and up projections with minimal CPU-GPU sync"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        gate_outputs = []
        up_outputs = []

        # Extract valid sizes and offsets (keep on GPU as long as possible)
        valid_sizes = m_sizes_gpu[valid_indices]
        valid_offsets = (
            m_offsets_gpu[valid_indices]
            if len(m_offsets_gpu) > len(valid_indices)
            else torch.cumsum(
                torch.cat([torch.tensor([0], device=device), valid_sizes[:-1]]), dim=0
            )
        )

        # Filter out zero-size experts on GPU before any CPU transfer
        nonzero_mask = valid_sizes > 0
        if not torch.any(nonzero_mask).item():  # Only sync needed for early exit
            return problem_sizes, strides_abc, ptrs_abc, gate_outputs, up_outputs

        # Apply mask to get final valid experts
        final_valid_indices = valid_indices[nonzero_mask]
        final_valid_sizes = valid_sizes[nonzero_mask]
        final_valid_offsets = valid_offsets[nonzero_mask]

        # Single batch CPU transfer at the end
        final_indices_cpu = final_valid_indices.cpu()
        final_sizes_cpu = final_valid_sizes.cpu()
        final_offsets_cpu = final_valid_offsets.cpu()

        # Convert to lists once
        indices_list = final_indices_cpu.tolist()
        sizes_list = final_sizes_cpu.tolist()
        offsets_list = final_offsets_cpu.tolist()

        # Now iterate with pre-transferred data
        for expert_idx, size, offset in zip(indices_list, sizes_list, offsets_list):
            # Get expert data
            expert_tokens = input_tokens[offset : offset + size].contiguous()
            # Original weight format: gate/up are [intermediate_size, hidden_size]
            # Need to transpose for matrix multiplication: tokens @ weight.t()
            gate_weight = (
                gate_weights[expert_idx].t().contiguous()
            )  # [hidden_size, intermediate_size]
            up_weight = (
                up_weights[expert_idx].t().contiguous()
            )  # [hidden_size, intermediate_size]

            if self.debug_shapes:
                print(f"[DEBUG] Gate/Up projection expert {expert_idx}")
                print(f"  - expert_tokens: {expert_tokens.shape}")
                print(f"  - gate_weight (after .t()): {gate_weight.shape}")
                print(f"  - up_weight (after .t()): {up_weight.shape}")

            M, K = expert_tokens.shape  # M = batch_size, K = hidden_size
            K_weight, N = (
                gate_weight.shape
            )  # K_weight = hidden_size, N = intermediate_size

            # Verify dimension compatibility
            if K != K_weight:
                raise ValueError(
                    f"Dimension mismatch in gate/up projections: "
                    f"input tokens have {K} features but weight expects {K_weight}. "
                    f"Tokens shape: {expert_tokens.shape}, Gate weight shape: {gate_weight.shape}"
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
        """Prepare metadata for down projection with minimal CPU-GPU sync"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        down_outputs = []

        # Filter valid indices to match hidden states length on GPU
        num_hidden_states = len(hidden_states)
        if num_hidden_states == 0:
            return problem_sizes, strides_abc, ptrs_abc, down_outputs

        # Limit valid indices to available hidden states (GPU operation)
        valid_indices_limited = valid_indices[:num_hidden_states]

        # Single batch CPU transfer
        valid_indices_cpu = valid_indices_limited.cpu().tolist()

        for i, expert_idx in enumerate(valid_indices_cpu):
            if i < num_hidden_states:
                hidden = hidden_states[i]
                # Original down weight format: [hidden_size, intermediate_size]
                # Need to transpose for matrix multiplication: hidden @ weight.t()
                down_weight = (
                    down_weights[expert_idx].t().contiguous()
                )  # [intermediate_size, hidden_size]

                if self.debug_shapes:
                    print(f"[DEBUG] Down projection expert {expert_idx}")
                    print(f"  - hidden: {hidden.shape}")
                    print(f"  - down_weight (after .t()): {down_weight.shape}")

                M, K = hidden.shape  # M = batch_size, K = intermediate_size
                K_weight, N = (
                    down_weight.shape
                )  # K_weight = intermediate_size, N = hidden_size

                # Verify dimension compatibility
                if K != K_weight:
                    raise ValueError(
                        f"Dimension mismatch in down projection: "
                        f"hidden states have {K} features but down_weight expects {K_weight} input features. "
                        f"Hidden shape: {hidden.shape}, Down weight shape: {down_weight.shape}"
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
        """Add a single projection to the metadata lists (weights are transposed at call site)."""
        M, K = input_tensor.shape  # M = batch_size, K = input_features
        K_weight, N = (
            weight_tensor.shape
        )  # K_weight = input_features, N = output_features
        L = 1

        # Debug print
        if self.debug_shapes:
            print(
                f"[DEBUG] Matrix mult: input {input_tensor.shape} @ weight {weight_tensor.shape} -> output {output_tensor.shape}"
            )

        # Verify dimension compatibility for matrix multiplication
        if K != K_weight:
            raise ValueError(
                f"Matrix multiplication dimension mismatch: "
                f"input has {K} features but weight expects {K_weight} input features. "
                f"Input shape: {input_tensor.shape}, Weight shape: {weight_tensor.shape}"
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
            logger.info(
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
            logger.info("Kernel compilation successful")

        return self._compiled_kernels[cache_key]

    def _create_initial_tensors(self, problem_shape, device):
        """Create initial CUTE tensors for kernel compilation."""
        M, N, K, L = problem_shape

        # Create tensors with standard PyTorch layout (weights will be transposed at runtime)
        tensors = [
            torch.randn(M, K, dtype=self.DTYPE_TORCH, device=device),  # A (input)
            torch.randn(
                K, N, dtype=self.DTYPE_TORCH, device=device
            ),  # B (transposed weight)
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
        """Reconstruct the full output tensor with minimal CPU-GPU sync"""
        if not final_outputs:
            return output

        # Find valid experts on GPU
        valid_mask = m_sizes_gpu > 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        valid_sizes = m_sizes_gpu[valid_indices]

        # Filter to match final_outputs length
        num_outputs = len(final_outputs)
        if num_outputs == 0:
            return output

        # Limit to available outputs
        valid_indices_limited = valid_indices[:num_outputs]
        valid_sizes_limited = valid_sizes[:num_outputs]

        # Compute offsets if not provided properly (GPU operations)
        if len(m_offsets_gpu) <= len(valid_indices_limited):
            valid_offsets_limited = torch.cumsum(
                torch.cat(
                    [
                        torch.tensor([0], device=m_sizes_gpu.device),
                        valid_sizes_limited[:-1],
                    ]
                ),
                dim=0,
            )
        else:
            valid_offsets_limited = m_offsets_gpu[valid_indices_limited]

        # Single batch CPU transfer for reconstruction
        valid_sizes_cpu = valid_sizes_limited.cpu().tolist()
        valid_offsets_cpu = valid_offsets_limited.cpu().tolist()

        # Reconstruct output using pre-transferred data
        for i, (size, offset) in enumerate(zip(valid_sizes_cpu, valid_offsets_cpu)):
            if i < len(final_outputs):
                output[offset : offset + size] = final_outputs[i]

        return output

    @staticmethod
    def is_available() -> bool:
        """Check if CUTLASS is available on the current system."""
        return HAS_CUTLASS
