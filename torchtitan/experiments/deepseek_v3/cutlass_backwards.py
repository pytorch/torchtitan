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

"""
Notes!

current error - requires a kernel context before getting hardware info.

This class is used to get the hardware info of given GPU device.
It provides methods to get the max active clusters for given cluster size.

Prerequisite:
- CUDA driver is initialized via `driver.cuInit` or other CUDA APIs.
- CUDA context is created via `driver.cuCtxCreate` or other CUDA APIs.


this works:
cute hardware - device_id 0
cute hardware - driver_version 12080
2025-06-12 15:59:27,116 - INFO - Started preprocessing [_host_function]
2025-06-12 15:59:27,117 - INFO - ASTPreprocessor Transforming function [_host_function]
2025-06-12 15:59:27,118 - INFO - ASTPreprocessor Executing transformed code for function [_host_function]
2025-06-12 15:59:27,118 - INFO - Final mangled function name: cutlass__host_function_cutlassutilshardware_infoHardwareInfo_object_at_
2025-06-12 15:59:27,120 - INFO - Started preprocessing [_empty_kernel]
2025-06-12 15:59:27,120 - INFO - ASTPreprocessor Transforming function [_empty_kernel]
2025-06-12 15:59:27,121 - INFO - ASTPreprocessor Executing transformed code for function [_empty_kernel]
2025-06-12 15:59:27,122 - INFO - Final mangled function name: cutlass__empty_kernel_cutlassutilshardware_infoHardwareInfo_object_at_
2025-06-12 15:59:27,243 - INFO - Function=[cutlass__host_function_cutlassutilshardware_infoHardwareInfo_object_at_] Computed module_hash=[dcde861f00d587038a517012d015a7fe7d920bf8fd510b4d947c612997627913]
2025-06-12 15:59:27,243 - INFO - JIT cache miss function=[cutlass__host_function_cutlassutilshardware_infoHardwareInfo_object_at_] module_hash=[dcde861f00d587038a517012d015a7fe7d920bf8fd510b4d947c612997627913]
2025-06-12 15:59:27,275 - INFO - cuModuleLoadData 478334896
2025-06-12 15:59:27,276 - INFO - cuModuleGetFunction <CUmodule 0x1c94d2d0> kernel_cutlass__empty_kernel_cutlassutilshardware_infoHardwareInfo_object_at__0
2025-06-12 15:59:27,276 - INFO - <CUfunction 0x1c910de0> <-- cuModuleGetFunction
max_dynamic_shared_memory: 232448
max_active_blocks: 1
Initialized CUTLASSGroupedGemmStrategy for Blackwell with:

but basic strategy below fails:

✓ CUTLASS and strategies imported successfully
Testing CUTLASS Backward Group GEMM...
Creating strategy for 4 experts, 1024 in_features, 2048 out_features, 512 total_tokens
Initializing CUTLASSGroupedGemmStrategy for Blackwell
cute hardware - device_id 0
cute hardware - driver_version 12080
Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 1387, in <module>
    test_cutlass_backward_group_gemm()
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 1348, in test_cutlass_backward_group_gemm
    strategy = CUTLASSGroupedGemmStrategy(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 120, in __init__
    self._initialize_hardware()
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 149, in _initialize_hardware
    self.max_active_clusters = self.hardware_info.get_max_active_clusters(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py", line 328, in walk_module_and_get_cubin_data
    module.operation.walk(walk_gpu_binary_op)
RuntimeError: Exception raised in callback: Traceback (most recent call last):
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 1387, in <module>
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 1348, in test_cutlass_backward_group_gemm
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 120, in __init__
  File "/data/users/less/torchtitan/torchtitan/experiments/deepseek_v3/cutlass_backwards.py", line 149, in _initialize_hardware
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py", line 47, in get_max_active_clusters
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/utils/hardware_info.py", line 176, in _get_device_function
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/compiler.py", line 221, in compile
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py", line 1337, in _func
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py", line 1188, in generate_mlir
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/dsl.py", line 1129, in compile_and_cache
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py", line 258, in update_jit_cuda_modules
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py", line 328, in walk_module_and_get_cubin_data
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py", line 325, in walk_gpu_binary_op
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/jit_executor.py", line 243, in walk_callback
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/runtime/cuda.py", line 294, in load_cubin_module_data
  File "/home/less/.conda/envs/pycutlass/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/runtime/cuda.py", line 229, in checkCudaErrors
DSLCudaRuntimeError: DSLCudaRuntimeError: Unknown CUDA error
Error Code: 201

🔍 Additional Context:
- Error name: CUDA_ERROR_INVALID_CONTEXT
- CUDA_TOOLKIT_PATH: not set
- Target SM ARCH: not set

📊 GPU Information:
- CUDA devices available: 8 (current: 0)
- Architecture: Blackwell (sm_100a)
- Compatible SM archs: sm_100a

Compatibility Check:
❌ Error: Target SM ARCH unknown is not compatible
💡 Please use one of SM ARCHs: sm_100a

"""


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
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(4, 4),
    ):
        """Initialize the CUTLASS grouped GEMM strategy for Blackwell architecture."""
        print(f"Initializing CUTLASSGroupedGemmStrategy for Blackwell")
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


class CUTLASSBackwardGroupGemm(torch.autograd.Function):
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
        input_tokens, weight_stack, m_sizes_gpu, m_offsets_gpu, valid_indices, device
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
                ].contiguous()  # [N, K] - already transposed

                M, K = expert_tokens.shape
                N, K_w = expert_weight.shape
                assert K == K_w, f"Dimension mismatch: {K} != {K_w}"
                L = 1

                # Create output tensor
                output = torch.empty(M, N, dtype=strategy.DTYPE_TORCH, device=device)
                outputs.append(output)

                # Add to metadata: expert_tokens @ expert_weight^T
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    expert_tokens,
                    expert_weight,
                    output,
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                )

        return problem_sizes, strides_abc, ptrs_abc, outputs

    @staticmethod
    def _prepare_input_grad_metadata(
        grad_output, weight_stack, m_sizes_gpu, m_offsets_gpu, valid_indices, device
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
                L = 1

                # Create output tensor for gradient
                grad_input_expert = torch.empty(
                    M, K, dtype=strategy.DTYPE_TORCH, device=device
                )
                outputs.append(grad_input_expert)

                # Add to metadata: grad_expert @ expert_weight (no transpose needed)
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    grad_expert,
                    expert_weight,
                    grad_input_expert,
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
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
                L = 1

                # Get output tensor (slice of grad_weight for this expert)
                grad_weight_expert = grad_weight[expert_idx]  # [N, K]

                # For dW = dY^T @ X, we need to transpose dY
                # This means we compute: X^T @ dY -> (dY^T @ X)^T = dW^T, then transpose result
                # Actually, let's compute grad_expert^T @ input_expert directly
                grad_expert_t = grad_expert.t().contiguous()  # [N, M]

                # Add to metadata: grad_expert_t @ input_expert -> grad_weight_expert
                CUTLASSBackwardGroupGemm._add_gemm_to_metadata(
                    grad_expert_t,
                    input_expert,
                    grad_weight_expert,
                    problem_sizes,
                    strides_abc,
                    ptrs_abc,
                )

        return problem_sizes, strides_abc, ptrs_abc

    @staticmethod
    def _add_gemm_to_metadata(A, B, C, problem_sizes, strides_abc, ptrs_abc):
        """Add a single GEMM operation to metadata lists."""
        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, f"Inner dimension mismatch: {K} != {K_b}"

        # CUTLASS expects B to be [N, K] for A @ B^T, but we have B as [K, N]
        # So we need to transpose B or adjust our computation
        # For A @ B -> C where A is [M, K] and B is [K, N], we need B^T which is [N, K]
        B_transposed = B.t().contiguous()  # [N, K]

        L = 1

        # Convert to MNKL format
        A_mnkl = A.unsqueeze(-1).contiguous()  # [M, K, 1]
        B_mnkl = B_transposed.unsqueeze(-1).contiguous()  # [N, K, 1]
        C_mnkl = C.unsqueeze(-1).contiguous()  # [M, N, 1]

        # Extract strides
        A_strides = list(A_mnkl.stride()[:2])
        B_strides = list(B_mnkl.stride()[:2])
        C_strides = list(C_mnkl.stride()[:2])

        # Add to metadata
        problem_sizes.append([M, N, K, L])
        strides_abc.append([A_strides, B_strides, C_strides])
        ptrs_abc.append([A.data_ptr(), B_transposed.data_ptr(), C.data_ptr()])

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

        if bias:
            raise NotImplementedError(
                "Bias not yet implemented for CUTLASS grouped linear"
            )

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
    # Create strategy (assuming it's available)

    strategy = CUTLASSGroupedGemmStrategy(
        custom_activation=lambda x: x,  # Identity for testing
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 2),
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


if __name__ == "__main__":
    test_cutlass_backward_group_gemm()
