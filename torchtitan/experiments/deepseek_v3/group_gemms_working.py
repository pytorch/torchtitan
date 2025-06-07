# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

"""

# init_logger()
import logging
from logging import Logger

import torch
import torch.nn as nn

# from torchtitan.tools.logging import init_logger, logger

# logger.setLevel("INFO")
# logging.getLogger("cutlass").setLevel(logging.WARNING)

try:
    import deep_gemm

    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_AVAILABLE = False

if DEEPGEMM_AVAILABLE:
    import dsgemm_kernels
    import dsgemm_utils

try:
    from torchao.float8.config import ScalingGranularity
    from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

    TORCHAO_FP8_GG_AVAILABLE = True

except ImportError:
    TORCHAO_FP8_GG_AVAILABLE = False

try:
    from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
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
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell_group_gemms.cute_grouped_gemm import (
        GroupedGemmKernel,
    )
    from torchtitan.experiments.kernels.blackwell_group_gemms.dense_gemm import (
        DenseGemmKernel,
    )
    from torchtitan.experiments.kernels.blackwell_group_gemms.tensor_cute_converter import (
        GemmTensorConverter,
    )

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    assert False, "CUTLASS is not available"


# Export all strategies
__all__ = [
    "ManualLoopGroupGEMM",
    "CuteDenseLoopingGroupGEMM",
    "TorchBF16GroupGEMM",
    "TorchAOBF16GroupGEMM",
    "TritonCGBF16GroupGEMM",
    "DSGroupGEMM",
]


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


# ========= Manual Looping Baselines ===================


class CuteDenseLoopingGroupGEMM(GroupGEMMStrategy):
    """
    Implementation of grouped GEMM using CUTLASS CUTE Dense GEMM kernel with manual looping.

    This class provides an optimized way to execute multiple GEMM operations for MoE models
    by using the CUTLASS CUTE API with Tensor Cores on NVIDIA GPUs.
    """

    def __init__(self, custom_activation):
        """
        Initialize the CuteDenseLoopingGroupGEMM.

        Args:
            custom_activation: Activation function to use between gate and up projections
        """
        super().__init__(custom_activation)
        self.alignment = 16
        self.dtype = torch.bfloat16
        self.cutlass_dtype = cutlass.BFloat16

        # Setup logging
        self.logger = logging.getLogger("CuteDenseLoopingGroupGEMM")

        # Create GEMM kernel with optimized parameters for Blackwell architecture
        try:
            self.gemm_kernel = DenseGemmKernel(
                acc_dtype=cutlass.Float32,  # Accumulator type
                use_2cta_instrs=False,  # Paired CTA
                mma_tiler_mn=(128, 128),  # Tile size
                cluster_shape_mn=(2, 2),  # Cluster shape
                use_tma_store=True,  # Use TMA for store operations
            )
            self.logger.debug("GEMM kernel created successfully")
        except Exception as e:
            self.logger.error(f"Kernel setup failed: {e}")
            raise RuntimeError(f"Failed to create GEMM kernel: {e}")

        # Setup CUDA stream
        torch_stream = torch.cuda.Stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Compiled kernel cache - keyed by input/output shapes
        self.kernel_cache = {}

        # Debug mode - set to False in production
        self.debug = False

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute grouped GEMM operations using manual loops over experts.

        Args:
            contig_tokens: Contiguous token tensor [total_tokens, hidden_size]
            m_sizes: List of token counts per expert
            m_offsets: List of token offsets per expert
            module: Module containing the expert weights

        Returns:
            Processed output tensor [total_tokens, hidden_size]
        """
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Prepare output tensor
        hidden_size = w_gate.shape[2]  # [num_experts, out_dim, in_dim]
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

                # Execute gate projection
                gate_out = self._execute_gemm(
                    expert_tokens,
                    gate_weight,
                    f"gate_{expert_idx}",
                    expert_tokens.shape[0],
                    gate_weight.shape[0],
                )

                # Execute up projection
                up_out = self._execute_gemm(
                    expert_tokens,
                    up_weight,
                    f"up_{expert_idx}",
                    expert_tokens.shape[0],
                    up_weight.shape[0],
                )

                # Apply activation and combine
                hidden = self.activation_function(gate_out) * up_out

                # Execute down projection
                expert_output = self._execute_gemm(
                    hidden,
                    down_weight,
                    f"down_{expert_idx}",
                    hidden.shape[0],
                    down_weight.shape[0],
                )

                # Store results
                output[offset : offset + size] = expert_output

            offset += size

        if self.debug:
            self.logger.debug(f"GEMM output shape: {output.shape}")

        return output

    def _execute_gemm(self, input_tensor, weight, kernel_name, M, N):
        """
        Execute a single GEMM operation using the CUTLASS CUTE Dense GEMM kernel.

        Args:
            input_tensor: Input tensor of shape [M, K]
            weight: Weight tensor of shape [N, K]
            kernel_name: Name for the kernel (for caching)
            M: Number of rows in input_tensor
            N: Number of rows in weight

        Returns:
            Output tensor of shape [M, N]
        """
        # Ensure tensors are contiguous
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        # Get dimensions
        K = input_tensor.shape[1]

        # Create output tensor with correct dimensions
        output = torch.zeros(
            (M, N), device=input_tensor.device, dtype=self.dtype, requires_grad=False
        )

        # Convert to MNKL format (add batch dimension)
        A_mnkl = input_tensor.unsqueeze(-1).contiguous().detach()  # [M, K, 1]
        B_mnkl = weight.unsqueeze(-1).contiguous().detach()  # [N, K, 1]
        C_mnkl = output.unsqueeze(-1).contiguous()  # [M, N, 1]

        # Create CUTE tensors
        A_cute = from_dlpack(A_mnkl, assumed_align=self.alignment)
        B_cute = from_dlpack(B_mnkl, assumed_align=self.alignment)
        C_cute = from_dlpack(C_mnkl, assumed_align=self.alignment)

        # Set data types
        A_cute.element_type = self.cutlass_dtype
        B_cute.element_type = self.cutlass_dtype
        C_cute.element_type = self.cutlass_dtype

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        # Get or compile kernel
        cache_key = (M, N, K, kernel_name)
        if cache_key not in self.kernel_cache:
            try:
                self.kernel_cache[cache_key] = cute.compile(
                    self.gemm_kernel, A_cute, B_cute, C_cute, self.stream
                )
                if self.debug:
                    self.logger.debug(
                        f"Compiled kernel for {kernel_name} with shape [{M}, {N}, {K}]"
                    )
            except Exception as e:
                self.logger.error(f"Kernel compilation failed for {kernel_name}: {e}")
                raise RuntimeError(f"Failed to compile kernel for {kernel_name}: {e}")

        # Execute kernel
        try:
            self.kernel_cache[cache_key](A_cute, B_cute, C_cute, self.stream)
            if self.debug:
                self.logger.debug(f"Executed kernel {kernel_name} successfully")
        except Exception as e:
            self.logger.error(f"Kernel execution failed for {kernel_name}: {e}")
            raise RuntimeError(f"Failed to execute kernel for {kernel_name}: {e}")

        # Return output tensor
        return C_mnkl.squeeze(-1)

    @staticmethod
    def is_available() -> bool:
        return True


# ======================== new
class CuteDenseLoopingGroupGEMM_detail(GroupGEMMStrategy):
    """
    Implementation of grouped GEMM using Blackwell Dense GEMM kernel with manual looping.

    This class provides a way to execute multiple GEMM operations with different problem
    sizes using the Blackwell Dense GEMM kernel. It loops through each expert and
    executes the kernel for each matrix multiplication.
    """

    def __init__(self, custom_activation):
        """
        Initialize the CuteDenseLoopingGroupGEMM.

        Args:
            custom_activation: Activation function to use
        """
        super().__init__(custom_activation)

        # Kernel configuration
        self.alignment = 16
        self.dtype = torch.bfloat16
        self.cutlass_dtype = cutlass.BFloat16

        # Initialize Dense GEMM kernel
        try:
            self.gemm_kernel = DenseGemmKernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=False,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                use_tma_store=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GEMM kernel: {e}")

        # Setup CUDA stream
        torch_stream = torch.cuda.Stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Cache for compiled kernels
        self._compiled_kernels = {}

        # Performance monitoring
        self.debug_mode = False

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def _create_cute_tensor(self, tensor: torch.Tensor) -> cute.Tensor:
        """
        Convert a PyTorch tensor to a CUTE tensor with proper formatting.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            CUTE tensor ready for kernel execution
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Convert to MNKL format
        tensor_mnkl = tensor.unsqueeze(-1).contiguous().detach()

        # Create CUTE tensor
        cute_tensor = from_dlpack(tensor_mnkl, assumed_align=self.alignment)
        cute_tensor.element_type = self.cutlass_dtype
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)

        return cute_tensor

    def _get_or_compile_kernel(self, a_cute, b_cute, c_cute, operation_name: str):
        """
        Get a compiled kernel from cache or compile a new one.

        Args:
            a_cute: Input tensor A in CUTE format
            b_cute: Input tensor B in CUTE format
            c_cute: Output tensor C in CUTE format
            operation_name: Name of the operation for caching

        Returns:
            Compiled CUTE kernel
        """
        cache_key = f"{operation_name}_{a_cute.shape}_{b_cute.shape}_{c_cute.shape}"

        if cache_key not in self._compiled_kernels:
            try:
                self._compiled_kernels[cache_key] = cute.compile(
                    self.gemm_kernel, a_cute, b_cute, c_cute, self.stream
                )
                if self.debug_mode:
                    print(f"✓ Compiled kernel for {operation_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to compile {operation_name} kernel: {e}")

        return self._compiled_kernels[cache_key]

    def _execute_gemm_operation(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, operation_name: str
    ) -> torch.Tensor:
        """
        Execute a single GEMM operation using the CUTE kernel.

        Args:
            input_tensor: Input tensor [M, K]
            weight: Weight tensor [N, K]
            operation_name: Name of the operation for debugging

        Returns:
            Output tensor [M, N]
        """
        batch_size, input_dim = input_tensor.shape
        output_dim = weight.shape[0]

        # Create output tensor
        output = torch.zeros(
            (batch_size, output_dim),
            device=input_tensor.device,
            dtype=self.dtype,
            requires_grad=False,
        )

        # Convert tensors to CUTE format
        try:
            a_cute = self._create_cute_tensor(input_tensor)
            b_cute = self._create_cute_tensor(weight)
            c_cute = self._create_cute_tensor(output)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CUTE tensors for {operation_name}: {e}"
            )

        # Get or compile kernel
        compiled_kernel = self._get_or_compile_kernel(
            a_cute, b_cute, c_cute, operation_name
        )

        # Execute kernel
        try:
            compiled_kernel(a_cute, b_cute, c_cute, self.stream)
            if self.debug_mode:
                print(f"✓ Executed {operation_name} kernel successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to execute {operation_name} kernel: {e}")

        return output.squeeze(-1) if output.dim() > 2 else output

    def _process_expert(
        self,
        expert_tokens: torch.Tensor,
        expert_idx: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process tokens through a single expert using CUTE kernels.

        Args:
            expert_tokens: Tokens for this expert [num_tokens, hidden_size]
            expert_idx: Index of the expert
            w_gate: Gate projection weights [intermediate_size, hidden_size]
            w_up: Up projection weights [intermediate_size, hidden_size]
            w_down: Down projection weights [hidden_size, intermediate_size]

        Returns:
            Expert output [num_tokens, hidden_size]
        """
        # Gate projection
        gate_out = self._execute_gemm_operation(
            expert_tokens, w_gate, f"gate_expert_{expert_idx}"
        )

        # Up projection
        up_out = self._execute_gemm_operation(
            expert_tokens, w_up, f"up_expert_{expert_idx}"
        )

        # Apply activation and combine
        hidden = self.activation_function(gate_out) * up_out

        # Down projection
        expert_output = self._execute_gemm_operation(
            hidden, w_down, f"down_expert_{expert_idx}"
        )

        return expert_output

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute the complete grouped GEMM operation.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: MoE module containing weights and parameters

        Returns:
            Processed tokens
        """
        try:
            # Get weights
            device = contig_tokens.device
            w_gate = module.get_parameter("gate_proj_weight")
            w_up = module.get_parameter("up_proj_weight")
            w_down = module.get_parameter("down_proj_weight")

            # Validate inputs
            if len(m_sizes) != w_gate.shape[0]:
                raise ValueError(
                    f"Number of experts mismatch: {len(m_sizes)} vs {w_gate.shape[0]}"
                )

            # Prepare output tensor
            hidden_size = w_gate.shape[2] if len(w_gate.shape) > 2 else w_gate.shape[1]
            output = torch.zeros(
                contig_tokens.shape[0],
                hidden_size,
                dtype=contig_tokens.dtype,
                device=device,
            )

            # Process each expert
            offset = 0
            active_experts = 0

            for expert_idx, size in enumerate(m_sizes):
                if size > 0:
                    # Get tokens and weights for this expert
                    expert_tokens = contig_tokens[offset : offset + size]
                    expert_gate_weight = w_gate[expert_idx]
                    expert_up_weight = w_up[expert_idx]
                    expert_down_weight = w_down[expert_idx]

                    # Process through expert
                    expert_output = self._process_expert(
                        expert_tokens,
                        expert_idx,
                        expert_gate_weight,
                        expert_up_weight,
                        expert_down_weight,
                    )

                    # Store results
                    output[offset : offset + size] = expert_output
                    active_experts += 1

                offset += size

            if self.debug_mode:
                print(
                    f"Processed {active_experts} active experts out of {len(m_sizes)} total"
                )

            return output

        except Exception as e:
            # Fallback to PyTorch implementation on error
            if self.debug_mode:
                print(f"CUTE kernel failed, falling back to PyTorch: {e}")
            return self._fallback_pytorch(contig_tokens, m_sizes, module)

    def _fallback_pytorch(self, contig_tokens, m_sizes, module):
        """
        Fallback implementation using standard PyTorch operations.

        Args:
            contig_tokens: Input tokens
            m_sizes: Group sizes
            module: MoE module

        Returns:
            Processed tokens using PyTorch mm
        """
        device = contig_tokens.device
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        hidden_size = w_gate.shape[2] if len(w_gate.shape) > 2 else w_gate.shape[1]
        output = torch.zeros(
            contig_tokens.shape[0],
            hidden_size,
            dtype=contig_tokens.dtype,
            device=device,
        )

        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                expert_tokens = contig_tokens[offset : offset + size]

                # Standard PyTorch forward pass
                gate_out = torch.mm(expert_tokens, w_gate[expert_idx].t())
                up_out = torch.mm(expert_tokens, w_up[expert_idx].t())
                hidden = self.activation_function(gate_out) * up_out
                expert_output = torch.mm(hidden, w_down[expert_idx].t())

                output[offset : offset + size] = expert_output

            offset += size

        return output

    def clear_cache(self):
        """Clear the compiled kernel cache to free memory."""
        self._compiled_kernels.clear()
        if self.debug_mode:
            print("Cleared compiled kernel cache")

    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode."""
        self.debug_mode = enabled

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system."""
        try:
            return CUTLASS_AVAILABLE and torch.cuda.is_available()
        except:
            return False


# =========
class ManualLoopGroupGEMM(GroupGEMMStrategy):
    """Manual looping baseline implementation for comparison"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using manual loops over experts"""
        # Get weights and ensure they're on the same device as tokens
        device = contig_tokens.device
        w_gate = module.get_parameter("gate_proj_weight").to(device)
        w_up = module.get_parameter("up_proj_weight").to(device)
        w_down = module.get_parameter("down_proj_weight").to(device)

        # Prepare output tensor
        hidden_size = w_gate.shape[
            2
        ]  # Assuming stacked weights shape [num_experts, out_dim, in_dim]
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
