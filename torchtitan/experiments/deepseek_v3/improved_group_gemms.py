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
