import torch
import triton
import triton.language as tl

from cg_forward import cg_grouped_gemm_forward, early_config_prune, STANDARD_CONFIGS

# ============ Triton kernel for contiguous grouped GEMM backward inputs ============


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_backward_inputs(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    b_ptr,  # expert weights [num_experts, N, K]
    grad_input_ptr,  # [M_TOTAL, K]
    # Pointer to indices array
    indices_ptr,  # [M_TOTAL / GROUP_SIZE_M]
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Computes the gradient with respect to the inputs (backward pass).
    Essentially performs: grad_input = grad_output @ expert_weights
    """
    pid = tl.program_id(0)

    # number of tiles per matrix dimension
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # 2D tile index from linear
    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx)

        # Initialize accumulator for the gradient
        grad_input = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

        # Compute gradient with respect to inputs in tiles along N dimension
        for n in range(0, N, BLOCK_SIZE_N):
            # offsets and mask for N dimension
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n
            mask_n = offs_n < N

            # Masks for grad_output and weights
            mask_go = mask_m[:, None] & mask_n[None, :]
            mask_w = mask_k[:, None] & mask_n[None, :]

            # Load grad_output with bounds checking
            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0)

            # Load expert weights for the expert assigned to this block
            # Note: transposed access pattern for matmul
            w_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Compute partial gradient for this N tile: grad_input += grad_output @ weights
            grad_input += tl.dot(go, w)

        # Store results with bounds checking
        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_gi = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_input_ptrs, grad_input, mask=mask_gi)


# ============ Triton kernel for contiguous grouped GEMM backward weights ============


@triton.jit
def _kernel_cg_backward_weights_single_group(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    # Expert index and group boundaries
    expert_idx: tl.constexpr,  # Expert ID for this kernel
    group_start: tl.constexpr,  # Start index of group
    group_size: tl.constexpr,  # Size of group
    # Matrix dimensions
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes the gradient with respect to weights for a single expert group.
    Each kernel instance handles exactly one group for one expert.
    """
    pid = tl.program_id(0)

    # 2D tile index within the expert's grid
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tile_n = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # Starting indices for this tile within the expert's weight matrix
    n_start = tile_n * BLOCK_SIZE_N
    k_start = tile_k * BLOCK_SIZE_K

    # Only process if the indices are in bounds
    if n_start < N and k_start < K:

        # Offsets for this tile
        offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        # Masks for bounds checking
        mask_n = offs_n < N
        mask_k = offs_k < K

        # Initialize accumulator for the gradient
        grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

        # Break the group into manageable blocks for processing
        for m_offset in range(0, group_size, BLOCK_SIZE_N):
            # Calculate current block size (might be smaller at the end)
            curr_block_size = min(BLOCK_SIZE_N, group_size - m_offset)

            # Calculate actual M indices
            offs_m = tl.arange(0, curr_block_size) + group_start + m_offset

            # Masks for data loading
            mask_m = offs_m < (group_start + group_size)
            mask_go = mask_m[:, None] & mask_n[None, :]
            mask_in = mask_m[:, None] & mask_k[None, :]

            # Load grad_output and inputs
            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0)

            in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
            inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

            # Accumulate gradient: grad_weights += grad_output.T @ inputs
            grad_weights += tl.dot(go.T, inp)

        # Store results with bounds checking
        grad_weights_ptrs = (
            grad_weights_ptr
            + expert_idx * N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )
        mask_gw = mask_n[:, None] & mask_k[None, :]

        # Atomic add to handle multiple groups updating the same expert weights
        tl.atomic_add(grad_weights_ptrs, grad_weights, mask=mask_gw)


# =============== Functions for backward pass =================


def cg_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,  # [M_total, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [num_groups]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to inputs.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        expert_weights: Expert weight tensor, shape [num_experts, N, K]
        expert_indices: Indices tensor mapping each group to its expert, shape [num_groups]
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_inputs: Gradient with respect to inputs, shape [M_total, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    num_experts, _, K = expert_weights.shape

    # Check if dimensions match
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Create output tensor for gradients
    grad_inputs = torch.empty(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Calculate grid size for the kernel
    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )

    # Launch kernel
    _kernel_cg_backward_inputs[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )

    return grad_inputs


def cg_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,  # [M_total, N]
    inputs: torch.Tensor,  # [M_total, K]
    expert_indices: torch.Tensor,  # [num_groups]
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to expert weights.
    Uses a separate kernel launch for each group to avoid dealing with
    dynamic control flow inside the kernel.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        inputs: Input tensor, shape [M_total, K]
        expert_indices: Indices tensor mapping each group to its expert, shape [num_groups]
        num_experts: Number of experts
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    _, K = inputs.shape
    num_groups = M_total // group_size_m

    # Check if dimensions match
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"
    assert (
        expert_indices.shape[0] == num_groups
    ), "Expert indices length must match number of groups"

    # Create output tensor for gradients (initialized to zeros)
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Process each group separately to avoid conditional control flow in kernel
    for group_idx in range(num_groups):
        # Get expert ID for this group
        expert_idx = int(expert_indices[group_idx].item())

        # Calculate start index for this group
        group_start = group_idx * group_size_m

        # Calculate grid size for this group - cover the expert's N×K matrix
        grid = (triton.cdiv(N, 64) * triton.cdiv(K, 64),)

        # Launch kernel for this specific group
        _kernel_cg_backward_weights_single_group[grid](
            grad_output,
            inputs,
            grad_weights,
            expert_idx=expert_idx,
            group_start=group_start,
            group_size=group_size_m,
            N=N,
            K=K,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=64,
        )

    return grad_weights


# =============== Update the autograd function =================


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with complete backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=128):
        """Forward pass for contiguous grouped GEMM."""
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.group_size_m = group_size_m

        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for contiguous grouped GEMM."""
        inputs, expert_weights, expert_indices = ctx.saved_tensors
        group_size_m = ctx.group_size_m

        # Get number of experts
        num_experts = expert_weights.shape[0]

        # Compute gradients
        grad_inputs = cg_grouped_gemm_backward_inputs(
            grad_output=grad_output,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

        grad_weights = cg_grouped_gemm_backward_weights(
            grad_output=grad_output,
            inputs=inputs,
            expert_indices=expert_indices,
            num_experts=num_experts,
            group_size_m=group_size_m,
        )

        # No gradient for expert_indices (it's just an index tensor)
        grad_indices = None

        # No gradient for group_size_m (it's just a parameter)
        grad_group_size_m = None

        return grad_inputs, grad_weights, grad_indices, grad_group_size_m


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM with full backward pass support.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [num_groups] mapping each group to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(
        inputs, expert_weights, expert_indices, group_size_m
    )


# =============== Test functions for verifying correctness =================


def verify_cg_gemm_backward(
    M_total=1024,
    N=512,
    K=512,
    num_experts=8,
    group_size_m=128,
    device="cuda",
    atol=1e-2,
    rtol=1e-2,
):
    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // group_size_m) * group_size_m
    num_groups = M_total // group_size_m

    # Create test tensors
    torch.manual_seed(0)
    inputs = torch.randn((M_total, K), device=device, requires_grad=True)
    expert_weights = torch.randn((num_experts, N, K), device=device, requires_grad=True)

    # Create group-level expert indices
    group_indices = torch.randint(
        0, num_experts, (num_groups,), device=device, dtype=torch.int32
    )

    # Expand to token-level indices for the forward function
    expert_indices = torch.zeros(M_total, device=device, dtype=torch.int32)
    for g in range(num_groups):
        group_start = g * group_size_m
        group_end = (g + 1) * group_size_m
        expert_indices[group_start:group_end] = group_indices[g]

    # Create a target for gradient computation
    target = torch.randn((M_total, N), device=device)

    # PyTorch reference implementation
    outputs_ref = torch.zeros((M_total, N), device=device)
    for g in range(num_groups):
        group_start = g * group_size_m
        group_end = (g + 1) * group_size_m
        expert_idx = group_indices[g].item()

        # Compute output for this group
        outputs_ref[group_start:group_end] = (
            inputs[group_start:group_end] @ expert_weights[expert_idx].t()
        )

    # Compute loss and gradients with PyTorch
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = expert_weights.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    expert_weights.grad.zero_()

    # Compute with our implementation
    outputs = cg_grouped_gemm(inputs, expert_weights, expert_indices, group_size_m)
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    # Check if outputs match
    outputs_match = torch.allclose(outputs, outputs_ref, atol=atol, rtol=rtol)
    print(f"Outputs match: {outputs_match}")

    # Check if gradients match
    inputs_grad_match = torch.allclose(
        inputs.grad, grad_inputs_ref, atol=atol, rtol=rtol
    )
    weights_grad_match = torch.allclose(
        expert_weights.grad, grad_weights_ref, atol=atol, rtol=rtol
    )

    print(f"Input gradients match: {inputs_grad_match}")
    print(f"Weight gradients match: {weights_grad_match}")

    if not inputs_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(inputs.grad - grad_inputs_ref))
        print(f"Max difference in input gradients: {max_diff}")

    if not weights_grad_match:
        # Compute max absolute difference for debugging
        max_diff = torch.max(torch.abs(expert_weights.grad - grad_weights_ref))
        print(f"Max difference in weight gradients: {max_diff}")

    return inputs_grad_match, weights_grad_match


if __name__ == "__main__":
    # Run verification test
    verify_cg_gemm_backward()
