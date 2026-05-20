import torch
import triton
import triton.language as tl


@triton.jit
def _compute_permutation_indices_kernel(
    # Output pointer for permuted indices
    permuted_indices_ptr,  # [total_indices]
    # Output pointer for expert token sizes
    m_sizes_ptr,  # [experts_per_rank]
    # Input pointer for tokens per expert group
    tokens_per_expert_group_ptr,  # [n_routed_experts]
    # Dimensions
    n_routed_experts,
    experts_per_rank,
    ep_size,
    total_padded_tokens,
    alignment: tl.constexpr,
    max_experts: tl.constexpr,  # Maximum number of experts (must be power of 2)
    pad_value: tl.constexpr = -1,
):
    """
    Kernel to compute permutation indices for MoE token routing.

    This kernel implements the PyTorch-based permutation logic in Triton:
    1. Computes offsets from tokens_per_expert_group
    2. For each local expert, collects indices from all remote ranks
    3. Adds padding to align each expert's token group

    Args:
        permuted_indices_ptr: Output pointer for permuted indices
        m_sizes_ptr: Output pointer for expert token sizes (padded)
        tokens_per_expert_group_ptr: Input pointer for tokens per expert group
        n_routed_experts: Total number of experts across all ranks
        experts_per_rank: Number of experts per rank
        ep_size: Number of expert parallel ranks
        total_padded_tokens: Total size of the permuted indices array
        alignment: Alignment requirement for token group sizes (e.g., 128)
        max_experts: Maximum number of experts (must be power of 2)
        pad_value: Value to use for padding (-1)
    """
    # Get the local expert ID this program instance handles
    local_expert_idx = tl.program_id(0)

    # Process only if within valid range
    if local_expert_idx < experts_per_rank:
        # Step 1: Compute prefix sum offsets for all expert groups
        # Use a fixed-size array that's a power of 2
        offsets = tl.zeros([1024], dtype=tl.int32)

        # Compute prefix sum manually
        running_sum = 0
        for i in range(n_routed_experts):
            if i < max_experts:  # Ensure we don't go out of bounds
                tokens = tl.load(tokens_per_expert_group_ptr + i)
                offsets[i] = running_sum
                running_sum += tokens

        # Step 2: Calculate the output offset for this local expert
        output_offset = 0
        for e in range(local_expert_idx):
            e_tokens_total = 0
            # Calculate total tokens for previous experts (across all ranks)
            for r in range(ep_size):
                e_idx = r * experts_per_rank + e
                if e_idx < n_routed_experts:
                    e_tokens = tl.load(tokens_per_expert_group_ptr + e_idx)
                    e_tokens_total += e_tokens

            # Add padding for alignment and update offset
            padded_size = (
                e_tokens_total + (alignment - e_tokens_total % alignment) % alignment
            )
            output_offset += padded_size

        # Step 3: Process tokens for this expert across all remote ranks
        total_expert_tokens = 0
        cur_pos = output_offset

        # For each remote rank, gather indices
        for r in range(ep_size):
            expert_idx = r * experts_per_rank + local_expert_idx

            # Skip if expert index is out of bounds
            if expert_idx < n_routed_experts:
                # Get number of tokens and starting offset for this expert on this rank
                num_tokens = tl.load(tokens_per_expert_group_ptr + expert_idx)
                start_offset = 0
                if expert_idx < max_experts:  # Safety check
                    start_offset = offsets[expert_idx]

                # Generate indices for this expert group and write them to output
                # Process in blocks of max_block_size
                max_block_size = 128  # Smaller block size for better parallelism
                for block_start in range(0, num_tokens, max_block_size):
                    # Create block indices
                    block_idx = block_start + tl.arange(0, max_block_size)
                    # Mask for valid indices in this block
                    mask = block_idx < num_tokens
                    if tl.sum(mask) > 0:  # Only continue if any thread has valid work
                        # Calculate source index with masking
                        src_idx = tl.where(mask, start_offset + block_idx, 0)

                        # Ensure source index is never negative (though this should never happen)
                        src_idx = tl.maximum(src_idx, 0)

                        # Calculate destination with masking
                        dst_idx = tl.where(mask, cur_pos + block_idx, 0)

                        # Write with masking
                        tl.store(permuted_indices_ptr + dst_idx, src_idx, mask=mask)

                # Update position and count
                cur_pos += num_tokens
                total_expert_tokens += num_tokens

        # Step 4: Add padding for alignment
        padding_needed = (alignment - total_expert_tokens % alignment) % alignment

        # Write padding values (pad_value) at the end of this expert's section
        # Process padding in blocks
        max_block_size = 128  # Smaller block size for better parallelism
        for block_start in range(0, padding_needed, max_block_size):
            # Create block indices
            block_idx = block_start + tl.arange(0, max_block_size)
            # Mask for valid indices in this block
            mask = block_idx < padding_needed
            if tl.sum(mask) > 0:  # Only continue if any thread has valid work
                dst_idx = tl.where(mask, cur_pos + block_idx, 0)
                tl.store(permuted_indices_ptr + dst_idx, pad_value, mask=mask)

        # Step 5: Store the padded size for this expert
        total_padded = total_expert_tokens + padding_needed
        tl.store(m_sizes_ptr + local_expert_idx, total_padded)


@triton.jit
def _apply_permutation_kernel(
    # Input pointer
    input_ptr,  # [total_tokens, hidden_dim]
    # Output pointer
    output_ptr,  # [total_tokens, hidden_dim]
    # Permutation indices
    indices_ptr,  # [total_tokens]
    # Dimensions
    total_tokens,
    hidden_dim,
    BLOCK_SIZE_M: tl.constexpr,  # Block size for token dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for hidden dimension
):
    """
    Kernel to apply permutation to tokens.

    Args:
        input_ptr: Input tensor pointer [total_tokens, hidden_dim]
        output_ptr: Output tensor pointer [total_tokens, hidden_dim]
        indices_ptr: Permutation indices [total_tokens]
        total_tokens: Total number of tokens
        hidden_dim: Hidden dimension size
        BLOCK_SIZE_M: Block size for token dimension
        BLOCK_SIZE_N: Block size for hidden dimension
    """
    # Get program ID
    pid_m = tl.program_id(0)  # Block index in token dimension
    pid_n = tl.program_id(1)  # Block index in hidden dimension

    # Calculate start indices
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Create offsets
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create masks for valid elements
    mask_m = offs_m < total_tokens
    mask_n = offs_n < hidden_dim

    # Process tokens in a vectorized way where possible
    # Load all permutation indices for this block
    token_indices = start_m + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_indices < total_tokens

    # Process each token in the block
    for m_idx in range(BLOCK_SIZE_M):
        if start_m + m_idx >= total_tokens:
            break

        # Load permutation index
        perm_idx = tl.load(indices_ptr + start_m + m_idx)

        # Process only valid source indices (not padding)
        if perm_idx >= 0:
            # Load input values with mask for hidden dimension
            input_row_ptr = input_ptr + perm_idx * hidden_dim

            # Load the entire row with mask
            input_values = tl.load(input_row_ptr + offs_n, mask=mask_n, other=0.0)

            # Store to output with mask
            output_row_ptr = output_ptr + (start_m + m_idx) * hidden_dim
            tl.store(output_row_ptr + offs_n, input_values, mask=mask_n)


def compute_permutation_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
    alignment: int = 128,
    pad_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute permutation indices for MoE token routing using Triton.

    Args:
        tokens_per_expert_group: Number of tokens per expert group [n_routed_experts]
        experts_per_rank: Number of experts per rank
        ep_size: Number of expert parallel ranks
        alignment: Alignment requirement for token group sizes (default: 128)
        pad_value: Value to use for padding (default: -1)

    Returns:
        Tuple of (permuted_indices, m_sizes)
    """
    device = tokens_per_expert_group.device
    n_routed_experts = tokens_per_expert_group.shape[0]

    # Estimate total number of tokens (including padding)
    total_tokens = tokens_per_expert_group.sum().item()
    # Add potential padding (worst case: each expert needs alignment-1 padding)
    # Also ensure we have a bit of extra space to avoid any potential overflow
    total_tokens_padded = total_tokens + experts_per_rank * alignment

    # Allocate output tensors
    permuted_indices = torch.full(
        (total_tokens_padded,), pad_value, dtype=torch.int32, device=device
    )
    m_sizes = torch.empty(experts_per_rank, dtype=torch.int32, device=device)

    # Launch kernel
    grid = (experts_per_rank,)
    _compute_permutation_indices_kernel[grid](
        permuted_indices,
        m_sizes,
        tokens_per_expert_group,
        n_routed_experts,
        experts_per_rank,
        ep_size,
        total_tokens_padded,
        alignment,
        pad_value,
    )

    return permuted_indices, m_sizes


def apply_permutation(
    input_tensor: torch.Tensor,
    permuted_indices: torch.Tensor,
    output_shape: torch.Size,
) -> torch.Tensor:
    """
    Apply permutation to tokens using Triton.

    Args:
        input_tensor: Input tensor [total_tokens, hidden_dim]
        permuted_indices: Permutation indices [total_tokens]
        output_shape: Shape of output tensor

    Returns:
        Permuted tokens
    """
    device = input_tensor.device
    total_tokens, hidden_dim = input_tensor.shape

    # Allocate output tensor
    output_tensor = torch.zeros(output_shape, dtype=input_tensor.dtype, device=device)

    # Calculate optimal block sizes
    BLOCK_SIZE_M = 32  # Token dimension block size
    BLOCK_SIZE_N = min(
        128, triton.next_power_of_2(hidden_dim)
    )  # Hidden dimension block size

    # Calculate grid size
    grid = (
        triton.cdiv(total_tokens, BLOCK_SIZE_M),
        triton.cdiv(hidden_dim, BLOCK_SIZE_N),
    )

    # Launch kernel
    _apply_permutation_kernel[grid](
        input_tensor,
        output_tensor,
        permuted_indices,
        total_tokens,
        hidden_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )

    return output_tensor


def optimized_token_permutation(
    token_gather_buf: torch.Tensor,
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
    alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized token permutation for MoE routing using Triton.

    Args:
        token_gather_buf: Gathered tokens [total_tokens, hidden_dim]
        tokens_per_expert_group: Number of tokens per expert group [n_routed_experts]
        experts_per_rank: Number of experts per rank
        ep_size: Number of expert parallel ranks
        alignment: Alignment requirement for token group sizes (default: 128)

    Returns:
        Tuple of (permuted_tokens, m_sizes)
    """
    # Step 1: Compute permutation indices
    permuted_indices, m_sizes = compute_permutation_indices(
        tokens_per_expert_group,
        experts_per_rank,
        ep_size,
        alignment,
        pad_value=-1,
    )

    # For debugging: compare with PyTorch reference implementation
    # This section can be uncommented to help debug
    """
    with torch.no_grad():
        # Prefix sum to get the start indices
        offsets = torch.cumsum(tokens_per_expert_group, 0)
        offsets = torch.cat([torch.zeros(1, dtype=offsets.dtype, device=offsets.device), offsets])
        offsets = offsets.tolist()

        # Create indices chunk by chunk
        indices = [
            torch.arange(offsets[i], offsets[i+1], device=tokens_per_expert_group.device)
            for i in range(len(offsets)-1)
        ]

        pt_permuted_indices = []
        pt_m_sizes = []

        # For each local expert
        for e in range(experts_per_rank):
            len_for_e = 0
            indices_for_e = []
            # For each remote rank
            for r in range(ep_size):
                i = r * experts_per_rank + e
                if i < len(indices):
                    # Append the real index chunks
                    indices_for_e.append(indices[i])
                    len_for_e += indices[i].shape[0]

            # Prepare padding
            fill_len = (alignment - len_for_e % alignment) % alignment
            fill = torch.full((fill_len,), -1, dtype=torch.int32, device=tokens_per_expert_group.device)
            indices_for_e.append(fill)

            # The group's token length is the sum of the real tokens and the padding
            pt_m_sizes.append(len_for_e + fill_len)
            pt_permuted_indices.append(torch.cat(indices_for_e))

        pt_permuted_indices = torch.cat(pt_permuted_indices)
        pt_m_sizes = torch.tensor(pt_m_sizes, dtype=torch.int32, device=tokens_per_expert_group.device)

        # Print comparison
        print(f"PyTorch indices shape: {pt_permuted_indices.shape}, Triton indices shape: {permuted_indices.shape}")
        print(f"Indices match: {torch.allclose(pt_permuted_indices, permuted_indices)}")
        print(f"M_sizes match: {torch.allclose(pt_m_sizes, m_sizes)}")

        if not torch.allclose(pt_permuted_indices, permuted_indices):
            # Find first mismatch
            mismatch_idx = (pt_permuted_indices != permuted_indices).nonzero(as_tuple=True)[0]
            if len(mismatch_idx) > 0:
                first_idx = mismatch_idx[0].item()
                print(f"First mismatch at index {first_idx}:")
                print(f"PyTorch: {pt_permuted_indices[first_idx]}, Triton: {permuted_indices[first_idx]}")
    """

    # Step 2: Apply permutation
    permuted_tokens = apply_permutation(
        token_gather_buf,
        permuted_indices,
        token_gather_buf.shape,
    )

    return permuted_tokens, m_sizes


# Test helper function to create realistic test data
def create_test_data(
    batch_size: int = 2,
    seq_len: int = 128,  # Reduced sequence length to avoid OOM
    hidden_dim: int = 128,  # Reduced hidden dimension
    n_experts: int = 8,
    ep_size: int = 2,
    top_k: int = 2,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create realistic test data for MoE token routing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension size
        n_experts: Total number of experts
        ep_size: Number of expert parallel ranks
        top_k: Top-k routing
        device: Device to create tensors on

    Returns:
        Tuple of (token_gather_buf, tokens_per_expert_group, router_probs)
    """
    # Create input tensor
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Create router probabilities (simulating softmax output from router)
    router_probs = torch.softmax(
        torch.randn(batch_size, seq_len, n_experts, device=device), dim=-1
    )

    # Get top-k experts and probabilities
    topk_probs, topk_idx = torch.topk(router_probs, top_k, dim=-1)

    # Simulate token assignment to experts
    # Count tokens per expert
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(top_k):
                expert_idx = topk_idx[b, s, k].item()
                tokens_per_expert[expert_idx] += 1

    # Create token gather buffer (this would normally be done in the router)
    total_tokens = tokens_per_expert.sum().item()
    token_gather_buf = torch.randn(total_tokens, hidden_dim, device=device)

    return token_gather_buf, tokens_per_expert, router_probs


# Example validation function to compare with PyTorch implementation
def validate_permutation_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
    alignment: int = 128,
) -> bool:
    """
    Validate that the Triton implementation produces the same indices as PyTorch.

    Args:
        tokens_per_expert_group: Number of tokens per expert group [n_routed_experts]
        experts_per_rank: Number of experts per rank
        ep_size: Number of expert parallel ranks
        alignment: Alignment requirement for token group sizes (default: 128)

    Returns:
        True if the implementations match, False otherwise
    """
    # Triton implementation
    triton_indices, triton_m_sizes = compute_permutation_indices(
        tokens_per_expert_group,
        experts_per_rank,
        ep_size,
        alignment,
        pad_value=-1,
    )

    # PyTorch reference implementation
    with torch.no_grad():
        offsets = torch.cumsum(tokens_per_expert_group, 0)
        offsets = torch.cat(
            [torch.zeros(1, dtype=offsets.dtype, device=offsets.device), offsets]
        )

        # Create indices chunk by chunk
        indices = [
            torch.arange(
                offsets[i].item(),
                offsets[i + 1].item(),
                device=tokens_per_expert_group.device,
            )
            for i in range(len(offsets) - 1)
        ]

        pt_permuted_indices = []
        pt_m_sizes = []

        # For each local expert
        for e in range(experts_per_rank):
            len_for_e = 0
            indices_for_e = []
            # For each remote rank
            for r in range(ep_size):
                i = r * experts_per_rank + e
                if i < len(indices):
                    # Append the real index chunks
                    indices_for_e.append(indices[i])
                    len_for_e += indices[i].shape[0]

            # Prepare padding
            fill_len = (alignment - len_for_e % alignment) % alignment
            fill = torch.full(
                (fill_len,),
                -1,
                dtype=torch.int32,
                device=tokens_per_expert_group.device,
            )
            indices_for_e.append(fill)

            # The group's token length is the sum of the real tokens and the padding
            pt_m_sizes.append(len_for_e + fill_len)
            pt_permuted_indices.append(torch.cat(indices_for_e))

        pt_permuted_indices = torch.cat(pt_permuted_indices)
        pt_m_sizes = torch.tensor(
            pt_m_sizes, dtype=torch.int32, device=tokens_per_expert_group.device
        )

    # Check if shapes match
    shape_match = (
        pt_permuted_indices.shape == triton_indices.shape
        and pt_m_sizes.shape == triton_m_sizes.shape
    )

    # Check if values match
    values_match = torch.all(pt_permuted_indices == triton_indices) and torch.all(
        pt_m_sizes == triton_m_sizes
    )

    # If not matching, print detailed diagnostics
    if not values_match:
        print(
            f"PyTorch indices shape: {pt_permuted_indices.shape}, Triton indices shape: {triton_indices.shape}"
        )

        if pt_permuted_indices.shape == triton_indices.shape:
            # Find first mismatch
            mismatch_idx = (pt_permuted_indices != triton_indices).nonzero(
                as_tuple=True
            )[0]
            if len(mismatch_idx) > 0:
                first_idx = mismatch_idx[0].item()
                print(f"First mismatch at index {first_idx}:")
                print(
                    f"PyTorch: {pt_permuted_indices[first_idx]}, Triton: {triton_indices[first_idx]}"
                )

                # Show surrounding values for context
                start = max(0, first_idx - 5)
                end = min(len(pt_permuted_indices), first_idx + 6)
                print(
                    f"PyTorch indices [{start}:{end}]: {pt_permuted_indices[start:end]}"
                )
                print(f"Triton indices [{start}:{end}]: {triton_indices[start:end]}")

        if pt_m_sizes.shape == triton_m_sizes.shape:
            if not torch.all(pt_m_sizes == triton_m_sizes):
                print(
                    f"m_sizes mismatch: PyTorch: {pt_m_sizes}, Triton: {triton_m_sizes}"
                )

    return shape_match and values_match


def test_moe_routing_pipeline(
    batch_size: int = 2,
    seq_len: int = 128,  # Reduced for testing
    hidden_dim: int = 128,  # Reduced for testing
    n_experts: int = 8,
    experts_per_rank: int = 4,  # n_experts // ep_size
    ep_size: int = 2,
    top_k: int = 2,
    alignment: int = 128,
    device: str = "cuda",
    debug: bool = True,  # Enable additional debug output
):
    """
    Test the full MoE routing pipeline with both PyTorch and Triton implementations.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension size
        n_experts: Total number of experts
        experts_per_rank: Number of experts per rank (n_experts // ep_size)
        ep_size: Number of expert parallel ranks
        top_k: Top-k routing
        alignment: Alignment requirement
        device: Device to run on
    """
    print("Creating test data...")
    try:
        token_gather_buf, tokens_per_expert, _ = create_test_data(
            batch_size, seq_len, hidden_dim, n_experts, ep_size, top_k, device
        )

        print("Token counts per expert:", tokens_per_expert)
        print("Total tokens:", tokens_per_expert.sum().item())

        # Verify dimensionality
        print(f"token_gather_buf shape: {token_gather_buf.shape}")
        print(f"tokens_per_expert shape: {tokens_per_expert.shape}")

        # Safety check - ensure we have at least some tokens for each expert
        if (tokens_per_expert == 0).any():
            print(
                "Warning: Some experts have 0 tokens, adding 1 token to each for stability"
            )
            tokens_per_expert = tokens_per_expert + 1
            # Expand token_gather_buf to match the new token count
            new_total = tokens_per_expert.sum().item()
            new_buf = torch.randn(new_total, hidden_dim, device=device)
            new_buf[: token_gather_buf.shape[0]] = token_gather_buf
            token_gather_buf = new_buf
            print(f"New total tokens: {new_total}")

    except Exception as e:
        print(f"Error creating test data: {str(e)}")
        raise

    print("\nRunning PyTorch implementation...")
    # PyTorch implementation
    with torch.no_grad():
        # Prefix sum to get the start indices
        offsets = torch.cumsum(tokens_per_expert, 0)
        offsets = torch.cat(
            [torch.zeros(1, dtype=offsets.dtype, device=offsets.device), offsets]
        )

        # Create indices chunk by chunk
        indices = [
            torch.arange(
                offsets[i].item(),
                offsets[i + 1].item(),
                device=tokens_per_expert.device,
            )
            for i in range(len(offsets) - 1)
        ]

        pt_permuted_indices = []
        pt_m_sizes = []

        # For each local expert
        for e in range(experts_per_rank):
            len_for_e = 0
            indices_for_e = []
            # For each remote rank
            for r in range(ep_size):
                i = r * experts_per_rank + e
                if i < len(indices):
                    # Append the real index chunks
                    expert_indices = indices[i]
                    if debug:
                        print(
                            f"Expert {e}, Rank {r}, Expert idx {i}: {expert_indices.shape[0]} tokens"
                        )
                        if expert_indices.shape[0] > 0:
                            print(
                                f"  Index range: {expert_indices[0].item()} to {expert_indices[-1].item()}"
                            )

                    indices_for_e.append(expert_indices)
                    len_for_e += expert_indices.shape[0]

            # Prepare padding
            fill_len = (alignment - len_for_e % alignment) % alignment
            fill = torch.full(
                (fill_len,), -1, dtype=torch.int32, device=tokens_per_expert.device
            )
            if debug:
                print(f"Expert {e}: {len_for_e} tokens, {fill_len} padding")

            indices_for_e.append(fill)

            # The group's token length is the sum of the real tokens and the padding
            padded_size = len_for_e + fill_len
            pt_m_sizes.append(padded_size)
            concatenated = torch.cat(indices_for_e)

            if debug:
                print(f"Expert {e}: Final size {concatenated.shape[0]}")
                # Check if any indices are out of bounds
                valid_indices = concatenated[concatenated >= 0]
                if len(valid_indices) > 0:
                    max_idx = valid_indices.max().item()
                    if max_idx >= token_gather_buf.shape[0]:
                        print(
                            f"  WARNING: Expert {e} has index {max_idx} >= {token_gather_buf.shape[0]}"
                        )

            pt_permuted_indices.append(concatenated)

        pt_permuted_indices = torch.cat(pt_permuted_indices)
        pt_m_sizes = torch.tensor(
            pt_m_sizes, dtype=torch.int32, device=tokens_per_expert.device
        )

        # Apply permutation
        pt_permuted_tokens = torch.zeros_like(token_gather_buf)
        for i in range(len(pt_permuted_indices)):
            idx = pt_permuted_indices[i].item()
            # Skip padding and ensure index is within bounds
            if idx >= 0 and idx < token_gather_buf.shape[0]:
                pt_permuted_tokens[i] = token_gather_buf[idx]
            elif idx >= 0:
                print(
                    f"Warning: PyTorch implementation - index {idx} out of bounds (max: {token_gather_buf.shape[0]-1})"
                )

    print("\nRunning Triton implementation...")
    # Triton implementation
    triton_permuted_tokens, triton_m_sizes = optimized_token_permutation(
        token_gather_buf,
        tokens_per_expert,
        experts_per_rank,
        ep_size,
        alignment,
    )

    # Get the permutation indices from the compute_permutation_indices function
    triton_indices, _ = compute_permutation_indices(
        tokens_per_expert,
        experts_per_rank,
        ep_size,
        alignment,
        pad_value=-1,
    )

    print("\nComparing results...")
    # Check if shapes match
    print(
        f"Permuted indices shapes match: {pt_permuted_indices.shape == triton_indices.shape}"
    )
    print(f"m_sizes shapes match: {pt_m_sizes.shape == triton_m_sizes.shape}")
    print(
        f"Permuted tokens shapes match: {pt_permuted_tokens.shape == triton_permuted_tokens.shape}"
    )

    # Check if values match
    indices_match = torch.all(pt_permuted_indices == triton_indices)
    m_sizes_match = torch.all(pt_m_sizes == triton_m_sizes)
    tokens_match = torch.allclose(
        pt_permuted_tokens, triton_permuted_tokens, rtol=1e-5, atol=1e-5
    )

    print(f"Permuted indices match: {indices_match}")
    print(f"m_sizes match: {m_sizes_match}")
    print(f"Permuted tokens match: {tokens_match}")

    if not indices_match:
        # Find first mismatch
        mismatch_idx = (pt_permuted_indices != triton_indices).nonzero(as_tuple=True)[0]
        if len(mismatch_idx) > 0:
            first_idx = mismatch_idx[0].item()
            print(f"First indices mismatch at index {first_idx}:")
            print(
                f"PyTorch: {pt_permuted_indices[first_idx]}, Triton: {triton_indices[first_idx]}"
            )

    if not m_sizes_match:
        print(f"m_sizes mismatch:")
        print(f"PyTorch: {pt_m_sizes}")
        print(f"Triton: {triton_m_sizes}")

    if not tokens_match:
        # Find max difference in tokens
        token_diff = (pt_permuted_tokens - triton_permuted_tokens).abs()
        max_diff_idx = token_diff.sum(dim=1).argmax().item()
        print(f"Max token difference at index {max_diff_idx}:")
        print(f"Max difference: {token_diff[max_diff_idx].max().item()}")

    success = indices_match and m_sizes_match and tokens_match
    print(f"\nOverall test {'passed' if success else 'failed'}")
    return success


def test_with_controlled_data():
    """
    Test with manually controlled data to ensure deterministic behavior.
    """
    print("Running controlled test case...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a simple controlled test case
    experts_per_rank = 2
    ep_size = 2
    alignment = 8  # Smaller alignment for easier debugging

    # Create tokens_per_expert with a known pattern
    tokens_per_expert = torch.tensor([5, 3, 4, 2], dtype=torch.int32, device=device)

    # Create a simple token buffer
    total_tokens = tokens_per_expert.sum().item()  # Should be 14
    token_gather_buf = torch.arange(
        0, total_tokens, dtype=torch.float32, device=device
    ).unsqueeze(1)

    print("Input configuration:")
    print(f"  tokens_per_expert: {tokens_per_expert}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  experts_per_rank: {experts_per_rank}")
    print(f"  ep_size: {ep_size}")
    print(f"  alignment: {alignment}")

    # Expected results based on PyTorch implementation
    # Calculate expected permuted indices
    # For expert 0: tokens from expert 0 (rank 0) and expert 2 (rank 1)
    # For expert 1: tokens from expert 1 (rank 0) and expert 3 (rank 1)

    # Manual calculation for verification:
    # For expert 0:
    #   - Tokens from expert 0 (rank 0): indices 0-4
    #   - Tokens from expert 2 (rank 1): indices 8-11
    #   - Total: 9 tokens, alignment to 16 requires 7 padding (-1s)
    # For expert 1:
    #   - Tokens from expert 1 (rank 0): indices 5-7
    #   - Tokens from expert 3 (rank 1): indices 12-13
    #   - Total: 5 tokens, alignment to 8 requires 3 padding (-1s)

    # Run the Triton implementation
    print("\nRunning Triton implementation...")
    triton_indices, triton_m_sizes = compute_permutation_indices(
        tokens_per_expert,
        experts_per_rank,
        ep_size,
        alignment,
        pad_value=-1,
    )

    # Run the PyTorch implementation
    print("\nRunning PyTorch implementation...")
    with torch.no_grad():
        # Prefix sum to get the start indices
        offsets = torch.cumsum(tokens_per_expert, 0)
        offsets = torch.cat(
            [torch.zeros(1, dtype=offsets.dtype, device=offsets.device), offsets]
        )

        # Create indices chunk by chunk
        indices = [
            torch.arange(
                offsets[i].item(),
                offsets[i + 1].item(),
                device=tokens_per_expert.device,
            )
            for i in range(len(offsets) - 1)
        ]

        pt_permuted_indices = []
        pt_m_sizes = []

        # For each local expert
        for e in range(experts_per_rank):
            len_for_e = 0
            indices_for_e = []
            # For each remote rank
            for r in range(ep_size):
                i = r * experts_per_rank + e
                if i < len(indices):
                    # Append the real index chunks
                    print(
                        f"Expert {e}, Rank {r}, Expert idx {i}: {indices[i].shape[0]} tokens"
                    )
                    indices_for_e.append(indices[i])
                    len_for_e += indices[i].shape[0]

            # Prepare padding
            fill_len = (alignment - len_for_e % alignment) % alignment
            fill = torch.full(
                (fill_len,), -1, dtype=torch.int32, device=tokens_per_expert.device
            )
            indices_for_e.append(fill)

            # The group's token length is the sum of the real tokens and the padding
            pt_m_sizes.append(len_for_e + fill_len)
            pt_permuted_indices.append(torch.cat(indices_for_e))

            print(
                f"Expert {e}: {len_for_e} tokens + {fill_len} padding = {len_for_e + fill_len} total"
            )

        pt_permuted_indices = torch.cat(pt_permuted_indices)
        pt_m_sizes = torch.tensor(
            pt_m_sizes, dtype=torch.int32, device=tokens_per_expert.device
        )

    # Compare results
    print("\nComparing results:")
    print("PyTorch indices:", pt_permuted_indices)
    print("Triton indices:", triton_indices)
    print("PyTorch m_sizes:", pt_m_sizes)
    print("Triton m_sizes:", triton_m_sizes)

    # Check if values match
    indices_match = torch.all(pt_permuted_indices == triton_indices)
    m_sizes_match = torch.all(pt_m_sizes == triton_m_sizes)

    print(f"Indices match: {indices_match}")
    print(f"m_sizes match: {m_sizes_match}")

    # Apply permutation
    print("\nApplying permutation...")

    # PyTorch implementation
    pt_permuted_tokens = torch.zeros_like(token_gather_buf)
    for i in range(len(pt_permuted_indices)):
        idx = pt_permuted_indices[i].item()
        if (
            idx >= 0 and idx < token_gather_buf.shape[0]
        ):  # Skip padding and invalid indices
            pt_permuted_tokens[i] = token_gather_buf[idx]

    # Triton implementation
    triton_permuted_tokens = apply_permutation(
        token_gather_buf,
        triton_indices,
        token_gather_buf.shape,
    )

    # Compare results
    tokens_match = torch.allclose(pt_permuted_tokens, triton_permuted_tokens)
    print(f"Tokens match: {tokens_match}")

    # Print the first few tokens for verification
    print("\nFirst 8 permuted tokens:")
    print("PyTorch:", pt_permuted_tokens[:8].flatten().tolist())
    print("Triton:", triton_permuted_tokens[:8].flatten().tolist())

    success = indices_match and m_sizes_match and tokens_match
    print(f"\nControlled test {'passed' if success else 'failed'}")
    return success


if __name__ == "__main__":
    test_with_controlled_data()
