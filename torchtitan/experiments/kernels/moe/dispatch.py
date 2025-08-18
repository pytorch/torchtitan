# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# Adding out-of-tree ops to the `symm_mem` library
lib = torch.library.Library("symm_mem", "FRAGMENT")  # noqa: TOR901

"""
all_to_all_vdev_2d_offset_copy:
Copy data from `input` to `symm_in_buf` and call `all_to_all_vdev_2d_offset` to shuffle data
"""
lib.define(
    "all_to_all_vdev_2d_offset_copy("
    "Tensor input, Tensor symm_in_buf, Tensor(a!) out, "
    "Tensor in_splits_offsets, Tensor(a!) out_splits_offsets, str group_name) -> ()",
    tags=[torch._C.Tag.needs_exact_strides],
)


@torch.library.impl(lib, "all_to_all_vdev_2d_offset_copy", "CUDA")
def _all_to_all_vdev_2d_offset_copy_cuda(
    input: torch.Tensor,
    symm_in_buf: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: str,
) -> None:
    if symm_in_buf.shape[0] < input.shape[0]:
        raise RuntimeError(
            f"symm_in_buf with dim-0 length {symm_in_buf.shape[0]} cannot fit input with dim-0 length  {input.shape[0]}"
        )
    if symm_in_buf.shape[1:] != input.shape[1:]:
        raise RuntimeError(
            f"symm_in_buf non-0 dims do not match that of input: {symm_in_buf.shape[1:]} vs {input.shape[1:]}"
        )
    if symm_in_buf.dtype != input.dtype:
        raise RuntimeError(
            f"symm_in_buf dtype {symm_in_buf.dtype} does not match input dtype {input.dtype}"
        )

    symm_in_buf.narrow(0, 0, input.shape[0]).copy_(input)
    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
        symm_in_buf,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group_name,
    )


class AllToAllVDev2d(torch.autograd.Function):
    """
    Autograd function for `all_to_all_vdev_2d`
    """

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        input: torch.Tensor,
        out: torch.Tensor,
        in_splits: torch.Tensor,
        out_splits_offsets: torch.Tensor,
        group_name: str,
        major_align: int,
        # Buffers needed for backward pass
        grad_out_buf: torch.Tensor,
        grad_in_buf: torch.Tensor,
        grad_in_splits_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Functionality is the same as `all_to_all_vdev_2d` but with functionalization.
        """
        # Shuffle input to output
        torch.ops.symm_mem.all_to_all_vdev_2d(
            input, out, in_splits, out_splits_offsets, group_name, major_align
        )

        # Output splits in forward is the input splits in backward
        ctx.save_for_backward(
            out_splits_offsets, grad_out_buf, grad_in_buf, grad_in_splits_offsets
        )
        ctx.group_name = group_name
        return out

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None]:
        """
        Backward pass of `all_to_all_vdev_2d` is `all_to_all_vdev_2d_offset`.

        Args:
            `grad_output`: gradients of output passed back from the downstream.

        Returns:
            `grad_input`: gradients of input.
        """
        # Splits info
        # Splits/offsets of grad_out is the same as out splits/offsets in forward
        (
            grad_out_splits_offsets,
            grad_out_buf,
            grad_in_buf,
            grad_in_splits_offsets,
        ) = ctx.saved_tensors

        # Shuffle gradients back to the input
        torch.ops.symm_mem.all_to_all_vdev_2d_offset_copy(
            grad_output,
            grad_out_buf,
            grad_in_buf,
            grad_out_splits_offsets,
            grad_in_splits_offsets,
            group_name=ctx.group_name,
        )
        return grad_in_buf, None, None, None, None, None, None, None, None


class TokenDispatcher(torch.nn.Module):
    """
    Dispatch tokens to different experts, with backward pass to shuffle gradients back to the input.
    Args:
        `group_name`: name of the group to use for communication.
        `align`: alignment of the token offsets for each receiving expert. If
                 using Grouped Gemm next, this should be the same as Grouped Gemm's
                 alignment.
        `in_len`: length of the input.
        `out_len`: length of the output.
        `token_shape`: shape of the tokens.
        `num_ranks`: number of ranks in the group.
        `num_local_experts`: number of local experts.
        `dtype`: data type of the input/output.
        `device`: device to use for communication.
    """

    def __init__(
        self,
        group_name: str,
        align: int,
        in_len,
        out_len,
        token_shape,
        num_ranks,
        num_local_experts,
        dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.group_name = group_name
        self.align = align
        self.grad_out_buf = symm_mem.empty(
            out_len, *token_shape, dtype=dtype, device=device
        )
        self.grad_in_buf = symm_mem.empty(
            in_len, *token_shape, dtype=dtype, device=device
        )
        self.nsplits = num_ranks * num_local_experts
        self.grad_in_splits_offsets = symm_mem.empty(
            (2, self.nsplits), dtype=torch.int64, device=device
        )

    def forward(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        in_splits: torch.Tensor,
        out_splits_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            `inp`: input tensor.
            `out`: buffer for output tensor.
            `in_splits`: splits of the input tensor.
            `out_splits_offsets`: splits and offsets of the output tensor.
            See `all_to_all_vdev_2d` for more details.
        Note:
            All tensor arguments must be symmetrically allocated, i.e.
            >>> inp = symm_mem.empty(max_inp_len, dtype=dtype, device=device)
            >>> out = symm_mem.empty(max_out_len, dtype=dtype, device=device)
            >>> in_splits = symm_mem.empty(
            ...     nsplits, dtype=torch.int64, device=device)
            >>> out_splits_offsets = symm_mem.empty(
            ...     (2, nsplits), dtype=torch.int64, device=device)
        """

        if in_splits.numel() != self.nsplits:
            raise ValueError(f"Expected {self.nsplits} splits, got {in_splits.numel()}")
        if out_splits_offsets.shape != (2, self.nsplits):
            raise ValueError(
                f"Expected shape (2, {self.nsplits}), got {out_splits_offsets.shape}"
            )

        return AllToAllVDev2d.apply(
            inp,
            out,
            in_splits,
            out_splits_offsets,
            self.group_name,
            self.align,
            self.grad_out_buf,
            self.grad_in_buf,
            self.grad_in_splits_offsets,
        )


def test_token_dispatch() -> None:
    # Init
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    # NVSHMEM backend specific
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    # Set NVSHMEM as SymmMem backend
    symm_mem.set_backend("NVSHMEM")

    # Mimics Group GEMM alignment
    align = 8
    torch.manual_seed(42 + rank)

    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    dtype = torch.float
    # Number of experts per rank
    ne = 8
    nsplits = ne * world_size

    # Number of elements for an expert is random between [0, k)
    k = 10
    inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=device)

    # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
    max_inp_len = k * nsplits
    # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
    overflow_factor = world_size  # worst case: one rank receives all data
    max_out_len = max_inp_len * overflow_factor

    hid = 4096
    inp = symm_mem.empty(max_inp_len, hid, dtype=dtype, device=device)
    out = symm_mem.empty(max_out_len, hid, dtype=dtype, device=device)
    in_splits = symm_mem.empty(nsplits, dtype=torch.int64, device=device).copy_(
        inp_splits
    )
    # 2 rows: output splits, output offsets
    out_splits_offsets = symm_mem.empty((2, nsplits), dtype=torch.int64, device=device)

    dispatcher = TokenDispatcher(
        group_name,
        align,
        max_inp_len,
        max_out_len,
        inp.shape[1:],
        world_size,
        ne,
        dtype,
        device,
    )

    compiled_dispatcher = torch.compile(
        dispatcher,
        fullgraph=True,
    )

    # Perform a Dot product with output, so that gradients passed back from
    # different ranks are different
    weight = torch.empty(max_out_len, dtype=dtype, device=device).fill_(rank + 1)

    # Run a few iterations
    iters = 2
    for i in range(iters):
        # Test if gradients would be passed back from inp to tokens
        tokens = torch.randn(
            max_inp_len, hid, dtype=dtype, device=device
        ).requires_grad_(True)
        tokens.grad = None
        inp.copy_(tokens)
        output = compiled_dispatcher(inp, out, in_splits, out_splits_offsets)
        p = torch.matmul(weight, output)
        p.sum().backward()

    # Check gradients
    start = 0
    for i, split in enumerate(in_splits.tolist()):
        grad_chunk = tokens.grad[start : start + split]
        dst_rank = i // ne
        torch.testing.assert_close(
            grad_chunk,
            torch.empty(split, hid, device=device).fill_(dst_rank + 1),
        )
        start += split

    dist.destroy_process_group()
    print(f"Rank {rank} passed")


if __name__ == "__main__":
    # To run this test, use the following command:
    #   torchrun --nproc-per-node 4 --standalone dispatch.py
    test_token_dispatch()
