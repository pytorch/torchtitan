# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``FP32OutputLinear``: a ``Linear`` with an fp32 output and close to fp32 gradients."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.autograd.function import once_differentiable

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.linear import Linear

# Shape suffix legend for FP32OutputLinear:
#   T = num tokens, D = model dimension, O = output features, P = grad_output pieces (2 or 3)


class FP32OutputLinear(Linear):
    """``Linear`` with an fp32 output and close to fp32 gradients, at close to bf16 speed. Useful
    for layers that need higher precision, e.g. an LM head or a MoE router gate.

    Forward: bf16 input and weight, a bf16 GEMM that accumulates in fp32, fp32 output.
    Backward: approximates an fp32 backward with bf16 GEMMs, and returns grad_weight in fp32.

    Falls back to slower fp32 matmuls when:
    (a) the input is not on CUDA,
    (b) the input or the weight is not bf16, or
    (c) batch-invariant mode is on.

    Accuracy and timings: https://github.com/pytorch/torchtitan/pull/4923
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        higher_precision_bwd: bool = False
        """Split grad_output into 3 bf16 pieces (exact) instead of 2. Slower (1.1-1.5x backward),
        and the gain may be too small to notice. It matters most for small out_features, e.g. a
        router, not an LM head: the GEMM's own rounding grows with out_features and hides the
        gain."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.higher_precision_bwd = config.higher_precision_bwd

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # torch.mm takes 2D inputs, so flatten the input: [B, S, D] -> [B * S, D]. The weight is
        # already 2D: Linear.forward flattens a stacked [num_linears, O, D] to [num_linears * O, D].
        output = _FP32OutputLinearFunction.apply(
            input.reshape(-1, input.shape[-1]), weight, self.higher_precision_bwd
        )
        output = output.reshape(*input.shape[:-1], -1)
        return output if bias is None else output + bias.float()


@spmd.register_local_autograd_function
class _FP32OutputLinearFunction(torch.autograd.Function):
    """``output = input @ weight.T`` in fp32, with bf16 GEMMs. See ``FP32OutputLinear``."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_TD: torch.Tensor,
        weight_OD: torch.Tensor,
        higher_precision_bwd: bool,
    ) -> torch.Tensor:
        """``output = input @ weight.T``: a bf16 GEMM that accumulates in fp32 and returns fp32.

        ``torch.mm(..., out_dtype=torch.float32)`` has no autograd formula ("derivative for
        aten::mm is not implemented"), hence this Function:

            input (bf16) --+
                           +--> bf16 GEMM, fp32 accumulate --> output (fp32)
            weight (bf16) -+

        The product of two bf16 numbers is exact in fp32, so upcasting input and weight first
        would add no precision, only an fp32 copy of the weight. Accumulating in fp32 is enough.
        """
        ctx.use_bf16_gemm = (
            # aten::mm.dtype (bf16 inputs, fp32 output) is only implemented for CUDA/ROCm.
            input_TD.is_cuda
            and input_TD.dtype == weight_OD.dtype == torch.bfloat16
            # TODO: batch-invariant mode can't use this op (cuBLAS's out_dtype GEMM isn't
            # batch-invariant), so it takes the slow fallback. A bf16-input, fp32-output matmul
            # in batch_invariant_ops would let it take the optimized path.
            and not is_in_batch_invariant_mode()
        )
        ctx.higher_precision_bwd = higher_precision_bwd
        ctx.save_for_backward(input_TD, weight_OD)
        if ctx.use_bf16_gemm:
            return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)
        # Slow fallback: upcast the input and weight to fp32.
        return torch.mm(input_TD.float(), weight_OD.float().T)

    @staticmethod
    # Some methods may need the gradient of a gradient. This backward calls torch.mm(out_dtype=),
    # which has no derivative formula, so it can only be differentiated once.
    @once_differentiable
    def backward(ctx, grad_output_TO: torch.Tensor):  # pyrefly: ignore[bad-override]
        """``grad_input = grad_output @ weight``, ``grad_weight = grad_output.T @ input``.

        Close to an fp32 backward, at a fraction of its cost (table below).

        The problem: we have 3 tensors,
            weight:       bf16
            input:        bf16
            grad_output:  fp32, because the output was fp32

        We want, accumulated in fp32:
            grad_input  = grad_output @ weight     rounded once to the input's dtype
            grad_weight = grad_output.T @ input    returned in fp32

        How can we do it fast, with close to fp32 precision?

        Option 1: round grad_output to bf16, then bf16 GEMMs. GPUs run those on tensor cores, ~25x
        faster than fp32 matmuls on GB300. Fast, but loses precision: grad_output keeps only 8 of
        its 24 significant bits.

        Option 2: upcast weight and input, then do all matmuls in fp32. Precise, but slow (table
        below).

        Option 3 (this function): split grad_output into 3 (or 2) bf16 pieces that sum to it.
        weight and input already are bf16, so only grad_output needs splitting.

            fp32:  sign | exponent (8 bits) | mantissa (23 bits)   1 + 23 = 24 significant bits
            bf16:  sign | exponent (8 bits) | mantissa  (7 bits)   1 +  7 =  8 significant bits

        bf16 keeps 8 of fp32's 24 significant bits but has the same exponent, so each piece keeps
        its own scale, and 3 pieces hold an fp32 exactly (``_split_into_bf16_pieces_eager``):

            3 pieces, exact:  0.1 = 0.100097656 - 0.000097752 + 0.000000097   (hi + mid + lo)
            2 pieces:         0.1 ~ 0.100097656 - 0.000097752                 (hi + lo, off by 1e-7)

        Substituting grad_output = hi + mid + lo turns each fp32 GEMM into bf16 GEMMs:

            grad_input  = grad_output @ weight
                        = (hi + mid + lo) @ weight
                        = hi @ weight + mid @ weight + lo @ weight      3 bf16 GEMMs (2 without mid)

            grad_weight = grad_output.T @ input
                        = hi.T @ input + mid.T @ input + lo.T @ input   3 bf16 GEMMs (2 without mid)

        2 or 3 pieces: the GEMM rounds as it accumulates, losing precision, and more so for
        longer sums. So the third piece helps less the longer the sum: a little for a router's
        128 experts, nothing for a 152k vocab. It also makes the backward up to ~1.5x slower.

        Relative error vs fp64, and backward time as a multiple of a bf16 Linear's (GB300):

                                    relative error             backward time
                                    grad_input  grad_weight    eager   compiled
            LM head (Qwen3-8B, 2048 tokens; bf16 backward: 2.8 ms)
              bf16 grad_output      1.5e-3      1.3e-3          1.0x    1.0x
              2 pieces              2.1e-4      8.4e-6          2.4x    2.5x
              3 pieces              2.1e-4      8.6e-6          3.6x    3.8x
              fp32 matmul (IEEE)    7.9e-5      2.3e-6         26.2x   27.4x
            router (2048 -> 128; errors on 16k tokens, times on 64k; bf16 backward: 0.24 ms)
              bf16 grad_output      1.7e-3      1.2e-3          1.0x    1.0x
              2 pieces              2.4e-6      2.9e-6          2.2x    1.6x
              3 pieces              3.5e-7      2.1e-6          2.8x    1.7x
              fp32 matmul (IEEE)    5.1e-8      2.9e-7          5.1x    4.5x

        Stacking: every piece multiplies the same weight (for grad_input) or input (for
        grad_weight). Concatenating the pieces runs one GEMM per gradient instead of one per
        piece (1.4x faster for the LM head's grad_weight), but the GEMM then needs the shared
        operand repeated once per piece: a copy, which costs memory and time. We concatenate
        along the dim that makes that copy small.
        With 2 pieces (hi, lo: [T, O]; x = input: [T, D]; W = weight: [O, D]):

            LM head (T = 2048 tokens, O = 152k vocab, D = 4096): concatenate along tokens
                grad_input  = cat([hi, lo], dim=0) @ W               [2T, D]: add its two halves
                grad_weight = cat([hi, lo], dim=0).T @ cat([x, x])   copies x: 32 MiB
                (concatenating along O would copy W instead: 2.3 GiB)
            router (T = 64k tokens, O = 128 experts, D = 2048): concatenate along out_features
                grad_input  = cat([hi, lo], dim=1) @ cat([W, W])     copies W: 1 MiB
                (concatenating along T would copy x instead: 512 MiB)
                grad_weight = hi.T @ x + lo.T @ x                    one small GEMM per piece
        """
        input_TD, weight_OD = ctx.saved_tensors
        needs_grad_input, needs_grad_weight, _ = ctx.needs_input_grad
        # .float() is a no-op unless autocast made the fallback's output bf16.
        grad_output_TO = grad_output_TO.float()
        num_tokens, out_features = grad_output_TO.shape

        if not ctx.use_bf16_gemm:  # cases (a)-(c) in FP32OutputLinear
            grad_input_TD, grad_weight_OD = _fp32_backward(
                grad_output_TO,
                input_TD,
                weight_OD,
                needs_grad_input=needs_grad_input,
                needs_grad_weight=needs_grad_weight,
            )
        # For speed, stack the pieces along the smaller dim (see "Stacking").
        elif out_features > num_tokens:
            grad_input_TD, grad_weight_OD = _wide_backward(
                grad_output_TO,
                input_TD,
                weight_OD,
                higher_precision_bwd=ctx.higher_precision_bwd,
                needs_grad_input=needs_grad_input,
                needs_grad_weight=needs_grad_weight,
            )
        else:
            grad_input_TD, grad_weight_OD = _narrow_backward(
                grad_output_TO,
                input_TD,
                weight_OD,
                higher_precision_bwd=ctx.higher_precision_bwd,
                needs_grad_input=needs_grad_input,
                needs_grad_weight=needs_grad_weight,
            )

        # TODO: grad_weight is fp32, but autograd rounds its values to bf16 (.grad stays fp32) when
        # - this Function runs inside a torch.compile region:
        #   https://github.com/pytorch/pytorch/pull/197381
        # - the weight is computed from the parameter instead of being the parameter itself
        #   (e.g. a view of a num_linears > 1 parameter):
        #   https://github.com/pytorch/pytorch/issues/189633
        return grad_input_TD, grad_weight_OD, None


def _fp32_backward(
    grad_output_TO: torch.Tensor,
    input_TD: torch.Tensor,
    weight_OD: torch.Tensor,
    *,
    needs_grad_input: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Slow fallback: upcast the input and weight, then fp32 matmuls for both gradients.

    In batch-invariant mode, batch_invariant_ops' Triton matmul runs them in TF32.
    """
    grad_input_TD = grad_weight_OD = None
    if needs_grad_input:
        grad_input_TD = torch.mm(grad_output_TO, weight_OD.float()).to(input_TD.dtype)
    if needs_grad_weight:
        grad_weight_OD = torch.mm(grad_output_TO.T, input_TD.float())
    return grad_input_TD, grad_weight_OD


def _wide_backward(
    grad_output_TO: torch.Tensor,
    input_TD: torch.Tensor,
    weight_OD: torch.Tensor,
    *,
    higher_precision_bwd: bool,
    needs_grad_input: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """More outputs than tokens (e.g. an LM head): stack the pieces along tokens and copy the
    input (see "Stacking" in ``backward``)."""
    num_pieces = 3 if higher_precision_bwd else 2
    num_tokens = grad_output_TO.shape[0]
    grad_input_TD = grad_weight_OD = None

    # ==== Split grad_output into P bf16 pieces, stacked along tokens: [P * T, O] ====
    stacked_PTO = _split_into_bf16_pieces(grad_output_TO, higher_precision_bwd, dim=0)

    # ==== grad_input: one GEMM over the stack, then sum the per-piece results ====
    if needs_grad_input:
        # TODO: sum out_features in 8192-wide chunks with addmm(out=), to cut the GEMM's
        # accumulation error: grad_input error 2.1e-4 -> 1.2e-5 (bf16 values equal to the
        # exact gradient's: 96.2% -> 99.6%) for +3% backward time (GB300). Inductor can't
        # lower addmm(out_dtype=) yet (https://github.com/pytorch/pytorch/pull/190936), so
        # wrap it in a custom op.
        # Better still: a fused kernel that splits grad_output in registers and adds each
        # partial sum in fp32 outside the tensor core, as vLLM does for a router:
        # https://github.com/vllm-project/vllm/pull/55899.
        grad_input_PTD = torch.mm(stacked_PTO, weight_OD, out_dtype=torch.float32)
        grad_input_PTD = grad_input_PTD.unflatten(0, (num_pieces, num_tokens))
        grad_input_TD = grad_input_PTD.sum(dim=0).to(input_TD.dtype)

    # ==== grad_weight: one GEMM of the stack against P copies of the input ====
    if needs_grad_weight:
        # TODO: with ChunkedLossWrapper, autograd adds each chunk's grad_weight into
        # weight.grad in a separate kernel. Adding in place with addmm(out=weight.grad) and
        # returning None (autograd may copy a returned buffer) saves 0.9 ms and a 2.3 GiB
        # temporary per Qwen3-8B chunk (GB300). Eager only: graph_trainer's compiled pass
        # (https://github.com/pytorch/torchtitan/pull/4768) only adds a bf16 GEMM into a
        # bf16 .grad. fp32 into fp32 would also need Inductor to compile addmm(out_dtype=).
        grad_weight_OD = torch.mm(
            stacked_PTO.T,
            # Copying x beats one GEMM per piece + add: 1.4x at 2048 tokens, 1.02-1.17x
            # at 8k-32k (Qwen3-8B head, GB300).
            torch.cat([input_TD] * num_pieces),
            out_dtype=torch.float32,
        )
    return grad_input_TD, grad_weight_OD


def _narrow_backward(
    grad_output_TO: torch.Tensor,
    input_TD: torch.Tensor,
    weight_OD: torch.Tensor,
    *,
    higher_precision_bwd: bool,
    needs_grad_input: bool,
    needs_grad_weight: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """More tokens than outputs (e.g. a router): stack the pieces along out_features and copy
    the weight (see "Stacking" in ``backward``)."""
    num_pieces = 3 if higher_precision_bwd else 2
    out_features = grad_output_TO.shape[1]
    grad_input_TD = grad_weight_OD = None

    # ==== Split grad_output into P bf16 pieces, stacked along out_features: [T, P * O] ====
    stacked_TPO = _split_into_bf16_pieces(grad_output_TO, higher_precision_bwd, dim=1)

    # ==== grad_input: one GEMM of the stack against P copies of the weight ====
    if needs_grad_input:
        grad_input_TD = torch.mm(stacked_TPO, torch.cat([weight_OD] * num_pieces))

    # ==== grad_weight: one small GEMM per piece, added in fp32 ====
    if needs_grad_weight:
        pieces_TO = stacked_TPO.split(out_features, dim=1)  # views, no copy
        grad_weight_OD = torch.mm(pieces_TO[0].T, input_TD, out_dtype=torch.float32)
        for piece_TO in pieces_TO[1:]:
            grad_weight_OD += torch.mm(piece_TO.T, input_TD, out_dtype=torch.float32)
    return grad_input_TD, grad_weight_OD


# =============================== Split grad_output into bf16 pieces ===============================
# Compiled, the split + cat is one kernel instead of 6 (10 with 3 pieces): 2.10 -> 0.51 ms, -14-17%
# per Qwen3-8B head + loss chunk (2 pieces, GB300), bitwise equal. Only the split: compiling the
# Function rounds grad_weight to bf16 (see the TODO in backward). Always compiled, like
# FlexAttention: a @local_compile region stays eager unless the model lists it. No fullgraph: past
# the recompile limit or with TORCH_COMPILE_DISABLE=1, it runs eagerly.
#
# make_fx and FakeTensorMode can't run a compiled kernel inside the backward: real tracing records
# its output as a constant, fake tracing crashes. They record a custom op as one op instead:
# `custom_op` runs the compiled kernel on real tensors; `register_fake` gives tracers the output's
# shape and dtype without running it.


@torch.library.custom_op("torchtitan::split_into_bf16_pieces", mutates_args=())
def _split_into_bf16_pieces(
    grad_output_TO: torch.Tensor, higher_precision_bwd: bool, dim: int
) -> torch.Tensor:
    """``_split_into_bf16_pieces_eager``, compiled. See the comment above."""
    # out_features stays static: a symbolic out_features makes the split 1.3-3.8x slower (GB300).
    # The token dim starts static and turns symbolic once it changes, so new token counts don't
    # recompile.
    torch._dynamo.mark_static(grad_output_TO, 1)
    return _compiled_split_into_bf16_pieces(grad_output_TO, higher_precision_bwd, dim)


@_split_into_bf16_pieces.register_fake
def _(
    grad_output_TO: torch.Tensor, higher_precision_bwd: bool, dim: int
) -> torch.Tensor:
    shape = list(grad_output_TO.shape)
    shape[dim] *= 3 if higher_precision_bwd else 2
    return grad_output_TO.new_empty(shape, dtype=torch.bfloat16)


def _split_into_bf16_pieces_eager(
    grad_output_TO: torch.Tensor, higher_precision_bwd: bool, dim: int
) -> torch.Tensor:
    """Split an fp32 tensor into bf16 pieces that sum back to it, [hi, lo] or [hi, mid, lo], and
    concatenate them along ``dim``.

    Args:
        grad_output_TO: fp32 tensor to split, [T, O].
        higher_precision_bwd: False stops after the second piece: about 16 of the 24
            significant bits, off by 1e-7 below. True keeps all 3: exact (for |x| >= 2^-110).
        dim: 0 stacks the P pieces along tokens, [P * T, O]; 1 along out_features, [T, P * O].

    Example, x = 0.1:
        hi  = nearest bf16 to x            =  0.100097656   (a bit too big)
        mid = nearest bf16 to x - hi       = -0.000097752   (negative: corrects hi)
        lo  = x - hi - mid                 =  0.000000097

    Each piece can truncate (drop fp32's low 16 bits) or round to the nearest bf16. We round, as
    Triton and XLA do: LM-head grad_weight error 1.6e-5 (truncating) -> 8.4e-6 (rounding), GB300.
    ``x.to(torch.bfloat16)`` would round, but compiled, Inductor drops the round trip and
    x - x.to(bf16).float() becomes 0 (https://github.com/pytorch/pytorch/pull/180575). So
    ``round_to_bf16`` rounds with integer bit ops: add half a bf16 step, then drop the low bits.
    """

    def round_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
        # Nearest bf16 value, ties away from zero, kept in fp32: add half a bf16 ulp, then keep
        # the bits bf16 has (sign, exponent, top 7 mantissa bits: 0xFFFF0000 == -65536).
        return ((tensor.view(torch.int32) + 0x8000) & -65536).view(torch.float32)

    hi = round_to_bf16(grad_output_TO)
    rest = grad_output_TO - hi
    if higher_precision_bwd:
        mid = round_to_bf16(rest)
        pieces = [hi, mid, rest - mid]
    else:
        pieces = [hi, rest]
    return torch.cat([piece.to(torch.bfloat16) for piece in pieces], dim=dim)


_compiled_split_into_bf16_pieces = torch.compile(_split_into_bf16_pieces_eager)

# ======================================== End of the split ========================================


__all__ = ["FP32OutputLinear"]
