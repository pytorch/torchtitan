# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- Standard ``nn.Linear`` parameter and state-dict behavior is retained.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat
from torch.autograd.function import once_differentiable

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = num tokens, D = model dimension, O = output features, P = grad_output pieces (2 or 3)


class Linear(nn.Linear, Module):
    """Configurable linear that can store multiple stacked projections.

    A single projection keeps the standard ``[out_features, in_features]``
    parameter shape. Multiple projections use
    ``[num_linears, out_features, in_features]``, keeping each projection
    contiguous for blockwise weight quantization, and return
    ``[..., num_linears, out_features]``.

    The local projection is the remat region ``<fqn>.linear``, so every
    subclass, including quantized and LoRA ones that override ``_linear``,
    declares the same region. Tensor-parallel subclasses declare their
    collectives as sibling regions. Callers must not wrap a ``Linear`` call in
    another region: a saved outer region cannot contain a recomputed one.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        if config.num_linears > 1:
            self.weight = nn.Parameter(
                self.weight.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.bias.requires_grad,
                )

    def reset_parameters(self) -> None:
        # Flattening handles both ordinary and stacked projections while
        # keeping fan-in equal to in_features.
        nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _flatten_weight_and_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        weight = self.weight.flatten(0, -2)
        bias = None if self.bias is None else self.bias.flatten()
        return weight, bias

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, self.weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(input, weight, bias)
        return self._unflatten_output(output)

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply local projection compute without outer communication.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        Explicit operands let those boundaries adjust an operand's SPMD type
        before invoking the selected local compute implementation.
        """
        return F.linear(input, weight, bias)


def maybe_gather_tp_input(module: Module, x: torch.Tensor) -> torch.Tensor:
    """Redistribute a TP input to ``Replicate`` in the ``<fqn>.tp_gather`` region.

    ``<fqn>`` is ``module``'s remat name. This is an all-gather under sequence
    parallelism, and otherwise a forward no-op whose backward all-reduces.
    Modules whose projections share one input call this once at their common
    boundary. Returns ``x`` unchanged without TP.
    """
    tp_group = spmd_mesh_group(MeshAxisName.TP)
    if tp_group is None:
        return x
    return remat.region(
        spmd.redistribute,
        module.remat_region_name("tp_gather"),
        recompute=module.remat_should_recompute("tp_gather"),
    )(
        x,
        tp_group,
        src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
        dst=spmd.R,
        backward_options={"op_dtype": x.dtype},
    )


class ColumnParallelLinear(Linear):
    """Prepare an input for a column-parallel Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. The same module handles both tensor-parallel modes.
    With sequence parallelism, ``Shard(0) -> Replicate`` is an input all-gather.
    Without sequence parallelism, ``Invariant -> Replicate`` is a forward no-op
    whose backward performs the required all-reduce.

    The redistribution is the remat region ``<fqn>.tp_gather``, a sibling of
    the projection region ``<fqn>.linear``. When the projection is saved and
    the redistribution is recomputed, the projection does not retain the
    gathered input for its weight gradient; replay re-gathers it from the
    sequence shard.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(maybe_gather_tp_input(self, input))


class SharedExpertRowParallelLinear(Linear):
    """Row-parallel shared-expert projection with a conditional reduction.

    With sequence parallelism, the output is reduce-scattered from Partial to
    Shard(0). Otherwise it remains Partial so the MoE can combine routed and
    shared partials before one all-reduce.

    Like ``RowParallelLinear``, the projection and the reduction are the
    separately controlled remat regions ``<fqn>.linear`` and
    ``<fqn>.tp_reduce``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        linear_fn = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(linear_fn(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias))
        if tp_group is None or not spmd_dense_sp_enabled():
            return output
        # A recomputed reduction after a saved projection keeps the TP-times
        # larger partial output for replay; save both to avoid it.
        return remat.region(
            spmd.redistribute,
            self.remat_region_name("tp_reduce"),
            recompute=self.remat_should_recompute("tp_reduce"),
        )(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0),
            backward_options={"op_dtype": output.dtype},
        )


class RowParallelLinear(Linear):
    """Reduce the partial output of an independently configured Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. ``Partial -> Shard(0)`` is a reduce-scatter, while
    ``Partial -> Invariant`` is an all-reduce without it. Dense SP state selects
    between the two. An invariant bias is converted to a partial contribution
    before local compute so the reduction adds it exactly once.

    The projection and the reduction are the separately controlled remat
    regions ``<fqn>.linear`` and ``<fqn>.tp_reduce``. Saving only the
    projection keeps its TP-times larger partial output for the replayed
    reduction; saving only the reduction skips the collective during replay.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        linear_fn = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # Its row-sharded operands and bias jointly produce a partial output.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(linear_fn(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias))
        if tp_group is None:
            return output

        # A recomputed reduction after a saved projection keeps the TP-times
        # larger partial output for replay; save both to avoid it.
        return remat.region(
            spmd.redistribute,
            self.remat_region_name("tp_reduce"),
            recompute=self.remat_should_recompute("tp_reduce"),
        )(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
            backward_options={"op_dtype": output.dtype},
        )


class GroupedLinear(Module):
    """A collection of linears selected by cumulative group offsets.

    Like :class:`Linear`, ``num_linears`` retains a projection axis in parameter
    storage. For example, a fused gate/up projection stores ``[E, 2, F, D]``
    and returns ``[R, 2, F]`` while grouped GEMM consumes its zero-copy
    ``[E, 2F, D]`` view.

    The grouped matmul is the remat region ``<fqn>.grouped_mm``, so every
    subclass, including quantized and LoRA ones that override ``_grouped_mm``,
    declares the same region.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure a grouped linear.

        Attributes:
            group_size: Number of independently selected linear weights.
            in_features: Input features for each linear.
            out_features: Output features for each linear.
            num_linears: Number of stacked linears per element in the group.
                Values greater than one retain a projection axis before
                ``out_features``.
        """

        group_size: int
        in_features: int
        out_features: int
        num_linears: int = 1

    def __init__(self, config: Config):
        super().__init__()
        self.group_size = config.group_size
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        output_shape = (
            (config.out_features,)
            if config.num_linears == 1
            else (config.num_linears, config.out_features)
        )
        self.weight = nn.Parameter(
            torch.empty(config.group_size, *output_shape, config.in_features)
        )

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        """Apply each grouped linear to rows selected by cumulative offsets.

        Args:
            input_RI: Input rows grouped by the selected linear.
            offsets_E: Exclusive cumulative row end for each group.

        Returns:
            Output rows with an optional ``num_linears`` axis before the output
            feature axis.
        """
        output_shape = self.weight.shape[1:-1]
        weight_EOI = self.weight.flatten(1, -2)
        output_RO = remat.region(
            self._grouped_mm,
            self.remat_region_name("grouped_mm"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )(input_RI=input_RI, weight_EOI=weight_EOI, offsets_E=offsets_E)
        return output_RO.reshape(*output_RO.shape[:-1], *output_shape)

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Execute ``input_RI @ weight_EOI.transpose(-2, -1)`` by expert."""
        return torch._grouped_mm(
            input_RI,
            weight_EOI.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )


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
        """Split grad_output into 3 bf16 pieces (exact) instead of 2. Slower (about 1.4x backward),
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

        Option 1: round grad_output to bf16, then bf16 GEMMs. GPUs run those on tensor cores, ~15x
        faster than fp32 matmuls on H100. Fast, but loses precision: grad_output keeps only 8 of
        its 24 significant bits.

        Option 2: upcast weight and input, then do all matmuls in fp32. Precise, but slow (table
        below).

        Option 3 (this function): split grad_output into 3 (or 2) bf16 pieces that sum to it.
        weight and input already are bf16, so only grad_output needs splitting.

            fp32:  sign | exponent (8 bits) | mantissa (23 bits)   1 + 23 = 24 significant bits
            bf16:  sign | exponent (8 bits) | mantissa  (7 bits)   1 +  7 =  8 significant bits

        bf16 keeps 8 of fp32's 24 significant bits but has the same exponent, so each piece keeps
        its own scale, and 3 pieces hold an fp32 exactly (``_split_into_bf16_pieces``):

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
        128 experts, nothing for a 152k vocab. It also makes the backward ~1.4x slower.

        Relative error vs fp64, and backward time as a multiple of a bf16 Linear's (H100):

                                    relative error             backward time
                                    grad_input  grad_weight    eager   compiled
            LM head (Qwen3-8B, 2048 tokens; bf16 backward: 7.3 ms)
              bf16 grad_output      1.5e-3      1.3e-3          1.0x    1.0x
              2 pieces              2.1e-4      8.3e-6          2.5x    2.4x
              3 pieces              2.1e-4      8.8e-6          3.5x    3.5x
              fp32 matmul (IEEE)    7.9e-5      2.3e-6         13.2x   13.1x
            router (2048 -> 128; errors on 16k tokens, times on 64k; bf16 backward: 0.29 ms)
              bf16 grad_output      1.7e-3      1.4e-3          1.0x    1.0x
              2 pieces              2.5e-6      3.3e-6          1.7x    1.7x
              3 pieces              3.5e-7      2.6e-6          2.3x    2.3x
              fp32 matmul (IEEE)    5.1e-8      3.1e-7          5.5x    5.4x

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
        # exact gradient's: 96.2% -> 99.6%) for +3% backward time (H100). Inductor can't
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
        # returning None (autograd may copy a returned buffer) saves 2.5 ms and a 2.3 GiB
        # temporary per Qwen3-8B chunk (H100). Eager only: graph_trainer's compiled pass
        # (https://github.com/pytorch/torchtitan/pull/4768) only adds a bf16 GEMM into a
        # bf16 .grad. fp32 into fp32 would also need Inductor to compile addmm(out_dtype=).
        grad_weight_OD = torch.mm(
            stacked_PTO.T,
            # Copying x beats one GEMM per piece + add: 1.4x at 2048 tokens, 1.02-1.13x
            # at 8k-32k (Qwen3-8B head, H100).
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
# Compiled, the split + cat is one kernel instead of 6 (10 with 3 pieces): 6.78 -> 1.66 ms, -15-17%
# per Qwen3-8B head + loss chunk (2 pieces, H100), bitwise equal. Only the split: compiling the
# Function rounds grad_weight to bf16 (see the TODO in backward). Always compiled, like
# FlexAttention: a @local_compile region stays eager unless the model lists it. No fullgraph: past
# the recompile limit or with TORCH_COMPILE_DISABLE=1, it runs eagerly.
#
# make_fx (graph_trainer), FakeTensorMode and an outer torch.compile record ops instead of running
# them, and can't see inside a compiled kernel. So callers go through a custom op, which they
# record as one op: `custom_op` runs the compiled kernel on real tensors; `register_fake` gives
# them the output's shape and dtype without running it.


@torch.library.custom_op("torchtitan::split_into_bf16_pieces", mutates_args=())
def _split_into_bf16_pieces(
    grad_output_TO: torch.Tensor, higher_precision_bwd: bool, dim: int
) -> torch.Tensor:
    """``_split_into_bf16_pieces_eager``, compiled. See the comment above."""
    # out_features stays static: a symbolic out_features makes the split 2.4-4x slower (H100).
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
    Triton and XLA do: LM-head grad_weight error 1.4e-5 (truncating) -> 8.3e-6 (rounding), H100.
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


__all__ = [
    "ColumnParallelLinear",
    "GroupedLinear",
    "FP32OutputLinear",
    "Linear",
    "RowParallelLinear",
    "maybe_gather_tp_input",
]
