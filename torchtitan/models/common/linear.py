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

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.inplace_wgrad_accum import (
    running_grad,
    uses_inplace_wgrad_accum,
)
from torchtitan.protocols.module import Module


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
        inplace_wgrad_accum: bool = True
        """Whether a later WGRAD (a microbatch with gradient sync disabled, or a
        ``ChunkedLossWrapper`` chunk) is added into the existing ``weight.grad``
        in the GEMM epilogue instead of a separate AccumulateGrad add and its
        full-size temporary. Tracing cannot represent this, so torch.compile and
        GraphTrainer must set it to False; forward raises if it is traced with
        this on. Subclasses whose compute cannot do it reject True."""

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        self.inplace_wgrad_accum = config.inplace_wgrad_accum
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

    @staticmethod
    def _flatten_weight_and_bias(
        weight: torch.Tensor, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        return weight.flatten(0, -2), None if bias is None else bias.flatten()

    def _unflatten_output(
        self, output: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Read each parameter once: a data parallel parametrization (SimpleFSDP)
        # all-gathers on every read.
        weight, bias = self.weight, self.bias
        output = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(input, weight, bias)
        return self._unflatten_output(output, weight)

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

        ``weight`` and ``bias`` are the parameters as stored, ``[N, K]`` and
        ``[N]``, or ``[num_linears, N, K]`` and ``[num_linears, N]`` for a
        stacked projection; implementations flatten them. Passing the
        parameter itself lets a backward add into its ``.grad`` in place and
        return a gradient in its shape.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        Explicit operands let those boundaries adjust an operand's SPMD type
        before invoking the selected local compute implementation.
        """
        if uses_inplace_wgrad_accum(self.inplace_wgrad_accum, weight, self):
            # The Function's backward does not replicate autocast's casts, and
            # its in-place addmm(out=) is not autocast-eligible.
            device_type = input.device.type
            if torch.amp.is_autocast_available(
                device_type
            ) and torch.is_autocast_enabled(device_type):
                raise RuntimeError(
                    f"{type(self).__qualname__} has inplace_wgrad_accum=True but "
                    "runs under autocast; set inplace_wgrad_accum=False."
                )
            return _LinearFunction.apply(input, weight, bias)
        return F.linear(input, *self._flatten_weight_and_bias(weight, bias))


@spmd.register_local_autograd_function
class _LinearFunction(torch.autograd.Function):
    """``F.linear`` whose backward adds WGRAD into ``weight.grad`` in place.

    ``Linear._linear`` uses it only when in-place WGRAD accumulation is on.
    Without a running gradient it computes the same GEMMs as ``F.linear``'s
    autograd formula, so the gradients are bitwise identical. Later
    contributions use ``addmm(out=weight.grad)``, which adds the GEMM's fp32
    accumulator into the running gradient instead of first rounding the WGRAD
    to the weight dtype.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        weight_NK, bias_N = Linear._flatten_weight_and_bias(weight, bias)
        ctx.save_for_backward(input, weight_NK)
        # Kept on ctx rather than saved: backward needs this exact parameter
        # object to read and clear its .grad, and saved-tensor hooks may unpack
        # a different one. A leaf parameter does not reference its graph, so
        # this forms no cycle.
        ctx.weight_param = weight
        ctx.bias_shape = None if bias is None else bias.shape
        return F.linear(input, weight_NK, bias_N)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input, weight_NK = ctx.saved_tensors
        weight = ctx.weight_param
        grad_MN = grad_output.reshape(-1, grad_output.shape[-1])
        input_MK = input.reshape(-1, input.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_MN.mm(weight_NK).view(input.shape)
        if ctx.needs_input_grad[1]:
            grad = running_grad(weight)
            if grad is None:
                grad_weight = grad_MN.t().mm(input_MK).view(weight.shape)
            else:
                # A later microbatch or chunk: add into the running gradient in
                # the GEMM epilogue. Then hand the same buffer back and clear
                # the parameter, so AccumulateGrad reattaches it instead of
                # adding it to itself.
                grad_NK = grad.view(-1, grad.shape[-1])
                # Add in the running gradient's dtype. out_dtype only when it
                # differs from the inputs': CPU has no addmm(out_dtype=) kernel,
                # even when it matches them.
                if grad.dtype == input_MK.dtype:
                    torch.addmm(grad_NK, grad_MN.t(), input_MK, out=grad_NK)
                else:
                    torch.addmm(
                        grad_NK,
                        grad_MN.t(),
                        input_MK,
                        out_dtype=grad.dtype,
                        out=grad_NK,
                    )
                weight.grad = None
                grad_weight = grad
        if ctx.needs_input_grad[2]:
            grad_bias = grad_MN.sum(0).view(ctx.bias_shape)
        return grad_input, grad_weight, grad_bias


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
        weight, bias = self.weight, self.bias
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
                output = self._unflatten_output(linear_fn(input, weight, bias), weight)
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias), weight)
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
        weight, bias = self.weight, self.bias
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
                output = self._unflatten_output(linear_fn(input, weight, bias), weight)
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias), weight)
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


__all__ = [
    "ColumnParallelLinear",
    "GroupedLinear",
    "Linear",
    "RowParallelLinear",
    "maybe_gather_tp_input",
]
