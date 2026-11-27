# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``FullPrecisionLinear``: a ``Linear`` computed in fp32 on any floating input."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import Linear


class FullPrecisionLinear(Linear):
    """``Linear`` computed in fp32 (or wider): the input is upcast to at least fp32, the weight
    is kept in fp32, and the output and weight gradient are fp32. The input gradient is
    returned in the input's dtype. An fp64 input and weight stay fp64.

    The weight and bias stay fp32 under mixed precision: ``preserve_dtype_params`` defaults to
    ``("weight",)`` and always gains ``"bias"`` when there is one, so FSDP all-gathers them in
    fp32 instead of ``param_dtype``. A weight in another dtype fails the matmul's dtype check
    rather than silently running in lower precision.

    Use it where ``HiMidLoLinear``'s bf16 GEMMs do not apply: on CPU, in batch-invariant mode,
    or where fp32 products are wanted. It runs fp32 matmuls (no tensor cores on most GPUs), so
    it suits small projections such as a router gate; for an LM head prefer ``HiMidLoLinear``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        preserve_dtype_params: tuple[str, ...] = ("weight",)

        def __post_init__(self) -> None:
            Linear.Config.__post_init__(self)
            # A bf16-rounded bias would add a bf16 error to the fp32 output.
            if self.bias and "bias" not in self.preserve_dtype_params:
                self.preserve_dtype_params = (*self.preserve_dtype_params, "bias")

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = torch.promote_types(input.dtype, torch.float32)
        return super()._linear(
            input.to(dtype), weight, None if bias is None else bias.to(dtype)
        )


__all__ = ["FullPrecisionLinear"]
