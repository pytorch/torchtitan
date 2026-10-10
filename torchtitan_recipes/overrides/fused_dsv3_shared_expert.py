# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental DSv3 shared-expert GEMM epilogues with explicit autograd.

The specialization is T4096/D7168/F2048 on GB300, with the existing MXFP8
linears and fused SwiGLU activation. Both flags default off. See
``docs/fused-dsv3-shared-expert.md`` for the numerical and roofline gates.
"""

import logging
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    SharedExpertRowParallelLinear,
)
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan.quantization.utils import get_quantized_linear

from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU

from ._dsv3_shared_expert.autograd import (
    FusedDSv3SharedExpertFunction,
    prepare_shared_expert,
    shared_expert_linear,
)
from ._dsv3_shared_expert.ops import ACCEPTED


__all__ = [
    "ACCEPTED",
    "FusedDSv3SharedExpert",
    "FusedDSv3SharedExpertFunction",
    "fused_dsv3_shared_expert",
]

logger = logging.getLogger(__name__)


def _supports_hooks(module):
    hooks = torch.nn.modules.module
    if any(
        (
            hooks._global_forward_pre_hooks,
            hooks._global_forward_hooks,
            hooks._global_backward_pre_hooks,
            hooks._global_backward_hooks,
        )
    ):
        return False
    for child in (module, module.w13, module.w2):
        if child._backward_hooks or child._backward_pre_hooks:
            return False
        for hook in (
            *child._forward_pre_hooks.values(),
            *child._forward_hooks.values(),
        ):
            if not getattr(hook, "__module__", "").startswith(
                "torch.distributed.fsdp."
            ):
                return False
    return True


class _SharedExpertOutputLinear(MXFP8Linear):
    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):
        pass

    def forward(self, input: torch.Tensor, *, prepared=None) -> torch.Tensor:
        if prepared is None:
            return super().forward(input)
        if self.weight.shape != (7168, 2048) or self.weight.dtype != torch.bfloat16:
            raise ValueError("Shared W2 requires its gathered BF16 [7168,2048] weight")
        return shared_expert_linear(
            input,
            self.weight,
            prepared,
            hidden_save_format=self.input_activation_format_for_backward,
        )


class FusedDSv3SharedExpert(FeedForward):
    """Fuse the ordinary shared FFN, keeping its checkpoint names and W2 call."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        shared_forward_quant: bool = False
        shared_backward_quant: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.shared_forward_quant = config.shared_forward_quant
        self.shared_backward_quant = config.shared_backward_quant

    def _supports_fusion(self, x):
        return (
            (self.shared_forward_quant or self.shared_backward_quant)
            and self.training
            and type(x) in (torch.Tensor, FakeTensor)
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape in ((4096, 7168), (1, 4096, 7168))
            and x.is_contiguous()
            and x.storage_offset() % 8 == 0
            and not torch.is_autocast_enabled("cuda")
            and isinstance(self.w13, MXFP8Linear)
            and isinstance(self.w2, _SharedExpertOutputLinear)
            and type(self.activation_fn) is FusedSwiGLU
            and self.w13.weight.shape == (2, 2048, 7168)
            and self.w13.weight.dtype == torch.bfloat16
            and self.w13.bias is None
            and self.w2.bias is None
            and spmd_mesh_size("tp") == 1
            and spmd_mesh_size("cp") == 1
            # W13 is consumed in its owner's unshard scope. An independently
            # sharded child needs its own module boundary, so it stays native.
            and not isinstance(self.w13, FSDPModule)
            # Selective region policies require the original linear boundaries.
            and not self.w13._remat_save_patterns
            and not self.w13._remat_recompute_patterns
            and not self.w2._remat_save_patterns
            and not self.w2._remat_recompute_patterns
            and _supports_hooks(self)
            and (
                isinstance(x, FakeTensor)
                or torch.cuda.get_device_capability(x.device) == (10, 3)
            )
        )

    def forward(self, x: torch.Tensor, *, prepared_input=None) -> torch.Tensor:
        if self._supports_fusion(x):
            prepared = prepare_shared_expert(
                x,
                self.w13.weight,
                fused_forward=self.shared_forward_quant,
                fused_backward=self.shared_backward_quant,
                input_save_format=self.w13.input_activation_format_for_backward,
                prepared_input=prepared_input,
            )
            return self.w2(prepared.hidden, prepared=prepared)
        return super().forward(x)


@override(
    target=FeedForward.Config,
    exact=True,
    fqns=["*.moe.shared_experts"],
    description="Fuse DSv3 shared-expert MXFP8 GEMM epilogues and their backward.",
)
def fused_dsv3_shared_expert(
    cfg: FeedForward.Config,
    *,
    forward_quant: bool = False,
    backward_quant: bool = False,
) -> FeedForward.Config:
    if not (forward_quant or backward_quant):
        return cfg
    w13_owners = (MXFP8Linear, get_quantized_linear(MXFP8Linear, ColumnParallelLinear))
    w2_owners = (
        MXFP8Linear,
        get_quantized_linear(MXFP8Linear, SharedExpertRowParallelLinear),
    )
    if (
        cfg.w13._owner not in w13_owners
        or cfg.w2._owner not in w2_owners
        or type(cfg.activation_fn) is not FusedSwiGLU.Config
        or (cfg.w13.num_linears, cfg.w13.in_features, cfg.w13.out_features)
        != (2, 7168, 2048)
        or (cfg.w2.num_linears, cfg.w2.in_features, cfg.w2.out_features)
        != (1, 2048, 7168)
        or cfg.w13.bias
        or cfg.w2.bias
    ):
        logger.warning(
            "Shared-expert fusion needs the DSv3 7168/2048 MXFP8 linears and "
            "an already configured FusedSwiGLU activation; keeping native modules."
        )
        return cfg
    owner = cfg.w2._owner
    assert owner is not None and issubclass(owner, Linear)
    output_class = get_quantized_linear(_SharedExpertOutputLinear, owner)
    return derive(
        cfg,
        FusedDSv3SharedExpert.Config,
        w2=derive(cfg.w2, output_class.Config),
        shared_forward_quant=forward_quant,
        shared_backward_quant=backward_quant,
    )
