# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 (weight) + MXFP8 (activation) fake-quant QAT for Kimi K3.

The K3-faithful quantization path (vs the NF4 QLoRA convenience in
``lora.py``): K3 is MXFP4-QAT from SFT (MXFP4 weights, MXFP8
activations, OCP microscaling, block 32). This module provides an
EMULATED fake-quant so QAT runs on any GPU (fake-quant is bf16 compute;
FP4 hardware only speeds deployment, not QAT).

Fidelity scope (honest):
- Emulated MX rounding targets the OCP spec but is NOT verified
  bit-identical to Moonshot's kernels -> "MX-deployable", not
  "K3-QAT-bit-parity".
- Continued QAT from K3's shipped packed MXFP4 starts from an
  already-degraded master (K3's bf16 master is not released).
- torchao provides the MX primitives (MXTensor.to_mx / dequantize).

The wrapper does straight-through fake-quant: forward uses
dequant(quant(w)) so the loss sees quantized weights, while the bf16
master trains (STE via detach trick).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.tools.logging import logger

_WEIGHT_ELEM = torch.float4_e2m1fn_x2  # MXFP4
_ACT_ELEM = torch.float8_e4m3fn  # MXFP8
_BLOCK = 32  # OCP microscaling block


_warned_shapes: set[tuple] = set()


def _warn_unquantized(shape: tuple, block_size: int) -> None:
    if shape in _warned_shapes:
        return
    _warned_shapes.add(shape)
    logger.warning(
        "MXFP4 QAT: tensor of shape %s left UNQUANTIZED -- last dim %d is not a "
        "multiple of the MX block size %d. Under TP this is what a shard of "
        "w2_EDF looks like, so the effective quantization scope is narrower "
        "than requested and depends on the parallel layout. Choose an "
        "intermediate size divisible by block_size * tensor_parallel_degree to "
        "quantize it.",
        shape,
        shape[-1],
        block_size,
    )


def _fake_quant_mx(t: torch.Tensor, elem_dtype, block_size: int) -> torch.Tensor:
    """Straight-through emulated MX fake-quant: value = dequant(quant(t)),
    gradient = identity (STE)."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    if t.shape[-1] % block_size != 0:
        # Not blockable: leave in high precision. Warn rather than skip in
        # silence -- for w2_EDF the last dim IS the expert-TP-sharded one, so a
        # tensor that is blockable whole becomes non-blockable per shard and the
        # run would quietly train a different quantization scope than requested
        # (measured: moe_intermediate_size 224 is blockable, 224/2 under tp2 is
        # not). Once per shape, not once per forward.
        _warn_unquantized(tuple(t.shape), block_size)
        return t
    q = MXTensor.to_mx(
        t.contiguous().to(torch.bfloat16), elem_dtype=elem_dtype, block_size=block_size
    ).dequantize()
    # Emulated MX can overflow E2M1/E4M3 range on out-of-distribution
    # values (real QAT weights train in-range; random-init or exploding
    # activations do not). Never emit non-finite: fall back to the
    # high-precision value elementwise where quant blew up.
    q = q.to(t.dtype)
    q = torch.where(torch.isfinite(q), q, t)
    # STE: forward q, backward identity through t.
    return t + (q - t).detach()


class MXFP4QATLinear(nn.Module):
    """Fake-quant QAT wrapper over an nn.Linear.

    Weight is fake-quantized to MXFP4, activation to MXFP8, each forward.
    The underlying nn.Linear.weight stays the trainable bf16 master.
    """

    def __init__(self, base: nn.Linear, quantize_act: bool = True) -> None:
        super().__init__()
        self.base = base
        self.quantize_act = quantize_act

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    # Passthroughs, not conveniences: callers that reach for .weight on a
    # projection get None from a bare wrapper and silently skip it. That is how
    # tag_per_head_muon lost every wrapped Q/K/V projection, degrading Per-Head
    # Muon to full-matrix Muon with no warning (the warning only fires when
    # NOTHING is tagged, and unwrapped projections still tag). Returning the
    # base parameter itself, not a copy, keeps attribute tagging effective.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _fake_quant_mx(self.base.weight, _WEIGHT_ELEM, _BLOCK)
        if self.quantize_act:
            x = _fake_quant_mx(x, _ACT_ELEM, _BLOCK)
        return F.linear(x, w, self.base.bias)


_qat_experts_cache: dict[type, type] = {}

_EXPERT_WEIGHT_NAMES = ("w1_EFD", "w2_EDF", "w3_EFD")


def _qat_grouped_experts_subclass(parent_cls: type) -> type:
    """Subclass of a ``GroupedExperts`` variant with MXFP4/MXFP8 fake-quant.

    Works for any GroupedExperts subclass, which matters because K3's routed
    experts are ``KimiSiTUGroupedExperts``, not the core class.

    The fake-quantized weights are installed into ``self.__dict__`` for the
    duration of forward, which shadows ``_parameters`` for normal attribute
    lookup, and removed afterwards. A class-level property would be simpler but
    is wrong here: FSDP2's ``reset_sharded_param`` does ``getattr(module,
    name)`` OUTSIDE forward and requires the DTensor parameter back, so a
    permanent shadow fails with "'Tensor' object has no attribute
    '_local_tensor'". Renaming the masters (as the NF4 packing path does) would
    also work but would break the state-dict adapter and the expert TP/EP
    layout, both of which key off these exact names.
    """
    if parent_cls in _qat_experts_cache:
        return _qat_experts_cache[parent_cls]

    class MXFP4QATGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        def forward(self, x_RD, num_tokens_per_expert_E):
            if self._qat_quantize_act:
                x_RD = _fake_quant_mx(x_RD, _ACT_ELEM, _BLOCK)
            for name in _EXPERT_WEIGHT_NAMES:
                w = self._parameters.get(name)
                if w is None:
                    continue
                if isinstance(w, DTensor):
                    # Under EP/TP the master is a DTensor and the parent
                    # forward would call to_local() itself; localize here so
                    # MX quantization sees a plain tensor. Bare to_local
                    # mirrors the parent: the gradient keeps the parameter's
                    # own placement, correct because each rank quantizes
                    # exactly its own shard.
                    #
                    # Per-shard quantization is NOT equivalent to quantizing
                    # the whole tensor: MX block scales come from the max-abs
                    # within each block, so a shard boundary that cuts across
                    # the blocked dim changes the scales. For w1_EFD/w3_EFD the
                    # blocked last dim is D, which expert TP does not shard, so
                    # they are unaffected. For w2_EDF the last dim is the
                    # intermediate size -- exactly what expert TP shards -- so
                    # w2 under TP is quantized per shard, and is skipped
                    # entirely (with a warning) when the shard stops being a
                    # multiple of the block size.
                    w = w.to_local()
                self.__dict__[name] = _fake_quant_mx(w, _WEIGHT_ELEM, _BLOCK)
            try:
                return super().forward(x_RD, num_tokens_per_expert_E)
            finally:
                for name in _EXPERT_WEIGHT_NAMES:
                    self.__dict__.pop(name, None)

    MXFP4QATGroupedExperts.__name__ = f"MXFP4QAT{parent_cls.__name__}"
    MXFP4QATGroupedExperts.__qualname__ = MXFP4QATGroupedExperts.__name__
    _qat_experts_cache[parent_cls] = MXFP4QATGroupedExperts
    return MXFP4QATGroupedExperts


# The pre-release default: every MLA + dense/shared-FFN Linear. This is very
# nearly the complement of what K3 quantizes, so it is available only as an
# explicit ablation scope, never as the default.
ALL_LINEAR_QAT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def apply_mxfp4_qat(
    model: nn.Module,
    *,
    scope: str = "k3_official",
    targets: tuple[str, ...] = ALL_LINEAR_QAT_TARGETS,
    quantize_act: bool = True,
) -> int:
    """Attach MXFP4-weight / MXFP8-activation fake-quant QAT. Returns count.

    ``scope="k3_official"`` (default) follows the released quantization_config
    and report sec 4.1.4: routed experts only, everything else in higher
    precision. See quant_scope.py for the derivation.

    ``scope="all_linear"`` wraps ``targets`` instead -- the MLA and dense/shared
    FFN projections. That is close to the complement of K3's scope, so it is an
    ablation knob, not a faithful configuration. KDA projections are never
    wrapped: fla reads ``.weight`` directly, bypassing module forward, so a
    wrapper there would be silently dead.
    """
    if scope == "k3_official":
        from torchtitan.models.kimi_k3.quant_scope import quantizable_modules

        candidates = quantizable_modules(model)
        if not candidates:
            raise ValueError(
                "apply_mxfp4_qat(scope='k3_official') found no routed experts "
                "to quantize; a dense model has nothing in K3's MXFP4 scope"
            )
        n = 0
        for _fqn, experts in candidates:
            if getattr(experts, "_mxfp4_qat", False):
                continue  # idempotent: re-application is a no-op, not an error
            experts._qat_quantize_act = quantize_act
            experts.__class__ = _qat_grouped_experts_subclass(type(experts))
            experts._mxfp4_qat = True
            n += 1
        logger.info(
            "MXFP4 QAT (K3 official scope): %d routed-expert modules, "
            "MXFP8 activations %s",
            n,
            "on" if quantize_act else "off",
        )
        return n

    if scope != "all_linear":
        raise ValueError(
            f"Unknown scope {scope!r}; expected 'k3_official' or 'all_linear'"
        )

    from torchtitan.models.kimi_k3.model import KimiDeltaAttention

    n = 0
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            continue
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, MXFP4QATLinear(child, quantize_act=quantize_act))
                n += 1
    if n == 0:
        raise ValueError("apply_mxfp4_qat matched no target Linears")
    return n
