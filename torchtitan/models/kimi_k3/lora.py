# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-level LoRA for the plain-module Kimi Linear model.

    Upstream's ``LoRAConverter`` works on ``Linear.Config`` trees and cannot reach
    directly-built modules, so ``apply_lora`` swaps target ``nn.Linear`` projections
    for :class:`KimiLoRALinear` after build. ``lora_b`` is zero-init, so step 0 is
    bit-identical to the base model.

    See ``phase13_k3like_48b_posttrain/LORA_MODULE_LEVEL.md``.
    """

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# KDA-internal projections are NOT targetable: KimiDeltaAttention reads
# ``linear.weight`` directly for the fla kernels (module forward is
# bypassed), so a wrapper there would be silently dead. apply_lora
# skips the KDA subtree structurally, so the set below only needs to
# cover MLA + dense/shared FFN + the latent MoE projections.
#
# Entries containing a dot match a qualified name suffix; bare entries
# match a leaf module name. The latent projections are named ``down`` /
# ``up``, which are too generic to match bare.
#
# Routed experts are absent by construction: they are GroupedExperts 3-D
# parameters, not nn.Linear, and get adapted (or quantized) through the
# grouped path instead -- see quant_scope.py.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    # MLA, direct-Q path (48B-A3B, q_lora_rank=None)
    "q_proj",
    # MLA, compressed-Q path (K3 ships q_lora_rank=1536)
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    # K3's Gated MLA output gate (report Eq. 7). Grafting a gate onto a
    # checkpoint that has none makes this a NEW param that must be
    # full-param trainable instead -- pass fullparam_markers then.
    "attn_gate_proj",
    # dense FFN and shared experts
    "gate_proj",
    "up_proj",
    "down_proj",
    # latent MoE projections (K3's Eq. 11 shared W_down / W_up)
    "latent.down",
    "latent.up",
)

# Params that stay full-param trainable under base-freeze: the AttnRes
# graft set (new zero-init params; the "alpha-fullparam exception").
_FULLPARAM_EXCEPTION_MARKERS: tuple[str, ...] = (
    "attention_res",
    "ffn_res",
    "output_res",
)


class KimiLoRALinear(nn.Module):
    """LoRA wrapper over an existing ``nn.Linear``.

    ``forward = base(x) + (alpha / rank) * lora_b(lora_a(x))`` with
    ``lora_a`` kaiming-init and ``lora_b`` zero-init (identity at
    step 0). Adapters are raw parameters (not nn.Linear children) so
    the model's generic init pass does not blindly re-init them;
    :meth:`reset_parameters` is dispatched from
    ``KimiK3Model.init_weights`` by class name.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        quantize_base: str | None = None,
        quantize_act: bool = False,
    ) -> None:
        super().__init__()
        assert rank > 0
        self.base = base
        self.base.weight.requires_grad_(False)
        # Capture before quantization: mxfp4 drops base.weight (split
        # storage), so dtype/device must be read first. Adapters match the
        # base compute dtype, else a bf16 base + fp32 adapter mismatches in
        # the forward matmul.
        pdtype = base.weight.dtype
        dev = base.weight.device
        # K3 trains the BACKBONE in MXFP4 weights + MXFP8 activations
        # (report sec 4.1.4); that is a property of the model, not of LoRA. When
        # LoRA attaches to a base that is already packed MXFP4, the activations
        # it sees should be MXFP8 too, or the adapter trains against numerics
        # the deployed model never sees. Off by default because the released
        # checkpoint is weights-only (input_activations: null), so a frozen-base
        # load without QAT semantics is also a legitimate configuration.
        self._quantize_act = quantize_act
        self._quantize_base = None
        if quantize_base == "nf4":
            self.quantize_base_nf4()
        elif quantize_base == "mxfp4":
            self.quantize_base_mxfp4()
        elif quantize_base is not None:
            raise ValueError(f"Unsupported quantize_base={quantize_base!r}")
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._lora_scaling = alpha / rank
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=dev, dtype=pdtype)
        )
        self.lora_b = nn.Parameter(
            torch.empty(base.out_features, rank, device=dev, dtype=pdtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lora_a.device.type != "meta":
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    @torch.no_grad()
    def quantize_base_nf4(self) -> bool:
        """Pack the frozen base to NF4 (torchao). Idempotent.

        QLoRA is lossy by design -- the step-0 identity anchor holds
        only for the unquantized gated graft; QLoRA trades exactness for
        a ~4x cut in memory AND (on comms-bound fabrics) in FSDP
        all-gather traffic. Callable at build (over default weights) or
        post-load (over checkpoint weights) -- the latter is the correct
        trainer order, so real weights, not init noise, get quantized.

        torchao NF4 double-quant requires numel divisible by
        block_size(64) * scaler_block_size(256) = 16384. Dims that don't
        divide are left in bf16 (a real torchao constraint, not all
        model dims are NF4-friendly); returns False in that case.
        """
        from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

        if isinstance(self.base.weight, NF4Tensor):
            self._quantize_base = "nf4"
            return True  # already packed
        self._nf4_ok = self.base.weight.numel() % 16384 == 0
        if not self._nf4_ok:
            self._quantize_base = None  # leave bf16
            return False
        self.base.weight = nn.Parameter(
            to_nf4(self.base.weight.data.to(torch.bfloat16)),
            requires_grad=False,
        )
        self._quantize_base = "nf4"
        return True

    @torch.no_grad()
    def quantize_base_mxfp4(self) -> bool:
        """Pack the frozen base to MXFP4 (torchao MX, block 32) -- K3's
        native weight format (FP4 E2M1 + MX E8M0 block scale). Idempotent.

        Split storage: MXTensor's packed qdata is half-width, so the logical
        weight view is non-contiguous and FSDP2 rejects it as a param. Store
        qdata (uint8) and scale (E8M0 bytes viewed as uint8, since FSDP2's
        all-gather has no float8_e8m0fnu copy kernel) as plain contiguous
        frozen params + the flatten ctx, and reconstruct the MXTensor via
        __tensor_unflatten__ after all-gather. block_size 32 needs
        in_features % 32 == 0 (all K3 dims satisfy this); else stays bf16.
        """
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        if getattr(self, "_mx_ctx", None) is not None:
            self._quantize_base = "mxfp4"
            return True  # already packed
        w = self.base._parameters.get("weight")
        if w is None or w.shape[-1] % 32 != 0:
            self._quantize_base = None
            return False
        if w.is_meta:
            # Meta-first trainer flow: register the PACKED LAYOUT only
            # (qdata [out, in/2] uint8 + scale [out, in/32] e8m0-as-uint8)
            # so FSDP shards the packed bytes; the actual quantized values
            # arrive via DCP checkpoint load (stream_quantize_mxfp4_dcp.py
            # converts a bf16 checkpoint to this layout). Valid because
            # MX block-32 quantization is row-blockwise, so it commutes
            # with FSDP2's Shard(0) row sharding: quantize-then-shard ==
            # shard-then-load-quantized-rows. The flatten ctx carries no
            # shape/data, so a 1x32 dummy reproduces it exactly.
            out_f, in_f = w.shape
            dummy = MXTensor.to_mx(
                torch.zeros(1, 32, dtype=torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
            )
            _, self._mx_ctx = dummy.__tensor_flatten__()
            self._mx_scale_dtype = dummy.scale.dtype
            self.base_qdata = nn.Parameter(
                torch.empty(out_f, in_f // 2, dtype=torch.uint8, device="meta"),
                requires_grad=False,
            )
            self.base_scale = nn.Parameter(
                torch.empty(out_f, in_f // 32, dtype=torch.uint8, device="meta"),
                requires_grad=False,
            )
            del self.base._parameters["weight"]
            self._quantize_base = "mxfp4"
            return True
        mx = MXTensor.to_mx(
            w.data.to(torch.bfloat16),
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=32,
        )
        _, self._mx_ctx = mx.__tensor_flatten__()
        self._mx_scale_dtype = mx.scale.dtype
        self.base_qdata = nn.Parameter(mx.qdata.contiguous(), requires_grad=False)
        self.base_scale = nn.Parameter(
            mx.scale.view(torch.uint8).contiguous(), requires_grad=False
        )
        # Drop the bf16 base weight so FSDP shards only the packed bytes.
        del self.base._parameters["weight"]
        self._quantize_base = "mxfp4"
        return True

    def apply_packed_mxfp4_tp(self, tp_mesh, colwise: bool) -> None:
        """TP-shard the packed MXFP4 base (call at parallelize time).

        Colwise (out-sharded): qdata/scale shard on dim 0 -- MX block-32
        quantization is row-blockwise, so row sharding is exact. Rowwise
        (in-sharded): shard on dim 1; requires (in_features // tp) % 32
        == 0 so the shard boundary lands on whole MX blocks (then the
        qdata byte boundary in/2/tp and the scale boundary in/32/tp are
        integral too). Registered as DTensor so DCP resharding of the
        packed checkpoint keeps working; the forward computes on the
        LOCAL shard (see the packed-TP branch in :meth:`forward`).
        """
        from torch.distributed.tensor import distribute_tensor, Shard

        tp = tp_mesh.size()
        out_f, in_f = self.base.out_features, self.base.in_features
        if colwise:
            if out_f % tp != 0:
                raise ValueError(
                    f"packed-MXFP4 colwise TP: out_features {out_f} not "
                    f"divisible by tp={tp}"
                )
            placements = [Shard(0)]
        else:
            if in_f % tp != 0 or (in_f // tp) % 32 != 0:
                raise ValueError(
                    f"packed-MXFP4 rowwise TP: in_features {in_f} must be "
                    f"divisible by tp={tp} with (in/tp) % 32 == 0 (MX "
                    "block alignment)"
                )
            placements = [Shard(1)]
        self.base_qdata = nn.Parameter(
            distribute_tensor(self.base_qdata, tp_mesh, placements),
            requires_grad=False,
        )
        self.base_scale = nn.Parameter(
            distribute_tensor(self.base_scale, tp_mesh, placements),
            requires_grad=False,
        )
        self._tp_style = "colwise" if colwise else "rowwise"
        self._tp_mesh = tp_mesh

    def _maybe_quantize_act(self, x: torch.Tensor) -> torch.Tensor:
        """MXFP8 fake-quant on the input, when the base is packed MXFP4.

        Shares ``mxfp4_qat``'s emulated MX rounding so the QAT path and the
        packed-base path cannot drift apart.
        """
        if not self._quantize_act or self._quantize_base != "mxfp4":
            return x
        from torchtitan.models.kimi_k3.mxfp4_qat import (
            _ACT_ELEM,
            _BLOCK,
            _fake_quant_mx,
        )

        return _fake_quant_mx(x, _ACT_ELEM, _BLOCK)

    def _dequant_base_mxfp4(self) -> torch.Tensor:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        qdata, scale = self.base_qdata, self.base_scale
        if getattr(self, "_tp_style", None) is not None:
            # TP-sharded packed base: dequantize this rank's LOCAL shard
            # (row rows for colwise, whole-block column slice for
            # rowwise); the forward's packed-TP branch does the local
            # matmul + collective.
            qdata = qdata.to_local() if hasattr(qdata, "to_local") else qdata
            scale = scale.to_local() if hasattr(scale, "to_local") else scale
        else:
            if hasattr(qdata, "full_tensor"):
                qdata = qdata.full_tensor()
            if hasattr(scale, "full_tensor"):
                scale = scale.full_tensor()
        mx = MXTensor.__tensor_unflatten__(
            {"qdata": qdata, "scale": scale.view(self._mx_scale_dtype)},
            self._mx_ctx,
            None,
            None,
        )
        return mx.dequantize()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def bias(self):
        """Transparent passthrough, so callers that inspect the wrapped
        Linear keep working. init_weights' graft-gate branch reads
        ``gate_proj.bias`` to decide whether the gate is the near-identity
        variant, and attn_gate_proj is a LoRA target."""
        return self.base.bias

    @property
    def weight(self):
        """Transparent passthrough to the base weight.

        Returns None when the base is packed (quantize_base_mxfp4 deletes
        ``base.weight`` in favour of split qdata/scale storage), which is the
        signal callers already use to skip init for packed bases."""
        return self.base._parameters.get("weight")

    def _forward_packed_tp(self, x: torch.Tensor) -> torch.Tensor:
        """TP forward for the packed-MXFP4 base: local dequant + local
        matmul, DTensor only at the boundary.

        Colwise: x is replicated (DTensor(Replicate) or plain local);
        each rank computes its out/tp columns; returns
        DTensor(Shard(-1)) to match ColwiseParallel(use_local_output=
        False) consumers. Rowwise: x is the in/tp local shard (plain, or
        DTensor(Shard(-1))); local partial matmul, ONE all-reduce over
        tp for base+adapter combined (linearity: sum commutes), returns
        a plain replicated tensor to match RowwiseParallel(
        output_layouts=Replicate, use_local_output=True).

        Backward: explicit grad_placements make the tp reductions
        happen -- replicated operands used by all ranks (colwise x and
        lora_a, rowwise lora_b) carry Partial gradients that must
        all-reduce; a bare to_local() would silently skip it (same trap
        as the attn_res pseudo-query note).
        """
        from torch.distributed.tensor import DTensor, Partial, Replicate

        colwise = self._tp_style == "colwise"
        tp_mesh = self._tp_mesh

        if isinstance(x, DTensor):
            grad_pl = (Partial(),) if colwise else None
            x_loc = x.to_local(grad_placements=grad_pl)
        else:
            x_loc = x

        x_loc = self._maybe_quantize_act(x_loc)
        w_loc = self._dequant_base_mxfp4().to(x_loc.dtype)

        la, lb = self.lora_a, self.lora_b
        if colwise:
            # lora_a Replicate (grads sum over tp), lora_b Shard(0) local.
            la = (
                la.to_local(grad_placements=(Partial(),))
                if isinstance(la, DTensor)
                else la
            )
            lb = lb.to_local() if isinstance(lb, DTensor) else lb
        else:
            # lora_a Shard(1) local, lora_b Replicate (grads sum over tp).
            la = la.to_local() if isinstance(la, DTensor) else la
            lb = (
                lb.to_local(grad_placements=(Partial(),))
                if isinstance(lb, DTensor)
                else lb
            )
        if la.dtype != x_loc.dtype:
            la = la.to(x_loc.dtype)
            lb = lb.to(x_loc.dtype)

        out_loc = F.linear(x_loc, w_loc) + self._lora_scaling * F.linear(
            F.linear(x_loc, la), lb
        )
        bias = self.base.bias
        if colwise:
            from torch.distributed.tensor import Shard

            # Colwise shards the OUTPUT features, so this rank's bias slice matches its
            # output slice and is added locally.
            if bias is not None:
                b = bias.to_local() if isinstance(bias, DTensor) else bias
                out_loc = out_loc + b.to(out_loc.dtype)
            return DTensor.from_local(
                out_loc, tp_mesh, [Shard(out_loc.dim() - 1)], run_check=False
            )
        # Rowwise: local outputs are partial sums over the in/tp shards.
        out = DTensor.from_local(out_loc, tp_mesh, [Partial()], run_check=False)
        out = out.redistribute(tp_mesh, [Replicate()]).to_local()
        # Rowwise does NOT shard the output, so the bias must be added AFTER the partial
        # sums are reduced -- adding it to out_loc would apply it once per TP rank.
        if bias is not None:
            b = bias.full_tensor() if isinstance(bias, DTensor) else bias
            out = out + b.to(out.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor, Replicate, Shard

        x_is_dt = isinstance(x, DTensor)
        if self._quantize_base == "nf4":
            from torchao.dtypes.nf4tensor import linear_nf4

            base_out = linear_nf4(x, self.base.weight)
            # linear_nf4 takes weight only, so the bias has to be added here as the
            # mxfp4 and unquantized branches do. Omitting it shifted every biased
            # projection (attn_gate_proj) by a constant with no error.
            if self.base.bias is not None:
                base_out = base_out + self.base.bias
        elif self._quantize_base == "mxfp4":
            if getattr(self, "_tp_style", None) is not None:
                return self._forward_packed_tp(x)
            # No weight-only MXFP4 linear in torchao yet: dequant then
            # matmul (memory/comms win from the packed base still holds).
            x = self._maybe_quantize_act(x)
            w = self._dequant_base_mxfp4().to(x.dtype)
            if x_is_dt:
                # Dequant densifies the packed params, but a NoParallel
                # descent (MoE shared experts) hands us a DTensor input:
                # replicate w so the matmul stays DTensor x DTensor.
                mesh = x.device_mesh
                w = DTensor.from_local(
                    w, mesh, [Replicate()] * mesh.ndim, run_check=False
                )
            base_out = F.linear(x, w)
            if self.base.bias is not None:
                base_out = base_out + self.base.bias
        else:
            bw = self.base.weight
            if not x_is_dt and isinstance(bw, DTensor):
                # Plain input, DTensor base weight: reduce iff Rowwise (module docstring).
                bb = self.base.bias
                if isinstance(bb, DTensor):
                    bb = bb.to_local()
                if any(p.is_shard() and p.dim == 1 for p in bw.placements):
                    mesh = bw.device_mesh
                    x_dt = DTensor.from_local(
                        x, mesh, (Shard(x.dim() - 1),), run_check=False
                    )
                    # Partial -> Replicate all-reduce, then plain to match the
                    # style's use_local_output=True convention. Bias is added
                    # AFTER the reduction, or it would be counted once per rank.
                    base_out = F.linear(x_dt, bw).full_tensor()
                    if bb is not None:
                        base_out = base_out + bb
                else:
                    base_out = F.linear(x, bw.to_local(), bb)
            else:
                base_out = self.base(x)

        # TP: align the adapters with the input's tensor kind so the matmul
        # isn't mixed Tensor/DTensor. Colwise/Rowwise-styled projections get
        # DTensor adapters (distributed in parallelize) and a DTensor input;
        # NoParallel descents (MoE shared experts) may leave the raw adapter
        # params plain while the input is a DTensor, or run plain input
        # against distributed adapters -- handle both directions.
        la, lb = self.lora_a, self.lora_b
        if x_is_dt:
            mesh = x.device_mesh
            repl = [Replicate()] * mesh.ndim
            if not isinstance(la, DTensor):
                la = DTensor.from_local(la, mesh, repl, run_check=False)
            if not isinstance(lb, DTensor):
                lb = DTensor.from_local(lb, mesh, repl, run_check=False)
        elif isinstance(la, DTensor) and any(p.is_shard() for p in la.placements):
            # x is plain but lora_a is sharded on the contracted axis -- the
            # rowwise case, and o_proj is the only site where it happens (the
            # MLA attention output is built in plain-tensor land). Unwrapping
            # both adapters here would make the product each rank's PARTIAL
            # contribution with no DTensor left to sum it: RowwiseParallel
            # all-reduces the base only, so the adapter rides outside it and
            # lora_b's gradient comes back short by ~sqrt(tp).
            #
            # Lift x into the adapters' mesh instead of dropping them out of
            # it. Then DTensor owns the whole product and gets BOTH gradients
            # right, which one reduction on their shared output cannot: lora_a
            # is the local shard of a Shard(1) parameter and its gradient is
            # already complete per rank, while lora_b is Replicate and its
            # gradient is a sum across ranks.
            mesh = la.device_mesh
            shard_axis = next(p.dim for p in la.placements if p.is_shard())
            x = DTensor.from_local(x, mesh, (Shard(x.dim() - 1),), run_check=False)
            del shard_axis
        else:
            if isinstance(la, DTensor):
                la = la.to_local()
            if isinstance(lb, DTensor):
                lb = lb.to_local()
        if la.dtype != x.dtype:
            # Frozen-base LoRA: trainable adapters stay fp32 masters while
            # the frozen base (and thus x) is bf16. Under FSDP the
            # mixed-precision policy casts adapters for compute; without
            # FSDP (dp_shard=1 debug runs) align here instead.
            la = la.to(x.dtype)
            lb = lb.to(x.dtype)
        lora_out = F.linear(F.linear(x, la), lb)
        # DTensor adapter output but a plain base output (a use_local_output
        # style). Match the base's locality -- which of the two ways depends on
        # the style, and getting it backwards is a shape error, not a silent
        # one:
        #   Rowwise: base_out is the FULL width (already all-reduced), and the
        #     adapter is Partial, so full_tensor() to all-reduce it.
        #   Colwise: base_out is this rank's SHARD, and the adapter is Shard on
        #     the output features, so to_local() to take the matching shard.
        #     full_tensor() here all-gathers to the global width and fails
        #     against the narrower base (e.g. 512 vs 256 at tp=2, which is what
        #     attn_gate_proj hit once it became a LoRA target).
        if isinstance(lora_out, DTensor) and not isinstance(base_out, DTensor):
            from torch.distributed.tensor import Shard

            if any(isinstance(p, Shard) for p in lora_out.placements):
                lora_out = lora_out.to_local()
            else:
                lora_out = lora_out.full_tensor()
        return base_out + self._lora_scaling * lora_out


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    freeze_base: bool = True,
    quantize_base: str | None = None,
    quantize_act: bool = False,
    fullparam_markers: tuple[str, ...] = _FULLPARAM_EXCEPTION_MARKERS,
) -> int:
    """Swap target Linears for LoRA wrappers; optionally freeze the base.

    Returns the number of wrapped modules. Freezing covers every
    parameter except LoRA adapters and the AttnRes graft params
    (alpha-fullparam exception).
    """
    from torchtitan.models.kimi_k3.model import KimiDeltaAttention

    leaf_targets = frozenset(t for t in targets if "." not in t)
    suffix_targets = tuple(f".{t}" for t in targets if "." in t)

    num_wrapped = 0
    for parent_fqn, module in model.named_modules():
        if isinstance(module, KimiDeltaAttention):
            # Structural skip -- see DEFAULT_LORA_TARGETS note.
            continue
        for child_name, child in list(module.named_children()):
            fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            matched = child_name in leaf_targets or fqn.endswith(suffix_targets)
            if matched and isinstance(child, nn.Linear):
                setattr(
                    module,
                    child_name,
                    KimiLoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        quantize_base=quantize_base,
                        quantize_act=quantize_act,
                    ),
                )
                num_wrapped += 1
    if num_wrapped == 0:
        raise ValueError(f"apply_lora matched no target Linears (targets={targets}).")

    if freeze_base:
        for name, p in model.named_parameters():
            if "lora_a" in name or "lora_b" in name:
                continue
            if any(m in name for m in fullparam_markers):
                continue
            p.requires_grad_(False)
            # Frozen params need no fp32 master copy: keep them bf16
            # resident. At 48B this is the difference between 12 GiB/card
            # sharded (fast, no offload) and 24.6 GiB fp32 shards that
            # force CPU offload (~5 min/step over PCIe). HF checkpoints
            # are bf16, so the load path is dtype-exact too.
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
    return num_wrapped


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA-only checkpoint payload: adapters + AttnRes graft params.

    This is the unit a veRL trainer->rollout weight sync ships when the
    base is frozen (LoRA-only DCP leg of the P0 trio).
    """
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


_nf4_experts_cls_cache: dict[type, type] = {}


def _nf4_experts_subclass(cls: type) -> type:
    """Subclass with dequant properties over the NF4-packed expert params."""
    if cls in _nf4_experts_cls_cache:
        return _nf4_experts_cls_cache[cls]

    def _make_fget(name: str):
        def fget(self):
            from torch.distributed.tensor import DTensor
            from torchao.dtypes.nf4tensor import NF4Tensor

            t = self._parameters[name + "_nf4"]
            if isinstance(t, DTensor):
                # Pre-unshard access (outside FSDP's forward window):
                # gather explicitly. During forward FSDP2 exposes the
                # plain unsharded NF4.
                t = t.full_tensor()
            if isinstance(t, NF4Tensor):
                t = t.get_original_weight()
            return t.view(self._nf4_shapes[name])

        return fget

    sub = type(
        f"NF4{cls.__name__}",
        (cls,),
        {n: property(_make_fget(n)) for n in ("w1_EFD", "w2_EDF", "w3_EFD")},
    )
    _nf4_experts_cls_cache[cls] = sub
    return sub


def quantize_grouped_experts_nf4(model: nn.Module) -> int:
    """Pack every GroupedExperts weight to NF4 (the 48B memory/comms bulk).

    3-D [E, A, B] params pack as a 2-D (E*A, B) NF4 view; a dequant
    property restores the logical shape at forward time (GroupedExperts
    reads self.w1_EFD etc. and casts to bf16 anyway). Params stay
    registered (frozen) so FSDP can shard the packed bytes.
    """
    from torchao.dtypes.nf4tensor import to_nf4

    from torchtitan.models.common.moe import GroupedExperts

    num_quantized = 0
    for m in model.modules():
        if isinstance(m, GroupedExperts) and not hasattr(m, "_nf4_shapes"):
            shapes: dict[str, tuple[int, ...]] = {}
            for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
                p = m._parameters.get(name)
                if p is None:
                    continue
                shapes[name] = tuple(p.shape)
                packed = to_nf4(p.data.reshape(-1, p.shape[-1]).to(torch.bfloat16))
                # Store under a distinct name: the logical name becomes
                # a dequant property, and FSDP shards the packed param.
                del m._parameters[name]
                m.register_parameter(
                    name + "_nf4", nn.Parameter(packed, requires_grad=False)
                )
            m._nf4_shapes = shapes
            m.__class__ = _nf4_experts_subclass(type(m))
            num_quantized += 1
    return num_quantized


_mxfp4_experts_cls_cache: dict[type, type] = {}


def _mxfp4_experts_subclass(cls: type) -> type:
    """Subclass with dequant properties over MXFP4-packed expert params."""
    if cls in _mxfp4_experts_cls_cache:
        return _mxfp4_experts_cls_cache[cls]

    def _make_fget(name: str):
        def fget(self):
            from torch.distributed.tensor import DTensor
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            qdata = self._parameters[name + "_qdata"]
            scale = self._parameters[name + "_scale"]
            if isinstance(qdata, DTensor):
                # Pre-unshard access (outside FSDP's forward window): gather
                # explicitly. During forward FSDP2 exposes plain unsharded
                # tensors, mirroring the NF4 path above.
                qdata = qdata.full_tensor()
                scale = scale.full_tensor()
            mx = MXTensor.__tensor_unflatten__(
                {"qdata": qdata, "scale": scale.view(self._mx_scale_dtype)},
                self._mx_ctx,
                None,
                None,
            )
            return mx.dequantize().view(self._mxfp4_shapes[name])

        return fget

    sub = type(
        f"MXFP4{cls.__name__}",
        (cls,),
        {n: property(_make_fget(n)) for n in ("w1_EFD", "w2_EDF", "w3_EFD")},
    )
    _mxfp4_experts_cls_cache[cls] = sub
    return sub


def quantize_grouped_experts_mxfp4(model: nn.Module) -> int:
    """Pack routed-expert weights to MXFP4 -- K3's actual quantization scope.

    This is the QLoRA counterpart of ``apply_mxfp4_qat``: real packing (the
    memory win) rather than fake-quant, for a frozen base. Only modules in
    K3's official scope are touched (see quant_scope.py), so the attention,
    latent, shared-expert, router and lm_head weights the release keeps in
    higher precision stay bf16.

    A 3-D ``[E, A, B]`` param packs as a 2-D ``(E*A, B)`` MX view: MX blocks
    run along the last dim, so flattening the leading dims is exact and the
    per-expert boundary always falls on a block boundary. Split storage
    (qdata uint8 + scale-as-uint8) matches ``KimiLoRALinear``: MXTensor's
    packed qdata is half-width, so the logical view is non-contiguous and
    FSDP2 rejects it as a param. Requires ``B % 32 == 0``; other params stay
    bf16.
    """
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    from torchtitan.models.kimi_k3.quant_scope import (
        MXFP4_GROUP_SIZE,
        quantizable_modules,
    )

    num_quantized = 0
    for _fqn, m in quantizable_modules(model):
        if hasattr(m, "_mxfp4_shapes"):
            continue  # idempotent
        shapes: dict[str, tuple[int, ...]] = {}
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            p = m._parameters.get(name)
            if p is None or p.shape[-1] % MXFP4_GROUP_SIZE != 0:
                continue
            shapes[name] = tuple(p.shape)
            mx = MXTensor.to_mx(
                p.data.reshape(-1, p.shape[-1]).to(torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=MXFP4_GROUP_SIZE,
            )
            _, m._mx_ctx = mx.__tensor_flatten__()
            m._mx_scale_dtype = mx.scale.dtype
            del m._parameters[name]
            m.register_parameter(
                name + "_qdata",
                nn.Parameter(mx.qdata.contiguous(), requires_grad=False),
            )
            m.register_parameter(
                name + "_scale",
                nn.Parameter(
                    mx.scale.view(torch.uint8).contiguous(), requires_grad=False
                ),
            )
        if not shapes:
            continue
        m._mxfp4_shapes = shapes
        m.__class__ = _mxfp4_experts_subclass(type(m))
        num_quantized += 1
    return num_quantized


def quantize_lora_bases(
    model: nn.Module, *, mode: str = "nf4", experts: bool = True
) -> int:
    """Post-load QLoRA hook: quantize every LoRA base after weights load.

    The titan trainer's meta-first flow builds, then materializes real
    weights (init or checkpoint), THEN should quantize -- packing at
    build time (KimiLoRALinear(quantize_base=...)) quantizes init noise /
    meta storage, not the loaded checkpoint, and breaks ``init_weights``.
    Call this AFTER load and BEFORE fully_shard so FSDP shards the packed
    bytes. ``mode`` is ``nf4`` (torchao QLoRA codebook, titan customer
    option) or ``mxfp4`` (K3's native FP4 format). Idempotent; returns the
    number of bases packed (wrapped linears + grouped experts when
    ``experts``). Non-alignable dims stay bf16.

    Scope note: this packs every LoRA base, which is BROADER than K3's own
    scope (routed experts only -- quant_scope.py). That is deliberate. K3
    quantizes as part of full-param QAT; QLoRA here is our memory-reduction
    path for adapting a frozen base, and which projections are bases at all
    is already the caller's choice via ``apply_lora(targets=...)``. For a
    faithful reproduction of K3's quantization use ``apply_mxfp4_qat``, whose
    default scope is the released one.
    """
    if mode not in ("nf4", "mxfp4"):
        raise ValueError(f"Unsupported quantize mode {mode!r}")
    packed = 0
    for module in model.modules():
        if not isinstance(module, KimiLoRALinear):
            continue
        did = (
            module.quantize_base_nf4()
            if mode == "nf4"
            else module.quantize_base_mxfp4()
        )
        packed += int(did)
    if experts:
        packed += (
            quantize_grouped_experts_nf4(model)
            if mode == "nf4"
            else quantize_grouped_experts_mxfp4(model)
        )
    return packed


@torch.no_grad()
def _materialize(t: torch.Tensor) -> torch.Tensor:
    """A full, plain tensor -- ``full_tensor()`` on a DTensor, else unchanged.

    Same idiom :meth:`KimiLoRALinear._dequant_base_mxfp4` already uses on the
    packed base. It matters on both sides of the merge: mixing a materialized
    base with sharded adapters either raises on the add, or -- worse -- produces
    a rank-local shard that then gets written under a full-tensor key, so the
    exported checkpoint silently holds one rank's slice.

    This is a collective, and every rank walks the same modules in the same
    order, so the calls line up. Export runs outside autograd, hence no
    grad_placements.
    """
    return t.full_tensor() if hasattr(t, "full_tensor") else t


# Wrapper segments that appear in ``named_modules()`` paths but NOT in
# ``state_dict()`` keys, because each wrapper installs a hook that strips its own
# prefix. Activation checkpointing, FSDP and torch.compile all do this.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
)


def _state_dict_prefix(mod_name: str, sd: dict) -> str:
    """The state-dict prefix for a module reached at ``mod_name``.

    These two namings differ once anything wraps the module: activation
    checkpointing turns ``layers.0.feed_forward.gate_proj`` into
    ``layers.0._checkpoint_wrapped_module.feed_forward.gate_proj`` in ``named_modules()``,
    while ``state_dict()`` strips it back out. Composing keys from the module path
    then writes a name nothing else recognises AND leaves the adapter keys in place,
    because the pops miss too. Observed as
    ``ValueError: Unmapped tt key: 'layers.0._checkpoint_wrapped_module.feed_forward.gate_proj.weight'``
    from a GRPO weight sync -- the merge had silently produced both a bogus merged
    key and the original LoRA triple.

    An unknown wrapper raises rather than guessing: a wrong name here is a weight
    that never reaches the rollout engine, which is not a failure that announces
    itself.
    """
    stripped = ".".join(p for p in mod_name.split(".") if p not in _WRAPPER_SEGMENTS)
    for candidate in (stripped, mod_name):
        if any(
            f"{candidate}{suffix}" in sd
            for suffix in (".base.weight", ".base_qdata", ".lora_a")
        ):
            return candidate
    raise KeyError(
        f"LoRA module at {mod_name!r} has no matching state_dict entry (tried "
        f"{stripped!r}); an unrecognised module wrapper is in the path, and "
        "merging under a guessed name would ship weights nothing can load"
    )


def merge_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Fold LoRA adapters into base weights and return a plain state dict
    keyed by ORIGINAL param names (no ``.base``/``lora_a``/``lora_b``).

    For each wrapped linear, ``W_merged = W_base + scaling * (B @ A)``.
    This is the deployable/exportable form: feed it straight to
    ``KimiLinearStateDictAdapter.to_hf`` to save a trained LoRA back to
    HF format (the raw adapter drops lora_* keys, so without merge a
    trained LoRA cannot be exported). NF4-quantized bases are
    dequantized to bf16 before merge.
    """
    # Start from the full state dict (includes tied params like a tied
    # lm_head and buffers), then overwrite each LoRA slot with its merged
    # weight and drop the adapter keys.
    sd = dict(model.state_dict())
    for mod_name, module in model.named_modules():
        if not isinstance(module, KimiLoRALinear):
            continue
        # named_modules() and state_dict() disagree once a wrapper is in the path.
        prefix = _state_dict_prefix(mod_name, sd)
        if module._quantize_base == "mxfp4":
            base_w = module._dequant_base_mxfp4()
        elif module._quantize_base == "nf4":
            from torchao.dtypes.nf4tensor import NF4Tensor

            base_w = module.base.weight
            if isinstance(base_w, NF4Tensor):
                base_w = base_w.get_original_weight()
        else:
            base_w = module.base.weight
        base_w = _materialize(base_w)
        out_dtype = base_w.dtype if base_w.dtype != torch.uint8 else torch.bfloat16
        # fp32 delta for deployable precision, cast back to base dtype. Both
        # adapters are materialized first: under TP they are DTensors while the
        # dequantized base is already plain.
        lora_b = _materialize(module.lora_b)
        lora_a = _materialize(module.lora_a)
        delta = module._lora_scaling * (lora_b.float() @ lora_a.float())
        sd[f"{prefix}.weight"] = (base_w.float() + delta).to(out_dtype).contiguous()
        for suffix in (
            ".base.weight",
            ".base.bias",
            ".base_qdata",
            ".base_scale",
            ".lora_a",
            ".lora_b",
        ):
            sd.pop(f"{prefix}{suffix}", None)
    return sd
