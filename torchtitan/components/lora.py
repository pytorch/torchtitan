# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, fields

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def _lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """Derive LoRA adapter sharding from the base linear's TP sharding.

    For colwise base linears, ``lora_a`` is TP-replicated and ``lora_b``
    mirrors the base output-dim shard.

    For rowwise base linears, ``lora_a`` mirrors the base input-dim shard and
    ``lora_b`` is TP-replicated, producing the same partial-output shape as the
    base linear.
    """
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None, None

    replicated_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    if base_weight_sharding == dense_param_placement(tp=spmd.R):
        # TP-invariant base (e.g. a compression that is rank-sized, not
        # head-sized): the adapters replicate with it.
        return replicated_weight, replicated_weight
    if base_weight_sharding == dense_param_placement(tp=spmd.S(0)):
        lora_b_sharding = ShardingConfig(
            state_shardings={"weight": base_weight_sharding},
        )
        return replicated_weight, lora_b_sharding
    else:
        assert base_weight_sharding == dense_param_placement(tp=spmd.S(1))
        lora_a_sharding = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        )
        return lora_a_sharding, replicated_weight


class LoRALinearBase:
    """Marker base of every dynamically created LoRA linear class.

    ``_get_lora_cls`` builds one subclass per parent linear class, so there is
    no single concrete class to isinstance against; this empty base is baked
    into each of them and is the stable way to find LoRA modules on a built
    model (``merge_lora_state_dict`` walks it).
    """


_lora_class_cache: dict[type, type] = {}
_frozen_config_class_cache: dict[type, type] = {}


def _get_lora_cls(parent_cls: type) -> type:
    """Get or create a LoRA subclass for *parent_cls* (e.g. Linear, Float8Linear).

    The returned class has a proper ``Config`` that extends the parent's Config
    with ``rank`` and ``alpha``.  Adapters are built in ``__init__`` from the
    base config's dimensions and sharding.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]

    class LoRALinear(parent_cls, LoRALinearBase):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float
            quantize_base: str | None = None

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            for param in nn.Module.parameters(self):
                param.requires_grad_(False)
            self._lora_scaling = config.alpha / config.rank
            if config.quantize_base not in (None, "nf4", "mxfp4"):
                raise ValueError(
                    f"quantize_base must be None, 'nf4' or 'mxfp4', got "
                    f"{config.quantize_base!r}"
                )
            self._quantize_base_requested = config.quantize_base
            lora_a_sharding, lora_b_sharding = _lora_adapter_sharding(
                config.sharding_config
            )
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=config.rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={
                    "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
                },
            ).build()
            self.lora_b = Linear.Config(
                in_features=config.rank,
                out_features=config.out_features,
                bias=False,
                sharding_config=lora_b_sharding,
                param_init={"weight": nn.init.zeros_},
            ).build()
            if config.quantize_base == "mxfp4":
                # BUILD-time swap, before any parallelize: FSDP2 then shards
                # the packed bytes natively (pack-then-shard), which is the
                # order the nf4 path cannot reach. Works on meta too -- the
                # packed LAYOUT registers now, values arrive at init or load.
                # AFTER the adapter derivation above: the swap rewrites the
                # sharding_config's weight entry for the packed pair.
                self._config_sharding = config.sharding_config
                self._swap_in_packed_mxfp4_layout()

        _quantize_base: str | None = None

        def _swap_in_packed_mxfp4_layout(self) -> None:
            """Replace the base weight with MXFP4 split storage (torchao MX,
            block 32): ``base_qdata`` [out, in/2] uint8 and ``base_scale``
            [out, in/32] e8m0-bytes-as-uint8, both plain contiguous frozen
            params. MXTensor itself cannot be a param -- its packed qdata
            makes the logical view non-contiguous and FSDP2 rejects it -- so
            the tensor is reconstructed via ``__tensor_unflatten__`` at use.
            The scale is stored viewed as uint8 because FSDP2's all-gather
            has no float8_e8m0fnu copy kernel. block_size 32 needs
            in_features % 32 == 0; other dims stay bf16.
            """
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            w = self._parameters.get("weight")
            if w is None or w.shape[-1] % 32 != 0:
                self._quantize_base_requested = None
                return
            # The flatten ctx carries no shape or data, so a dummy reproduces
            # it exactly; this also works while w is still on meta.
            dummy = MXTensor.to_mx(
                torch.zeros(1, 32, dtype=torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
            )
            _, self._mx_ctx = dummy.__tensor_flatten__()
            self._mx_scale_dtype = dummy.scale.dtype
            out_f, in_f = w.shape
            if w.is_meta:
                qdata = torch.empty(out_f, in_f // 2, dtype=torch.uint8, device="meta")
                scale = torch.empty(out_f, in_f // 32, dtype=torch.uint8, device="meta")
            else:
                mx = MXTensor.to_mx(
                    w.data.to(torch.bfloat16),
                    elem_dtype=torch.float4_e2m1fn_x2,
                    block_size=32,
                )
                qdata = mx.qdata.contiguous()
                scale = mx.scale.view(torch.uint8).contiguous()
            self.base_qdata = nn.Parameter(qdata, requires_grad=False)
            self.base_scale = nn.Parameter(scale, requires_grad=False)
            del self._parameters["weight"]
            sharding = getattr(self, "_config_sharding", None)
            base_weight_sharding = (
                sharding.state_shardings.get("weight") if sharding else None
            )
            self._packed_tp_style = None
            if base_weight_sharding is not None:
                # The declarative system requires a placement for every param
                # once a sharding_config exists. The packed pair mirrors the
                # base weight's TP layout: colwise (S(0)) shards packed ROWS,
                # exact because MX block-32 is row-blockwise; rowwise (S(1))
                # shards packed columns, exact only when the local in-features
                # stay block-divisible -- checked at first dequant, when the
                # actual shard is known. Replicate replicates.
                if base_weight_sharding == dense_param_placement(tp=spmd.S(0)):
                    pair = dense_param_placement(tp=spmd.S(0))
                    self._packed_tp_style = "colwise"
                elif base_weight_sharding == dense_param_placement(tp=spmd.S(1)):
                    pair = dense_param_placement(tp=spmd.S(1))
                    self._packed_tp_style = "rowwise"
                else:
                    pair = dense_param_placement(tp=spmd.R)
                sharding.state_shardings["base_qdata"] = pair
                sharding.state_shardings["base_scale"] = pair
                del sharding.state_shardings["weight"]
            self._quantize_base = "mxfp4"

        def _init_packed_mxfp4_values(self) -> None:
            """Materialize from-scratch values for the packed base.

            MX block-32 quantization is row-blockwise, so it commutes with
            FSDP2's Shard(0) row sharding: each rank draws ITS rows in bf16
            with the parent's weight init and quantizes them locally.
            """
            from torch.distributed.tensor import DTensor
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            init_fn = self._mx_weight_init or (
                lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            )
            qdata, scale = self.base_qdata, self.base_scale
            q_local = qdata.to_local() if isinstance(qdata, DTensor) else qdata
            s_local = scale.to_local() if isinstance(scale, DTensor) else scale
            rows = q_local.shape[0]
            w_rows = torch.empty(
                rows,
                q_local.shape[1] * 2,
                dtype=torch.bfloat16,
                device=q_local.device,
            )
            init_fn(w_rows)
            mx = MXTensor.to_mx(
                w_rows, elem_dtype=torch.float4_e2m1fn_x2, block_size=32
            )
            with torch.no_grad():
                q_local.copy_(mx.qdata)
                s_local.copy_(mx.scale.view(torch.uint8))

        def _dequant_base_mxfp4(self) -> torch.Tensor:
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            qdata, scale = self.base_qdata, self.base_scale
            if getattr(self, "_packed_tp_style", None) is not None:
                # TP-sharded packed base: dequantize this rank's LOCAL shard;
                # the packed-TP forward does the local matmul + collective.
                qdata = qdata.to_local() if hasattr(qdata, "to_local") else qdata
                scale = scale.to_local() if hasattr(scale, "to_local") else scale
                if scale.shape[-1] * 32 != qdata.shape[-1] * 2:
                    raise ValueError(
                        "packed-MXFP4 rowwise TP shard is not MX-block "
                        f"aligned: local qdata {tuple(qdata.shape)} vs scale "
                        f"{tuple(scale.shape)}. in_features per TP rank must "
                        "be a multiple of 32."
                    )
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

        def init_states(self, **kwargs) -> None:
            if self._quantize_base == "mxfp4":
                # Config.build installs _param_init AFTER __init__, so the
                # swap could not rewrite it there. The parent's per-param
                # dict raises on names it does not know: hand it placeholder
                # zeros for the packed pair (real values land just below)
                # and keep the weight's init fn for that draw. New dict --
                # the config's may be shared.
                self._mx_weight_init = (self._param_init or {}).get("weight")
                if self._param_init is not None:
                    self._param_init = {
                        k: v for k, v in self._param_init.items() if k != "weight"
                    }
                    self._param_init["base_qdata"] = nn.init.zeros_
                    self._param_init["base_scale"] = nn.init.zeros_
            super().init_states(**kwargs)
            if self._quantize_base == "mxfp4":
                self._init_packed_mxfp4_values()
                return
            if self._quantize_base_requested is None:
                return
            from torch.distributed.tensor import DTensor

            if isinstance(self.weight, DTensor):
                # A DTensor base here means the module is FSDP/TP-managed, and
                # FSDP2's lazy_init re-reads the sharded param it registered
                # (reset_sharded_param): swapping in packed bytes after the
                # fact breaks it even on a 1-rank mesh, in two tested ways
                # (plain NF4 param: no _local_tensor; NF4 inside the DTensor
                # shell: invalid storage). NF4 under FSDP needs the
                # pack-then-shard order via split-storage params, which the
                # MXFP4 port will bring; refusing beats a broken lazy_init.
                raise NotImplementedError(
                    "quantize_base='nf4' does not support FSDP/TP-managed "
                    "base weights yet: build without parallelize (library "
                    "use), or drop quantize_base."
                )
            self.quantize_base_nf4()

        @torch.no_grad()
        def quantize_base_nf4(self) -> bool:
            """Pack the frozen base to NF4 (torchao). Idempotent.

            QLoRA is lossy by design: it trades exactness for a ~4x cut in
            memory and, on comms-bound fabrics, in FSDP all-gather traffic.
            Call AFTER weights load (quantizing at build packs init noise)
            and BEFORE fully_shard, so FSDP shards the packed bytes.

            torchao NF4 double-quant requires numel divisible by
            block_size(64) * scaler_block_size(256) = 16384; dims that do
            not divide stay bf16 (returns False).
            """
            from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

            if isinstance(self.weight, NF4Tensor):
                self._quantize_base = "nf4"
                return True
            if self.weight.numel() % 16384 != 0:
                return False
            self.weight = nn.Parameter(
                to_nf4(self.weight.data.to(torch.bfloat16)),
                requires_grad=False,
            )
            self._quantize_base = "nf4"
            return True

        def _forward_packed_tp(self, x: torch.Tensor) -> torch.Tensor:
            """TP forward for the packed base: local dequant + local matmul,
            DTensor only at the boundary.

            Colwise: x replicated; each rank computes its out/tp columns;
            returns DTensor(Shard(-1)). Rowwise: x is the in/tp shard; local
            partial matmul, ONE reduction over tp for base+adapters combined
            (linearity: the sum commutes), returns DTensor redistributed to
            Replicate.

            Backward: explicit grad_placements make the tp reductions happen
            -- operands used identically by all ranks (colwise x and lora_a,
            rowwise lora_b) carry Partial gradients that must all-reduce; a
            bare to_local() would silently skip it.
            """
            import torch.nn.functional as F
            from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

            colwise = self._packed_tp_style == "colwise"
            qdata = self.base_qdata
            assert isinstance(qdata, DTensor)
            tp_mesh = qdata.device_mesh

            if isinstance(x, DTensor):
                if not colwise and all(p.is_replicate() for p in x.placements):
                    # A rowwise input can arrive replicated (an attention
                    # output all-gathered before the projection); the
                    # unquantized path never notices because the base linear
                    # redistributes for itself. Here the weight dequantizes
                    # into a local tensor, so the shapes must agree first.
                    x = x.redistribute(tp_mesh, [Shard(x.dim() - 1)])
                grad_pl = (Partial(),) if colwise else None
                x_loc = x.to_local(grad_placements=grad_pl)
            else:
                x_loc = x

            w_loc = self._dequant_base_mxfp4().to(x_loc.dtype)
            la = self.lora_a.weight
            lb = self.lora_b.weight
            if colwise:
                # lora_a Replicate (grads sum over tp), lora_b Shard(0) local.
                if isinstance(la, DTensor):
                    la = la.to_local(grad_placements=(Partial(),))
                if isinstance(lb, DTensor):
                    lb = lb.to_local()
            else:
                # lora_a Shard(1) local, lora_b Replicate (grads sum over tp).
                if isinstance(la, DTensor):
                    la = la.to_local()
                if isinstance(lb, DTensor):
                    lb = lb.to_local(grad_placements=(Partial(),))
            la = la.to(x_loc.dtype)
            lb = lb.to(x_loc.dtype)

            out_loc = F.linear(x_loc, w_loc) + self._lora_scaling * F.linear(
                F.linear(x_loc, la), lb
            )
            bias = getattr(self, "bias", None)
            if colwise:
                # Colwise shards the OUTPUT features: this rank's bias slice
                # matches its output slice and adds locally.
                if bias is not None:
                    b = bias.to_local() if isinstance(bias, DTensor) else bias
                    out_loc = out_loc + b.to(out_loc.dtype)
                return DTensor.from_local(
                    out_loc, tp_mesh, [Shard(out_loc.dim() - 1)], run_check=False
                )
            # Rowwise: local outputs are partial sums over the in/tp shards.
            # The declared contract expects Partial(sum) out -- the module
            # boundary owns the reduction. The bias therefore adds LOCALLY as
            # bias/tp (the same convention as common Linear's forward), so
            # the reduction sums it back to exactly one bias.
            if bias is not None:
                b = bias.to_local() if isinstance(bias, DTensor) else bias
                out_loc = out_loc + b.to(out_loc.dtype) / tp_mesh.size()
            return DTensor.from_local(out_loc, tp_mesh, [Partial()], run_check=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if (
                self._quantize_base == "mxfp4"
                and getattr(self, "_packed_tp_style", None) is not None
                and isinstance(self.base_qdata, torch.Tensor)
                and hasattr(self.base_qdata, "device_mesh")
            ):
                return self._forward_packed_tp(input)
            if self._quantize_base == "mxfp4":
                from torch.distributed.tensor import DTensor, Replicate

                # No weight-only MXFP4 linear in torchao yet: dequantize,
                # then matmul. The memory and FSDP all-gather win from the
                # packed base still holds -- the dense weight is transient.
                w = self._dequant_base_mxfp4().to(input.dtype)
                if isinstance(input, DTensor):
                    mesh = input.device_mesh
                    w = DTensor.from_local(
                        w, mesh, [Replicate()] * mesh.ndim, run_check=False
                    )
                base_out = torch.nn.functional.linear(input, w)
                if getattr(self, "bias", None) is not None:
                    base_out = base_out + self.bias
            elif self._quantize_base == "nf4":
                from torchao.dtypes.nf4tensor import linear_nf4

                # linear_nf4 takes the weight only; a bias is added here,
                # exactly as the unquantized branch's F.linear would.
                base_out = linear_nf4(input, self.weight)
                if getattr(self, "bias", None) is not None:
                    base_out = base_out + self.bias
            else:
                base_out = super().forward(input)
            lora_out = self.lora_b(self.lora_a(input))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


_EXPERT_WEIGHT_NAMES = ("w1_EFD", "w2_EDF", "w3_EFD")
_mxfp4_experts_cls_cache: dict[type, type] = {}


class MXFP4ExpertsBase:
    """Marker base of every packed-experts class (see LoRALinearBase)."""


def _get_mxfp4_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP4 split-storage subclass of a grouped-experts class.

    Same build-time pack-then-shard order as the LoRA linear's mxfp4 path: the
    3-D ``[E, A, B]`` weights become ``(E*A, B/2)`` uint8 qdata plus
    ``(E*A, B/32)`` e8m0-as-uint8 scale at __init__, so FSDP2 shards packed
    bytes. MX blocks run along the last dim, so flattening the leading dims is
    exact and every expert boundary falls on a block boundary. The logical
    names become dequant properties -- the forward reads ``self.w1_EFD`` and
    casts to bf16 anyway, so it is unchanged.
    """
    if parent_cls in _mxfp4_experts_cls_cache:
        return _mxfp4_experts_cls_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    def _make_fget(name: str):
        def fget(self):
            from torch.distributed.tensor import DTensor
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            if name not in self._mxfp4_shapes:
                # This weight's last dim did not divide the MX block: it
                # stayed a plain bf16 param, which the class property must
                # hand through.
                return self._parameters[name]
            qdata = self._parameters[name + "_qdata"]
            scale = self._parameters[name + "_scale"]
            if isinstance(qdata, DTensor):
                # Pre-unshard access (outside FSDP's forward window): gather
                # explicitly. During forward FSDP2 exposes plain unsharded
                # tensors.
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

    class MXFP4Experts(parent_cls, MXFP4ExpertsBase):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config) -> None:
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            super().__init__(config)
            dummy = MXTensor.to_mx(
                torch.zeros(1, 32, dtype=torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
            )
            _, self._mx_ctx = dummy.__tensor_flatten__()
            self._mx_scale_dtype = dummy.scale.dtype
            self._mxfp4_shapes: dict[str, tuple[int, ...]] = {}
            for name in _EXPERT_WEIGHT_NAMES:
                p = self._parameters.get(name)
                if p is None or p.shape[-1] % 32 != 0:
                    continue
                self._mxfp4_shapes[name] = tuple(p.shape)
                rows = p.numel() // p.shape[-1]
                cols = p.shape[-1]
                if p.is_meta:
                    qdata = torch.empty(
                        rows, cols // 2, dtype=torch.uint8, device="meta"
                    )
                    scale = torch.empty(
                        rows, cols // 32, dtype=torch.uint8, device="meta"
                    )
                else:
                    mx = MXTensor.to_mx(
                        p.data.reshape(rows, cols).to(torch.bfloat16),
                        elem_dtype=torch.float4_e2m1fn_x2,
                        block_size=32,
                    )
                    qdata = mx.qdata.contiguous()
                    scale = mx.scale.view(torch.uint8).contiguous()
                del self._parameters[name]
                self.register_parameter(
                    name + "_qdata", nn.Parameter(qdata, requires_grad=False)
                )
                self.register_parameter(
                    name + "_scale", nn.Parameter(scale, requires_grad=False)
                )
                sharding = getattr(config, "sharding_config", None)
                entry = (
                    sharding.state_shardings.pop(name, None)
                    if sharding and sharding.state_shardings
                    else None
                )
                if entry is not None:
                    # Translate the declared 3-D placement to the flattened
                    # packed pair. Replicate/invariant carries over; a shard
                    # on the EXPERT dim maps to row-shard (experts are
                    # contiguous row blocks in the (E*A, B) flatten); a shard
                    # on an inner dim does not survive the flatten and needs
                    # the packed expert-TP unit.
                    from torchtitan.distributed.parallel_dims import MeshAxisName

                    tp_type = entry.axis_types.get(MeshAxisName.TP)
                    if isinstance(tp_type, spmd.Shard) and tp_type.dim != 0:
                        raise NotImplementedError(
                            f"quantize_experts='mxfp4': {name} is declared "
                            "TP-sharded on an inner dim, which the packed "
                            "flatten cannot express yet. Run without "
                            "expert-TP, or drop quantize_experts."
                        )
                    # Entries for axes absent from the mesh (e.g. the EP axis
                    # in a TP-only run) are ignored at resolve time and pass
                    # through; packed experts UNDER EP are not validated yet.
                    sharding.state_shardings[name + "_qdata"] = entry
                    sharding.state_shardings[name + "_scale"] = entry

        def init_states(self, **kwargs) -> None:
            # Config.build installs _param_init after __init__; rewrite here,
            # before the parent's dict-driven init would raise on the packed
            # names. Placeholder zeros first, real values drawn just below.
            self._mx_weight_inits = {}
            if self._param_init is not None:
                new_init = dict(self._param_init)
                for name in self._mxfp4_shapes:
                    self._mx_weight_inits[name] = new_init.pop(name, None)
                    new_init[name + "_qdata"] = nn.init.zeros_
                    new_init[name + "_scale"] = nn.init.zeros_
                self._param_init = new_init
            super().init_states(**kwargs)
            self._init_packed_values()

        def _init_packed_values(self) -> None:
            """From-scratch values: MX block-32 is row-blockwise, so each rank
            draws ITS rows in bf16 with the weight's init fn and quantizes
            locally (commutes with Shard(0))."""
            from torch.distributed.tensor import DTensor
            from torchao.prototype.mx_formats.mx_tensor import MXTensor

            for name, shape in self._mxfp4_shapes.items():
                init_fn = self._mx_weight_inits.get(name) or (
                    lambda w: nn.init.trunc_normal_(w, std=0.02)
                )
                qdata = self._parameters[name + "_qdata"]
                scale = self._parameters[name + "_scale"]
                q_local = qdata.to_local() if isinstance(qdata, DTensor) else qdata
                s_local = scale.to_local() if isinstance(scale, DTensor) else scale
                w_rows = torch.empty(
                    q_local.shape[0],
                    q_local.shape[1] * 2,
                    dtype=torch.bfloat16,
                    device=q_local.device,
                )
                init_fn(w_rows)
                mx = MXTensor.to_mx(
                    w_rows, elem_dtype=torch.float4_e2m1fn_x2, block_size=32
                )
                with torch.no_grad():
                    q_local.copy_(mx.qdata)
                    s_local.copy_(mx.scale.view(torch.uint8))

    for name in _EXPERT_WEIGHT_NAMES:
        setattr(MXFP4Experts, name, property(_make_fget(name)))
    MXFP4Experts.__name__ = f"MXFP4{parent_cls.__name__}"
    MXFP4Experts.__qualname__ = f"MXFP4{parent_cls.__name__}"
    _mxfp4_experts_cls_cache[parent_cls] = MXFP4Experts
    return MXFP4Experts


def _get_frozen_config_cls(
    config_cls: type[Module.Config],
) -> type[Module.Config]:
    """Get or create a config subclass that freezes direct build parameters."""
    if config_cls in _frozen_config_class_cache:
        return _frozen_config_class_cache[config_cls]

    class FrozenConfig(config_cls):  # type: ignore[valid-type, misc]
        def build(self, **kwargs):
            instance = config_cls.build(self, **kwargs)
            for param in instance.parameters(recurse=False):
                param.requires_grad_(False)
            return instance

    FrozenConfig.__name__ = f"Frozen{config_cls.__name__}"
    FrozenConfig.__qualname__ = f"Frozen{config_cls.__qualname__}"
    _frozen_config_class_cache[config_cls] = FrozenConfig
    return FrozenConfig


def _make_frozen_config(cfg: Module.Config) -> Module.Config:
    """Create a frozen config that still passes checks for the original type."""
    frozen_cls = _get_frozen_config_cls(type(cfg))
    return frozen_cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init})


class LoRAConverter(ModelConfigConverter):
    """Apply LoRA adapters to Linear layers in a model.

    Operates on the model config tree: target Linear configs are replaced
    with ``LoRALinear.Config`` (which builds a LoRA subclass with frozen base
    and trainable adapters). Non-target modules are replaced with dynamic
    frozen config subclasses that freeze direct parameters at build time.

    When ``target_modules`` is None (default), every ``Linear.Config`` is
    converted.  When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        rank: int = 8
        """Rank of the LoRA matrices."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Module names to apply LoRA to (matched against the last segment of the FQN).
        None means all Linear layers. An empty list means no layers."""

        quantize_experts: str | None = None
        """Pack frozen grouped-expert weights to MXFP4 split storage at build
        (the experts are a MoE model's parameter bulk, so this is where the
        QLoRA memory and FSDP all-gather win mostly lives). Same
        pack-then-shard order as quantize_base='mxfp4'."""

        quantize_base: str | None = None
        """Pack frozen base weights at init ('nf4', via torchao). QLoRA:
        lossy by design, ~4x memory cut on the bases. Library-level for now:
        the packing refuses FSDP/TP-managed bases (FSDP2's lazy_init cannot
        take a post-hoc packed param) until the split-storage port lands, so
        it fits models built without parallelize; adapting a LOADED
        checkpoint uses quantize_lora_bases after the load instead."""

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules is not None else None
        )
        if config.quantize_experts not in (None, "mxfp4"):
            raise ValueError(
                f"quantize_experts must be None or 'mxfp4', got "
                f"{config.quantize_experts!r}"
            )
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def _make_lora_config(self, cfg: Linear.Config):
        """Create a LoRALinear.Config from a base Linear.Config."""
        assert cfg._owner is not None
        lora_cls = _get_lora_cls(cfg._owner)
        return lora_cls.Config(  # pyrefly: ignore [missing-attribute]
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=self.rank,
            alpha=self.alpha,
            quantize_base=self.config.quantize_base,
        )

    def convert(self, model_config: Module.Config) -> Module.Config:
        """Walk the module config tree from leaves to root.

        Target Linear modules get their config replaced with
        ``LoRALinear.Config``. All other module configs become frozen config
        subclasses so LoRA training updates only adapter parameters.
        """
        converted_root = model_config
        matched = set()
        configs = list(model_config.traverse(Module.Config, recurse=True))

        for fqn, cfg, parent, attr in reversed(configs):
            assert isinstance(cfg, Module.Config)
            last_segment = fqn.rsplit(".", 1)[-1]
            is_target = isinstance(cfg, Linear.Config) and (
                self.target_modules is None or last_segment in self.target_modules
            )

            if is_target:
                new_cfg = self._make_lora_config(cfg)
                matched.add(last_segment)
            elif (
                self.config.quantize_experts == "mxfp4"
                and isinstance(cfg, GroupedExperts.Config)
                and cfg.dim % 32 == 0
                and cfg.hidden_dim % 32 == 0
            ):
                # The packed subclass creates its params frozen; no frozen
                # wrap needed on top.
                assert cfg._owner is not None
                experts_cls = _get_mxfp4_experts_cls(cfg._owner)
                new_cfg = experts_cls.Config(
                    **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
                )
            else:
                new_cfg = _make_frozen_config(cfg)

            if parent is None:
                converted_root = new_cfg
            elif isinstance(parent, list):
                assert isinstance(attr, int)
                parent[attr] = new_cfg
            else:
                assert isinstance(attr, str)
                setattr(parent, attr, new_cfg)

        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"Linear.Config in the model config tree."
            )
        return converted_root


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter-only checkpoint payload: every parameter left trainable.

    Under LoRA that is the adapters (plus anything a model deliberately
    unfroze); shipping only these is what makes adapter checkpoints and
    trainer-to-rollout weight syncs light.
    """
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


# Wrapper segments that appear in ``named_modules()`` paths but NOT in
# ``state_dict()`` keys, because each wrapper installs a hook that strips its
# own prefix. Activation checkpointing, FSDP and torch.compile all do this.
_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
)


def _state_dict_prefix(mod_name: str, sd: dict) -> str:
    """The state-dict prefix for a module reached at ``mod_name``.

    The two namings differ once anything wraps the module: activation
    checkpointing turns ``layers.0.feed_forward.w1`` into
    ``layers.0._checkpoint_wrapped_module.feed_forward.w1`` in
    ``named_modules()``, while ``state_dict()`` strips it back out. Composing
    keys from the module path then writes a name nothing else recognises AND
    leaves the adapter keys in place, because the pops miss too.

    An unknown wrapper raises rather than guessing: a wrong name here is a
    weight that never reaches the consumer, which is not a failure that
    announces itself.
    """
    stripped = ".".join(p for p in mod_name.split(".") if p not in _WRAPPER_SEGMENTS)
    for candidate in (stripped, mod_name):
        if f"{candidate}.lora_a.weight" in sd:
            return candidate
    raise KeyError(
        f"LoRA module at {mod_name!r} has no matching state_dict entry (tried "
        f"{stripped!r}); an unrecognised module wrapper is in the path, and "
        "merging under a guessed name would ship weights nothing can load"
    )


def merge_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Fold LoRA adapters into base weights and return a state dict keyed by
    ORIGINAL param names (no ``lora_a``/``lora_b``). The model is unchanged
    when this returns.

    For each LoRA linear, ``W_merged = W_base + (alpha / rank) * (B @ A)``.
    This is the deployable/exportable form: a state-dict adapter can convert
    it exactly as it converts the unadapted model's, whereas the raw state
    dict carries adapter keys nothing downstream recognises.

    The merge happens IN the modules (temporarily, restored from clones on
    the way out) rather than in the returned dict, because a base linear's
    serialization is not necessarily ``<fqn>.weight``: the fused attention
    linear exports split ``wq``/``wk``/``wv`` keys through a state-dict hook,
    and writing composed key names would miss every such hook. Taking
    ``state_dict()`` with the merged weights in place lets each module's own
    serialization produce the right keys.

    Under TP nothing needs materializing: the adapter shardings mirror the
    base (``_lora_adapter_sharding``), so ``B @ A`` composes to the base
    weight's placement and the add dispatches as DTensors.
    """
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinearBase)
    ]
    packed_experts = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MXFP4ExpertsBase)
    ]
    # The merged weight goes in as a NEW Parameter object and the original is
    # re-bound afterwards. Swapping the object (rather than copy_ into the
    # storage) keeps the returned dict from aliasing anything that gets
    # reverted, and never writes THROUGH a quantized tensor -- copy_ into an
    # NF4 base would silently re-quantize the merged value.
    originals = [module._parameters.get("weight") for _, module in lora_modules]
    with torch.no_grad():
        for _, module in lora_modules:
            if module._quantize_base == "mxfp4":
                # No weight param exists on a packed module: the merged
                # weight goes in as a fresh one and the restore removes it.
                base_w = module._dequant_base_mxfp4()
            else:
                base_w = module.weight
            if module._quantize_base == "nf4":
                base_w = base_w.get_original_weight()
            # fp32 delta for deployable precision, cast back to the base dtype.
            delta = module._lora_scaling * (
                module.lora_b.weight.float() @ module.lora_a.weight.float()
            )
            module.weight = nn.Parameter(
                (base_w.float() + delta).to(base_w.dtype).contiguous(),
                requires_grad=False,
            )
        for _, module in packed_experts:
            for wname, shape in module._mxfp4_shapes.items():
                # The property dequantizes; register the dense weight as a
                # temporary param so state_dict emits the ORIGINAL key.
                dense = getattr(module, wname).contiguous()
                # Straight into _parameters: register_parameter refuses the
                # name because the class dequant property answers hasattr.
                module._parameters[wname] = nn.Parameter(dense, requires_grad=False)
    try:
        sd = dict(model.state_dict())
    finally:
        for (_, module), original in zip(lora_modules, originals):
            if original is None:
                del module._parameters["weight"]
            else:
                module.weight = original
        for _, module in packed_experts:
            for wname in module._mxfp4_shapes:
                del module._parameters[wname]
    for mod_name, module in packed_experts:
        stripped = ".".join(
            part for part in mod_name.split(".") if part not in _WRAPPER_SEGMENTS
        )
        for wname in module._mxfp4_shapes:
            for suffix in ("_qdata", "_scale"):
                sd.pop(f"{stripped}.{wname}{suffix}", None)
    for mod_name, _ in lora_modules:
        # named_modules() and state_dict() disagree once a wrapper is in the
        # path.
        prefix = _state_dict_prefix(mod_name, sd)
        for suffix in (
            ".lora_a.weight",
            ".lora_b.weight",
            ".base_qdata",
            ".base_scale",
        ):
            sd.pop(f"{prefix}{suffix}", None)
    return sd


def quantize_lora_bases(model: nn.Module) -> int:
    """Post-load QLoRA hook: pack every LoRA base weight to NF4.

    The trainer's meta-first flow builds, then materializes real weights
    (init or checkpoint), THEN quantizes -- packing at build time would
    quantize init noise, not the loaded checkpoint. Call AFTER load and
    BEFORE fully_shard so FSDP shards the packed bytes. Idempotent; returns
    the number of bases packed. Dims torchao cannot block-quantize stay bf16.
    """
    packed = 0
    for module in model.modules():
        if isinstance(module, LoRALinearBase):
            packed += int(module.quantize_base_nf4())
    return packed
