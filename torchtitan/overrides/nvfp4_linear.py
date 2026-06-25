# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NVFP4 sequence-parallel attention / feed-forward block path.

This swaps the stock in-layer ``Linear`` projections of the attention and
feed-forward blocks for :class:`NVFP4Linear`, which keeps a bf16 ``weight`` (and
optional ``bias``) and quantizes activations, weights, and gradients to NVFP4.
The quantization is TorchAO's ``torchao.prototype.moe_training.nvfp4_training``.
This module only adapts it to TorchTitan's ``Module`` protocol and the override
mechanism.

Usage:

    torchtitan_train --module llama3 --config llama3_8b \\
        --override.imports torchtitan.overrides.nvfp4_linear

Why a parent-block override (not a child-Linear one)?
The NVFP4 communication optimization all-gathers fp4 codes/scales not bf16
activations in the column-parallel forward. This is an advantage if activations
stay sequence-sharded at the attention / FFN block boundary. Stock TorchTitan
sharding bf16-gathers them to TP Replicate first (parent ``in_dst -> tp=R``),
destroying the opportunity. We override the PARENT ``FeedForward`` and
``GQAttention`` configs (``in_dst = in_src`` keeps the sequence-parallel input)
and replace their child linears with ``NVFP4Linear``. This must be parent-only.
A parent override plus a child ``Linear`` override would be an
ancestor-descendant pair, which TorchTitan's override conflict check rejects.

The TorchAO kernels target NVIDIA Blackwell (sm_100+) and CUDA only. Forward
delegates to TorchAO ops registered as compile-safe, so the override composes
with ``torch.compile``. NVFP4 passes a per-module RHT sign-vector tuple as a
graph constant under ``fullgraph=True``. Those constants plus async DTensor
collectives can exceed Dynamo's default recompile limit before the first step
on large TP runs, so we raise it once at import.

Tensor parallelism -- TorchTitan owns module sharding and boundary
redistribution (weight/bias placement, the parent block's SP input boundary, the
rowwise output contract). NVFP4 owns quantization semantics: the RHT amax + its
TP MAX all-reduce, stochastic rounding, the RHT sign vector, scaled_mm, and the
fp4 code/scale collectives. We reuse torchao's column/row-parallel autograd
Functions, which all-gather fp4 codes and reduce-scatter bf16 partials
internally, wrapped as thin DTensor <-> local bridges. Validation is
TP-degree-dependent and runs in ``NVFP4Linear.parallelize`` where
``parallel_dims`` is known. The override factories only see the config node and
so cannot gate on TP. Validation requires each local GEMM dim is divisible by
128 and sequence parallelism is required under TP.

NOTE (checkpoint compatibility) -- this is a *dual* contract:

* Native NVFP4 checkpoints (the default ``state_dict()`` / DCP path) keep the
  ``_sr_seed`` and ``_rht_sign_vector`` buffers persistent and explicitly
  sharded, so a resumed run restores the saved stochastic-rounding seed and RHT
  basis. Full stochastic-rounding stream replay also depends on TorchAO's
  counter and offset semantics.
* The stock and HF export boundary strips them automatically. The Llama 3 HF
  state-dict adapter only emits keys it knows how to map, so the NVFP4 runtime
  buffers (under ``layers.*``) are dropped and the exported checkpoint contains
  only the stock ``weight`` / ``bias`` keys. Loading a stock checkpoint back
  into NVFP4 is a non-strict load; ``init_states`` redraws the buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard

from torchtitan.config import derive, override
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import (
    spmd_distribute_tensor,
    spmd_layout_to_dtensor_placements,
)
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FusedQKVLinear, GQAttention, QKVLinear
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.protocols.sharding import resolve_placements, ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    _TORCHAO_IMPORT_ERROR: ImportError | None = None
else:
    try:
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
            nvfp4_linear,
        )
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
            nvfp4_col_parallel_linear,
            nvfp4_row_parallel_linear,
        )
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
            _make_rht_sign_vector,
            _rht_sign_vector_to_tuple,
        )
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )

        _TORCHAO_IMPORT_ERROR = None
    except ImportError as e:
        _TORCHAO_IMPORT_ERROR = e

__all__ = ["NVFP4Linear", "nvfp4_attention", "nvfp4_feed_forward"]

TP = MeshAxisName.TP

# torchao's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

# NVFP4 passes per-module Python sign-vector tuples into the compiled block.
# Under fullgraph=True those constants plus async DTensor collectives can exceed
# Dynamo's default limit before step 1 on large TP runs; raise it once at import.
_NVFP4_RECOMPILE_LIMIT = 64
if torch._dynamo.config.recompile_limit < _NVFP4_RECOMPILE_LIMIT:
    torch._dynamo.config.recompile_limit = _NVFP4_RECOMPILE_LIMIT


def _replicated_nvfp4_state_placement() -> SpmdLayout:
    return dense_param_placement(tp=spmd.R)


def _infer_tp_style(
    sharding_config: ShardingConfig | None,
) -> Literal["colwise", "rowwise"] | None:
    """Infer colwise vs rowwise from the weight's declared TP placement.

    Read from the ShardingConfig (not a runtime DTensor): colwise shards the
    weight output dim (Shard(0)), rowwise shards the input/contraction dim
    (Shard(1)).
    """
    if sharding_config is None:
        return None
    weight_layout = sharding_config.state_shardings.get("weight")
    if weight_layout is None:
        return None
    tp_placement = spmd_layout_to_dtensor_placements(weight_layout).get(TP)
    if isinstance(tp_placement, Shard) and tp_placement.dim == 1:
        return "rowwise"
    return "colwise"


def _is_seq_parallel(layout) -> bool:
    """True if the activation layout shards the sequence (dim 1) on the TP axis.

    Activations are [B, L, D]; sequence is dim 1. The feature shard (colwise
    output) is Shard(-1), so a bare Shard check is not enough -- match dim 1.
    """
    if layout is None:
        return False
    tp_placement = spmd_layout_to_dtensor_placements(layout).get(TP)
    return isinstance(tp_placement, Shard) and tp_placement.dim == 1


def _nvfp4_colwise_sp(x_BLD, w_local, bias, sr_seed, sign_vector, tp_group, world_size):
    """Colwise NVFP4 over a local sequence shard, returning a feature shard.

    x_BLD is the local SP activation shard [B, L_local, D]. Transpose the
    sequence to dim 0 before flattening so torchao's all-gather over the TP group
    (gather dim 0) reconstructs the full sequence in order -- rank-major
    concatenation equals sequence-global ordering -- even for batch > 1. torchao
    all-gathers fp4 codes/scales (not bf16) and returns the full-sequence,
    feature-sharded output [m, n_local].
    """
    B, L_local, D = x_BLD.shape
    x_2d = x_BLD.transpose(0, 1).reshape(L_local * B, D)
    if x_2d.shape[0] % _NVFP4_BLOCK:
        raise ValueError(
            f"NVFP4 colwise TP requires flattened local M divisible by "
            f"{_NVFP4_BLOCK}; got B={B}, L_local={L_local}, M={x_2d.shape[0]}."
        )
    out_2d = nvfp4_col_parallel_linear(
        x_2d,
        w_local,
        bias,
        sr_seed=sr_seed,
        tp_group=tp_group,
        world_size=world_size,
        sign_vector=sign_vector,
    )
    n_local = out_2d.shape[-1]
    l_full = out_2d.shape[0] // B
    return out_2d.reshape(l_full, B, n_local).transpose(0, 1)


def _nvfp4_rowwise_sp(x_BLD, w_local, bias, sr_seed, sign_vector, tp_group, world_size):
    """Rowwise NVFP4 over a full-sequence feature shard, returning a seq shard.

    x_BLD is the full-sequence, feature-sharded activation [B, L, K_local].
    torchao computes the local partial outer product and reduce-scatters the bf16
    result along the sequence dim internally, returning the SP seq shard
    [m/w, n]. Sequence-first flatten makes the reduce-scatter split on dim 0
    yield clean per-rank sequence shards even for batch > 1. When the inherited
    output contract is the SP seq shard, the override declares out_dst=None so
    TorchTitan does not reduce again; a non-SP contract keeps its out_dst and
    TorchTitan redistributes this SP shard to it (see _to_nvfp4_rowwise).
    """
    B, L, K_local = x_BLD.shape
    x_2d = x_BLD.transpose(0, 1).reshape(L * B, K_local)
    if x_2d.shape[0] % (world_size * _NVFP4_BLOCK):
        local_m = x_2d.shape[0] // world_size
        raise ValueError(
            f"NVFP4 rowwise TP requires flattened M divisible by "
            f"TP * {_NVFP4_BLOCK} so the reduce-scattered local M stays "
            f"{_NVFP4_BLOCK}-aligned; got B={B}, L={L}, M={x_2d.shape[0]}, "
            f"TP={world_size}, reduce-scattered local M={local_m}."
        )
    out_2d = nvfp4_row_parallel_linear(
        x_2d,
        w_local,
        bias,
        sr_seed=sr_seed,
        tp_group=tp_group,
        world_size=world_size,
        sign_vector=sign_vector,
    )
    n = out_2d.shape[-1]
    l_local = out_2d.shape[0] // B
    return out_2d.reshape(l_local, B, n).transpose(0, 1)


def _swap_tp_placement(dtensor: DTensor, tp_placement) -> tuple:
    """Return (mesh, placements) of ``dtensor`` with the TP axis swapped.

    Activations are replicated over data axes (DP/FSDP) and only the TP axis
    carries the linear's sharding. Reusing the input's own mesh keeps the output
    DTensor on the same mesh (1-D under tp-only, 2-D under tp+fsdp) so it composes
    with the surrounding activations.
    """
    mesh = dtensor.device_mesh
    names = mesh.mesh_dim_names
    assert (
        names is not None and "tp" in names
    ), f"NVFP4 TP path requires a 'tp' mesh axis, got {names}"
    placements = list(dtensor.placements)
    placements[names.index("tp")] = tp_placement
    return mesh, tuple(placements)


class NVFP4Linear(Linear):
    """NVFP4 Linear satisfying TorchTitan's Module protocol.

    Inherits TorchTitan's ``Linear`` (a flat ``nn.Linear`` + ``Module`` leaf), so
    weight/bias are sharded by ``_distribute_states`` from the inherited
    colwise/rowwise ``sharding_config``. NVFP4 runtime buffers start as ``None``
    (skipped by ``_distribute_states``) and are materialized in
    ``_init_self_buffers``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        # colwise -> w1/w3, qkv; rowwise -> w2, wo. Inferred from the weight
        # placement at parallelize() time when None.
        tensor_parallel_style: Literal["colwise", "rowwise"] | None = None
        # Whether the parent block keeps a sequence-parallel input (set by the
        # override factory from the block's in_src). The fp4-all-gather TP path is
        # only correct when activations stay sequence-sharded; parallelize() uses
        # this to fail loudly on TP-without-SP rather than silently miscompute.
        expects_sequence_parallel: bool = False
        # Persistent NVFP4 runtime-buffer placements. These feed directly into
        # sharding_config.state_shardings so checkpoint metadata matches the
        # buffers' actual state contract.
        sr_seed_state_sharding: SpmdLayout = field(
            default_factory=_replicated_nvfp4_state_placement
        )
        rht_sign_vector_state_sharding: SpmdLayout = field(
            default_factory=_replicated_nvfp4_state_placement
        )

    def __init__(self, config: "NVFP4Linear.Config"):
        super().__init__(config)
        self.kernel_preference = KernelPreference.TRITON
        self.tensor_parallel_style = config.tensor_parallel_style
        self.expects_sequence_parallel = config.expects_sequence_parallel
        self.sr_seed_state_sharding = config.sr_seed_state_sharding
        self.rht_sign_vector_state_sharding = config.rht_sign_vector_state_sharding
        self.tp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self._nvfp4_state_specs: dict[str, tuple] = {}
        self._nvfp4_replicate_mesh = None
        self._nvfp4_spmd_backend: Literal[
            "default", "full_dtensor", "spmd_types"
        ] = "default"
        # Persistent for native NVFP4 TorchTitan checkpoint/resume. Start as None
        # so _distribute_states() skips distribution; materialized later in
        # _init_self_buffers().
        self.register_buffer("_sr_seed", None, persistent=True)
        self.register_buffer("_rht_sign_vector", None, persistent=True)
        self._rht_sign_vector_tuple: tuple[int, ...] | None = None

    def _local_sr_seed(self) -> torch.Tensor:
        return (
            self._sr_seed.to_local()
            if isinstance(self._sr_seed, DTensor)
            else self._sr_seed
        )

    def _local_rht_sign_vector(self) -> torch.Tensor:
        sign_vector = (
            self._rht_sign_vector.to_local()
            if isinstance(self._rht_sign_vector, DTensor)
            else self._rht_sign_vector
        )
        if sign_vector is not None and sign_vector.device.type != "meta":
            sign_vector = sign_vector.reshape(-1)
        return sign_vector

    def _refresh_rht_sign_vector_tuple(self) -> None:
        sign_vector = self._local_rht_sign_vector()
        self._rht_sign_vector_tuple = (
            None if sign_vector is None else _rht_sign_vector_to_tuple(sign_vector)
        )

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_rht_sign_vector_tuple()

    @property
    def rht_sign_vector(self) -> tuple[int, ...]:
        if self._rht_sign_vector_tuple is None:
            self._refresh_rht_sign_vector_tuple()
        if self._rht_sign_vector_tuple is None:
            raise RuntimeError("rht_sign_vector is not materialized")
        return self._rht_sign_vector_tuple

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        # Cache the TP group before super().parallelize() distributes states,
        # mirroring Embedding.parallelize(). The NVFP4 TP path owns the amax
        # all-reduce and fp4 collectives over this group.
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
            self.world_size = tp_mesh.size()
            if self.tensor_parallel_style is None:
                self.tensor_parallel_style = _infer_tp_style(self._sharding_config)
        self._validate(parallel_dims)
        self._cache_nvfp4_state_specs(parallel_dims)
        super().parallelize(parallel_dims)

    def _validate(self, parallel_dims: ParallelDims) -> None:
        # TP-degree-dependent contracts the override factory could not check (it
        # only sees the config node, not parallel_dims).
        if parallel_dims.tp_enabled and parallel_dims.spmd_backend == "spmd_types":
            raise ValueError(
                "NVFP4 + TP does not support parallelism.spmd_backend='spmd_types' "
                "yet. The NVFP4 TP path currently wraps outputs as DTensors using "
                "the input DTensor placements, while spmd_types activations are "
                "plain tensors with SPMD annotations. Use the default or "
                "full_dtensor backend for NVFP4 + TP."
            )
        tp = self.world_size
        if self.tensor_parallel_style == "rowwise":
            if self.in_features % tp:
                raise ValueError(
                    f"NVFP4 rowwise TP requires in_features divisible by TP before "
                    f"checking the local {_NVFP4_BLOCK}-block contract; got "
                    f"in_features={self.in_features}, TP={tp}."
                )
            local_in, local_out = self.in_features // tp, self.out_features
        else:
            if self.out_features % tp:
                raise ValueError(
                    f"NVFP4 colwise TP requires out_features divisible by TP before "
                    f"checking the local {_NVFP4_BLOCK}-block contract; got "
                    f"out_features={self.out_features}, TP={tp}."
                )
            local_in, local_out = self.in_features, self.out_features // tp
        if local_in % _NVFP4_BLOCK or local_out % _NVFP4_BLOCK:
            raise ValueError(
                f"NVFP4 requires each local GEMM dim divisible by {_NVFP4_BLOCK}; "
                f"{self.tensor_parallel_style or 'no-TP'} linear "
                f"in={self.in_features} out={self.out_features} under TP={tp} gives "
                f"local (in={local_in}, out={local_out}). Disable the NVFP4 "
                "override for this module or pick TP/feature dims that divide 128."
            )
        if parallel_dims.tp_enabled and not self.expects_sequence_parallel:
            raise ValueError(
                "NVFP4 + TP requires sequence parallelism (the block activation "
                "must stay sequence-sharded so the TP collectives move fp4 codes, "
                "not bf16). Set parallelism.enable_sequence_parallel=True."
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None:
            # Single-GPU or FSDP-only: no TP collectives. FSDP all-gathers the
            # full weight before forward, so a local tensor is expected here.
            weight = (
                self.weight.to_local()
                if isinstance(self.weight, DTensor)
                else self.weight
            )
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            return nvfp4_linear(
                input,
                weight,
                bias,
                kernel_preference=self.kernel_preference,
                sr_seed=self._local_sr_seed(),
                sign_vector=self.rht_sign_vector,
            )

        # The TP path always receives a DTensor activation: _validate rejects
        # spmd_types+TP, and the default/full_dtensor backends produce DTensor
        # activations. Pin that invariant here -- the output wrap below reads
        # input.device_mesh/placements and would otherwise fail opaquely.
        assert isinstance(
            input, DTensor
        ), f"NVFP4 TP forward requires a DTensor activation; got {type(input).__name__}."

        # weight/bias are plain Shard (colwise Shard(0), rowwise Shard(1)); torchao
        # returns their gradients in the same shard, so to_local()'s default grad
        # placement is already correct.
        w_local = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        bias_local = (
            self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        )
        # torchao's Functions return dx already in the input's placement (colwise
        # reduce-scatters dx to the seq shard; rowwise dx is a full-seq feature
        # shard). Pass grad_placements explicitly (per distributed rules) to pin the
        # backward to that contract instead of relying on to_local()'s default.
        x_local = input.to_local(grad_placements=input.placements)

        if self.tensor_parallel_style == "rowwise":
            out = _nvfp4_rowwise_sp(
                x_local,
                w_local,
                bias_local,
                self._local_sr_seed(),
                self.rht_sign_vector,
                self.tp_group,
                self.world_size,
            )
            out_tp_placement = Shard(1)  # SP: sequence shard
        else:
            out = _nvfp4_colwise_sp(
                x_local,
                w_local,
                bias_local,
                self._local_sr_seed(),
                self.rht_sign_vector,
                self.tp_group,
                self.world_size,
            )
            out_tp_placement = Shard(-1)  # feature shard

        mesh, placements = _swap_tp_placement(input, out_tp_placement)
        return DTensor.from_local(out, mesh, placements, run_check=False)

    def _cache_nvfp4_state_specs(self, parallel_dims: ParallelDims) -> None:
        self._nvfp4_state_specs = {}
        self._nvfp4_spmd_backend = parallel_dims.spmd_backend
        if parallel_dims.spmd_backend == "default":
            dense_axes = ["dp_replicate", "fsdp", "tp"]
        elif parallel_dims.spmd_backend == "full_dtensor":
            dense_axes = ["dp_replicate", "dp_shard", "cp", "tp"]
        else:
            dense_axes = ["dp", "cp", "tp"]
        self._nvfp4_replicate_mesh = parallel_dims.get_activated_mesh(dense_axes)

        if self._sharding_config is None:
            return
        for name in ("_sr_seed", "_rht_sign_vector"):
            layout = self._sharding_config.state_shardings.get(name)
            if layout is None:
                continue
            mesh = parallel_dims.resolve_mesh(layout.axes())
            if mesh is None:
                continue
            self._nvfp4_state_specs[name] = (
                mesh,
                resolve_placements(layout, mesh),
                layout,
            )

    @staticmethod
    def _num_state_shards(mesh, placements: tuple) -> int:
        shards = 1
        for axis, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shards *= mesh.size(axis)
        return shards

    def _state_buffer_shape(self, name: str) -> tuple[int, ...]:
        spec = self._nvfp4_state_specs.get(name)
        shards = 1 if spec is None else self._num_state_shards(spec[0], spec[1])
        if name == "_sr_seed":
            return (shards,)
        if shards == 1:
            return (16,)
        return (shards, 16)

    def _replicate_initial_state(self, tensor: torch.Tensor) -> torch.Tensor:
        mesh = self._nvfp4_replicate_mesh
        if mesh is None:
            return tensor
        replicated = distribute_tensor(
            tensor, mesh, [Replicate() for _ in range(mesh.ndim)]
        )
        return replicated.to_local()

    def _materialize_state_buffer(
        self, name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        tensor = self._replicate_initial_state(tensor)
        spec = self._nvfp4_state_specs.get(name)
        if spec is None:
            return tensor
        mesh, placements, layout = spec
        if self._nvfp4_spmd_backend == "spmd_types":
            return spmd_distribute_tensor(tensor, mesh, layout)
        return distribute_tensor(tensor, mesh, list(placements))

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # Materialize NVFP4 runtime buffers after parallelize() + to_empty().
        dev = buffer_device or self.weight.device
        self._sr_seed = self._materialize_state_buffer(
            "_sr_seed",
            torch.randint(
                -(2**63),
                2**63 - 1,
                self._state_buffer_shape("_sr_seed"),
                dtype=torch.int64,
                device=dev,
            ),
        )
        rht_shape = self._state_buffer_shape("_rht_sign_vector")
        if len(rht_shape) == 1:
            rht_sign_vector = _make_rht_sign_vector(None, device=dev)
        else:
            rht_sign_vector = torch.stack(
                [_make_rht_sign_vector(None, device=dev) for _ in range(rht_shape[0])]
            )
        self._rht_sign_vector = self._materialize_state_buffer(
            "_rht_sign_vector", rht_sign_vector
        )
        self._refresh_rht_sign_vector_tuple()


def _with_nvfp4_buffers(state: dict, cfg: NVFP4Linear.Config) -> dict:
    # _distribute_states() requires a placement entry for every direct buffer
    # name. The placements live on NVFP4Linear.Config so the buffer checkpoint
    # contract is explicit at the same level as the rest of the replacement
    # module's state.
    s = dict(state)
    s["_sr_seed"] = cfg.sr_seed_state_sharding
    s["_rht_sign_vector"] = cfg.rht_sign_vector_state_sharding
    return s


def _to_nvfp4_colwise(
    cfg: Linear.Config, *, sequence_parallel: bool
) -> NVFP4Linear.Config:
    base = cfg.sharding_config
    nvfp4_cfg = derive(
        cfg,
        NVFP4Linear.Config,
        tensor_parallel_style="colwise",
        expects_sequence_parallel=sequence_parallel,
    )
    sc = ShardingConfig(
        state_shardings=_with_nvfp4_buffers(base.state_shardings, nvfp4_cfg),
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_dst_shardings,
        out_src_shardings=base.out_src_shardings,
        out_dst_shardings=base.out_dst_shardings,
        local_map=base.local_map,
    )
    return derive(
        nvfp4_cfg,
        NVFP4Linear.Config,
        sharding_config=sc,
    )


def _to_nvfp4_rowwise(
    cfg: Linear.Config, *, sequence_parallel: bool
) -> NVFP4Linear.Config:
    base = cfg.sharding_config
    nvfp4_cfg = derive(
        cfg,
        NVFP4Linear.Config,
        tensor_parallel_style="rowwise",
        expects_sequence_parallel=sequence_parallel,
    )
    # torchao's row-parallel Function reduce-scatters the bf16 output internally
    # and returns the SP sequence shard, so out_src is ALWAYS that SP placement.
    # If the inherited contract already targets the SP seq shard (Shard(seq) -- the
    # dense output_sp=True case), the reduce-scatter is the final placement:
    # out_dst=None so TorchTitan does NOT reduce again (no double-reduce).
    # Otherwise honor the inherited out_dst so TorchTitan redistributes the SP
    # shard to the declared placement (e.g. all-gather -> Replicate for
    # output_sp=False). This composes with the torchao backward, which always
    # expects an SP-shard grad: TorchTitan's SP->out_dst redistribute is
    # differentiable and reduce-scatters the grad back to the SP shard.
    # (If such a non-SP output ever needs the reduction FUSED rather than adapted,
    # the contract-pure fix is a deferred-reduction nvfp4_row_parallel_mm variant
    # -- forward returns the Partial, backward skips the dy all-gather -- in
    # torchao or here; out of scope, no in-tree model produces that contract.)
    out_dst = (
        None if _is_seq_parallel(base.out_dst_shardings) else base.out_dst_shardings
    )
    sc = ShardingConfig(
        state_shardings=_with_nvfp4_buffers(base.state_shardings, nvfp4_cfg),
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_dst_shardings,
        out_src_shardings=dense_sequence_parallel_placement(),
        out_dst_shardings=out_dst,
        local_map=base.local_map,
    )
    return derive(
        nvfp4_cfg,
        NVFP4Linear.Config,
        sharding_config=sc,
    )


def _keep_sp_input(base: ShardingConfig | None) -> ShardingConfig | None:
    """Rewrite a parent block config to keep its SP input (no bf16 gather)."""
    if base is None:
        return None
    return ShardingConfig(
        state_shardings=base.state_shardings,
        in_src_shardings=base.in_src_shardings,
        in_dst_shardings=base.in_src_shardings,  # keep SP; was tp=R (bf16 gather)
        out_src_shardings=base.out_src_shardings,
        out_dst_shardings=base.out_dst_shardings,
        local_map=base.local_map,
    )


def _block_input_is_sp(base: ShardingConfig | None, arg_name: str) -> bool:
    in_src = base.in_src_shardings if base is not None else None
    return _is_seq_parallel(in_src.get(arg_name)) if in_src is not None else False


def _require_torchao() -> None:
    if _TORCHAO_IMPORT_ERROR is not None:
        raise ImportError(
            "nvfp4 override was requested but torchao's NVFP4 training prototype "
            "is not importable; install a torchao build that provides "
            "torchao.prototype.moe_training.nvfp4_training."
        ) from _TORCHAO_IMPORT_ERROR


def _require_sharding_config(base: ShardingConfig | None) -> None:
    # The colwise/rowwise rewrites read base.state_shardings / out_dst_shardings
    # directly. TorchTitan populates sharding_config in update_from_config, which
    # the Trainer runs before applying overrides; fail cleanly (not with an opaque
    # AttributeError) when the override is applied off that path.
    if base is None:
        raise ValueError(
            "nvfp4 override requires the block's sharding_config to be populated. "
            "TorchTitan sets it in update_from_config, which the Trainer runs "
            "before applying overrides; apply this override from the Trainer path "
            "(or after update_from_config), not to a bare config tree."
        )


@override(
    "nvfp4_feed_forward",
    target=FeedForward.Config,
    fqns=["layers.*.feed_forward", "*.layers.*.feed_forward"],
    description="NVFP4 sequence-parallel FFN block (fp4 all-gather, no bf16 gather).",
)
def nvfp4_feed_forward(cfg: FeedForward.Config) -> FeedForward.Config:
    _require_torchao()
    _require_sharding_config(cfg.sharding_config)
    sp = _block_input_is_sp(cfg.sharding_config, "x")
    return derive(
        cfg,
        FeedForward.Config,
        w1=_to_nvfp4_colwise(cfg.w1, sequence_parallel=sp),
        w3=_to_nvfp4_colwise(cfg.w3, sequence_parallel=sp),
        w2=_to_nvfp4_rowwise(cfg.w2, sequence_parallel=sp),
        sharding_config=_keep_sp_input(cfg.sharding_config),
    )


def _qkv_linears(qkv) -> list[Linear.Config] | None:
    if isinstance(qkv, QKVLinear.Config):
        return [qkv.wq, qkv.wkv]
    if isinstance(qkv, FusedQKVLinear.Config):
        return [qkv.wqkv]
    return None


@override(
    "nvfp4_attention",
    target=GQAttention.Config,
    description="NVFP4 sequence-parallel attention block (fp4 all-gather).",
)
def nvfp4_attention(cfg: GQAttention.Config) -> GQAttention.Config:
    _require_torchao()
    if _qkv_linears(cfg.qkv_linear) is None:
        logger.warning(
            "nvfp4 override skipped an attention block: qkv_linear type %s is not "
            "QKVLinear/FusedQKVLinear, so its projections stay bf16.",
            type(cfg.qkv_linear).__name__,
        )
        return cfg
    _require_sharding_config(cfg.sharding_config)
    sp = _block_input_is_sp(cfg.sharding_config, "x_BLD")

    qkv = cfg.qkv_linear
    if isinstance(qkv, QKVLinear.Config):
        new_qkv = derive(
            qkv,
            QKVLinear.Config,
            wq=_to_nvfp4_colwise(qkv.wq, sequence_parallel=sp),
            wkv=_to_nvfp4_colwise(qkv.wkv, sequence_parallel=sp),
        )
    else:
        new_qkv = derive(
            qkv,
            FusedQKVLinear.Config,
            wqkv=_to_nvfp4_colwise(qkv.wqkv, sequence_parallel=sp),
        )

    return derive(
        cfg,
        GQAttention.Config,
        qkv_linear=new_qkv,
        wo=_to_nvfp4_rowwise(cfg.wo, sequence_parallel=sp),
        sharding_config=_keep_sp_input(cfg.sharding_config),
    )
