# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantization converter.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4Linear`, which keeps a bf16
weight and quantizes activations, weights, and gradients to NVFP4 on the fly via
TorchAO's ``nvfp4_training`` kernels (NVIDIA Blackwell / sm_100+, CUDA only).

Like :class:`MXFP8LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Under tensor
parallelism the block boundary keeps its stock bf16 collectives (all-gather /
reduce-scatter); NVFP4 does not move fp4 codes over the wire. NVFP4Linear only
bridges TorchAO's functional op (opaque to DTensor) to torchtitan's DTensor-based
TP: it runs the GEMM on local shards and returns the output with the colwise /
rowwise placement so DTensor performs the (bf16) reduction.
"""

from dataclasses import dataclass, field, replace
from typing import Literal

import spmd_types as spmd
import torch
from torch.distributed.tensor import distribute_tensor, DTensor, Shard

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import spmd_layout_to_dtensor_placements
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import (
    LocalMapConfig,
    resolve_placements,
    ShardingConfig,
    SpmdLayout,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

TP = MeshAxisName.TP

# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

try:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
        nvfp4_linear,
        nvfp4_mm_triton,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        _make_rht_sign_vector,
        _rht_sign_vector_to_tuple,
        NVFP4Linear as TorchAONVFP4Linear,
    )

    # The NVFP4 GEMM is a raw autograd Function that runs on local shards inside
    # the spmd.local_map region (see _augment_sharding_config). Mark it local-safe
    # so SPMD type checking propagates through it; the local_map boundary declares
    # the real colwise/rowwise output and input-gradient types.
    spmd.register_local_autograd_function(nvfp4_mm_triton)

    class NVFP4Linear(TorchAONVFP4Linear, Module):
        """NVFP4 Linear satisfying torchtitan's Module protocol.

        Reuses TorchAO's ``NVFP4Linear`` (weight/bias, the ``_sr_seed`` /
        ``_rht_sign_vector`` runtime buffers, RHT logic, functional forward) and
        adds torchtitan's meta-init buffer protocol and a DTensor <-> local TP
        bridge. Buffers are replicated over every mesh axis (the RHT block is
        size-16 and contraction-agnostic; the seed is per-module).
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds NVFP4Linear."""

            pass

        def __init__(self, config: "NVFP4Linear.Config"):
            TorchAONVFP4Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )
            # TorchAO created the runtime buffers on the (meta) build device.
            # Re-register them as None so ``_distribute_states`` skips them and
            # ``_init_self_buffers`` materializes them on the real device, per
            # torchtitan's buffer protocol.
            # _sr_seed is a per-rank stochastic-rounding seed: distinct across
            # ranks (SR stays unbiased and NVFP4 never communicates quantized
            # values, so per-rank seeds are correct) and non-persistent (a Philox
            # key needs no checkpointing). Re-register it None so it is not
            # distributed and is re-drawn per rank in _init_self_buffers.
            self.register_buffer("_sr_seed", None, persistent=False)
            self._rht_sign_vector = None
            self._rht_sign_vector_tuple = None
            self._buffer_spec: tuple | None = None

        # -- runtime-buffer accessors (buffers may be replicated DTensors) --

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

        # -- parallelization --

        def parallelize(self, parallel_dims: ParallelDims) -> None:
            if self._sharding_config is not None:
                tp_style = _infer_tp_style(self._sharding_config)
                self._validate(parallel_dims, tp_style)
                self._sharding_config = self._augment_sharding_config(
                    self._sharding_config, tp_style
                )
                self._cache_buffer_spec(parallel_dims)
            super().parallelize(parallel_dims)

        def _augment_sharding_config(
            self, sc: ShardingConfig, tp_style: Literal["colwise", "rowwise"] | None
        ) -> ShardingConfig:
            """Turn the stock colwise/rowwise Linear config into a local_map
            region for the opaque ``nvfp4_linear`` op.

            ``nvfp4_linear`` is opaque to the SPMD type system, so TP runs it on
            local shards inside ``spmd.local_map``: the framework converts the
            activation to local on entry and re-types the output on exit. We add
            the activation's input layout (colwise consumes TP-Replicate; rowwise
            consumes TP-Shard(-1)) and the input-gradient layout (colwise grad is
            TP-Partial -- a per-rank partial sum over the output shard that must
            be all-reduced; rowwise grad stays TP-Shard(-1)). ``out_src`` /
            ``out_dst`` are inherited from colwise_config()/rowwise_config(). The
            runtime buffers are declared replicated so _distribute_states and DCP
            handle them alongside weight/bias.
            """
            if tp_style == "rowwise":
                in_layout = dense_activation_placement(tp=spmd.S(-1))
                in_grad = dense_activation_placement(tp=spmd.S(-1))
            else:
                in_layout = dense_activation_placement(tp=spmd.R)
                in_grad = dense_activation_placement(tp=spmd.P)
            return replace(
                sc,
                state_shardings={
                    **sc.state_shardings,
                    # _sr_seed is a non-persistent per-rank buffer (drawn in
                    # _init_self_buffers), so it is not a distributed state here.
                    "_rht_sign_vector": _replicated_layout(),
                },
                in_src_shardings={**(sc.in_src_shardings or {}), "x": in_layout},
                in_dst_shardings={**(sc.in_dst_shardings or {}), "x": in_layout},
                local_map=LocalMapConfig(in_grad_placements=(in_grad,)),
            )

        def _validate(
            self,
            parallel_dims: ParallelDims,
            tp_style: Literal["colwise", "rowwise"] | None,
        ) -> None:
            if parallel_dims.tp_enabled and parallel_dims.spmd_backend != "spmd_types":
                raise ValueError(
                    "NVFP4 tensor parallelism requires "
                    "parallelism.spmd_backend='spmd_types' (got "
                    f"'{parallel_dims.spmd_backend}'). The NVFP4 GEMM is opaque "
                    "to DTensor; TP runs through the spmd_types local_map path."
                )
            tp = parallel_dims.tp if parallel_dims.tp_enabled else 1
            if tp_style == "rowwise":
                if self.in_features % tp:
                    raise ValueError(
                        f"NVFP4 rowwise TP requires in_features divisible by TP; "
                        f"got in_features={self.in_features}, TP={tp}."
                    )
                local_in, local_out = self.in_features // tp, self.out_features
            else:
                if self.out_features % tp:
                    raise ValueError(
                        f"NVFP4 colwise TP requires out_features divisible by TP; "
                        f"got out_features={self.out_features}, TP={tp}."
                    )
                local_in, local_out = self.in_features, self.out_features // tp
            if local_in % _NVFP4_BLOCK or local_out % _NVFP4_BLOCK:
                raise ValueError(
                    f"NVFP4 requires each local GEMM dim divisible by "
                    f"{_NVFP4_BLOCK}; {tp_style or 'no-TP'} linear "
                    f"in={self.in_features} out={self.out_features} under TP={tp} "
                    f"gives local (in={local_in}, out={local_out})."
                )

        def _cache_buffer_spec(self, parallel_dims: ParallelDims) -> None:
            layout = _replicated_layout()
            mesh = parallel_dims.resolve_mesh(layout.axes())
            self._buffer_spec = (
                None if mesh is None else (mesh, resolve_placements(layout, mesh))
            )

        def _materialize_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
            if self._buffer_spec is None:
                return tensor
            mesh, placements = self._buffer_spec
            return distribute_tensor(tensor, mesh, list(placements))

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            dev = buffer_device or self.weight.device
            # Per-rank seed: a plain local tensor (not distributed), so each rank
            # draws its own. _rht_sign_vector stays replicated.
            self._sr_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
            )
            self._rht_sign_vector = self._materialize_buffer(
                _make_rht_sign_vector(None, device=dev)
            )
            self._refresh_rht_sign_vector_tuple()

        # -- forward --

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Runs on local tensors. Under TP the framework's spmd.local_map has
            # already converted the activation to a local shard and will re-type
            # the output; the weight/buffers are plain locals (spmd_types). Under
            # FSDP-only / single-GPU the activation is already local and the
            # weight is a DTensor that FSDP has all-gathered, so unwrap it.
            weight = (
                self.weight.to_local()
                if isinstance(self.weight, DTensor)
                else self.weight
            )
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            sr_seed = self._local_sr_seed()
            if spmd.is_type_checking():
                # The per-rank seed varies across ranks; declare it Varying so the
                # local type rule for the opaque nvfp4_mm_triton does not treat it
                # as an Invariant, which cannot mix with the R/V data operands.
                spmd.assert_type(
                    sr_seed,
                    {MeshAxisName.DP: spmd.V, MeshAxisName.CP: spmd.V, TP: spmd.V},
                )
            return nvfp4_linear(
                x,
                weight,
                bias,
                sr_seed=sr_seed,
                sign_vector=self.rht_sign_vector,
            )

except ImportError:
    NVFP4Linear = None


def _replicated_layout() -> SpmdLayout:
    """Replicated over every dense mesh axis."""
    return SpmdLayout({MeshAxisName.DP: spmd.R, MeshAxisName.CP: spmd.R, TP: spmd.R})


def _infer_tp_style(
    sharding_config: ShardingConfig | None,
) -> Literal["colwise", "rowwise"] | None:
    """Infer colwise vs rowwise from the weight's declared TP placement.

    The weight placement is torchtitan's canonical colwise/rowwise encoding:
    colwise shards the output dim (Shard(0)), rowwise the input dim (Shard(1)).
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


class NVFP4LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply NVFP4 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted. The LM head must be excluded
        (NVFP4 requires each GEMM dim divisible by 128; vocab is not).
        """

    def __init__(self, config: Config):
        self.config = config

        if NVFP4Linear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 training "
                "prototype. Install a torchao build with "
                "torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert NVFP4Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = NVFP4Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config
