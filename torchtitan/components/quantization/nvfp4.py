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
from torch.distributed.tensor import distribute_tensor, DTensor, Partial, Shard

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import spmd_layout_to_dtensor_placements
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import resolve_placements, ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

TP = MeshAxisName.TP

# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

try:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_linear
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        _make_rht_sign_vector,
        _rht_sign_vector_to_tuple,
        NVFP4Linear as TorchAONVFP4Linear,
    )

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
            self._sr_seed = None
            self._rht_sign_vector = None
            self._rht_sign_vector_tuple = None
            self._tp_active = False
            self._tp_style: Literal["colwise", "rowwise"] | None = None
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
            self._tp_active = parallel_dims.tp_enabled
            if self._sharding_config is not None:
                # Declare the runtime buffers (replicated) so _distribute_states
                # and DCP handle them alongside weight/bias.
                sc = self._sharding_config
                self._sharding_config = replace(
                    sc,
                    state_shardings={
                        **sc.state_shardings,
                        "_sr_seed": _replicated_layout(),
                        "_rht_sign_vector": _replicated_layout(),
                    },
                )
                self._tp_style = _infer_tp_style(self._sharding_config)
                self._validate(parallel_dims)
                self._cache_buffer_spec(parallel_dims)
            super().parallelize(parallel_dims)

        def _validate(self, parallel_dims: ParallelDims) -> None:
            if parallel_dims.tp_enabled and parallel_dims.spmd_backend == "spmd_types":
                raise ValueError(
                    "NVFP4 + TP does not support parallelism.spmd_backend="
                    "'spmd_types' yet. The NVFP4 TP bridge wraps outputs as "
                    "DTensors; use the default or full_dtensor backend."
                )
            tp = parallel_dims.tp if parallel_dims.tp_enabled else 1
            if self._tp_style == "rowwise":
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
                    f"{_NVFP4_BLOCK}; {self._tp_style or 'no-TP'} linear "
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
            self._sr_seed = self._materialize_buffer(
                torch.randint(
                    -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
                )
            )
            self._rht_sign_vector = self._materialize_buffer(
                _make_rht_sign_vector(None, device=dev)
            )
            self._refresh_rht_sign_vector_tuple()

        # -- forward --

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self._tp_active:
                # Single-GPU or FSDP-only: FSDP all-gathers the full weight
                # before forward, so a local tensor is expected here.
                weight = (
                    self.weight.to_local()
                    if isinstance(self.weight, DTensor)
                    else self.weight
                )
                bias = (
                    self.bias.to_local()
                    if isinstance(self.bias, DTensor)
                    else self.bias
                )
                return nvfp4_linear(
                    x,
                    weight,
                    bias,
                    sr_seed=self._local_sr_seed(),
                    sign_vector=self.rht_sign_vector,
                )

            # TP: the functional op is opaque to DTensor, so run it on local
            # shards and return the colwise/rowwise placement; the framework's
            # output redistribution does the (bf16) reduction.
            assert isinstance(x, DTensor), (
                f"NVFP4 TP forward requires a DTensor activation; got "
                f"{type(x).__name__}."
            )
            w_local = self.weight.to_local()
            bias_local = self.bias.to_local() if self.bias is not None else None
            # Pin the input-grad placement per TP style (distributed rules require
            # this for to_local). A colwise linear receives a TP-Replicate
            # activation, so each rank's local input-grad is only a partial sum
            # over the output shard and must be all-reduced: grad is Partial on
            # the TP axis. A rowwise linear receives a TP-Shard(-1) activation
            # whose grad stays Shard(-1). Defaulting to x.placements would
            # mislabel the colwise Partial grad as Replicate and skip the
            # reduction, corrupting the gradient into the upstream residual.
            if self._tp_style == "colwise":
                _, x_grad_placements = _swap_tp_placement(x, Partial())
            else:
                x_grad_placements = x.placements
            x_local = x.to_local(grad_placements=x_grad_placements)
            y = nvfp4_linear(
                x_local,
                w_local,
                bias_local,
                sr_seed=self._local_sr_seed(),
                sign_vector=self.rht_sign_vector,
            )
            out_tp = Shard(-1) if self._tp_style == "colwise" else Partial()
            mesh, placements = _swap_tp_placement(x, out_tp)
            return DTensor.from_local(y, mesh, placements, run_check=False)

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


def _swap_tp_placement(dtensor: DTensor, tp_placement) -> tuple:
    """Return (mesh, placements) of ``dtensor`` with the TP axis swapped.

    Activations are replicated over data axes (DP/FSDP) and only the TP axis
    carries the linear's sharding; reusing the input's mesh keeps the output on
    the same 1-D (tp-only) or 2-D (tp+fsdp) mesh as the surrounding activations.
    """
    mesh = dtensor.device_mesh
    names = mesh.mesh_dim_names
    assert (
        names is not None and "tp" in names
    ), f"NVFP4 TP path requires a 'tp' mesh axis, got {names}"
    placements = list(dtensor.placements)
    placements[names.index("tp")] = tp_placement
    return mesh, tuple(placements)


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
