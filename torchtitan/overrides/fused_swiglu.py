# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example override: a SwiGLU feed-forward with a single fused gate+up weight.

This is the worked example referenced in ``torchtitan/config/OVERRIDE.md``. It
demonstrates the pieces a non-trivial fused module needs to plug in via the
override mechanism, *without touching core*:

1. Custom ``__init__`` parametrization — ``w1`` and ``w3`` of the stock
   :class:`FeedForward` are fused into one parameter ``w13`` of shape
   ``(hidden_dim, 2, dim)`` (``w13[:, 0]`` is the gate / stock ``w1``,
   ``w13[:, 1]`` the up / stock ``w3``). The gate+up projection is a single GEMM.
2. Weight initialization via the ``Module`` protocol (``param_init``).
3. A ``sharding_config`` that supports both FSDP and tensor parallelism.
4. Registration via ``@override`` targeting ``FeedForward.Config``.

Tensor parallelism — the ``(hidden_dim, 2, dim)`` layout is what makes TP work.
``w13`` is sharded ``Shard(0)`` on the ``hidden_dim`` axis, so each TP rank holds
``(hidden_dim/tp, 2, dim)`` — a matching slice of *both* the gate and up
projections (the Megatron column-parallel layout for gated MLPs). A flat
``(2*hidden, dim)`` weight has no correct TP sharding (``Shard(0)`` there would
give one rank all of ``w1`` and another all of ``w3``); the explicit ``2`` dim
fixes that. ``hidden_dim`` is also dim 0, so FSDP shards the large axis cleanly
at any degree. The layout is contiguous and transpose-free, so it costs nothing
at compute time: the single GEMM is expressed with ``einsum`` (which contracts
``dim`` and keeps ``hidden`` sharded), and never reshapes across the sharded axis.

NOTE (checkpoint compatibility) -- although the parameter is the fused ``w13``,
this module checkpoints in the stock ``FeedForward`` layout
(``w1.weight`` / ``w3.weight``), so its checkpoints **are** interchangeable with
the non-fused module (and with the HF adapter, which targets the stock layout). A
``register_state_dict_post_hook`` splits ``w13`` into ``w1.weight``/``w3.weight``
on save, and a ``register_load_state_dict_pre_hook`` merges them back into ``w13``
on load (a native ``w13`` key is still accepted for back-compat). Resolves
https://github.com/pytorch/torchtitan/issues/3569. See ``OVERRIDE.md``
"Checkpoint Compatibility".
"""

from collections.abc import Callable
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = ["FusedSwiGLU"]


class FusedSwiGLU(Module):
    """SwiGLU FFN with the gate and up projections fused into one parameter.

    ``w13`` has shape ``(hidden_dim, 2, dim)``: ``w13[:, 0]`` is the gate
    projection (the stock ``w1``) and ``w13[:, 1]`` the up projection (the stock
    ``w3``). A single GEMM computes both; the result is split into gate/up. The
    down projection ``w2`` is reused as-is from the stock config. ``hidden_dim``
    is dim 0, so TP shards it (``Shard(0)``, matching gate/up slices per rank) and
    FSDP shards it cleanly at any degree (see module docstring).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        w2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        # Fused gate+up weight, gate=w13[:, 0], up=w13[:, 1] (see class docstring).
        self.w13 = torch.nn.Parameter(torch.empty(config.hidden_dim, 2, config.dim))
        self.w2 = config.w2.build()
        # Checkpoint in the stock FeedForward layout (w1.weight / w3.weight) so
        # checkpoints interoperate with the non-fused module. See class docstring.
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One GEMM contracting `dim`; keeps the `hidden` axis (TP-sharded),
        # without ever reshaping across the sharded axis.
        gate, up = torch.einsum("...d,hgd->...hg", x, self.w13).unbind(-1)
        return self.w2(F.silu(gate) * up)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Emit the fused ``w13`` as the stock ``w1.weight`` / ``w3.weight``.

        Runs after the default ``state_dict`` (which produced ``{prefix}w13``),
        so the saved checkpoint is in the stock FeedForward layout. ``w13[:, 0]``
        is the gate (stock ``w1``) and ``w13[:, 1]`` the up (stock ``w3``).
        """
        w13 = state_dict.pop(f"{prefix}w13")
        state_dict[f"{prefix}w1.weight"] = w13[:, 0].contiguous()
        state_dict[f"{prefix}w3.weight"] = w13[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Merge stock ``w1.weight`` / ``w3.weight`` back into the fused ``w13``.

        Runs before the default ``_load_from_state_dict``, so it loads ``w13``
        (the real parameter) normally afterwards. A native ``w13`` key is left
        untouched, keeping back-compat with checkpoints saved as ``w13``.
        """
        w1_key, w3_key = f"{prefix}w1.weight", f"{prefix}w3.weight"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)], dim=1
            )


@override(
    "fused_swiglu",
    target=FeedForward.Config,
    description="Fuse SwiGLU gate+up into one weight (FSDP + TP).",
)
def fused_swiglu(cfg: FeedForward.Config) -> FusedSwiGLU.Config:
    # Initialize each half of the fused weight with its own initializer:
    # `w13[:, 0]` is the gate (stock `w1`) and `w13[:, 1]` the up (stock `w3`).
    # In stock FeedForward these differ — `w1` uses the plain linear init while
    # `w3` shares `w2`'s depth-scaled init — so applying one to the whole tensor
    # would mis-initialize the up half. Keyed by the fused param name "w13" for
    # the Module init protocol.
    w1_init = (cfg.w1.param_init or {}).get("weight")
    w3_init = (cfg.w3.param_init or {}).get("weight")
    param_init = None
    if w1_init is not None and w3_init is not None:
        # Rebind to non-Optional locals: type narrowing from the `if` above does
        # not propagate into the nested closure that captures these.
        gate_init: Callable = w1_init
        up_init: Callable = w3_init

        def _init_w13(t: torch.Tensor) -> None:
            gate_init(t[:, 0, :])  # gate (stock w1)
            up_init(t[:, 1, :])  # up (stock w3)

        param_init = {"w13": _init_w13}

    # `derive` copies fields shared by name (e.g. `w2`) structurally; only the
    # genuinely-different fields are passed as deltas.
    fused = derive(
        cfg,
        FusedSwiGLU.Config,
        dim=cfg.w1.in_features,
        hidden_dim=cfg.w1.out_features,
        param_init=param_init,
    )

    # Shard `w13` on the hidden axis (dim 0) under TP — matching gate/up slices
    # per rank. Reuse the original FFN's input-activation redistribution (set on
    # `cfg.sharding_config` by the model's sharding pass) if present; `w2`
    # carries its stock rowwise sharding (copied by `derive`).
    base = cfg.sharding_config
    fused.sharding_config = ShardingConfig(
        state_shardings={"w13": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings=base.in_src_shardings if base is not None else None,
        in_dst_shardings=base.in_dst_shardings if base is not None else None,
    )
    return fused
