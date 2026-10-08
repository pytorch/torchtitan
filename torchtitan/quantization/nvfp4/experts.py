# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 grouped-linear building blocks."""

# Shape suffix legend: R = routed tokens, E = experts, I = input features,
# O = combined output features.

from dataclasses import dataclass
from typing import cast

import torch
from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
    _to_nvfp4_rht_rs_then_scaled_grouped_mm,
)

from torchtitan.protocols.module import Module

from .linear import _HARDCODED_SIGN_VECTOR


_nvfp4_grouped_linear_cache: dict[type, type] = {}


def _get_nvfp4_grouped_linear_cls(parent_cls: type) -> type:
    """Get or create an NVFP4-quantized subclass of a GroupedLinear class."""
    if parent_cls in _nvfp4_grouped_linear_cache:
        return _nvfp4_grouped_linear_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class NVFP4GroupedLinear(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config: Config):
            super().__init__(config)
            module = cast(Module, self)
            module.register_buffer("_sr_seed", None, persistent=False)
            module.register_buffer("_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None

        def _refresh_rht_sign_vector_tuple(self) -> None:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            self._rht_sign_vector_tuple = (
                None
                if sign_vector is None or sign_vector.device.type == "meta"
                else tuple(int(value) for value in sign_vector.tolist())
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

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            super()._init_self_buffers(buffer_device=buffer_device)
            dev = (
                buffer_device
                if buffer_device is not None
                else next(cast(Module, self).parameters()).device
            )
            self._sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            self._rht_sign_vector = torch.tensor(
                _HARDCODED_SIGN_VECTOR, dtype=torch.int8, device=dev
            )
            self._refresh_rht_sign_vector_tuple()

        def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
            return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                input_RI,
                weight_EOI,
                self.rht_sign_vector,
                self._sr_seed,
                offs=offsets_E,
                pad_token_groups_for_grouped_mm=False,
            )

    NVFP4GroupedLinear.__name__ = f"NVFP4{parent_cls.__name__}"
    NVFP4GroupedLinear.__qualname__ = f"NVFP4{parent_cls.__name__}"
    _nvfp4_grouped_linear_cache[parent_cls] = NVFP4GroupedLinear
    return NVFP4GroupedLinear
